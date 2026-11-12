#Requires -Version 7.0
<#
.SYNOPSIS
  INTEGRADOR-T4: lleva el arbol vivo (disco + indice compartido) de -Desde a -Hasta
  sin pisar trabajo ajeno.

.DESCRIPTION
  Por cada ruta de `git diff --raw Desde Hasta`:
    - disco igual a Desde (o ausente en Desde y en disco) e indice igual a Desde o a Hasta:
      se lleva a Hasta en disco y en el indice compartido;
    - disco ya igual a Hasta: solo se pone al dia el indice si estaba en Desde;
    - disco distinto de Desde y de Hasta, o indice con otra version preparada, o conflicto:
      NO se toca y se lista con su probable dueno.
  Mientras escribe el indice compartido toma `.git/index.lock` (lo crea en exclusiva) y lo
  suelta siempre (try/finally). No usa checkout, reset, stash, clean ni read-tree -u.
  Registro: .agent_tx/diagnostics/sincronizar_arbol_vivo.json.
  Ultimo SHA sincronizado: .agent_tx/diagnostics/arbol_vivo_sha.txt (solo sin -Simular).

.EXAMPLE
  pwsh -File scripts/sincronizar_arbol_vivo.ps1 -Simular
  pwsh -File scripts/sincronizar_arbol_vivo.ps1 -Desde 48eada605 -Hasta HEAD
#>
[CmdletBinding()]
param(
    [string]$Desde,
    [string]$Hasta = 'HEAD',
    [switch]$Simular,
    [string]$Repo = (Get-Location).Path,
    [int]$EsperaLockSegundos = 10,
    # Solo para pruebas: lanza una excepcion tras escribir el disco y antes de instalar el indice.
    [switch]$FallarAMitad
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$CERO = '0000000000000000000000000000000000000000'
$utf8 = [Text.UTF8Encoding]::new($false)

function Invoke-Git {
    param(
        [Parameter(Mandatory)][string[]]$Argumentos,
        [byte[]]$Entrada,
        [hashtable]$Entorno = @{},
        [switch]$PermitirFallo
    )
    $psi = [Diagnostics.ProcessStartInfo]::new('git')
    foreach ($a in $Argumentos) { $psi.ArgumentList.Add($a) }
    $psi.WorkingDirectory = $script:Raiz
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = $utf8
    $psi.StandardErrorEncoding = $utf8
    foreach ($k in $Entorno.Keys) { $psi.Environment[$k] = [string]$Entorno[$k] }
    $p = [Diagnostics.Process]::Start($psi)
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    if ($Entrada) { $p.StandardInput.BaseStream.Write($Entrada, 0, $Entrada.Length) }
    $p.StandardInput.Close()
    $p.WaitForExit()
    $r = [pscustomobject]@{ Codigo = $p.ExitCode; Salida = $out.Result; Error = $err.Result }
    if ($r.Codigo -ne 0 -and -not $PermitirFallo) {
        throw "git $($Argumentos -join ' ') fallo ($($r.Codigo)): $($r.Error.Trim())"
    }
    $r
}

function Get-Sha([string]$rev) {
    (Invoke-Git @('rev-parse', '--verify', '--quiet', "$rev^{commit}")).Salida.Trim()
}

# Hash del disco con los filtros de la ruta (autocrlf, .gitattributes), igual que git status.
# Devuelve hashtable ruta -> sha, $null si no existe, 'DIR' si es una carpeta.
function Get-HashDisco([string[]]$rutas) {
    $res = @{}
    $existentes = [Collections.Generic.List[string]]::new()
    foreach ($r in $rutas) {
        $full = Join-Path $script:Raiz $r
        if (Test-Path -LiteralPath $full -PathType Leaf) { $existentes.Add($r) }
        elseif (Test-Path -LiteralPath $full -PathType Container) { $res[$r] = 'DIR' }
        else { $res[$r] = $null }
    }
    if ($existentes.Count -gt 0) {
        $bytes = $utf8.GetBytes(($existentes -join "`n") + "`n")
        $sal = @((Invoke-Git @('hash-object', '--stdin-paths') -Entrada $bytes).Salida -split "`r?`n" |
                Where-Object { $_ })
        for ($i = 0; $i -lt $existentes.Count; $i++) { $res[$existentes[$i]] = $sal[$i] }
    }
    $res
}

# Entradas del indice: ruta -> @{Modo; Sha; Conflicto}
function Get-Indice([string]$archivoIndice) {
    $res = @{}
    $sal = (Invoke-Git @('ls-files', '-s', '-z') -Entorno @{ GIT_INDEX_FILE = $archivoIndice }).Salida
    foreach ($reg in $sal.Split([char]0)) {
        if (-not $reg) { continue }
        $tab = $reg.IndexOf("`t")
        $meta = $reg.Substring(0, $tab) -split ' '
        $ruta = $reg.Substring($tab + 1)
        if ($meta[2] -ne '0') { $res[$ruta] = @{ Modo = $meta[0]; Sha = $meta[1]; Conflicto = $true }; continue }
        if (-not $res.ContainsKey($ruta)) { $res[$ruta] = @{ Modo = $meta[0]; Sha = $meta[1]; Conflicto = $false } }
    }
    $res
}

$script:AREAS = @(
    @{ Patron = '^crates/(telemetry-server|telemetry-engine|audit-engine|flight-recorder)/'; Dueno = 'Sol (M1)' },
    @{ Patron = '^(crates/(execution-engine|risk-engine)/|tests/host_oms_contract\.rs$)'; Dueno = 'Codex QRA (M2)' },
    @{ Patron = '^crates/(signal-engine|strategy-core|feature-engine|dark-alpha-engine)/'; Dueno = 'AGY' },
    @{ Patron = '^crates/(data-ingest|data-pipeline|storage-engine|backtest-engine)/'; Dueno = 'Codex RB' },
    @{ Patron = '^(crates/god-engine-core/|launchers/|config_dir/|LAUNCH_[^/]*\.bat$)'; Dueno = 'ZCode (M3)' },
    @{ Patron = '^(Cargo\.toml$|Cargo\.lock$|\.cargo/|\.github/|scripts/|AGENT_SYNC\.md$|\.agents/)'; Dueno = 'Integrador' }
)
function Get-Dueno([string]$ruta, [string]$hasta) {
    $area = 'N/D'
    foreach ($a in $script:AREAS) { if ($ruta -match $a.Patron) { $area = $a.Dueno; break } }
    $ult = (Invoke-Git @('log', '-1', '--format=%h %s', $hasta, '--', $ruta) -Entorno @{ GIT_LITERAL_PATHSPECS = '1' } -PermitirFallo).Salida.Trim()
    "$area; ultimo commit: $(if ($ult) { $ult } else { 'N/D' })"
}

function Get-Clasificacion($e, $disco, $indice) {
    # $e: entrada del diff (ModoA, ShaA, ModoB, ShaB, Ruta). Devuelve accion + motivo.
    $d = $disco[$e.Ruta]
    $i = $indice[$e.Ruta]
    if ($e.ModoA -in '160000', '120000' -or $e.ModoB -in '160000', '120000') {
        return @{ Accion = 'pendiente'; Motivo = 'submodulo o enlace simbolico: no lo maneja la regla' }
    }
    if ($i -and $i.Conflicto) { return @{ Accion = 'pendiente'; Motivo = 'indice compartido con conflicto sin resolver' } }
    if ($d -eq 'DIR') { return @{ Accion = 'pendiente'; Motivo = 'en disco es una carpeta' } }
    $iSha = if ($i) { $i.Sha } else { $null }
    $iModo = if ($i) { $i.Modo } else { $null }
    if ($d -eq $e.ShaB) { $disco_ = 'ok' }
    elseif ($d -eq $e.ShaA) { $disco_ = 'actualizar' }
    else { return @{ Accion = 'pendiente'; Motivo = "disco distinto de Desde y de Hasta (disco=$(if ($d) { $d.Substring(0, 9) } else { 'ausente' }))" } }
    if ($iSha -eq $e.ShaB -and ($null -eq $e.ShaB -or $iModo -eq $e.ModoB)) { $indice_ = 'ok' }
    elseif ($iSha -eq $e.ShaA -or $iSha -eq $e.ShaB) { $indice_ = 'actualizar' }
    else { return @{ Accion = 'pendiente'; Motivo = "indice compartido con otra version preparada (indice=$(if ($iSha) { $iSha.Substring(0, 9) } else { 'ausente' }))" } }
    if ($disco_ -eq 'ok' -and $indice_ -eq 'ok') { return @{ Accion = 'al_dia'; Motivo = '' } }
    @{ Accion = 'actualizar'; Disco = ($disco_ -eq 'actualizar'); Indice = ($indice_ -eq 'actualizar'); Motivo = '' }
}

# ---------------------------------------------------------------------------------------------
$script:Raiz = (& git -C $Repo rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw "No es un repo git: $Repo" }
$script:Raiz = [IO.Path]::GetFullPath($script:Raiz)
$gitIndice = [IO.Path]::GetFullPath((Join-Path $script:Raiz (Invoke-Git @('rev-parse', '--git-path', 'index')).Salida.Trim()))
$diag = Join-Path $script:Raiz '.agent_tx/diagnostics'
$archivoSha = Join-Path $diag 'arbol_vivo_sha.txt'
$archivoLog = Join-Path $diag 'sincronizar_arbol_vivo.json'

if (-not $Desde) {
    if (-not (Test-Path -LiteralPath $archivoSha)) {
        throw "Falta -Desde y no existe $archivoSha (ultimo SHA sincronizado)."
    }
    $Desde = (Get-Content -LiteralPath $archivoSha -TotalCount 1).Trim()
}
$shaDesde = Get-Sha $Desde
$shaHasta = Get-Sha $Hasta
if (-not $shaDesde -or -not $shaHasta) { throw "No resuelvo Desde=$Desde o Hasta=$Hasta" }

$raw = (Invoke-Git @('diff', '--raw', '-z', '--no-renames', '--no-abbrev', $shaDesde, $shaHasta)).Salida
$tok = $raw.Split([char]0)
$entradas = [Collections.Generic.List[object]]::new()
for ($k = 0; $k + 1 -lt $tok.Length; $k += 2) {
    if (-not $tok[$k]) { break }
    $m = $tok[$k].TrimStart(':') -split ' '
    $entradas.Add([pscustomobject]@{
            ModoA = $m[0]; ModoB = $m[1]
            ShaA = $(if ($m[2] -eq $CERO) { $null } else { $m[2] })
            ShaB = $(if ($m[3] -eq $CERO) { $null } else { $m[3] })
            Estado = $m[4]; Ruta = $tok[$k + 1]
        })
}

$inicio = Get-Date
$disco = Get-HashDisco ($entradas | ForEach-Object Ruta)
$indice = Get-Indice $gitIndice
$plan = foreach ($e in $entradas) {
    $c = Get-Clasificacion $e $disco $indice
    [pscustomobject]@{ Entrada = $e; Clase = $c }
}

$resultado = [ordered]@{
    fecha = $inicio.ToString('o'); desde = $shaDesde; hasta = $shaHasta; simular = [bool]$Simular
    rutas_en_diff = $entradas.Count; actualizadas_disco = @(); actualizadas_indice = @()
    al_dia = 0; pendientes = @(); errores = @(); lock = 'no usado'
}
$resultado.al_dia = @($plan | Where-Object { $_.Clase.Accion -eq 'al_dia' }).Count
$aActualizar = @($plan | Where-Object { $_.Clase.Accion -eq 'actualizar' })
$pend = @($plan | Where-Object { $_.Clase.Accion -eq 'pendiente' })

$lockPath = "$gitIndice.lock"
$tmpIndice = "$gitIndice.sincronizar.tmp"
$lockMio = $false
$fs = $null
try {
    if (-not $Simular -and $aActualizar.Count -gt 0) {
        $limite = (Get-Date).AddSeconds($EsperaLockSegundos)
        while (-not $lockMio) {
            try { $fs = [IO.File]::Open($lockPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None); $lockMio = $true }
            catch [IO.IOException] {
                if ((Get-Date) -gt $limite) { $resultado.lock = 'ocupado'; throw "Indice compartido bloqueado ($lockPath existe) tras $EsperaLockSegundos s; no toco nada." }
                Start-Sleep -Milliseconds 250
            }
        }
        $resultado.lock = 'tomado'
        Copy-Item -LiteralPath $gitIndice -Destination $tmpIndice -Force

        # Con el lock tomado, recomprobar disco e indice: alguien pudo cambiarlos tras el plan.
        $disco2 = Get-HashDisco ($aActualizar | ForEach-Object { $_.Entrada.Ruta })
        $indice2 = Get-Indice $tmpIndice
        $listos = foreach ($x in $aActualizar) {
            $c2 = Get-Clasificacion $x.Entrada $disco2 $indice2
            if ($c2.Accion -eq 'actualizar') { [pscustomobject]@{ Entrada = $x.Entrada; Clase = $c2 } }
            elseif ($c2.Accion -eq 'pendiente') { $pend += [pscustomobject]@{ Entrada = $x.Entrada; Clase = @{ Accion = 'pendiente'; Motivo = "cambio durante la sincronizacion: $($c2.Motivo)" } } }
        }
        $listos = @($listos)

        # 1) Disco: escribir las de Hasta con git restore (indice temporal; el compartido esta bloqueado).
        $escribir = @($listos | Where-Object { $_.Clase.Disco -and $_.Entrada.ShaB })
        if ($escribir.Count -gt 0) {
            $specs = Join-Path $diag 'sincronizar_arbol_vivo.pathspec.tmp'
            New-Item -ItemType Directory -Force -Path $diag | Out-Null
            [IO.File]::WriteAllText($specs, (($escribir | ForEach-Object { $_.Entrada.Ruta + [char]0 }) -join ''), $utf8)
            try {
                Invoke-Git @('restore', "--source=$shaHasta", '--worktree', "--pathspec-from-file=$specs", '--pathspec-file-nul') `
                    -Entorno @{ GIT_INDEX_FILE = $tmpIndice; GIT_LITERAL_PATHSPECS = '1' } | Out-Null
            } finally { Remove-Item -LiteralPath $specs -Force -ErrorAction SilentlyContinue }
        }
        foreach ($x in @($listos | Where-Object { $_.Clase.Disco -and -not $_.Entrada.ShaB })) {
            $full = Join-Path $script:Raiz $x.Entrada.Ruta
            Remove-Item -LiteralPath $full -Force
            $dir = Split-Path -Parent $full
            while ($dir.Length -gt $script:Raiz.Length -and (Test-Path -LiteralPath $dir) -and -not (Get-ChildItem -LiteralPath $dir -Force | Select-Object -First 1)) {
                Remove-Item -LiteralPath $dir -Force; $dir = Split-Path -Parent $dir
            }
        }
        $verif = Get-HashDisco ($listos | Where-Object { $_.Clase.Disco } | ForEach-Object { $_.Entrada.Ruta })
        foreach ($x in @($listos | Where-Object { $_.Clase.Disco })) {
            if ($verif[$x.Entrada.Ruta] -ne $x.Entrada.ShaB) { $resultado.errores += "disco no quedo en Hasta: $($x.Entrada.Ruta)" }
            else { $resultado.actualizadas_disco += $x.Entrada.Ruta }
        }
        if ($FallarAMitad) { throw 'FallarAMitad: fallo simulado antes de instalar el indice' }

        # 2) Indice compartido: preparar en la copia y luego instalar por el lock (como hace git).
        $reg = [Text.StringBuilder]::new()
        foreach ($x in $listos) {
            $e = $x.Entrada
            if ($e.ShaB) { [void]$reg.Append("$($e.ModoB) $($e.ShaB)`t$($e.Ruta)").Append([char]0) }
            else { [void]$reg.Append("0 $CERO`t$($e.Ruta)").Append([char]0) }
        }
        if ($reg.Length -gt 0) {
            Invoke-Git @('update-index', '-z', '--index-info') -Entrada $utf8.GetBytes($reg.ToString()) `
                -Entorno @{ GIT_INDEX_FILE = $tmpIndice } | Out-Null
        }
        $bytes = [IO.File]::ReadAllBytes($tmpIndice)
        $fs.Write($bytes, 0, $bytes.Length); $fs.Flush($true); $fs.Dispose(); $fs = $null
        # Un lector con el indice abierto puede impedir el reemplazo un instante: reintentar.
        for ($n = 1; ; $n++) {
            try { [IO.File]::Move($lockPath, $gitIndice, $true); break }
            catch [IO.IOException], [UnauthorizedAccessException] { if ($n -ge 20) { throw }; Start-Sleep -Milliseconds 100 }
        }
        $lockMio = $false
        $resultado.lock = 'soltado'
        $resultado.actualizadas_indice = @($listos | ForEach-Object { $_.Entrada.Ruta })
    }
} catch {
    $resultado.errores += $_.Exception.Message
    throw
} finally {
    if ($fs) { $fs.Dispose() }
    if ($lockMio -and (Test-Path -LiteralPath $lockPath)) { Remove-Item -LiteralPath $lockPath -Force; $resultado.lock = 'soltado tras fallo' }
    Remove-Item -LiteralPath $tmpIndice -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath "$tmpIndice.lock" -Force -ErrorAction SilentlyContinue

    $resultado.pendientes = @($pend | ForEach-Object {
            [ordered]@{ ruta = $_.Entrada.Ruta; motivo = $_.Clase.Motivo; dueno = (Get-Dueno $_.Entrada.Ruta $shaHasta) }
        })
    if ($Simular) {
        $resultado['se_actualizarian'] = @($aActualizar | ForEach-Object {
                [ordered]@{ ruta = $_.Entrada.Ruta; disco = [bool]$_.Clase.Disco; indice = [bool]$_.Clase.Indice }
            })
    }
    $resultado['segundos'] = [math]::Round(((Get-Date) - $inicio).TotalSeconds, 2)
    New-Item -ItemType Directory -Force -Path $diag | Out-Null
    $hist = @()
    if (Test-Path -LiteralPath $archivoLog) {
        try { $prev = Get-Content -LiteralPath $archivoLog -Raw | ConvertFrom-Json; if ($prev.PSObject.Properties['historial']) { $hist = @($prev.historial) } } catch { }
    }
    $hist = @([ordered]@{ fecha = $resultado.fecha; desde = $shaDesde; hasta = $shaHasta; simular = [bool]$Simular
            disco = $resultado.actualizadas_disco.Count; indice = $resultado.actualizadas_indice.Count
            pendientes = $resultado.pendientes.Count; errores = $resultado.errores.Count
        }) + $hist | Select-Object -First 50
    $resultado['historial'] = $hist
    [IO.File]::WriteAllText($archivoLog, ($resultado | ConvertTo-Json -Depth 6), $utf8)
}

if (-not $Simular -and $resultado.errores.Count -eq 0) {
    [IO.File]::WriteAllText($archivoSha, "$shaHasta`n", $utf8)
}

$modo = if ($Simular) { 'SIMULAR' } else { 'APLICADO' }
Write-Host "sincronizar_arbol_vivo [$modo] $($shaDesde.Substring(0,9))..$($shaHasta.Substring(0,9)): $($entradas.Count) rutas en el diff, $($resultado.al_dia) al dia"
if ($Simular) {
    foreach ($x in $aActualizar) { Write-Host ("  actualizaria  disco={0,-5} indice={1,-5} {2}" -f [bool]$x.Clase.Disco, [bool]$x.Clase.Indice, $x.Entrada.Ruta) }
} else {
    Write-Host "  disco actualizado: $($resultado.actualizadas_disco.Count); indice actualizado: $($resultado.actualizadas_indice.Count)"
}
foreach ($p in $resultado.pendientes) { Write-Host "  PENDIENTE $($p.ruta) -- $($p.motivo) -- $($p.dueno)" }
foreach ($er in $resultado.errores) { Write-Host "  ERROR $er" }
Write-Host "  registro: $archivoLog"
if ($resultado.errores.Count -gt 0) { exit 1 }
