#Requires -Version 7.0
<#
.SYNOPSIS
  INTEGRADOR-T4: cola de integracion. Fusiona -Rama en main segun el mandato 9bdf.

.DESCRIPTION
  1. `git merge-tree --write-tree` de la rama sobre refs/heads/main. Si hay conflictos, para y los lista.
  2. Diff del arbol fusionado contra cada padre (se guarda en .agent_tx/diagnostics/integrar_rama/).
  3. Worktree temporal con ese arbol exacto: `cargo check --workspace --all-targets --locked` y
     `cargo test --locked -p <crates que cambian>`, con el target/ compartido (CARGO_TARGET_DIR).
  4. Commit con los dos padres y CAS de refs/heads/main (solo si main no cambio).
  5. scripts/sincronizar_arbol_vivo.ps1 para llevar el arbol vivo al nuevo main.
  Con -Simular no crea commit ni toca refs (si compila, salvo -SinCompilar).
  Sin push. Registro: .agent_tx/diagnostics/integrar_rama.json.

.EXAMPLE
  pwsh -File scripts/integrar_rama.ps1 -Rama agy/m3-warmstart-funnel -Simular -SinCompilar
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Rama,
    [switch]$Simular,
    [switch]$SinCompilar,
    [string]$Repo = (Get-Location).Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8 = [Text.UTF8Encoding]::new($false)

function G {
    $o = & git @args 2>&1
    $err = @($o | Where-Object { $_ -is [Management.Automation.ErrorRecord] })
    if ($LASTEXITCODE -ne 0) { throw "git $args fallo ($LASTEXITCODE): $err" }
    $o | Where-Object { $_ -isnot [Management.Automation.ErrorRecord] }
}

$Raiz = [IO.Path]::GetFullPath((& git -C $Repo rev-parse --show-toplevel).Trim())
Push-Location $Raiz
$sello = Get-Date -Format 'yyyyMMdd_HHmmss'
$diag = Join-Path $Raiz '.agent_tx/diagnostics'
$dirRun = Join-Path $diag ("integrar_rama/{0}_{1}" -f $sello, ($Rama -replace '[^A-Za-z0-9._-]', '_'))
New-Item -ItemType Directory -Force -Path $dirRun | Out-Null
$reg = [ordered]@{ fecha = (Get-Date).ToString('o'); rama = $Rama; simular = [bool]$Simular; sin_compilar = [bool]$SinCompilar
    main = $null; rama_sha = $null; arbol = $null; conflictos = @(); crates = @(); paquetes = @(); pasos = [ordered]@{}
    commit = $null; resultado = $null; carpeta = $dirRun
}
$script:codigo = 0
function Invoke-Integracion {
$wt = $null
try {
    $H = (G rev-parse --verify 'refs/heads/main^{commit}').Trim()
    $R = (G rev-parse --verify "$Rama^{commit}").Trim()
    $reg.main = $H; $reg.rama_sha = $R
    & git merge-base --is-ancestor $R $H
    if ($LASTEXITCODE -eq 0) { $reg.resultado = 'ya integrada (la rama es ancestro de main)'; return }

    # 1) Fusion en memoria
    $mt = @(& git merge-tree --write-tree --name-only --no-messages $H $R 2>&1)
    $rcMt = $LASTEXITCODE
    if ($rcMt -gt 1) { throw "merge-tree fallo ($rcMt): $mt" }
    $tree = "$($mt[0])".Trim()
    $reg.arbol = $tree
    if ($rcMt -eq 1) {
        $reg.conflictos = @($mt | Select-Object -Skip 1 | ForEach-Object { "$_".Trim() } | Where-Object { $_ } | Sort-Object -Unique)
        $reg.resultado = 'conflictos'; $script:codigo = 2
        Write-Host "CONFLICTOS al fusionar $Rama ($($R.Substring(0,9))) sobre main ($($H.Substring(0,9))):"
        $reg.conflictos | ForEach-Object { Write-Host "  $_" }
        return
    }

    # 2) Diff contra cada padre
    $cambios = @(G diff --name-only $H $tree)
    G diff --stat=200 $H $tree | Set-Content -Encoding utf8 (Join-Path $dirRun 'diff_vs_main.stat.txt')
    G diff --stat=200 $R $tree | Set-Content -Encoding utf8 (Join-Path $dirRun 'diff_vs_rama.stat.txt')
    & git diff $H $tree | Set-Content -Encoding utf8 (Join-Path $dirRun 'diff_vs_main.patch')
    & git diff $R $tree | Set-Content -Encoding utf8 (Join-Path $dirRun 'diff_vs_rama.patch')
    $reg.pasos['diff_vs_main'] = (G diff --shortstat $H $tree | Out-String).Trim()
    $reg.pasos['diff_vs_rama'] = (G diff --shortstat $R $tree | Out-String).Trim()
    $crates = @($cambios | Where-Object { $_ -match '^crates/([^/]+)/' } | ForEach-Object { $Matches[1] } | Sort-Object -Unique)
    $reg.crates = $crates
    $paquetes = foreach ($c in $crates) {
        $toml = & git show "${tree}:crates/$c/Cargo.toml" 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
        $enPackage = $false
        foreach ($l in $toml) {
            if ($l -match '^\s*\[package\]') { $enPackage = $true; continue }
            if ($l -match '^\s*\[') { $enPackage = $false }
            if ($enPackage -and $l -match '^\s*name\s*=\s*"([^"]+)"') { $Matches[1]; break }
        }
    }
    $reg.paquetes = @($paquetes)
    Write-Host "Fusion limpia de $Rama ($($R.Substring(0,9))) sobre main ($($H.Substring(0,9))): arbol $($tree.Substring(0,9))"
    Write-Host "  contra main: $($reg.pasos['diff_vs_main'])"
    Write-Host "  contra rama: $($reg.pasos['diff_vs_rama'])"
    Write-Host "  crates que cambian: $(if ($crates) { $crates -join ', ' } else { 'ninguno' })"
    $raizCambia = @($cambios | Where-Object { $_ -match '^(Cargo\.toml|Cargo\.lock|\.cargo/|src/|tests/)' })
    if ($raizCambia) { Write-Host "  tambien cambia la raiz: $($raizCambia -join ', ')" }

    # 3) Validacion sobre el arbol exacto
    if (-not $SinCompilar) {
        $wt = Join-Path ([IO.Path]::GetTempPath()) "integrar_rama_$sello"
        G worktree add --detach --no-checkout $wt $H | Out-Null
        Push-Location $wt
        try {
            G read-tree --reset -u $tree | Out-Null
            $env:CARGO_TARGET_DIR = Join-Path $Raiz 'target'
            $log = Join-Path $dirRun 'cargo_check.log'
            $t = Measure-Command { & cargo check --workspace --all-targets --locked *> $log }
            $reg.pasos['cargo_check'] = @{ codigo = $LASTEXITCODE; segundos = [math]::Round($t.TotalSeconds) }
            Write-Host "  cargo check --workspace --all-targets: exit $LASTEXITCODE en $([math]::Round($t.TotalSeconds)) s"
            if ($LASTEXITCODE -ne 0) { $reg.resultado = 'cargo check fallo'; $script:codigo = 4; return }
            if ($reg.paquetes.Count -gt 0) {
                $pargs = @($reg.paquetes | ForEach-Object { '-p'; $_ })
                $log = Join-Path $dirRun 'cargo_test.log'
                $t = Measure-Command { & cargo test --locked @pargs *> $log }
                $reg.pasos['cargo_test'] = @{ codigo = $LASTEXITCODE; segundos = [math]::Round($t.TotalSeconds) }
                Write-Host "  cargo test $($pargs -join ' '): exit $LASTEXITCODE en $([math]::Round($t.TotalSeconds)) s"
                if ($LASTEXITCODE -ne 0) { $reg.resultado = 'cargo test fallo'; $script:codigo = 4; return }
            }
            if ((G write-tree).Trim() -ne $tree) { throw 'el worktree de validacion ya no coincide con el arbol fusionado' }
        } finally {
            Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
            Pop-Location
        }
    }

    if ($Simular) { $reg.resultado = 'simulado: fusion limpia' + $(if ($SinCompilar) { ' (sin compilar)' } else { ', check y pruebas en verde' }); return }

    # 4) Commit con dos padres y CAS
    $msg = "INTEGRAR: $Rama ($($R.Substring(0,9))) en main`n`nmerge-tree limpio sobre $($H.Substring(0,9)); check --all-targets y pruebas de: $($reg.paquetes -join ', ')`nRegistro: $($dirRun.Substring($Raiz.Length + 1))"
    $C = (G commit-tree $tree -p $H -p $R -m $msg).Trim()
    & git update-ref -m "integrar_rama: $Rama" refs/heads/main $C $H
    if ($LASTEXITCODE -ne 0) {
        $reg.resultado = "CAS fallo: main avanzo desde $($H.Substring(0,9)); vuelve a correr para recomponer y validar"
        $script:codigo = 3; return
    }
    $reg.commit = $C
    Write-Host "  main: $($H.Substring(0,9)) -> $($C.Substring(0,9)) (CAS ok, sin push)"

    # 5) Arbol vivo al dia
    $sinc = Join-Path $PSScriptRoot 'sincronizar_arbol_vivo.ps1'
    $archivoSha = Join-Path $diag 'arbol_vivo_sha.txt'
    $ultimo = if (Test-Path -LiteralPath $archivoSha) { (Get-Content -LiteralPath $archivoSha -TotalCount 1).Trim() } else { $null }
    if ($ultimo -and $ultimo -ne $H) {
        & pwsh -NoProfile -File $sinc -Repo $Raiz -Desde $ultimo -Hasta $H
        $reg.pasos['sincronizar_previo'] = $LASTEXITCODE
    }
    & pwsh -NoProfile -File $sinc -Repo $Raiz -Desde $H -Hasta $C
    $reg.pasos['sincronizar'] = $LASTEXITCODE
    $reg.resultado = 'integrada'
} catch {
    $reg.resultado = "error: $($_.Exception.Message)"; $script:codigo = 1
    Write-Host "ERROR: $($_.Exception.Message)"
} finally {
    if ($wt) {
        # El indice del worktree temporal tiene el arbol fusionado y su HEAD es main: volverlo a HEAD
        # (solo ese worktree propio) para que `worktree remove` no lo vea modificado; sin --force.
        if (Test-Path -LiteralPath $wt) { & git -C $wt read-tree --reset -u HEAD 2>&1 | Out-Null }
        $salWt = & git worktree remove $wt 2>&1
        if ($LASTEXITCODE -ne 0) { Write-Host "  AVISO: no pude retirar el worktree temporal $wt ($salWt); retiralo a mano tras revisarlo." }
    }
    Pop-Location
    [IO.File]::WriteAllText((Join-Path $diag 'integrar_rama.json'), ($reg | ConvertTo-Json -Depth 5), $utf8)
    Write-Host "Resultado: $($reg.resultado). Registro: $dirRun"
}
}
Invoke-Integracion
exit $script:codigo
