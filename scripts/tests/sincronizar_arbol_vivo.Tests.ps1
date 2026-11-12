#Requires -Version 7.0
<#
  INTEGRADOR-T4: pruebas de scripts/sincronizar_arbol_vivo.ps1 en un repo git temporal (fuera del repo real).
  Script de asserts sin Pester (solo esta Pester 3.4, el de serie de Windows): sale con codigo 1 si algo falla.
  Uso: pwsh -NoProfile -File scripts/tests/sincronizar_arbol_vivo.Tests.ps1
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\sincronizar_arbol_vivo.ps1'))
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("sincronizar_test_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
$utf8 = [Text.UTF8Encoding]::new($false)
$fallos = [Collections.Generic.List[string]]::new()
$pasadas = 0

function Afirmar([bool]$cond, [string]$que) {
    if ($cond) { $script:pasadas++; Write-Host "  ok   $que" } else { $script:fallos.Add($que); Write-Host "  FALLO $que" }
}
function G {
    $o = & git @args 2>&1
    $err = @($o | Where-Object { $_ -is [Management.Automation.ErrorRecord] })
    if ($LASTEXITCODE -ne 0) { throw "git $args : $err" }
    $o | Where-Object { $_ -isnot [Management.Automation.ErrorRecord] }
}
function Escribir([string]$ruta, [string]$txt) {
    $full = Join-Path $tmp $ruta
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $full) | Out-Null
    [IO.File]::WriteAllText($full, $txt, $utf8)
}
function Leer([string]$ruta) { [IO.File]::ReadAllText((Join-Path $tmp $ruta)).Replace("`r`n", "`n") }
function Blob([string]$txt) { $txt | Set-Content -NoNewline -LiteralPath (Join-Path $tmp '.git/blob.tmp'); (G hash-object -w --path x.txt (Join-Path $tmp '.git/blob.tmp')).Trim() }
function IndiceSha([string]$ruta) { $l = G ls-files -s -- ":(literal)$ruta"; if ($l) { ("$l" -split '\s')[1] } else { $null } }
function CommitPrivado([string]$padre, [hashtable]$cambios) {
    # Commit con indice privado (como los agentes): no toca disco ni indice compartido.
    $env:GIT_INDEX_FILE = Join-Path $tmp '.git/index_prueba'
    try {
        G read-tree $padre | Out-Null
        foreach ($k in $cambios.Keys) {
            if ($null -eq $cambios[$k]) { G update-index --force-remove -- $k | Out-Null }
            else { G update-index --add --cacheinfo "100644,$(Blob $cambios[$k]),$k" | Out-Null }
        }
        $arbol = (G write-tree).Trim()
    } finally { Remove-Item Env:GIT_INDEX_FILE }
    $c = (G commit-tree $arbol -p $padre -m "sintetico").Trim()
    G update-ref refs/heads/main $c $padre | Out-Null
    $c
}
function Correr([string[]]$extra) {
    & pwsh -NoProfile -File $script -Repo $tmp @extra *> (Join-Path $tmp '.git/ultima_corrida.log')
    $LASTEXITCODE
}

try {
    New-Item -ItemType Directory -Path $tmp | Out-Null
    Push-Location $tmp
    G init -q -b main | Out-Null
    G config user.email prueba@local | Out-Null; G config user.name prueba | Out-Null
    G config core.autocrlf true | Out-Null

    Escribir 'igual.txt' "v1`n"
    Escribir 'editado.txt' "v1`n"
    Escribir 'borrar.txt' "v1`n"
    Escribir 'con espacio/archivo con espacio.txt' "v1`n"
    Escribir 'indice_ajeno.txt' "v1`n"
    Escribir 'ya_en_disco.txt' "v1`n"
    G add -A | Out-Null
    G commit -q -m c1 | Out-Null
    $c1 = (G rev-parse HEAD).Trim()

    $c2 = CommitPrivado $c1 @{
        'igual.txt' = "v2`n"; 'editado.txt' = "v2`n"; 'borrar.txt' = $null
        'con espacio/archivo con espacio.txt' = "v2`n"; 'indice_ajeno.txt' = "v2`n"
        'ya_en_disco.txt' = "v2`n"
        'nuevo.txt' = "nuevo`n"; 'dir nuevo/otro archivo.txt' = "nuevo`n"; 'colision.txt' = "de main`n"
    }
    # Estado ajeno simulado: edicion a mano, version preparada en el indice compartido, archivo sin
    # seguimiento que choca con uno nuevo de Hasta, y una ruta que su agente ya dejo en Hasta en disco.
    Escribir 'editado.txt' "a mano`n"
    Escribir 'indice_ajeno.txt' "preparado por otro`n"; G add -- indice_ajeno.txt | Out-Null; Escribir 'indice_ajeno.txt' "v1`n"
    Escribir 'colision.txt' "sin seguimiento`n"
    Escribir 'ya_en_disco.txt' "v2`n"
    $idxAjeno = IndiceSha 'indice_ajeno.txt'

    Write-Host "1) -Simular no cambia nada"
    $hIdx = (Get-FileHash .git/index).Hash
    $rc = Correr @('-Desde', $c1, '-Hasta', $c2, '-Simular')
    Afirmar ($rc -eq 0) "simular sale con 0"
    Afirmar ((Get-FileHash .git/index).Hash -eq $hIdx) "simular no escribe el indice"
    Afirmar ((Leer 'igual.txt') -eq "v1`n") "simular no escribe el disco"
    Afirmar (-not (Test-Path .agent_tx/diagnostics/arbol_vivo_sha.txt)) "simular no guarda el SHA"
    $j = Get-Content .agent_tx/diagnostics/sincronizar_arbol_vivo.json -Raw | ConvertFrom-Json
    Afirmar ($j.simular -and @($j.se_actualizarian).Count -eq 6) "simular registra 6 rutas a actualizar"

    Write-Host "2) aplicar"
    $rc = Correr @('-Desde', $c1, '-Hasta', $c2)
    Afirmar ($rc -eq 0) "aplicar sale con 0"
    Afirmar ((Leer 'igual.txt') -eq "v2`n") "ruta igual a Desde se actualiza en disco"
    Afirmar ((IndiceSha 'igual.txt') -eq (G rev-parse "${c2}:igual.txt").Trim()) "ruta igual a Desde se actualiza en el indice"
    Afirmar ((Leer 'editado.txt') -eq "a mano`n") "ruta editada a mano no se toca en disco"
    Afirmar ((IndiceSha 'editado.txt') -eq (G rev-parse "${c1}:editado.txt").Trim()) "ruta editada a mano no se toca en el indice"
    Afirmar (-not (Test-Path 'borrar.txt')) "ruta borrada en Hasta se borra del disco"
    Afirmar ($null -eq (IndiceSha 'borrar.txt')) "ruta borrada en Hasta sale del indice"
    Afirmar ((Leer 'nuevo.txt') -eq "nuevo`n") "ruta creada en Hasta aparece en disco"
    Afirmar ((Leer 'dir nuevo/otro archivo.txt') -eq "nuevo`n") "ruta creada con espacios en carpeta nueva"
    Afirmar ((Leer 'con espacio/archivo con espacio.txt') -eq "v2`n") "ruta con espacios se actualiza"
    Afirmar ((IndiceSha 'dir nuevo/otro archivo.txt') -eq (G rev-parse "${c2}:dir nuevo/otro archivo.txt").Trim()) "ruta nueva con espacios entra al indice"
    Afirmar ((Leer 'indice_ajeno.txt') -eq "v1`n" -and (IndiceSha 'indice_ajeno.txt') -eq $idxAjeno) "indice con version ajena no se toca"
    Afirmar ((Leer 'colision.txt') -eq "sin seguimiento`n" -and $null -eq (IndiceSha 'colision.txt')) "archivo sin seguimiento distinto no se pisa"
    Afirmar ((IndiceSha 'ya_en_disco.txt') -eq (G rev-parse "${c2}:ya_en_disco.txt").Trim()) "disco ya en Hasta: solo se pone al dia el indice"
    Afirmar (-not (Test-Path .git/index.lock)) "lock soltado tras exito"
    Afirmar ((Get-Content .agent_tx/diagnostics/arbol_vivo_sha.txt -TotalCount 1).Trim() -eq $c2) "guarda el SHA sincronizado"
    $j = Get-Content .agent_tx/diagnostics/sincronizar_arbol_vivo.json -Raw | ConvertFrom-Json
    $pend = @($j.pendientes | ForEach-Object ruta) | Sort-Object
    Afirmar (($pend -join ',') -eq 'colision.txt,editado.txt,indice_ajeno.txt') "pendientes listados: $($pend -join ',')"
    # Lo no tocado sigue atrasado: colision.txt borrada en el indice (esta en HEAD) y editado.txt con c1 en el indice.
    $st = @(G status --porcelain -uno) | Sort-Object { $_.Substring(3) }
    Afirmar (($st -join '|') -eq 'D  colision.txt|MM editado.txt|MM indice_ajeno.txt') "git status solo muestra lo ajeno: $($st -join '|')"

    Write-Host "3) el lock se suelta si falla a mitad"
    $c3 = CommitPrivado $c2 @{ 'igual.txt' = "v3`n" }
    $hIdx = (Get-FileHash .git/index).Hash
    $rc = Correr @('-Hasta', $c3, '-FallarAMitad')
    Afirmar ($rc -ne 0) "fallo a mitad sale con codigo distinto de 0"
    Afirmar (-not (Test-Path .git/index.lock)) "lock soltado tras fallo"
    Afirmar ((Get-FileHash .git/index).Hash -eq $hIdx) "indice compartido intacto tras fallo"
    Afirmar ((Get-Content .agent_tx/diagnostics/arbol_vivo_sha.txt -TotalCount 1).Trim() -eq $c2) "SHA sincronizado no avanza tras fallo"

    Write-Host "4) lock ajeno: no se toca nada ni se borra el lock ajeno"
    Escribir '.git/index.lock' 'ajeno'
    $rc = Correr @('-Hasta', $c3, '-EsperaLockSegundos', '1')
    Afirmar ($rc -ne 0) "lock ajeno: sale con codigo distinto de 0"
    Afirmar ((Test-Path .git/index.lock) -and (Get-Content .git/index.lock -Raw) -eq 'ajeno') "lock ajeno sigue en su sitio"
    Remove-Item .git/index.lock

    Write-Host "5) Desde por defecto = ultimo SHA sincronizado"
    $rc = Correr @('-Hasta', $c3)
    if ($rc -ne 0) { Get-Content .git/ultima_corrida.log | Write-Host }
    Afirmar ($rc -eq 0) "corrida con Desde por defecto sale con 0"
    Afirmar ((Leer 'igual.txt') -eq "v3`n" -and (IndiceSha 'igual.txt') -eq (G rev-parse "${c3}:igual.txt").Trim()) "Desde por defecto lleva igual.txt a v3"
} catch {
    $fallos.Add("excepcion: $($_.Exception.Message)"); Write-Host "  FALLO excepcion: $_"
} finally {
    Pop-Location -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
}

Write-Host "`n$pasadas afirmaciones correctas, $($fallos.Count) fallos"
if ($fallos.Count -gt 0) { $fallos | ForEach-Object { Write-Host "  - $_" }; exit 1 }
exit 0
