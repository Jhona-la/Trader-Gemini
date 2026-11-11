#Requires -Version 7.0
<#
  INTEGRADOR-T4: pruebas de scripts/integrar_rama.ps1 (con -SinCompilar) en un repo git temporal.
  Script de asserts sin Pester: sale con codigo 1 si algo falla.
  Uso: pwsh -NoProfile -File scripts/tests/integrar_rama.Tests.ps1
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\integrar_rama.ps1'))
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("integrar_test_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
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
function Correr([string[]]$extra) {
    & pwsh -NoProfile -File $script -Repo $tmp @extra *> (Join-Path $tmp '.git/ultima_corrida.log')
    $LASTEXITCODE
}
function Main { (G rev-parse refs/heads/main).Trim() }

try {
    New-Item -ItemType Directory -Path $tmp | Out-Null
    Push-Location $tmp
    G init -q -b main | Out-Null
    G config user.email prueba@local | Out-Null; G config user.name prueba | Out-Null
    G config core.autocrlf true | Out-Null
    Escribir 'a.txt' "base`n"; Escribir 'crates/uno/Cargo.toml' "[package]`nname = `"uno-pkg`"`nversion = `"0.1.0`"`nedition = `"2021`"`n"; Escribir 'crates/uno/src/lib.rs' "// v1`n"
    Escribir 'Cargo.toml' "[workspace]`nmembers = [`"crates/uno`"]`nresolver = `"2`"`n"; Escribir '.gitignore' "/target`n"
    & cargo generate-lockfile --quiet 2>$null; if ($LASTEXITCODE -ne 0) { throw 'cargo generate-lockfile fallo' }
    G add -A | Out-Null; G commit -q -m base | Out-Null

    # Rama limpia: toca crates/uno; main avanza en otro archivo.
    G switch -q -c limpia | Out-Null
    Escribir 'crates/uno/src/lib.rs' "// v2`n"; G commit -q -am 'rama limpia' | Out-Null
    G switch -q main | Out-Null
    Escribir 'b.txt' "main`n"; G add b.txt | Out-Null; G commit -q -m 'main avanza' | Out-Null
    # Rama en conflicto con main sobre a.txt.
    G switch -q -c choque HEAD~1 | Out-Null
    Escribir 'a.txt' "choque`n"; G commit -q -am 'choque' | Out-Null
    G switch -q main | Out-Null
    Escribir 'a.txt' "main`n"; G commit -q -am 'main toca a.txt' | Out-Null

    Write-Host "1) conflicto: para y lista"
    $m0 = Main
    $rc = Correr @('-Rama', 'choque', '-SinCompilar')
    Afirmar ($rc -eq 2) "conflicto sale con 2 (sale $rc)"
    Afirmar ((Main) -eq $m0) "conflicto no mueve main"
    $j = Get-Content .agent_tx/diagnostics/integrar_rama.json -Raw | ConvertFrom-Json
    Afirmar ((@($j.conflictos) -join ',') -eq 'a.txt') "lista el conflicto a.txt"

    Write-Host "2) -Simular con rama limpia"
    $rc = Correr @('-Rama', 'limpia', '-Simular', '-SinCompilar')
    Afirmar ($rc -eq 0) "simular sale con 0 (sale $rc)"
    Afirmar ((Main) -eq $m0) "simular no mueve main"
    $j = Get-Content .agent_tx/diagnostics/integrar_rama.json -Raw | ConvertFrom-Json
    Afirmar ((@($j.crates) -join ',') -eq 'uno' -and (@($j.paquetes) -join ',') -eq 'uno-pkg') "detecta crate uno y paquete uno-pkg"
    Afirmar ((Leer 'crates/uno/src/lib.rs') -eq "// v1`n") "simular no toca el disco"

    Write-Host "3) integrar de verdad (sin compilar)"
    $rc = Correr @('-Rama', 'limpia', '-SinCompilar')
    if ($rc -ne 0) { Get-Content .git/ultima_corrida.log | Write-Host }
    Afirmar ($rc -eq 0) "integrar sale con 0 (sale $rc)"
    $padres = @((G rev-list --parents -n 1 (Main)) -split ' ')
    Afirmar ($padres.Count -eq 3 -and $padres[1] -eq $m0 -and $padres[2] -eq (G rev-parse limpia).Trim()) "commit con dos padres (main, rama)"
    Afirmar ((G show 'main:crates/uno/src/lib.rs') -eq '// v2' -and (G show 'main:a.txt') -eq 'main') "el arbol trae los cambios de los dos padres"
    Afirmar ((Leer 'crates/uno/src/lib.rs') -eq "// v2`n") "arbol vivo sincronizado en disco"
    Afirmar (-not (G status --porcelain -uno)) "arbol vivo e indice compartido iguales a HEAD"
    Afirmar ((Get-Content .agent_tx/diagnostics/arbol_vivo_sha.txt -TotalCount 1).Trim() -eq (Main)) "SHA sincronizado = nuevo main"

    Write-Host "4) rama ya integrada"
    $m1 = Main
    $rc = Correr @('-Rama', 'limpia', '-SinCompilar')
    Afirmar ($rc -eq 0 -and (Main) -eq $m1) "rama ya integrada: sale con 0 y no mueve main"
    Write-Host "5) con compilacion: worktree temporal, cargo check y cargo test del crate"
    G switch -q -c compila main | Out-Null
    Escribir 'crates/uno/src/lib.rs' "pub fn dos() -> u32 { 2 }`n#[cfg(test)] mod t { #[test] fn d() { assert_eq!(super::dos(), 2); } }`n"; G commit -q -am 'compila' | Out-Null
    G switch -q -c rota main | Out-Null
    Escribir 'crates/uno/src/lib.rs' "pub fn dos() -> u32 { `"no`" }`n"; G commit -q -am 'rota' | Out-Null
    G switch -q main | Out-Null
    $m2 = Main
    $rc = Correr @('-Rama', 'compila', '-Simular')
    if ($rc -ne 0) { Get-Content .git/ultima_corrida.log | Write-Host }
    $j = Get-Content .agent_tx/diagnostics/integrar_rama.json -Raw | ConvertFrom-Json
    Afirmar ($rc -eq 0 -and $j.pasos.cargo_check.codigo -eq 0 -and $j.pasos.cargo_test.codigo -eq 0) "rama que compila: check y test en verde (sale $rc)"
    Afirmar ((Main) -eq $m2) "simular con compilacion no mueve main"
    Afirmar (@(G worktree list).Count -eq 1) "worktree temporal retirado"
    $rc = Correr @('-Rama', 'rota', '-Simular')
    Afirmar ($rc -eq 4) "rama que no compila sale con 4 (sale $rc)"
    Afirmar (@(G worktree list).Count -eq 1) "worktree temporal retirado tras fallo"
} catch {
    $fallos.Add("excepcion: $($_.Exception.Message)"); Write-Host "  FALLO excepcion: $_"
} finally {
    Pop-Location -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
}

Write-Host "`n$pasadas afirmaciones correctas, $($fallos.Count) fallos"
if ($fallos.Count -gt 0) { $fallos | ForEach-Object { Write-Host "  - $_" }; exit 1 }
exit 0
