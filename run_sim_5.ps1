# F7: conservar CARGO_INCREMENTAL=0 solo durante este intento.
param([string]$BashPath = "$env:ProgramFiles\Git\usr\bin\bash.exe")

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$pathAnterior = $env:PATH
$jobsAnterior = $env:CARGO_BUILD_JOBS
$incrementalAnterior = $env:CARGO_INCREMENTAL
$ubicacionCambiada = $false
$codigoSalida = 1

try {
    if (Test-Path Env:RUSTFLAGS) {
        throw 'RUSTFLAGS de entorno no esta permitido por AGENT_SYNC; use la configuracion compartida'
    }
    if (-not (Test-Path -LiteralPath $BashPath -PathType Leaf)) {
        throw "No se encontro bash MSYS: $BashPath"
    }
    $bashExe = (Resolve-Path -LiteralPath $BashPath).ProviderPath
    Push-Location -LiteralPath $PSScriptRoot
    $ubicacionCambiada = $true
    if (-not (Test-Path -LiteralPath 'scripts/cargo_semaforo.sh' -PathType Leaf)) {
        throw 'No se encontro scripts/cargo_semaforo.sh en este checkout'
    }
    $env:PATH = [IO.Path]::GetDirectoryName($bashExe) + [IO.Path]::PathSeparator + $pathAnterior
    $env:CARGO_BUILD_JOBS = '1'
    # Adaptador V3 (AGENT_SYNC SS4): este launcher conserva su tope de
    # admision de 4 GiB declarandolo EXPLICITO. En el regimen 2026-11-06
    # (mandato del operador: cero restricciones de recursos) el default del
    # semaforo es SEMAFORO_RAM_MIN_GB=0 (sin puerta global) y cada agente que
    # quiera tope lo declara; sin esto, el contrato
    # test_denied_memory_returns_75_without_cargo no tiene quien lo sostenga.
    if (-not $env:SEMAFORO_RAM_MIN_GB) { $env:SEMAFORO_RAM_MIN_GB = '4' }
    $env:CARGO_INCREMENTAL = '0'

    & $bashExe --noprofile --norc 'scripts/cargo_semaforo.sh' run --bin multi_coin_simulator -j 1
    $codigoSalida = $LASTEXITCODE
} catch {
    Write-Error -ErrorRecord $_ -ErrorAction Continue
} finally {
    $env:PATH = $pathAnterior
    $env:CARGO_BUILD_JOBS = $jobsAnterior
    $env:CARGO_INCREMENTAL = $incrementalAnterior
    if ($ubicacionCambiada) {
        Pop-Location
    }
}

exit $codigoSalida
