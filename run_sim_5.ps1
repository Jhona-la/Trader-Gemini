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
