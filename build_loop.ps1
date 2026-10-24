# F7: un intento, con el turno y target compartidos del semaforo.
param([string]$BashPath = "$env:ProgramFiles\Git\usr\bin\bash.exe")

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$pathAnterior = $env:PATH
$jobsAnterior = $env:CARGO_BUILD_JOBS
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

    & $bashExe --noprofile --norc 'scripts/cargo_semaforo.sh' run --bin multi_coin_simulator -j 1
    $codigoSalida = $LASTEXITCODE
} catch {
    Write-Error -ErrorRecord $_ -ErrorAction Continue
} finally {
    $env:PATH = $pathAnterior
    $env:CARGO_BUILD_JOBS = $jobsAnterior
    if ($ubicacionCambiada) {
        Pop-Location
    }
}

exit $codigoSalida
