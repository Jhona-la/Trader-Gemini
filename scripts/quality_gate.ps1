# QUALITY GATE — Trader Gemini (F0.7)
# Uso: powershell -File scripts/quality_gate.ps1 [-ClippyDeny]
# Niveles:
#   Bloqueantes (siempre): cargo fmt --check, cargo check --all-targets, cargo test --lib
#   Clippy: cuenta warnings; compilacion fallida o sin turno siempre bloquea.
#   Con -ClippyDeny, los warnings de los paquetes saneados tambien bloquean.
#   La deuda de clippy legacy se liquida crate por crate al tocarlos en cada fase
#   del Plan Maestro; los crates ya saneados se agregan a $CLIPPY_DENY_LIST.
param(
    [switch]$ClippyDeny,
    [string]$BashPath = "$env:ProgramFiles\Git\usr\bin\bash.exe"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$pathAnterior = $env:PATH
$jobsAnterior = $env:CARGO_BUILD_JOBS
$ubicacionCambiada = $false
$codigoSalida = 1

# Crates ya saneados (clippy -D warnings bloqueante):
$CLIPPY_DENY_LIST = @(
    "dark-alpha-engine",
    "quantum-arena"
)

function Invoke-CargoStep {
    param([string]$Nombre, [string[]]$CargoArgs, [string]$Filtro)
    Write-Host "`n=== $Nombre ===" -ForegroundColor Cyan
    # Drenar incluso despues del diagnostico 30, sin guardar toda la salida.
    # Windows PowerShell 5 convierte stderr nativo en ErrorRecord: Continue
    # permite observar la salida real (incluido 75) antes de decidir el fallo.
    $script:totalWarnings = 0
    $mostrados = 0
    $accionAnterior = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $bashExe --noprofile --norc 'scripts/cargo_semaforo.sh' @CargoArgs 2>&1 | ForEach-Object {
            $linea = [string]$_
            if ($linea -match '^warning.*generated') { $script:totalWarnings++ }
            if (-not $Filtro) {
                Write-Output $linea
            } elseif ($mostrados -lt 30 -and $linea -match $Filtro) {
                Write-Output $linea
                $mostrados++
            }
        }
        $script:codigoCargo = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $accionAnterior
    }
    # Informativo son sus warnings, nunca una compilacion fallida o sin turno.
    if ($codigoCargo -ne 0) {
        $script:codigoSalida = $codigoCargo
        throw "FAIL: $Nombre (salida $codigoCargo)"
    }
}

try {
    if (Test-Path Env:RUSTFLAGS) {
        throw 'RUSTFLAGS de entorno no esta permitido; use la configuracion compartida'
    }
    if (-not (Test-Path -LiteralPath $BashPath -PathType Leaf)) {
        throw "No se encontro bash MSYS: $BashPath"
    }
    $bashExe = (Resolve-Path -LiteralPath $BashPath).ProviderPath
    Push-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
    $ubicacionCambiada = $true
    if (-not (Test-Path -LiteralPath 'scripts/cargo_semaforo.sh' -PathType Leaf)) {
        throw 'No se encontro scripts/cargo_semaforo.sh en este checkout'
    }
    $env:PATH = [IO.Path]::GetDirectoryName($bashExe) + [IO.Path]::PathSeparator + $pathAnterior
    $env:CARGO_BUILD_JOBS = '2'

    Write-Host "`n=== git diff --check ===" -ForegroundColor Cyan
    git -c core.whitespace=-blank-at-eof,-blank-at-eol diff --check HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Marcadores de conflicto o whitespace corrupto detectados' }
    Invoke-CargoStep -Nombre 'cargo fmt --check' -CargoArgs @('fmt', '--check')
    Invoke-CargoStep -Nombre 'cargo check --all-targets' -CargoArgs @('check', '--workspace', '--all-targets') -Filtro '^error'
    Invoke-CargoStep -Nombre 'cargo test --lib' -CargoArgs @('test', '--workspace', '--lib') -Filtro 'test result|FAILED'

    Invoke-CargoStep -Nombre 'clippy (informativo)' -CargoArgs @('clippy', '--workspace', '--all-targets') -Filtro 'generated .* warning'
    Write-Host "Crates con warnings: $totalWarnings (deuda decreciente por fase)"

    if ($ClippyDeny) {
        $denyArgs = @('clippy')
        foreach ($package in $CLIPPY_DENY_LIST) { $denyArgs += @('-p', $package) }
        $denyArgs += @('--all-targets', '--', '-D', 'warnings')
        Invoke-CargoStep -Nombre 'clippy -D warnings (lista de saneados)' -CargoArgs $denyArgs -Filtro '^error'
    }
    Write-Host "`nQUALITY GATE: PASS" -ForegroundColor Green
    $codigoSalida = 0
} catch {
    Write-Error -ErrorRecord $_ -ErrorAction Continue
} finally {
    $env:PATH = $pathAnterior
    $env:CARGO_BUILD_JOBS = $jobsAnterior
    if ($ubicacionCambiada) { Pop-Location }
}

exit $codigoSalida
