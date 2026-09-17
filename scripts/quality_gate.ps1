# QUALITY GATE — Trader Gemini (F0.7)
# Uso: powershell -File scripts/quality_gate.ps1 [-ClippyDeny]
# Niveles:
#   Bloqueantes (siempre): cargo fmt --check, cargo check --all-targets, cargo test --lib
#   Clippy: por defecto informativo (cuenta warnings). Con -ClippyDeny es bloqueante.
#   La deuda de clippy legacy se liquida crate por crate al tocarlos en cada fase
#   del Plan Maestro; los crates ya saneados se agregan a $CLIPPY_DENY_LIST.
param([switch]$ClippyDeny)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# Crates ya saneados (clippy -D warnings bloqueante):
$CLIPPY_DENY_LIST = @(
    "dark-alpha-engine",
    "quantum-arena"
)

function Step($name, $script) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $script
    if ($LASTEXITCODE -ne 0) { Write-Host "FAIL: $name" -ForegroundColor Red; exit 1 }
}

Step "cargo fmt --check" { cargo fmt --check 2>&1 | Tee-Object -Variable fmtOut
    if ($fmtOut) { Write-Host "Hay archivos sin formatear. Ejecuta: cargo fmt"; exit 1 } }
Step "cargo check --all-targets" { cargo check --workspace --all-targets 2>&1 | Select-String "^error" | Select-Object -First 5 }
Step "cargo test --lib" { cargo test --workspace --lib 2>&1 | Select-String "test result|FAILED" }

Write-Host "`n=== clippy (informativo) ===" -ForegroundColor Cyan
$total = (cargo clippy --workspace --all-targets 2>&1 | Select-String "^warning.*generated" | Measure-Object).Count
cargo clippy --workspace --all-targets 2>&1 | Select-String "generated .* warning" | Select-Object -First 30
Write-Host "Crates con warnings: $total (deuda decreciente por fase)"

if ($ClippyDeny) {
    Step "clippy -D warnings (lista de saneados)" {
        cargo clippy -p ($CLIPPY_DENY_LIST -join " -p ") --all-targets -- -D warnings 2>&1 | Select-String "^error" | Select-Object -First 5
    }
}

Write-Host "`nQUALITY GATE: PASS" -ForegroundColor Green
