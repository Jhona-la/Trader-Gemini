Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "🚀 TRADER GEMINI - AXIOM III (PURE METAL) 🚀" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Initializing God Engine natively..." -ForegroundColor Green

Write-Host "[1/2] Compiling Quantum Engine (Rust)..." -ForegroundColor Yellow
$compile = Start-Process -FilePath "cargo" -ArgumentList "build", "--release", "-j", "1" -Wait -NoNewWindow -PassThru
if ($compile.ExitCode -ne 0) {
    Write-Host "❌ Compilation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[2/2] Igniting God Engine..." -ForegroundColor Yellow
Start-Process -FilePath "cargo" -ArgumentList "run", "--release", "--bin", "god_engine" -NoNewWindow -Wait
