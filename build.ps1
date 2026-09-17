Write-Host "============================================================"
Write-Host "⚙️ INICIANDO COMPILACIÓN DEL CONSEJO DE SAGES (ULTRA-RELEASE)"
Write-Host "============================================================"

$env:RUSTFLAGS="-C target-cpu=native"

Write-Host ">> Ejecutando Cargo Build Workspace..."
cargo build --workspace --release

if ($LASTEXITCODE -eq 0) {
    Write-Host "============================================================"
    Write-Host "✅ ENSAMBLE COMPLETADO CON ÉXITO"
    Write-Host "============================================================"
    Write-Host "Los binarios ejecutables se encuentran en: target/release/"
} else {
    Write-Host "============================================================"
    Write-Host "❌ ERROR EN LA COMPILACIÓN"
    Write-Host "============================================================"
    exit 1
}
