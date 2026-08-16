#!/usr/bin/env bash
# build.sh - Script de compilación Ultra-Release HFT

echo "============================================================"
echo "⚙️ INICIANDO COMPILACIÓN DEL CONSEJO DE SAGES (ULTRA-RELEASE)"
echo "============================================================"

# Habilitar optimizaciones Target CPU nativo (LTO y codegen via Cargo.toml)
export RUSTFLAGS="-C target-cpu=native"

echo ">> Ejecutando Cargo Build Workspace..."
cargo build --workspace --release

if [ $? -eq 0 ]; then
    echo "============================================================"
    echo "✅ ENSAMBLE COMPLETADO CON ÉXITO"
    echo "============================================================"
    echo "Los binarios ejecutables se encuentran en: target/release/"
else
    echo "============================================================"
    echo "❌ ERROR EN LA COMPILACIÓN"
    echo "============================================================"
    exit 1
fi
