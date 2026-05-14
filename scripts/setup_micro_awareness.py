"""
Script de implementación rápida de conciencia micro
"""
import os
import subprocess
import sys

def setup_micro_awareness():
    """Configura el sistema con conciencia micro"""
    print("🚀 Implementando conciencia micro...")
    
    # Instalar dependencias si es necesario
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy"])
    
    # Crear estructura de directorios
    directories = [
        "core",
        "strategies", 
        "execution",
        "monitoring",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Conciencia micro implementada exitosamente")
    print("📊 Ejecute: python -m pytest tests/test_micro_awareness.py para validar")

if __name__ == "__main__":
    setup_micro_awareness()
