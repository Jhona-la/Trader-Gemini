import os
import shutil
from pathlib import Path

# QUÉ: Garbage Collector de archivos I/O grandes y carpetas temporales
# POR QUÉ: Los archivos jsonl de 1GB+ destruyen el SSD y agregan Jitter al CPU.
# PARA QUÉ: Mantener el sistema nano-speed purgado.

TARGET_DIR = Path("C:/Users/jhona/Documents/Proyectos/Trader Gemini/dashboard/data")

def clean_massive_files():
    print("[GC] Limpiando archivos gigantes...")
    if not TARGET_DIR.exists():
        return
        
    for file in TARGET_DIR.rglob("*"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            if size_mb > 100:  # Archivos de más de 100MB
                print(f"[GC] ELIMINANDO {file.name} ({size_mb:.2f} MB)")
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error borrando {file}: {e}")

def keep_latest_sessions():
    print("[GC] Limpiando carpetas de sesiones antiguas...")
    # Ejemplo de limpieza de subdirectorios
    patterns = ["futures_evo_*", "backtest_temp", "futures_mc_*"]
    for pattern in patterns:
        dirs = [d for d in TARGET_DIR.glob(pattern) if d.is_dir()]
        # Ordenar por fecha de modificación (más nuevos primero)
        dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Mantener solo los últimos 2 de cada patrón
        for d in dirs[2:]:
            print(f"[GC] BORRANDO DIRECTORIO OBSOLETO: {d.name}")
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"Error borrando {d}: {e}")

if __name__ == "__main__":
    clean_massive_files()
    keep_latest_sessions()
    print("[GC] Limpieza de I/O finalizada con éxito.")
