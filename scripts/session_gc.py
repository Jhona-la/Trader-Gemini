import os
import glob
import shutil
from pathlib import Path

def purge_io_hemorrhage(data_dir: str = "dashboard/data", keep_last: int = 5):
    """Elimina entornos de prueba viejos y trunca logs masivos > 100MB"""
    print(f"--- INICIANDO GARBAGE COLLECTOR DE SESIONES EN {data_dir} ---")
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"El directorio {data_dir} no existe.")
        return

    # 1. Truncar logs destructores de latencia (.jsonl y .log)
    for ext in ['*.jsonl', '*.log']:
        for log_file in data_path.glob(ext):
            try:
                size_mb = os.path.getsize(log_file) / (1024 * 1024)
                if size_mb > 50: # >50MB is too much for hot-path IO
                    print(f"Truncando archivo gigante ({size_mb:.2f} MB): {log_file}")
                    with open(log_file, 'w') as f:
                        f.write('{"event": "PURGED_FOR_LATENCY_OPTIMIZATION"}\n')
            except Exception as e:
                print(f"Error procesando {log_file}: {e}")
                
    # 2. Limpiar directorios de test huérfanos
    # Consideramos "test_env", "mc_", "temp"
    env_dirs = []
    for d in data_path.iterdir():
        if d.is_dir() and ("test_env" in d.name or "mc_" in d.name or "temp" in d.name or "evo_" in d.name):
            env_dirs.append(d)
            
    # Sort by modification time
    env_dirs = sorted(env_dirs, key=lambda x: os.path.getmtime(x))
    
    if len(env_dirs) > keep_last:
        to_delete = env_dirs[:-keep_last]
        for old_env in to_delete:
            print(f"Purgando sesión obsoleta: {old_env}")
            try:
                shutil.rmtree(old_env)
            except Exception as e:
                print(f"Error purgando {old_env}: {e}")
    else:
        print(f"Entornos de prueba ({len(env_dirs)}) dentro del límite de retención ({keep_last}).")

    print("--- PURGA FINALIZADA ---")

if __name__ == "__main__":
    purge_io_hemorrhage()
