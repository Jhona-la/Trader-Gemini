import subprocess
import time
import cProfile
import pstats
import sys

def main():
    print("[KILL-SWITCH] Iniciando backtest bajo el Dogma del Minuto...")
    
    cmd = [sys.executable, "-m", "cProfile", "-o", "backtest.prof", "scripts/run_god_mode_backtest.py", "--skip-training", "--days", "1"]
    
    start_time = time.time()
    process = subprocess.Popen(cmd)
    
    try:
        process.wait(timeout=60)
        print(f"[KILL-SWITCH] Terminado naturalmente en {time.time() - start_time:.2f}s")
    except subprocess.TimeoutExpired:
        print("[KILL-SWITCH] TIEMPO EXCEDIDO (60s). Ejecutando SIGTERM y generando autopsia...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            
    # Parse the profiler output if it exists
    try:
        p = pstats.Stats('backtest.prof')
        p.sort_stats('cumtime')
        print("\n\n=== 🔬 AUTOPSIA FORENSE (TOP 20 FUNCIONES POR CUMTIME) ===")
        p.print_stats(20)
        
        p.sort_stats('tottime')
        print("\n\n=== 🔬 AUTOPSIA FORENSE (TOP 20 FUNCIONES POR TOTTIME) ===")
        p.print_stats(20)
    except Exception as e:
        print(f"No se pudo leer el perfil (quizás el proceso fue matado abruptamente): {e}")

if __name__ == "__main__":
    main()
