import os
import sys
import subprocess
import time
import argparse
from datetime import datetime

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def print_banner(title):
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)

def run_script(script_path, args):
    cmd = [sys.executable, script_path] + args
    print(f"\n▶ Ejecutando: {' '.join(cmd)}")
    t0 = time.time()
    try:
        # Use subprocess to guarantee full memory release after the script finishes
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ [ERROR FATAL] Fallo en {os.path.basename(script_path)} con código {e.returncode}")
        sys.exit(1)
    
    elapsed = time.time() - t0
    print(f"✅ {os.path.basename(script_path)} completado exitosamente en {elapsed:.2f}s.")

def main():
    parser = argparse.ArgumentParser(description="MASTER ECOSYSTEM SYNERGY PIPELINE")
    parser.add_argument("--surrogate-days", type=int, default=30, help="Días para entrenar el Neural Surrogate")
    parser.add_argument("--blueprint-days", type=int, default=15, help="Días para el Blueprint Omni-Evolver")
    parser.add_argument("--blueprint-trials", type=int, default=200, help="Trials para el Blueprint Omni-Evolver")
    parser.add_argument("--fast-test", action="store_true", help="Modo ultra-rápido para validación de arquitectura")
    args = parser.parse_args()

    print_banner("INICIANDO SINERGIA CUÁNTICA DEL ECOSISTEMA TRADER GEMINI")
    print(f"📅 Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Entorno seguro de memoria (Subprocesamiento Activo para proteger los 16GB de RAM)")
    
    surrogate_days = 2 if args.fast_test else args.surrogate_days
    bp_days = 2 if args.fast_test else args.blueprint_days
    bp_trials = 2 if args.fast_test else args.blueprint_trials

    # FASE 1: ENTRENAMIENTO NEURONAL (MACHINE LEARNING WEIGHTS)
    print_banner("FASE 1: ENTRENAMIENTO DEL CEREBRO NEURONAL (GOD MODE SURROGATE)")
    print("Objetivo: Calibrar los 100 pesos de la Red Neuronal para Scalping y Swing de forma aislada.")
    surrogate_script = os.path.join(_project_root, "scripts", "god_mode_surrogate.py")
    run_script(surrogate_script, ["--days", str(surrogate_days), "--trials", str(bp_trials)])

    # Limpieza de memoria explícita entre procesos
    print("\n🧹 Liberando memoria caché antes de la Fase 2...")
    time.sleep(2)

    # FASE 2: EVOLUCIÓN ESTRUCTURAL Y DE CONSENSO (GLOBAL OMNI-EVOLVER)
    print_banner("FASE 2: SINERGIA GLOBAL Y OMNISCORE (GLOBAL EVOLVER)")
    print("Objetivo: Optimizar distribución de capital ($13) y los pesos de consenso (OmniScore: ML vs Tech).")
    blueprint_script = os.path.join(_project_root, "scripts", "global_omni_evolver.py")
    run_script(blueprint_script, ["--days", str(bp_days), "--trials", str(bp_trials)])

    # FASE 3: CIERRE DEL CICLO
    print_banner("LA SINERGIA ES PERFECTA")
    print("✔️ Genotipos Neurales guardados en: config/genotypes/")
    print("✔️ Master Global Omni-Evolver guardado en: data/omni_evolver_best_*.json")
    print("\n⚠️ El sistema está listo para producción. Ejecuta '.\\LAUNCH_GOD_MODE.bat' para inyectar todo en la matriz viva.")

if __name__ == "__main__":
    main()
