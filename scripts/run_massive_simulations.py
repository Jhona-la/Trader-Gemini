import os
import sys
import json
import itertools
import subprocess
from datetime import datetime

# ═════════════════════════════════════════════════════════════════════════════
# 🚀 MASSIVE SIMULATOR (GOD MODE ORCHESTRATOR)
# ═════════════════════════════════════════════════════════════════════════════
# QUÉ: Un orquestador que prueba combinaciones de hiper-parámetros iterativamente.
# POR QUÉ: Para descubrir los parámetros que aseguren el crecimiento exponencial.
# PARA QUÉ: Evitar hacer pruebas manuales; la máquina hace "fuerza bruta" inteligente.
# CÓMO: Genera variaciones en JSON, y llama a run_god_mode_backtest.py.
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation(simulation_id, params):
    """
    Ejecuta run_god_mode_backtest.py como un subprocess inyectando parámetros temporales.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 INICIANDO SIMULACIÓN {simulation_id}")
    print(f"Parámetros: {json.dumps(params)}")
    
    # Podemos inyectar las variables de entorno para que Config las tome,
    # o pasarlas por línea de comando si el script las acepta.
    # Por ahora, inyectamos por variables de entorno:
    env = os.environ.copy()
    
    # Risk Management overrides
    if "max_drawdown" in params:
        env["TRADER_MAX_DRAWDOWN"] = str(params["max_drawdown"])
    if "risk_per_trade" in params:
        env["TRADER_RISK_PER_TRADE"] = str(params["risk_per_trade"])
        
    # Strategy overrides (Ejemplo: Stop Loss o Take Profit genérico)
    if "tp_multiplier" in params:
        env["TRADER_TP_MULTIPLIER"] = str(params["tp_multiplier"])
    if "sl_multiplier" in params:
        env["TRADER_SL_MULTIPLIER"] = str(params["sl_multiplier"])
        
    # Ejecutar el backtest en modo SILENCIOSO o normal
    # Reducimos los días para simulaciones masivas iniciales (ej. 1 a 3 días)
    cmd = [
        sys.executable, 
        "scripts/run_god_mode_backtest.py",
        "--days", "0.1"  # Empezamos con 0.1 días para prueba rápida
    ]
    
    try:
        # Usamos subprocess para asegurar limpieza total de RAM (el OS mata el proceso al final)
        result = subprocess.run(
            cmd, 
            env=env,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Guardamos los logs de la simulación para auditoría
        log_dir = os.path.join("results", "simulations")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"sim_{simulation_id}.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== SIMULATION {simulation_id} ===\n")
            f.write(f"PARAMS: {json.dumps(params)}\n")
            f.write(f"RETURN CODE: {result.returncode}\n\n")
            f.write("--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
            
        print(f"✅ Simulación {simulation_id} finalizada. Logs en {log_file}")
        
    except Exception as e:
        print(f"❌ Error en simulación {simulation_id}: {e}")


def main():
    # Definimos el Grid de hiper-parámetros (Fricción, Riesgo, Estrategia)
    grid = {
        "max_drawdown": [0.015, 0.05],            # 1.5%, 5%
        "risk_per_trade": [0.01, 0.02],           # 1%, 2%
        "tp_multiplier": [1.5, 2.0],              # Ratio Riesgo/Beneficio
        "sl_multiplier": [1.0, 1.5]               # ATR Multiplier para SL
    }
    
    keys = grid.keys()
    values = grid.values()
    
    # Producto Cartesiano
    combinations = list(itertools.product(*values))
    print(f"🔬 Se han generado {len(combinations)} combinaciones para simular.")
    
    # Opcional: Solo correr las 2 primeras para validar el pipeline
    for i, combo in enumerate(combinations[:2]):
        params = dict(zip(keys, combo))
        run_simulation(f"BATCH_001_v{i}", params)


if __name__ == "__main__":
    main()
