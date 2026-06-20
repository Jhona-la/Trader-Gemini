import os
import sys
import json
import time
import random
import subprocess
import concurrent.futures
from pathlib import Path

# Fix path to root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Constraint: 16GB RAM -> Max 3 concurrent backtests
MAX_CONCURRENT = 3

def mutate_dna(base_dna):
    """Mutación genética ligera para simular 100 escenarios distintos."""
    mutated = base_dna.copy()
    mutated['rsi_buy'] = max(10, min(50, mutated['rsi_buy'] + random.randint(-3, 3)))
    mutated['rsi_sell'] = max(50, min(90, mutated['rsi_sell'] + random.randint(-3, 3)))
    mutated['bb_std'] = round(max(1.0, min(4.0, mutated['bb_std'] + random.uniform(-0.2, 0.2))), 3)
    mutated['tp_pct'] = round(max(0.0005, min(0.05, mutated['tp_pct'] * random.uniform(0.9, 1.1))), 4)
    mutated['sl_pct'] = round(max(0.005, min(0.1, mutated['sl_pct'] * random.uniform(0.9, 1.1))), 4)
    mutated['leverage'] = round(max(1.0, min(50.0, mutated['leverage'] + random.randint(-3, 3))), 1)
    return mutated

def run_single_backtest(iteration, dna):
    """
    Runs a single paramaterized God Mode Backtest.
    """
    env_id = f"mc_{iteration}"
    out_dir = os.path.join(project_root, "archive", "mc_results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"mc_{iteration}.json")
    
    # We skip permutations where sl_pct * leverage >= 100% (Instant Liquidation)
    if dna['sl_pct'] * dna['leverage'] >= 0.99:
        dna['leverage'] = 0.98 / dna['sl_pct']
        
    dna_str = json.dumps(dna)
    
    cmd = [
        sys.executable,
        os.path.join(project_root, "scripts", "run_god_mode_backtest.py"),
        "--env-id", env_id,
        "--days", "30",
        "--override", dna_str,
        "--output", out_file,
        "--quiet"
    ]
    
    # Run synchronously inside the worker thread
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # After it finishes, read the output
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                res = json.load(f)
                
            metrics = res["Metrics"]
            return {
                'iteration': iteration,
                'pnl': metrics["Total PnL"],
                'win_rate': metrics["Win Rate (%)"],
                'trades': metrics["Total Trades"],
                'sharpe': metrics["Sharpe Ratio"],
                'max_drawdown': metrics["Max Drawdown (%)"],
                'leverage': dna['leverage'],
                'tp_pct': dna['tp_pct'],
                'sl_pct': dna['sl_pct']
            }
        except Exception as e:
            return {'iteration': iteration, 'error': str(e)}
    else:
        return {'iteration': iteration, 'error': 'No output file generated'}

def run_100_backtests():
    print(f"⏳ Iniciando MONTE CARLO GENÉTICO: 100 Backtests a 30 Días...")
    print(f"🚀 Concurrencia limitada a {MAX_CONCURRENT} workers para proteger 16GB RAM.")
    
    # Load Base DNA
    try:
        with open(os.path.join(project_root, '.models', 'quantum_dna.json'), 'r') as f:
            base_dna = json.load(f)
            base_dna.setdefault('leverage', 14.0)
            base_dna.setdefault('tp_pct', 0.0020)
            base_dna.setdefault('sl_pct', 0.0452)
    except:
        base_dna = {
            'tp_pct': 0.0020, 'sl_pct': 0.0452, 'leverage': 26.0,
            'rsi_buy': 33, 'rsi_sell': 70, 'bb_std': 2.8, 'bb_period': 20,
            'ema_fast': 23, 'ema_slow': 86, 'strength_threshold': 0.50
        }
        
    results = []
    t0 = time.time()
    
    # Prepare all 100 tasks
    tasks = []
    for i in range(100):
        dna = base_dna if i == 0 else mutate_dna(base_dna)
        tasks.append((i + 1, dna))
        
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(run_single_backtest, t[0], t[1]): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if 'error' not in res:
                results.append(res)
            
            completed += 1
            if completed % 5 == 0:
                print(f"   [+] Completados {completed}/100 backtests paramétricos...")
            
    t1 = time.time()
    
    # Sort results by PnL
    results.sort(key=lambda x: x['pnl'], reverse=True)
    
    # Save to JSON for report generation
    out_path = os.path.join(project_root, 'archive', 'logs_historicos', '100_backtests_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'time_ms': (t1-t0)*1000, 'results': results}, f, indent=4)
        
    print(f"✅ 100 Backtests completados en {(t1-t0):.2f} segundos. Resultados guardados.")

if __name__ == '__main__':
    run_100_backtests()
