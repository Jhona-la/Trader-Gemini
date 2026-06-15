import time
import json
import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.quantum_engine import QuantumEngine

def mutate_dna(base_dna):
    """Mutación genética ligera para simular 100 escenarios distintos."""
    mutated = base_dna.copy()
    mutated['rsi_buy'] = max(10, min(50, mutated['rsi_buy'] + random.randint(-3, 3)))
    mutated['rsi_sell'] = max(50, min(90, mutated['rsi_sell'] + random.randint(-3, 3)))
    mutated['bb_std'] = round(max(1.0, min(4.0, mutated['bb_std'] + random.uniform(-0.2, 0.2))), 3)
    mutated['tp_pct'] = round(max(0.0005, min(0.05, mutated['tp_pct'] * random.uniform(0.9, 1.1))), 4)
    mutated['sl_pct'] = round(max(0.005, min(0.1, mutated['sl_pct'] * random.uniform(0.9, 1.1))), 4)
    mutated['leverage'] = round(max(1.0, min(50.0, mutated.get('leverage', 10.0) + random.randint(-3, 3))), 1)
    return mutated

def run_100_backtests():
    print("⏳ Iniciando MONTE CARLO GENÉTICO: 100 Backtests a 30 Días...")
    
    # Load Base DNA
    try:
        with open('.models/quantum_dna.json', 'r') as f:
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
        
    engine = QuantumEngine(capital=13.0)
    print(f"📊 Descargando datos para 30 días...")
    engine.load_data(days=30)
    
    results = []
    
    print("🚀 Ejecutando 100 simulaciones vectorizadas...")
    t0 = time.time()
    
    # Run 100 iterations
    for i in range(100):
        # Iteration 0 is the exact base DNA
        dna = base_dna if i == 0 else mutate_dna(base_dna)
        
        # We skip permutations where sl_pct * leverage >= 100% (Instant Liquidation)
        if dna['sl_pct'] * dna['leverage'] >= 0.99:
            dna['leverage'] = 0.98 / dna['sl_pct']
            
        res = engine.run_vectorized_backtest(dna=dna)
        
        results.append({
            'iteration': i + 1,
            'pnl': res['pnl'],
            'win_rate': res['win_rate'],
            'trades': res['trades'],
            'leverage': dna['leverage'],
            'tp_pct': dna['tp_pct'],
            'sl_pct': dna['sl_pct']
        })
        
        if (i + 1) % 10 == 0:
            print(f"   [+] Completados {i + 1}/100 backtests...")
            
    t1 = time.time()
    
    # Sort results by PnL
    results.sort(key=lambda x: x['pnl'], reverse=True)
    
    # Save to JSON for report generation
    out_path = 'archive/logs_historicos/100_backtests_results.json'
    os.makedirs('archive/logs_historicos', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'time_ms': (t1-t0)*1000, 'results': results}, f, indent=4)
        
    print(f"✅ 100 Backtests completados en {(t1-t0):.2f} segundos. Resultados guardados.")

if __name__ == '__main__':
    run_100_backtests()
