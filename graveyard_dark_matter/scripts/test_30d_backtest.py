import time
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.quantum_engine import QuantumEngine

def run_30_day_simulation():
    print("⏳ Iniciando simulación Cuántica Vectorizada a 30 días...")
    
    # Load DNA
    try:
        with open('.models/quantum_dna.json', 'r') as f:
            dna = json.load(f)
            # Add defaults if missing
            dna.setdefault('leverage', 14.0)
            dna.setdefault('tp_pct', 0.0011)
            dna.setdefault('sl_pct', 0.0484)
    except:
        dna = {
            'tp_pct': 0.0011, 'sl_pct': 0.0484, 'leverage': 14.0,
            'rsi_buy': 30, 'rsi_sell': 70, 'bb_std': 2.0, 'bb_period': 20,
            'ema_fast': 8, 'ema_slow': 21, 'strength_threshold': 0.50
        }
        
    engine = QuantumEngine(capital=13.0)
    
    # 30 Days
    days = 30
    print(f"📊 Descargando datos para {days} días...")
    engine.load_data(days=days)
    
    print("🚀 Evaluando parámetros Genéticos...")
    t0 = time.time()
                
    res = engine.run_vectorized_backtest(dna=dna)
    
    t1 = time.time()
    
    print("\n" + "═"*50)
    print("🏆 RESULTADOS BACKTEST SUPREMO (30 DÍAS)")
    print("═"*50)
    print(f"⏱️ Tiempo de Ejecución Motor: {(t1 - t0) * 1000:.3f} milisegundos")
    print(f"💰 Balance Final: ${res['pnl']:.2f} USD (Capital Inicial: $13.00)")
    print(f"🎯 Win Rate: {res['win_rate']:.2f}%")
    print(f"⚡ Operaciones (Trades): {res['trades']}")
    print(f"🧬 ADN Usado: Leverage {dna.get('leverage', 1.0)}x | TP: {dna.get('tp_pct', 0)*100:.2f}% | SL: {dna.get('sl_pct', 0)*100:.2f}%")
    print("═"*50)

if __name__ == '__main__':
    run_30_day_simulation()
