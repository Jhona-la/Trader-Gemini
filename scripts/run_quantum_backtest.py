#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 QUANTUM VECTORIZED BACKTEST (NANOSECOND SIMULATOR)
═══════════════════════════════════════════════════════════════════════════════
Ejecución masiva paralela sin Event Loop.
Reduce un backtest de 15 días (21.000 velas x 26 monedas) de segundos a milisegundos netos.
"""

import time
import argparse
import sys
import os

# Project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.quantum_engine import QuantumEngine
from utils.logger import logger

def parse_args():
    parser = argparse.ArgumentParser(description='Quantum Backtester')
    parser.add_argument('--days', type=int, default=15, help='Days of historical data')
    parser.add_argument('--capital', type=float, default=13.0, help='Initial capital')
    parser.add_argument('--horizon', type=str, default='SCALPING', help='Horizon mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"🌌 Inicializando Motor Cuántico | Capital: ${args.capital} | Horizon: {args.horizon} | Days: {args.days}")
    
    # 1. Instanciar Motor
    engine = QuantumEngine(capital=args.capital, horizon=args.horizon)
    
    # 2. Cargar Data (I/O)
    logger.info("📡 Inyectando Parquets en RAM (Vector Load)...")
    t0_io = time.perf_counter()
    engine.load_data(days=args.days)
    io_time_ms = (time.perf_counter() - t0_io) * 1000
    
    loaded_symbols = len(engine.data)
    if loaded_symbols == 0:
        logger.error("❌ No data loaded. Check 'data/historical' folder.")
        sys.exit(1)
        
    logger.info(f"✅ {loaded_symbols} Símbolos cargados en {io_time_ms:.2f} ms.")
    
    # 3. EJECUCIÓN CUÁNTICA (Cálculo Puro)
    logger.info("⚡ Ejecutando Álgebra Matricial Numba/NumPy (Vectorized Simulation)...")
    t0_calc = time.perf_counter()
    
    # Boom.
    results = engine.run_vectorized_backtest()
    
    calc_time_ms = (time.perf_counter() - t0_calc) * 1000
    
    # 4. Reporte Ultra-rápido
    logger.info("\n" + "="*50)
    logger.info("🏆 RESULTADOS CUÁNTICOS OBTENIDOS")
    logger.info("="*50)
    logger.info(f"⏳ Tiempo Matemático Neto : {calc_time_ms:.4f} milisegundos")
    logger.info(f"💰 Capital Inicial        : ${args.capital:.2f}")
    logger.info(f"💰 Capital Final          : ${results['final_capital']:.2f}")
    logger.info(f"📈 Beneficio Neto         : ${results['pnl']:.2f} ({(results['pnl']/args.capital)*100:.2f}%)")
    logger.info(f"⚔️ Trades Totales         : {results['trades']}")
    logger.info(f"🎯 Win Rate Global        : {results['win_rate']:.2f}%")
    from utils.logger import stop_logger
    stop_logger()

if __name__ == '__main__':
    main()
