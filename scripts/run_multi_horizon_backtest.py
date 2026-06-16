#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 QUANTUM MULTI-HORIZON BACKTEST (NANOSECOND SIMULATOR)
═══════════════════════════════════════════════════════════════════════════════
Ejecuta la simulación cuántica para Scalping, Microscalping y Swing simultáneamente
para verificar el crecimiento geométrico (Compounding) con comisiones Maker-Only.
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
    parser = argparse.ArgumentParser(description='Multi-Horizon Quantum Backtester')
    parser.add_argument('--days', type=int, default=15, help='Days of historical data')
    parser.add_argument('--capital', type=float, default=13.0, help='Initial capital')
    return parser.parse_args()

def main():
    args = parse_args()
    horizons = ["MICROSCALPING", "SCALPING", "SWING"]
    
    logger.info(f"🌌 Inicializando Multi-Horizon Quantum Test | Capital: ${args.capital} | Days: {args.days}")
    
    total_net_pnl = 0.0
    total_trades = 0
    
    for h_idx, horizon in enumerate(horizons):
        logger.info(f"\n[{h_idx+1}/{len(horizons)}] Evaluando Horizonte: {horizon} ...")
        
        engine = QuantumEngine(capital=args.capital, horizon=horizon)
        engine.load_data(days=args.days)
        
        if not hasattr(engine, 'data_1m') or (len(engine.data_1m) == 0 and len(engine.data_1h) == 0):
            logger.warning(f"⚠️ Sin datos para {horizon}. Omitiendo.")
            continue
            
        t0_calc = time.perf_counter()
        results = engine.run_vectorized_backtest()
        calc_time_ms = (time.perf_counter() - t0_calc) * 1000
        
        total_net_pnl += results['pnl']
        total_trades += results['trades']
        
        logger.info(f" > {horizon} Completado en {calc_time_ms:.2f} ms")
        logger.info(f" > Capital Final Horizonte: ${results['final_capital']:.2f} (Beneficio: ${results['pnl']:.2f})")
        logger.info(f" > Win Rate: {results['win_rate']:.2f}% | Trades: {results['trades']}")

    # Reporte Global Crecimiento Geométrico
    logger.info("\n" + "═"*60)
    logger.info("🏆 RESULTADOS MULTI-HORIZON (CRECIMIENTO COMPUESTO)")
    logger.info("═"*60)
    final_global_capital = args.capital + total_net_pnl
    growth_pct = (final_global_capital / args.capital - 1) * 100
    
    logger.info(f"💰 Capital Inicial        : ${args.capital:.2f}")
    logger.info(f"💰 Capital Final Simulado : ${final_global_capital:.2f}")
    logger.info(f"📈 Beneficio Neto Total   : ${total_net_pnl:.2f} (+{growth_pct:.2f}%)")
    logger.info(f"⚔️ Trades Totales Activos : {total_trades}")
    
    if growth_pct >= 100:
        logger.info(f"✅ ¡OBJETIVO LOGRADO! Capital duplicado exitosamente (+100%).")
    else:
        logger.warning(f"⚠️ Crecimiento inferior al objetivo de duplicación. Requiere refinamiento de LOB/Microestructura.")
        
    from utils.logger import stop_logger
    stop_logger()

if __name__ == '__main__':
    main()
