import sys
import os
import time
import json
import itertools
import multiprocessing as mp
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from concurrent.futures import ProcessPoolExecutor
from scripts.run_god_mode_backtest import run_global_backtest
# Fix deterministic environment
os.environ["TRADER_GEMINI_BACKTEST"] = "true"

def worker_backtest(params):
    worker_id, leverage, dynamic_multiplier, sl_pct, days, cache_path = params
    
    # Temporarily set environment/Config values for this worker process
    # Note: mp.Pool creates fresh processes in Windows, so this is safe per-process
    os.environ["BINANCE_LEVERAGE"] = str(leverage)
    os.environ["MAX_RISK_PER_TRADE"] = str(0.05 * dynamic_multiplier)
    
    try:
        # Import inside the worker to avoid pickling issues
        from core.backtest_infra import fetch_multi_symbol_data
        
        symbols = getattr(Config, 'TRADE_SYMBOLS', ["BTC/USDT", "ETH/USDT", "SOL/USDT"])[:3] # Test con 3 para velocidad
        
        # En Replay mode la descarga de datos es igual de rápida si está cacheada
        all_data = fetch_multi_symbol_data(symbols, days=days)
        
        # Execute Replay simulation
        res = run_global_backtest(
            all_data, 
            symbols, 
            days, 
            verbose=False,
            mode="REPLAY",
            signal_cache_path=cache_path
        )
        
        net_pnl = res['capital'] - res['initial_capital']
        roi = (net_pnl / res['initial_capital']) * 100
        
        return {
            "worker_id": worker_id,
            "params": {"lev": leverage, "mult": dynamic_multiplier, "sl": sl_pct},
            "roi": roi,
            "trades": res['total_trades'],
            "win_rate": res['win_rate'] * 100,
            "max_dd": res['max_drawdown'] * 100,
            "sharpe": res['sharpe'],
            "capital": res['capital']
        }
    except Exception as e:
        import traceback
        return {"error": f"{e}\n{traceback.format_exc()}", "worker_id": worker_id}

def run_super_massive_backtest():
    print(f"\n🚀 [SUPER MASSIVE BACKTEST] Optimizador Cuántico Activado...")
    print(f"Objetivo: Encontrar combinación para >100% ROI en 3 días (Crecimiento Compuesto)")
    
    days = 3
    symbols = getattr(Config, 'TRADE_SYMBOLS', ["BTC/USDT", "ETH/USDT", "SOL/USDT"])[:3]
    
    # PHASE 0: SIGNAL GENERATION (QUANTUM CACHE)
    cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "quantum_signal_cache.json"))
    print(f"======================================================================")
    print(f"🌌 FASE 0: GENERACIÓN DE CACHÉ CUÁNTICO (Solo Inteligencia Artificial)")
    print(f"======================================================================")
    from core.backtest_infra import fetch_multi_symbol_data
    all_data = fetch_multi_symbol_data(symbols, days=days + 70) # +70 for lookback parity
    
    run_global_backtest(
        all_data,
        symbols,
        days,
        verbose=True,
        mode="GENERATOR",
        signal_cache_path=cache_path
    )
    print(f"✅ FASE 0 COMPLETADA. Caché listo.\n")
    
    # Limit workers for low-resource environments (Avoid OOM when loading ML models)
    # NOW we can use ALL workers because RAM usage is negligible!
    max_workers = 12
    
    leverage_options = [10, 20, 30]
    multiplier_options = [1.0, 1.5]
    sl_options = [0.015, 0.025]
    
    combinations = list(itertools.product(leverage_options, multiplier_options, sl_options))
    tasks = []
    
    for i, (l, m, s) in enumerate(combinations):
        tasks.append((i, l, m, s, days, cache_path))
        
    print(f"📊 Ejecutando {len(tasks)} combinaciones en {max_workers} workers paralelos (REPLAY MODE)...")
    
    start_global = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker_backtest, tasks))
    
    end_global = time.perf_counter()
    
    print(f"\n======================================================================")
    print(f"🏆 RESULTADOS DEL OPTIMIZADOR CUÁNTICO ({(end_global - start_global):.2f}s)")
    print(f"======================================================================")
    
    valid_results = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    
    if errors:
        print(f"⚠️ {len(errors)} workers fallaron. Primer error:\n{errors[0]['error']}")
    
    valid_results.sort(key=lambda x: x['roi'], reverse=True)
    
    for r in valid_results[:5]:
        p = r['params']
        print(f"➤ Lev: {p['lev']}x | Mult: {p['mult']} | SL: {p['sl']*100:.1f}%")
        print(f"   ROI: {r['roi']:.2f}% | WinRate: {r['win_rate']:.1f}% | Trades: {r['trades']} | DD: {r['max_dd']:.2f}% | Capital: ${r['capital']:.2f}\n")
        
    if valid_results and valid_results[0]['roi'] > 100:
        print(f"🎯 ¡SANTO GRIAL ENCONTRADO! La combinación {valid_results[0]['params']} logró duplicar la cuenta en {days} días.")

if __name__ == "__main__":
    # Necesario en Windows para multiprocessing
    mp.freeze_support()
    run_super_massive_backtest()
