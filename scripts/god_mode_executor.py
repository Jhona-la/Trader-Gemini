import os
import sys
import time
import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# Aseguramos el PATH correcto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Evitamos que multiprocess cuelgue en Windows, usaremos ThreadPoolExecutor (o secuencial si hay problemas)
# ThreadPoolExecutor es más seguro para simulaciones I/O y backtesting intensivo en un solo OS proc.

os.environ["TRADER_GEMINI_BACKTEST"] = "true"

def worker_backtest(params):
    """
    Worker para backtest ejecutado en Hilos (Threading)
    Evita el bloqueo de multiprocessing de Windows.
    """
    worker_id, leverage, dynamic_multiplier, sl_pct, days = params
    
    # Imports locales para aislar estado
    from config import Config
    from scripts.run_god_mode_backtest import run_global_backtest
    from core.backtest_infra import fetch_multi_symbol_data
    
    try:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"] 
        all_data = fetch_multi_symbol_data(symbols, days=days)
        
        start_t = time.perf_counter()
        
        # Override config locally
        Config.BINANCE_LEVERAGE = leverage
        Config.MAX_RISK_PER_TRADE = 0.05 * dynamic_multiplier
        Config.STOP_LOSS_PCT = sl_pct
        
        results = run_global_backtest(
            all_data=all_data,
            symbols=symbols,
            days=days,
            initial_capital=13.0,
            verbose=False,
            seed=42,
            scenario="A",
            isolated_strategy="technical"
        )
        
        end_t = time.perf_counter()
        
        net_pnl = results.get("metrics", {}).get("net_pnl_usd", 0.0)
        roi_pct = (net_pnl / 13.0) * 100
        win_rate = results.get("metrics", {}).get("win_rate_pct", 0.0)
        
        return {
            "worker_id": worker_id,
            "leverage": leverage,
            "dynamic_multiplier": dynamic_multiplier,
            "sl_pct": sl_pct,
            "roi_pct": roi_pct,
            "net_pnl": net_pnl,
            "win_rate": win_rate,
            "elapsed_s": end_t - start_t
        }
        
    except Exception as e:
        return {"worker_id": worker_id, "error": str(e)}

def run_grid_search(max_workers=4):
    print("🚀 [GOD MODE EXECUTOR] Optimizador Cuántico Activado (Windows Safe Threading)...")
    print("Objetivo: Encontrar combinación para >100% ROI en 3 días (Crecimiento Compuesto)\n")
    
    leverages = [10, 20]
    multipliers = [1.0, 2.0] # Agresividad del Kelly
    sl_pcts = [0.002, 0.005] # Ajuste del riesgo asimétrico
    days = 3 
    
    combinations = list(itertools.product(leverages, multipliers, sl_pcts))
    
    tasks = []
    for i, (l, m, s) in enumerate(combinations):
        tasks.append((i, l, m, s, days))
        
    print(f"📊 Ejecutando {len(tasks)} combinaciones (Max Threads: {max_workers})...")
    
    start_global = time.perf_counter()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(worker_backtest, t): t for t in tasks}
        for future in as_completed(future_to_task):
            res = future.result()
            results.append(res)
            if "error" not in res:
                print(f"  [Worker-{res['worker_id']}] Completado en {res['elapsed_s']:.1f}s -> ROI: {res['roi_pct']:.2f}%")
            else:
                print(f"  [Worker-{res['worker_id']}] ❌ Error: {res['error']}")
    
    end_global = time.perf_counter()
    
    valid_results = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    
    valid_results.sort(key=lambda x: x["roi_pct"], reverse=True)
    
    print("\n" + "="*60)
    print("🏆 RESULTADOS DEL GOD MODE BACKTEST")
    print("="*60)
    
    for r in valid_results[:5]:
        print(f"🥇 ROI: {r['roi_pct']:.2f}% | PnL: ${r['net_pnl']:.2f} | WR: {r['win_rate']:.2f}% | Lev: {r['leverage']}x | Mult: {r['dynamic_multiplier']} | SL: {r['sl_pct']*100:.2f}%")
        
    if errors:
        print(f"\n⚠️ Tareas fallidas: {len(errors)}")
            
    print(f"\n⏱️ Tiempo Total Búsqueda: {end_global - start_global:.2f}s")

def run_mock():
    print("🚀 [GOD MODE EXECUTOR] Ejecutando Mock Masivo (Hyper-Speed)...")
    from scripts.super_massive_mock import run_super_massive_mock
    run_super_massive_mock()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['mock', 'backtest'], default='mock', help='Ejecutar simulación Mock o Grid Search Backtest')
    parser.add_argument('--threads', type=int, default=4, help='Número de hilos para el backtest')
    args = parser.parse_args()
    
    if args.mode == 'mock':
        run_mock()
    else:
        run_grid_search(max_workers=args.threads)
