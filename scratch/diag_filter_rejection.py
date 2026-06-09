"""
DIAGNOSTIC: Filter rejection counter — wraps run_global_backtest 
to count which filters kill the most signals in generate_order().
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import builtins
_rejection_counts = {}
_original_print = builtins.print

def counting_print(*args, **kwargs):
    msg = str(args[0]) if args else ""
    if "[RISK] Rejected by" in msg:
        parts = msg.split("Rejected by ")
        if len(parts) > 1:
            filter_name = parts[1].split(" for ")[0].strip()
            _rejection_counts[filter_name] = _rejection_counts.get(filter_name, 0) + 1
    # Suppress noisy output, only print important stuff
    if any(k in msg for k in ["===", "🚀", "TOTAL", "SUMMARY", "PnL", "Win Rate", "FILTER", "Equity"]):
        _original_print(*args, **kwargs)

builtins.print = counting_print

from scripts.run_god_mode_backtest import run_global_backtest, fetch_multi_symbol_data
from config import Config

if __name__ == "__main__":
    _original_print("=" * 70)
    _original_print("DIAGNOSTIC: Filter Rejection Counter (3-day backtest)")
    _original_print("=" * 70)
    
    symbols = Config.CORE_SYMBOLS[:3]  # BTC, ETH, SOL
    _original_print(f"Symbols: {symbols}")
    
    # Fetch data
    all_data = fetch_multi_symbol_data(symbols, days=3)
    
    # Run backtest
    results = run_global_backtest(all_data, symbols, days=3, verbose=False)
    
    # Report
    _original_print("\n" + "=" * 70)
    _original_print("🔍 FILTER REJECTION REPORT")
    _original_print("=" * 70)
    
    total_rejected = sum(_rejection_counts.values())
    _original_print(f"Total rejections: {total_rejected}")
    
    if results:
        trades = results.get('trades', [])
        _original_print(f"Total trades executed: {len(trades)}")
        _original_print(f"Rejection ratio: {total_rejected} rejections / {len(trades)} trades = {total_rejected / max(1, len(trades)):.1f}:1")
    
    _original_print(f"\nBreakdown by filter (sorted by kill count):")
    _original_print(f"{'Filter':<35} {'Count':>8} {'% of Total':>10}")
    _original_print("-" * 55)
    for name, count in sorted(_rejection_counts.items(), key=lambda x: -x[1]):
        pct = count / max(1, total_rejected) * 100
        _original_print(f"  {name:<33} {count:>8} {pct:>9.1f}%")
    
    builtins.print = _original_print
