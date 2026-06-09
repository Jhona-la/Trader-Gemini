"""
🧬 ADAPTIVE MULTI-HORIZON BACKTESTER
Runs backtests across 1D, 7D, 15D, 30D with the Adaptive Evolution Protocol
active, capturing PnL, Win Rate, Drawdown, and Sharpe Ratio.

HOW: Imports the full run_backtest pipeline and injects set_horizon_profile()
     for MarketRegimeDetector and SophiaIntelligence before each run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from io import StringIO
import contextlib

import logging
logging.getLogger('trader_gemini').setLevel(logging.WARNING)
# Horizons to test
HORIZONS = [1, 7, 15, 30]
SYMBOLS = ['BTC/USDT']  # Focus on BTC for speed

def run_single_horizon(symbol: str, days: int) -> dict:
    """Run a single backtest for a specific horizon with Adaptive Protocol active."""
    print(f"\n{'='*60}")
    print(f"🧬 ADAPTIVE BACKTEST: {symbol} | {days}D HORIZON")
    print(f"{'='*60}")
    
    # Import fresh each time to avoid state leaks
    from core.backtest_infra import fetch_binance_data, calculate_metrics, run_backtest
    
    # No monkey-patching needed — core/backtest_infra.py uses Config
    
    try:
        # 1. Fetch data
        t0 = time.time()
        data = fetch_binance_data(symbol, days)
        fetch_time = time.time() - t0
        
        if data.empty or len(data) < 100:
            return {'error': f'Insufficient data ({len(data)} bars)', 'days': days, 'symbol': symbol}
        
        # 2. Run backtest
        t1 = time.time()
        results = run_backtest(data, symbol)
        bt_time = time.time() - t1
        
        # 3. Calculate metrics
        portfolio = results['portfolio']
        metrics = calculate_metrics(portfolio)
        
        # 4. Add context
        metrics['days'] = days
        metrics['symbol'] = symbol
        metrics['bars_processed'] = results['bars_processed']
        metrics['signals_generated'] = results['signals']
        metrics['fetch_time_sec'] = round(fetch_time, 1)
        metrics['backtest_time_sec'] = round(bt_time, 1)
        
        return metrics
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'days': days, 'symbol': symbol}


def print_summary_table(all_results: list):
    """Print a formatted comparison table."""
    print("\n" + "="*90)
    print("📊 MULTI-HORIZON COMPARISON TABLE — ADAPTIVE EVOLUTION PROTOCOL")
    print("="*90)
    
    header = f"{'Horizon':<10} {'Trades':<8} {'Win Rate':<10} {'PnL %':<10} {'MaxDD %':<10} {'Sharpe':<10} {'PF':<8} {'Final $':<10}"
    print(header)
    print("-" * 90)
    
    for r in all_results:
        if 'error' in r:
            print(f"{r['days']}D{'':>6} ERROR: {r['error']}")
            continue
        
        pnl_icon = "🟢" if r['total_return'] > 0 else "🔴"
        dd_icon = "✅" if r['max_drawdown_pct'] < 1.5 else "⚠️"
        sr_icon = "✅" if r['sharpe_ratio'] > 2.0 else "⚠️"
        wr_icon = "✅" if r['win_rate'] > 50 else "⚠️"
        
        print(
            f"{r['days']}D{'':<7} "
            f"{r['total_trades']:<8} "
            f"{r['win_rate']:>5.1f}% {wr_icon} "
            f"{pnl_icon}{r['total_return']:>+7.2f}%  "
            f"{r['max_drawdown_pct']:>5.2f}% {dd_icon} "
            f"{r['sharpe_ratio']:>6.2f} {sr_icon} "
            f"{r.get('profit_factor', 0):>5.2f}  "
            f"${r['final_capital']:>7.2f}"
        )
    
    print("="*90)
    
    # Best horizon
    valid = [r for r in all_results if 'error' not in r and r['total_trades'] > 0]
    if valid:
        best_sharpe = max(valid, key=lambda x: x['sharpe_ratio'])
        best_wr = max(valid, key=lambda x: x['win_rate'])
        best_pnl = max(valid, key=lambda x: x['total_return'])
        lowest_dd = min(valid, key=lambda x: x['max_drawdown_pct'])
        
        print(f"\n🏆 BEST SHARPE:    {best_sharpe['days']}D ({best_sharpe['sharpe_ratio']:.2f})")
        print(f"🏆 BEST WIN RATE:  {best_wr['days']}D ({best_wr['win_rate']:.1f}%)")
        print(f"🏆 BEST PnL:       {best_pnl['days']}D ({best_pnl['total_return']:+.2f}%)")
        print(f"🏆 LOWEST DD:      {lowest_dd['days']}D ({lowest_dd['max_drawdown_pct']:.2f}%)")


if __name__ == '__main__':
    print("🧬 ADAPTIVE EVOLUTION PROTOCOL — MULTI-HORIZON BACKTESTER")
    print(f"Horizons: {HORIZONS}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for days in HORIZONS:
        for symbol in SYMBOLS:
            result = run_single_horizon(symbol, days)
            all_results.append(result)
            
            # Quick progress print
            if 'error' not in result:
                print(f"\n✅ {symbol} {days}D: WR={result['win_rate']:.1f}% | PnL={result['total_return']:+.2f}% | DD={result['max_drawdown_pct']:.2f}% | Sharpe={result['sharpe_ratio']:.2f}")
            else:
                print(f"\n❌ {symbol} {days}D: {result['error']}")
    
    # Final summary
    print_summary_table(all_results)
    
    # Save results to JSON
    output_file = f"adaptive_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {output_file}")
