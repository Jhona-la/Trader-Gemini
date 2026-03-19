import sys
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Root path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.run_backtest import fetch_binance_data, run_backtest, calculate_metrics

ALL_RESULTS = {}

def evaluate_horizon(days):
    print(f"\n" + "="*60)
    print(f"⏳ HORIZONTE: {days} DÍAS")
    print("="*60)
    
    from config import Config
    symbols = Config.TRADING_PAIRS
    for symbol in symbols:
        print(f"\n[+] {symbol} ({days}D)...", end=" ", flush=True)
        
        try:
            df = fetch_binance_data(symbol, days=days)
        except Exception as e:
            print(f"❌ Download fail: {e}")
            continue
            
        if df is None or df.empty:
            print(f"❌ No data")
            continue
            
        import contextlib, io
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            res = run_backtest(df, symbol)
            p = res['portfolio']
            metrics = calculate_metrics(p)
            
        pnl = p.current_capital - p.initial_capital
        total_trades = metrics['total_trades']
        win_rate = metrics['win_rate']
        sharpe = metrics['sharpe_ratio']
        max_dd = metrics['max_drawdown_pct']
        total_return = metrics['total_return']
        
        # Avg win / avg loss
        wins = [t for t in p.trades if t['pnl_usd'] > 0]
        losses = [t for t in p.trades if t['pnl_usd'] <= 0]
        avg_win = sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        avg_duration = sum(t.get('duration', 0) for t in p.trades) / len(p.trades) if p.trades else 0
        
        # Best and worst trade
        best_trade = max(p.trades, key=lambda t: t['pnl_usd'])['pnl_usd'] if p.trades else 0
        worst_trade = min(p.trades, key=lambda t: t['pnl_usd'])['pnl_usd'] if p.trades else 0
        
        result = {
            'symbol': symbol,
            'days': days,
            'pnl': round(pnl, 4),
            'total_return_pct': round(total_return, 2),
            'sharpe': round(sharpe, 2),
            'win_rate': round(win_rate, 1),
            'max_dd_pct': round(max_dd, 2),
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'payoff': round(payoff, 2),
            'avg_duration_min': round(avg_duration, 1),
            'best_trade': round(best_trade, 4),
            'worst_trade': round(worst_trade, 4),
        }
        
        key = f"{symbol}_{days}D"
        ALL_RESULTS[key] = result
        
        icon = "🟢" if pnl > 0 else "🔴"
        print(f"{icon} PnL=${pnl:+.2f} | WR={win_rate:.0f}% | Trades={total_trades} | Sharpe={sharpe:.2f} | DD={max_dd:.1f}% | Payoff={payoff:.2f}")

if __name__ == "__main__":
    t0 = time.time()
    
    evaluate_horizon(1)
    evaluate_horizon(15)
    evaluate_horizon(30)
    
    elapsed = time.time() - t0
    
    # Summary Table
    print("\n" + "="*100)
    print("📊 RESUMEN MULTI-HORIZONTE COMPLETO")
    print("="*100)
    print(f"{'Symbol':<12} {'Horizon':<8} {'PnL':>10} {'Return%':>9} {'WR%':>6} {'Trades':>7} {'Sharpe':>7} {'DD%':>7} {'Payoff':>8} {'AvgWin':>9} {'AvgLoss':>9}")
    print("-"*100)
    
    for key in sorted(ALL_RESULTS.keys()):
        r = ALL_RESULTS[key]
        icon = "🟢" if r['pnl'] > 0 else "🔴"
        print(f"{icon} {r['symbol']:<10} {r['days']:<6}D {r['pnl']:>+9.4f} {r['total_return_pct']:>+8.2f}% {r['win_rate']:>5.1f} {r['total_trades']:>7} {r['sharpe']:>7.2f} {r['max_dd_pct']:>6.2f}% {r['payoff']:>7.2f} {r['avg_win']:>+8.4f} {r['avg_loss']:>+8.4f}")
    
    # Per-symbol cross-horizon summary
    print(f"\n⏱️ Tiempo total: {elapsed:.0f}s")
    
    # Save to JSON for analysis
    with open('tests/multi_horizon_results.json', 'w') as fp:
        json.dump(ALL_RESULTS, fp, indent=2)
    print(f"💾 Resultados guardados en tests/multi_horizon_results.json")
    
    print("\n✅ MULTI-HORIZON AUDIT COMPLETED.")
