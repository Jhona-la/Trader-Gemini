import json
import pandas as pd

def check_zero_pnl():
    with open('backtest_results.json', 'r') as f:
        data = json.load(f)
    
    trades = data.get('detailed_trades', [])
    zero_pnl_count = 0
    non_zero_pnl_count = 0
    
    print(f"Total trades in json: {len(trades)}")
    
    for t in trades[:10]:
        print(f"Trade: {t['symbol']} | Side: {t['side']} | PnL %: {t['pnl_pct']} | PnL USD: {t['pnl_usd']}")
    
    for t in trades:
        if t['pnl_usd'] == 0.0 or t['pnl_usd'] == -0.0:
            zero_pnl_count += 1
        else:
            non_zero_pnl_count += 1
            
    print(f"\nZero PnL USD Trades: {zero_pnl_count}")
    print(f"Non-Zero PnL USD Trades: {non_zero_pnl_count}")

if __name__ == '__main__':
    check_zero_pnl()
