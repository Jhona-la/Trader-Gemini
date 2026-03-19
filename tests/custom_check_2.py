import json

def analyze_portfolio_bug():
    with open('backtest_results.json', 'r') as f:
        data = json.load(f)
    
    trades = data.get('detailed_trades', [])
    print(f"Total trades: {len(trades)}")
    
    cap = 15.0 # INITIAL_CAPITAL = 15.0 from run_backtest.py
    for i, t in enumerate(trades[:10]):
        # size_usd * pnl_pct - size_usd * commission = pnl_usd
        # size_usd * (pnl_pct - 0.0002) = pnl_usd
        pnl_pct_decimal = t['pnl_pct'] / 100.0  # Wait, in the json it's already pct?
        # Let's check from the script: pnl_pct was 0.499 in json... wait, is that 0.499% or 0.499 decimal (49.9%)?
        # run_backtest.py: 'pnl_pct': pnl_pct * 100
        # So it is 0.499%.
        
        commission = 0.0002
        # if pnl_usd = size * pnl_decimal - size * comm
        # size = pnl_usd / ((t['pnl_pct']/100.0) - commission)
        try:
            size_usd = t['pnl_usd'] / ((t['pnl_pct']/100.0) - commission)
        except ZeroDivisionError:
            size_usd = 0
            
        print(f"Trade {i}: PnL % {t['pnl_pct']:.4f} | PnL USD {t['pnl_usd']} | Est Size: {size_usd}")
        cap += t['pnl_usd']
        print(f"New Cap: {cap}")

if __name__ == '__main__':
    analyze_portfolio_bug()
