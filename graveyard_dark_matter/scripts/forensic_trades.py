"""
Deep forensic: Parse backtest trades to reconstruct PnL and understand losses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np

CSV_PATH = "dashboard/data/backtest_temp/bt_trades.csv"

def main():
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nTypes:\n{df['type'].value_counts()}")
    print(f"\nDirections:\n{df['direction'].value_counts()}")
    print(f"\nSymbols:\n{df['symbol'].value_counts()}")
    
    # The CSV has raw fills, no PnL. We must reconstruct trades.
    # Group by symbol, pair BUY+SELL fills chronologically
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['price','quantity','datetime'])
    df = df.sort_values('datetime')
    
    # Extract strategy from details
    df['strategy'] = df['details'].str.extract(r'^(.+?)\s+Exchange:', expand=False).str.strip()
    
    # Reconstruct round-trip trades per symbol
    trades = []
    positions = {}  # symbol -> {direction, entry_price, qty, entry_time, strategy}
    
    for _, row in df.iterrows():
        sym = row['symbol']
        direction = row['direction']
        price = row['price']
        qty = row['quantity']
        strategy = row['strategy'] or 'UNKNOWN'
        ts = row['datetime']
        
        if sym not in positions:
            # Opening new position
            positions[sym] = {
                'direction': direction,
                'entry_price': price,
                'qty': qty,
                'entry_time': ts,
                'strategy': strategy,
            }
        else:
            pos = positions[sym]
            if pos['direction'] != direction:
                # Closing position (opposite direction)
                entry_p = pos['entry_price']
                exit_p = price
                
                if pos['direction'] == 'BUY':
                    pnl_pct = (exit_p - entry_p) / entry_p
                else:
                    pnl_pct = (entry_p - exit_p) / entry_p
                
                notional = pos['qty'] * entry_p
                pnl_usd = pnl_pct * notional
                fee = notional * 0.0004  # round trip fee estimate
                net_pnl = pnl_usd - fee
                
                dur = (ts - pos['entry_time']).total_seconds()
                
                trades.append({
                    'symbol': sym,
                    'strategy': pos['strategy'],
                    'direction': 'LONG' if pos['direction'] == 'BUY' else 'SHORT',
                    'entry_price': entry_p,
                    'exit_price': exit_p,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'fee': fee,
                    'net_pnl': net_pnl,
                    'duration_s': dur,
                    'entry_time': pos['entry_time'],
                })
                del positions[sym]
            else:
                # Averaging into same direction (replace with latest)
                positions[sym] = {
                    'direction': direction,
                    'entry_price': (pos['entry_price'] * pos['qty'] + price * qty) / (pos['qty'] + qty),
                    'qty': pos['qty'] + qty,
                    'entry_time': pos['entry_time'],
                    'strategy': strategy,
                }
    
    if not trades:
        print("\nNO ROUND-TRIP TRADES RECONSTRUCTED")
        return
    
    tdf = pd.DataFrame(trades)
    total = len(tdf)
    wins = len(tdf[tdf['net_pnl'] > 0])
    losses = len(tdf[tdf['net_pnl'] <= 0])
    
    print(f"\n{'='*60}")
    print(f"RECONSTRUCTED TRADES: {total}")
    print(f"Wins: {wins} | Losses: {losses} | WR: {wins/total*100:.1f}%")
    print(f"Total Net PnL: ${tdf['net_pnl'].sum():.6f}")
    print(f"Total Fees: ${tdf['fee'].sum():.6f}")
    if wins > 0:
        print(f"Avg Win: ${tdf[tdf['net_pnl']>0]['net_pnl'].mean():.6f}")
    if losses > 0:
        print(f"Avg Loss: ${tdf[tdf['net_pnl']<=0]['net_pnl'].mean():.6f}")
    print(f"Avg Duration: {tdf['duration_s'].mean():.0f}s")
    
    print(f"\n=== WR POR ESTRATEGIA ===")
    for strat, grp in tdf.groupby('strategy'):
        w = len(grp[grp['net_pnl'] > 0])
        t = len(grp)
        wr = w/t*100 if t > 0 else 0
        pnl = grp['net_pnl'].sum()
        avg_dur = grp['duration_s'].mean()
        print(f"  {str(strat):45s} | {t:4d} trades | WR: {wr:5.1f}% | PnL: ${pnl:+.6f} | Avg Dur: {avg_dur:.0f}s")
    
    print(f"\n=== POR HORIZONTE (LONG vs SHORT) ===")
    for d, grp in tdf.groupby('direction'):
        w = len(grp[grp['net_pnl'] > 0])
        t = len(grp)
        wr = w/t*100 if t > 0 else 0
        print(f"  {d:8s} | {t:4d} trades | WR: {wr:.1f}% | PnL: ${grp['net_pnl'].sum():+.6f}")
    
    print(f"\n=== POR SYMBOL ===")
    for sym, grp in tdf.groupby('symbol'):
        w = len(grp[grp['net_pnl'] > 0])
        t = len(grp)
        wr = w/t*100 if t > 0 else 0
        print(f"  {sym:12s} | {t:4d} trades | WR: {wr:.1f}% | PnL: ${grp['net_pnl'].sum():+.6f}")
    
    print(f"\n=== 10 PEORES PERDIDAS ===")
    worst = tdf.nsmallest(10, 'net_pnl')
    print(worst[['symbol','strategy','direction','entry_price','exit_price','pnl_pct','net_pnl','duration_s']].to_string())
    
    print(f"\n=== 10 MEJORES GANANCIAS ===")
    best = tdf.nlargest(10, 'net_pnl')
    print(best[['symbol','strategy','direction','entry_price','exit_price','pnl_pct','net_pnl','duration_s']].to_string())

if __name__ == "__main__":
    main()
