import csv
import json

def match_trades():
    print("Matching trades from diag_trades.csv...")
    try:
        positions = {}
        trades = []
        
        with open('dashboard/data/diag_temp/diag_trades.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                sym = row['symbol']
                qty = float(row['quantity'])
                price = float(row['price'])
                direction = row['direction'] # 'BUY' or 'SELL'
                details_str = row['details']
                
                is_close = 'is_close": true' in details_str.lower() or 'is_close": True' in details_str or 'exit' in row['setup_type'].lower()
                is_exit = 'is_exit": true' in details_str.lower() or 'is_exit": True' in details_str
                
                if sym not in positions:
                    positions[sym] = {'qty': 0, 'entry_price': 0, 'direction': None, 'entry_time': None}
                    
                pos = positions[sym]
                
                # Check for position close
                if is_close or is_exit or (pos['qty'] != 0 and pos['direction'] != direction):
                    if pos['qty'] == 0:
                        continue # Orphaned close
                        
                    entry_price = pos['entry_price']
                    exit_price = price
                    
                    if pos['direction'] == 'BUY':
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                        
                    pnl_pct -= 0.0004 # Fees
                    
                    trades.append({
                        'symbol': sym,
                        'direction': pos['direction'],
                        'entry_time': pos['entry_time'],
                        'exit_time': row['datetime'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'setup': pos['setup_type'],
                        'exit_setup': row['setup_type']
                    })
                    
                    pos['qty'] = 0
                    pos['entry_price'] = 0
                    pos['direction'] = None
                else:
                    pos['qty'] = qty
                    pos['entry_price'] = price
                    pos['direction'] = direction
                    pos['entry_time'] = row['datetime']
                    pos['setup_type'] = row['setup_type']
                    
        if not trades:
            print("No completed trades found.")
            return
            
        wins = sum(1 for t in trades if t['pnl_pct'] > 0)
        total = len(trades)
        wr = (wins / total) * 100
        avg_pnl = sum(t['pnl_pct'] for t in trades) / total * 100
        
        print(f"\nTotal Matched Trades: {total}")
        print(f"Wins: {wins} | Losses: {total - wins}")
        print(f"Win Rate: {wr:.2f}%")
        print(f"Average PnL per trade: {avg_pnl:.3f}%")
        
        # Sort by PnL
        sorted_trades = sorted(trades, key=lambda x: x['pnl_pct'])
        print("\nWorst Trades:")
        for t in sorted_trades[:10]:
            print(f"{t['symbol']} | {t['direction']:4} | PnL: {t['pnl_pct']*100:6.3f}% | Setup: {t['setup']} -> {t['exit_setup']}")
            
        # Group by setup
        setups = {}
        for t in trades:
            s = t['setup']
            if s not in setups:
                setups[s] = {'total': 0, 'wins': 0, 'pnl_sum': 0}
            setups[s]['total'] += 1
            setups[s]['pnl_sum'] += t['pnl_pct']
            if t['pnl_pct'] > 0:
                setups[s]['wins'] += 1
                
        print("\nTrades by Setup:")
        for s, stats in setups.items():
            s_wr = (stats['wins'] / stats['total']) * 100
            s_avg = (stats['pnl_sum'] / stats['total']) * 100
            print(f"{s:30} | Total: {stats['total']:4} | WR: {s_wr:5.2f}% | AvgPnL: {s_avg:6.3f}%")
            
    except Exception as e:
        print("Error:", e)

match_trades()
