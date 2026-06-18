import pandas as pd
import sys
import os
import argparse
from typing import Dict, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="Toxicity Analyzer for Trader Gemini Backtests")
    parser.add_argument("--csv", type=str, default=r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\dashboard\data\backtest_temp\bt_trades.csv", help="Path to bt_trades.csv")
    parser.add_argument("--auto-ban-threshold", type=float, default=-5.0, help="PnL threshold to consider an asset toxic (percentage)")
    return parser.parse_args()

def analyze_trades(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    # Columns: Timestamp,Symbol,Action,Side,Quantity,Price,Value,Strategy,Reason,Version,ExecutionTime,Metadata
    columns = ["Timestamp", "Symbol", "Action", "Side", "Quantity", "Price", "Value", "Strategy", "Reason", "Version", "ExecutionTime", "Metadata"]
    
    try:
        df = pd.read_csv(csv_path, names=columns, header=None)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return
    
    # We want to match FILLs. 
    # Since backtest uses 'FLIP_EXIT' or normal 'FILL' for exits, we just track position sizes per symbol.
    
    symbol_stats = {}
    open_positions = {} # symbol -> {side, qty, avg_price}
    
    for idx, row in df.iterrows():
        if row['Action'] != 'FILL':
            continue
            
        sym = row['Symbol']
        side = row['Side'] # BUY or SELL
        qty = float(row['Quantity'])
        price = float(row['Price'])
        
        if sym not in symbol_stats:
            symbol_stats[sym] = {'trades': 0, 'wins': 0, 'losses': 0, 'realized_pnl_pct': 0.0, 'realized_pnl_usd': 0.0}
            open_positions[sym] = {'qty': 0.0, 'avg_price': 0.0}
            
        pos = open_positions[sym]
        
        # Determine if this is opening or closing
        is_closing = False
        if pos['qty'] > 0: # We are LONG
            if side == 'SELL':
                is_closing = True
        elif pos['qty'] < 0: # We are SHORT
            if side == 'BUY':
                is_closing = True
                
        if is_closing:
            # Calculate PnL
            # For simplicity in this script, assuming we close the exact same qty we opened.
            # If qty matches, we close. If it's a FLIP, it might be 2x. Let's just track the value.
            entry_price = pos['avg_price']
            exit_price = price
            
            close_qty = min(abs(pos['qty']), qty)
            
            if pos['qty'] > 0: # Long exit
                pnl_usd = (exit_price - entry_price) * close_qty
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else: # Short exit
                pnl_usd = (entry_price - exit_price) * close_qty
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                
            symbol_stats[sym]['trades'] += 1
            symbol_stats[sym]['realized_pnl_usd'] += pnl_usd
            symbol_stats[sym]['realized_pnl_pct'] += pnl_pct
            
            if pnl_usd > 0:
                symbol_stats[sym]['wins'] += 1
            else:
                symbol_stats[sym]['losses'] += 1
                
            # Update pos
            if pos['qty'] > 0:
                pos['qty'] -= close_qty
            else:
                pos['qty'] += close_qty
                
            # If it's a flip, the remaining qty is the new position
            remaining_qty = qty - close_qty
            if remaining_qty > 1e-8:
                pos['qty'] = remaining_qty if side == 'BUY' else -remaining_qty
                pos['avg_price'] = price
                
            if abs(pos['qty']) < 1e-8:
                pos['qty'] = 0.0
                pos['avg_price'] = 0.0
                
        else:
            # Opening or adding to position
            new_qty = pos['qty'] + (qty if side == 'BUY' else -qty)
            if pos['qty'] == 0:
                pos['avg_price'] = price
            else:
                # Weighted avg price
                total_val = abs(pos['qty']) * pos['avg_price'] + qty * price
                pos['avg_price'] = total_val / abs(new_qty)
            pos['qty'] = new_qty

    return symbol_stats

def main():
    args = parse_args()
    print("=" * 60)
    print("🔬 TOXICITY ANALYZER - Risk Manager Forensics")
    print("=" * 60)
    
    stats = analyze_trades(args.csv)
    if not stats:
        return
        
    print(f"\n{ 'Symbol':<10} | {'Trades':<8} | {'WinRate':<8} | {'PnL (USD)':<10} | {'PnL (%)':<10} | {'Status'}")
    print("-" * 70)
    
    toxic_assets = []
    
    for sym, s in sorted(stats.items(), key=lambda x: x[1]['realized_pnl_usd'], reverse=True):
        if s['trades'] == 0:
            continue
            
        wr = (s['wins'] / s['trades']) * 100
        pnl_usd = s['realized_pnl_usd']
        pnl_pct = s['realized_pnl_pct']
        
        status = "✅ HEALTHY"
        if pnl_pct <= args.auto_ban_threshold or (s['trades'] > 5 and wr < 40.0):
            status = "💀 TOXIC"
            toxic_assets.append(sym)
        elif pnl_pct < 0:
            status = "⚠️ WARNING"
            
        print(f"{sym:<10} | {s['trades']:<8} | {wr:>5.1f}%   | ${pnl_usd:>8.3f} | {pnl_pct:>8.2f}% | {status}")

    print("\n" + "=" * 60)
    if toxic_assets:
        print(f"🚨 FOUND {len(toxic_assets)} TOXIC ASSETS DRAINING CAPITAL:")
        for t in toxic_assets:
            print(f"   - {t} (Recommendation: Add to RiskManager Blacklist)")
    else:
        print("✅ No highly toxic assets detected yet. Portfolio is stable.")

if __name__ == "__main__":
    main()
