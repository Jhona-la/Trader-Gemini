
import json
import pandas as pd
from collections import Counter

def analyze_audit(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    summary = []
    all_decisions = []
    
    for entry in data:
        symbol = entry['symbol']
        tf = entry['timeframe']
        metrics = entry['metrics']
        
        row = {
            'symbol': symbol,
            'timeframe': tf,
            'pnl_usd': metrics['total_return'],
            'win_rate': metrics['win_rate'],
            'total_trades': metrics['total_trades'],
            'error': entry.get('error')
        }
        summary.append(row)
        
        for dec in entry['decisions']:
            if dec['pnl_usd'] < 0:
                reasoning = dec['reasoning']
                all_decisions.append({
                    'symbol': symbol,
                    'attribution': reasoning['attribution'],
                    'pnl_usd': dec.get('pnl_usd')
                })
    
    df_summary = pd.DataFrame(summary)
    df_losses = pd.DataFrame(all_decisions)
    
    print("=== GLOBAL PERFORMANCE SUMMARY ===")
    global_pnl = df_summary['pnl_usd'].sum()
    print(f"Total PnL: ${global_pnl:.2f}")
    
    print("\n=== PERFORMANCE BY SYMBOL (AVG ACROSS TF) ===")
    symbol_perf = df_summary.groupby('symbol')['pnl_usd'].sum().sort_values()
    print(symbol_perf)
    
    print("\n=== LOSS ATTRIBUTION ANALYSIS ===")
    if not df_losses.empty:
        attr_counts = df_losses['attribution'].value_counts()
        print(attr_counts)
        
        print("\n=== TOP 5 SYMBOLS BY LOSSES ===")
        print(symbol_perf.head(5))
    else:
        print("No losses recorded in decisions.")

if __name__ == "__main__":
    analyze_audit('massive_audit_raw.json')
