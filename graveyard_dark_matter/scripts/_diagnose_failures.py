import json
import pandas as pd
import argparse

def diagnose(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        print(f"================ SYMBOL FORENSIC AUDIT: {input_file} =================\n")
        
        # Check if new God Mode v3 output
        if 'trade_history' in data:
            print("Detected God Mode Unified output format.")
            for horizon, ledger in data['trade_history'].items():
                print(f"\n👉 HORIZON: {horizon.upper()}")
                for trade in ledger:
                    pnl = trade.get('net_pnl', 0)
                    wr = 100 if pnl > 0 else 0
                    strat = trade.get('metadata', {}).get('strategy', 'Unknown')
                    sym = trade.get('symbol', 'Unknown')
                    
                    status_icon = "❌" if pnl < 0 else "✅"
                    print(f"  🔹 {sym:10} [{strat:12}] {status_icon} Net PNL: ${pnl:8.4f}")
            return
            
        for horizon, symbols in data.items():
            print(f"👉 HORIZON: {horizon}")
            
            if not isinstance(symbols, dict): continue
            
            for symbol, strategies in symbols.items():
                if symbol == 'aggregated' or not isinstance(strategies, dict): continue
                
                print(f"  🔹 {symbol}")
                for strat_name, stats in strategies.items():
                    if not isinstance(stats, dict): continue
                    
                    pnl = stats.get('pnl_usd', 0)
                    wr = stats.get('win_rate', 0)
                    trades = stats.get('trades', 0)
                    dd = stats.get('max_drawdown', 0)
                    
                    status_icon = "❌" if pnl < 0 else "✅"
                    print(f"     [{strat_name:12}] {status_icon} PNL: ${pnl:8.4f} | WR: {wr:5.1f}% | Trades: {trades:3} | DD: {dd:5.2f}%")
            print("-" * 50)
    except Exception as e:
        print(f"Error parsing json: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose backtest failures')
    parser.add_argument('--input', type=str, required=True, help='Path to the results JSON file')
    args = parser.parse_args()
    diagnose(args.input)
