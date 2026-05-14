import json
import os
import glob
from collections import defaultdict

def audit_trades():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print("No logs directory found.")
        return

    # Find the most recent bot log file
    log_files = glob.glob(os.path.join(log_dir, "bot_*.json"))
    if not log_files:
        print("No bot_*.json files found.")
        return
        
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Auditing Log File: {latest_log}")
    
    trades = defaultdict(list)
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                event = data.get('event', '')
                
                if event == 'ORDER_FILLED':
                    trades[data.get('symbol', 'UNKNOWN')].append(data)
                elif event == 'SIGNAL_GENERATED':
                    pass
                elif 'error' in line.lower() or 'timeout' in line.lower():
                    pass # We could track errors here too
                    
            except json.JSONDecodeError:
                continue

    # Process and summarize
    total_trades = 0
    winners = 0
    losers = 0
    
    print("\n" + "="*50)
    print("🔬 AUTOPSIA CUÁNTICA DE TRADES")
    print("="*50)
    
    for symbol, events in trades.items():
        print(f"\n🪙 Símbolo: {symbol}")
        # Group fills by horizon/side or time proximity to reconstruct trades
        # As this is a generic parser, we will print the fill events to understand behavior
        for i, fill in enumerate(events[-20:]): # Last 20 fills
            side = fill.get('side', 'UNKNOWN')
            is_close = fill.get('is_close', False)
            price = fill.get('price', 0.0)
            qty = fill.get('quantity', 0.0)
            print(f"  [{i}] {'🔴 CIERRE' if is_close else '🟢 APERTURA'} | {side} | P: {price} | Q: {qty}")
            
if __name__ == "__main__":
    audit_trades()
