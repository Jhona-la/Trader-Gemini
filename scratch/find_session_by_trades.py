import os
import json

def find_session_by_trades(target_trades):
    path = "results/backtests"
    matches = []
    
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(path, filename), 'r') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    total_trades = metrics.get('total_trades', 0)
                    if total_trades == target_trades:
                        matches.append((filename, metrics.get('win_rate')))
            except:
                pass
                
    if matches:
        for m in matches:
            print(f"Match: {m[0]} | WR: {m[1]}%")
    else:
        print(f"No backtests found with {target_trades} trades.")

if __name__ == "__main__":
    find_session_by_trades(17)
