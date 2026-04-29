import os
import json

def find_highest_wr():
    path = "results/backtests"
    max_wr = 0
    best_file = None
    
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(path, filename), 'r') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    wr = metrics.get('win_rate', 0)
                    if wr > max_wr:
                        max_wr = wr
                        best_file = filename
            except:
                pass
                
    if best_file:
        print(f"Highest Global WR: {max_wr}% in {best_file}")
    else:
        print("No backtests found.")

if __name__ == "__main__":
    find_highest_wr()
