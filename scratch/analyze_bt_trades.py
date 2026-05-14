import pandas as pd
import json

csv_path = "dashboard/data/backtest_temp/bt_trades.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trades from {csv_path}")
    
    # Cast to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['fill_cost'] = pd.to_numeric(df['fill_cost'], errors='coerce')
    
    print("\nColumns:", list(df.columns))
    
    if 'details' in df.columns:
        print("\nExit strategies from details:")
        exit_counts = {}
        for d in df['details'].dropna():
            try:
                meta = json.loads(str(d).replace("'", '"'))
                reason = meta.get('close_reason', 'UNKNOWN')
                if meta.get('is_exit', False):
                    exit_counts[reason] = exit_counts.get(reason, 0) + 1
            except:
                pass
        for r, c in sorted(exit_counts.items(), key=lambda x: -x[1]):
            print(f"{r}: {c}")
            
except Exception as e:
    print(f"Failed to load: {e}")
