import pandas as pd
import json
import os
import glob

def analyze():
    # Find the latest subfolder in audits
    base_dir = "audits/forensic_audits"
    if not os.path.exists(base_dir):
        print("No forensic_audits directory found.")
        return

    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    if not subfolders:
        print("No audit subfolders found.")
        return

    latest_folder = max(subfolders, key=os.path.getmtime)
    print(f"Analyzing latest audit folder: {latest_folder}")

    # Read trade_replay.parquet
    trade_replay_path = os.path.join(latest_folder, "trade_replay.parquet")
    if os.path.exists(trade_replay_path):
        df = pd.read_parquet(trade_replay_path)
        print("\n--- Trade Replay Summary ---")
        print(f"Total Trades: {len(df)}")
        if len(df) > 0:
            print("\nExit Reasons:")
            print(df['exit_reason'].value_counts())
            print("\nSetup Types:")
            if 'setup_type' in df.columns:
                 print(df['setup_type'].value_counts())
            else:
                 print("No setup_type column found.")
            
            # Print losing trades details
            losers = df[df['pnl_pct'] < 0]
            if len(losers) > 0:
                print(f"\n--- Losing Trades ({len(losers)}) ---")
                cols = ['symbol', 'direction', 'entry_time', 'exit_time', 'exit_reason', 'pnl_pct']
                if 'setup_type' in df.columns:
                    cols.append('setup_type')
                print(losers[cols])
            else:
                print("\nNo losing trades found! 100% Win Rate.")
                print("PnL summary:")
                print(df['pnl_pct'].describe())
    else:
        print("No trade_replay.parquet found.")

if __name__ == "__main__":
    analyze()
