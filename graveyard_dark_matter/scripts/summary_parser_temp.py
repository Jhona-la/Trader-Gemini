import glob
import os
import json

newest = max(glob.glob('results/backtests/god_mode_*.json'), key=os.path.getctime)
print(f"File: {newest}")

with open(newest, "r") as f:
    data = json.load(f)

print(f"Metrics: {json.dumps(data['metrics'], indent=2)}")
print("\nStrategy Attribution:")
for k, v in data['strategy_attribution'].items():
    print(f"{k}: Trades {v['trades']}, WR {v['win_rate']:.2%}, PnL {v['pnl']:.4f}")

if 'forensic_v10' in data:
    print(f"\nRejections: {json.dumps(data['forensic_v10']['rejection_reasons'], indent=2)}")
