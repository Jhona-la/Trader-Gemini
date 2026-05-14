import json
import glob
import os
from collections import defaultdict

log_files = sorted(glob.glob("logs/bot_*.json*"), reverse=True)
if not log_files:
    print("No log files found.")
    exit()

recent_logs = log_files[:2] # Look at the most recent logs

trades = []
current_trade = defaultdict(dict)

for file_path in recent_logs:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                msg = data.get("msg", "")
                ts = data.get("ts", "")
                
                # Identify trade closes
                if "[Tracker]" in msg and ("WIN" in msg or "LOSS" in msg):
                    parts = msg.split("|")
                    if len(parts) >= 5:
                        reason = parts[0].split("]")[-1].strip()
                        symbol = parts[1].strip()
                        horizon = parts[2].strip()
                        result = parts[3].strip() # WIN or LOSS
                        pnl_part = parts[4].strip() if len(parts) > 4 else ""
                        
                        trades.append({
                            "ts": ts,
                            "symbol": symbol,
                            "horizon": horizon,
                            "reason": reason,
                            "result": result,
                            "pnl": pnl_part
                        })
            except Exception:
                continue

# Print last 50 trades
print(f"Total trades found in {recent_logs}: {len(trades)}")
print("=== ÚLTIMOS 50 TRADES ===")
for t in trades[-50:]:
    print(f"[{t['ts']}] {t['symbol']} ({t['horizon']}) - {t['result']} | Reason: {t['reason']} | PnL: {t['pnl']}")

# Summarize winning vs losing reasons
win_reasons = defaultdict(int)
loss_reasons = defaultdict(int)

for t in trades:
    if "WIN" in t['result']:
        win_reasons[t['reason']] += 1
    else:
        loss_reasons[t['reason']] += 1

print("\n=== RESUMEN GANADORES ===")
for r, c in win_reasons.items():
    print(f"{r}: {c}")

print("\n=== RESUMEN PERDEDORES ===")
for r, c in loss_reasons.items():
    print(f"{r}: {c}")

