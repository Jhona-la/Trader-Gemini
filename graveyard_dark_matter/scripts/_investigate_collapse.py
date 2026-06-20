import json

with open('scripts/backtest_v29_results.json', 'r') as f:
    d = json.load(f)

print("--- EQUITY CURVE ---")
ec = d['equity_curve_sample']
for i, e in enumerate(ec):
    print(f"[{i}] ${e:.4f}")

print("\n--- ALL TRADES ---")
all_trades = []
for horizon, trades in d['trade_history'].items():
    for t in trades:
        all_trades.append((horizon, t))

# Sort by closed_at time
all_trades.sort(key=lambda x: x[1]['closed_at'])

for horizon, t in all_trades[-10:]:  # Last 10 trades
    print(f"[{t.get('closed_at')}] {t.get('symbol')} {horizon} {t.get('direction')} | "
          f"Qty: {t.get('quantity')} | Entry: {t.get('entry_price')} | Exit: {t.get('exit_price')} | "
          f"Gross PnL: {t.get('gross_pnl')} | Net PnL: {t.get('net_pnl')} | Reason: {t.get('exit_reason')}")

print("\n--- BIGGEST LOSERS ---")
all_trades.sort(key=lambda x: x[1]['net_pnl'])
for horizon, t in all_trades[:5]:
    print(f"[{t.get('closed_at')}] {t.get('symbol')} {horizon} {t.get('direction')} | "
          f"Net PnL: {t.get('net_pnl')} | Reason: {t.get('exit_reason')}")

