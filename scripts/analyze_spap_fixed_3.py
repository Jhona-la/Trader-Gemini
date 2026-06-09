import json

f = 'results/backtests/god_mode_873d0a72_1d.json'
d = json.load(open(f))
m = d.get('metrics', {})
print(f'=== SPAP Unified Fixed ===')
print(f'Capital: ${m.get("final_capital", 0):.2f}')
print(f'Return: {m.get("total_return_pct", 0):.2f}%')
print(f'Trades: {m.get("wins",0)}W/{m.get("losses",0)}L WR={m.get("win_rate",0):.1f}%')

th = d.get('trade_history', {})
all_trades = []
for cat, trades in th.items():
    if isinstance(trades, list):
        all_trades.extend(trades)
all_trades.sort(key=lambda x: str(x.get('exit_time', x.get('entry_time', ''))))
print(f'\nTrade Details ({len(all_trades)} trades):')
for t in all_trades:
    pnl = t.get('net_pnl', t.get('pnl', 0))
    dur = t.get('duration_seconds', 0)
    sym = t.get('symbol', '?')
    d_dir = t.get('direction', '?')
    exit_r = t.get('exit_reason', '?')
    horizon = t.get('horizon', '?')
    print(f'  {sym:12s} {horizon:15s} {d_dir:6s} PnL=${pnl:+.4f} Dur={dur:6.0f}s Exit={exit_r}')
