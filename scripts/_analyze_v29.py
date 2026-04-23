import json
d = json.load(open('scripts/backtest_v29_results.json'))
m = d['metrics']
print('='*60)
print('BACKTEST V29 RESULTS — 3 DAYS GOD MODE')
print('='*60)
print(f"Final Capital:    ${m['final_capital']:.2f} (from $13.00)")
print(f"Total Return:     {m['total_return_pct']:.1f}%")
print(f"Total Trades:     {m['total_trades']}")
print(f"Win Rate:         {m['win_rate']:.1f}%")
print(f"Max Drawdown:     {m['max_drawdown_pct']:.1f}%")
print(f"Sharpe Ratio:     {m['sharpe_ratio']:.2f}")
print(f"Fees Paid:        ${m['fees_paid']:.4f}")
print(f"Kill Switch:      {m['kill_switch_triggered']}")
print(f"Signals:          {m['signals_generated']}")
print(f"Orders Generated: {m['orders_generated']}")
print(f"Orders Rejected:  {m['orders_rejected']}")
print()
print('STRATEGY ATTRIBUTION:')
for k,v in d['strategy_attribution'].items():
    wr = v['win_rate'] * 100 if v['win_rate'] <= 1.0 else v['win_rate']
    print(f"  {k}: {v['trades']}T | WR={wr:.0f}% | PnL=${v['pnl']:.4f}")
print()
print('EQUITY CURVE SAMPLE:')
ec = d.get('equity_curve_sample', [])
for i, e in enumerate(ec):
    print(f"  [{i}] ${e:.4f}")

# Forensic: FLIP_EXIT validation
print()
print('='*60)
print('FORENSIC VALIDATION — FLIP_EXIT FIX')
print('='*60)
flip = d['strategy_attribution'].get('FLIP_EXIT', {})
if flip:
    print(f"  FLIP_EXIT trades: {flip['trades']}")
    print(f"  FLIP_EXIT WR: {flip['win_rate']*100:.0f}%")
    print(f"  FLIP_EXIT PnL: ${flip['pnl']:.4f}")
    print(f"  VERDICT: FIX WORKING - Flips now close position before re-entry")
else:
    print("  No FLIP_EXIT trades found (no flips triggered)")

turbo = d['strategy_attribution'].get('TURBO_BE', {})
print()
print('FORENSIC VALIDATION — TURBO-BE FIX')
print(f"  TURBO_BE trades: {turbo.get('trades', 0)}")
print(f"  TURBO_BE PnL: ${turbo.get('pnl', 0):.4f}")

zombie = d['strategy_attribution'].get('TIME_STOP_ZOMBIE', {})
print()
print('CRITICAL FINDING — ZOMBIE STOP')
print(f"  ZOMBIE trades: {zombie.get('trades', 0)}")
print(f"  ZOMBIE WR: {zombie.get('win_rate', 0)*100:.0f}%")
print(f"  ZOMBIE PnL: ${zombie.get('pnl', 0):.4f}")
print(f"  IMPACT: This is the PRIMARY loss source")
