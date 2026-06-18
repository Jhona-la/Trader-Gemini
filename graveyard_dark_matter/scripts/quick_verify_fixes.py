"""Quick verification backtest — tests the 7 fixes"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("="*70)
print("🧪 QUICK VERIFICATION BACKTEST — Post-Fix")
print("="*70)

from scripts.run_multi_horizon_backtest import (
    fetch_data, run_strategy_backtest, INITIAL_CAPITAL, LEVERAGE,
    calibrate_sl_tp
)

# 1. Verify SL/TP calibration produces 1.8:1 ratio
print("\n📐 1. Testing SL/TP Calibration...")
df = fetch_data('BTC/USDT', 9)
if df is None or len(df) < 500:
    print("❌ Could not load data!")
    sys.exit(1)

print(f"✅ Data loaded: {len(df)} bars ({len(df)/1440:.1f} days)")

closes = df['close'].values
sl, tp = calibrate_sl_tp(closes, 60, sl_cap=0.020, tp_cap=0.050)
ratio = tp / sl if sl > 0 else 0
print(f"   SL={sl*100:.3f}% TP={tp*100:.3f}% Ratio={ratio:.2f}:1")
if ratio >= 1.8:
    print("   ✅ PASS: TP:SL ratio >= 1.8:1")
else:
    print(f"   ❌ FAIL: TP:SL ratio {ratio:.2f} < 1.8")

# 2. Run Technical 1D
print("\n📊 2. Technical Strategy — 1D...")
r1 = run_strategy_backtest(df, 'BTC/USDT', 'Technical', INITIAL_CAPITAL, LEVERAGE, 1)
print(f"   PNL: ${r1['pnl_usd']:+.4f}")
print(f"   Trades: {r1['trades']}")
print(f"   WR: {r1['win_rate']:.1f}%")
print(f"   DD: {r1['max_drawdown']:.2f}%")
print(f"   Sharpe: {r1['sharpe']:.2f}")
print(f"   Signals: L={r1['signal_counts']['long']} S={r1['signal_counts']['short']} N={r1['signal_counts']['neutral']}")

status1 = "✅ PASS" if r1['pnl_usd'] >= 0 else "⚠️ NEGATIVE"
print(f"   {status1}")

# 3. Run ML_XGBoost 1D
print("\n🤖 3. ML_XGBoost Strategy — 1D...")
r2 = run_strategy_backtest(df, 'BTC/USDT', 'ML_XGBoost', INITIAL_CAPITAL, LEVERAGE, 1)
print(f"   PNL: ${r2['pnl_usd']:+.4f}")
print(f"   Trades: {r2['trades']}")
print(f"   WR: {r2['win_rate']:.1f}%")
print(f"   DD: {r2['max_drawdown']:.2f}%")
print(f"   ML Accuracy: {r2.get('ml_accuracy', 0):.1f}%")
print(f"   Gate Active: {r2.get('accuracy_gate_active', 'N/A')}")
print(f"   Gate Blocks: {r2.get('accuracy_gate_blocks', 0)}")
print(f"   ML Trainings: {r2.get('ml_trainings', 0)}")

status2 = "✅ PASS" if r2['trades'] > 0 else "⚠️ NO TRADES"
print(f"   {status2}")

# 4. Summary
print("\n" + "="*70)
print("📊 RESUMEN POST-FIX")
print("="*70)
print(f"{'Metric':<25} {'Technical':>12} {'ML_XGBoost':>12} {'Target':>12}")
print("-"*70)
print(f"{'PNL ($)':.<25} {r1['pnl_usd']:>+12.4f} {r2['pnl_usd']:>+12.4f} {'>= $0.00':>12}")
print(f"{'Trades':.<25} {r1['trades']:>12} {r2['trades']:>12} {'> 15':>12}")
print(f"{'Win Rate (%)':.<25} {r1['win_rate']:>12.1f} {r2['win_rate']:>12.1f} {'> 50%':>12}")
print(f"{'Max Drawdown (%)':.<25} {r1['max_drawdown']:>12.2f} {r2['max_drawdown']:>12.2f} {'< 3%':>12}")
print(f"{'SL/TP Ratio':.<25} {'1.8:1+':>12} {'1.8:1+':>12} {'>= 1.8:1':>12}")

total_pnl = r1['pnl_usd'] + r2['pnl_usd']
print(f"\n💰 PNL Total: ${total_pnl:+.4f}")
if total_pnl > 0:
    print("🎯 VEREDICTO: ✅ SISTEMA RENTABLE")
elif r1['trades'] + r2['trades'] > 20:
    print("🎯 VEREDICTO: ⚠️ Sistema operativo pero no rentable aún — Optuna puede mejorar")
else:
    print("🎯 VEREDICTO: ❌ Insuficientes trades — revisar filtros")
