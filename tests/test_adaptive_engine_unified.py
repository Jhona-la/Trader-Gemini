"""Quick test for the unified AdaptiveMLParameterEngine"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.components.adaptive_engine import AdaptiveMLParameterEngine

# Test 1: Scalping via horizon_days
e1 = AdaptiveMLParameterEngine(horizon_days=1, profile_override={'ml_lookahead': 3})
print(f"Scalping: profile={e1.profile}, sl_range={e1.r['sl_mult']}, tp_range={e1.r['tp_mult']}")
print(f"  sl_mult={e1.get('sl_mult'):.4f}, tp_mult={e1.get('tp_mult'):.4f}")
print(f"  model_suffix={e1.get_model_suffix()}")
assert e1.profile == 'scalping'
assert e1.r['sl_mult'][1] <= 0.50, f"Scalping SL upper bound should be 0.50, got {e1.r['sl_mult'][1]}"
assert e1.get_model_suffix() == '_scalping'

# Test 2: Swing via horizon_str
e2 = AdaptiveMLParameterEngine(horizon_str='SWING')
print(f"Swing: profile={e2.profile}, sl_range={e2.r['sl_mult']}, tp_range={e2.r['tp_mult']}")
print(f"  model_suffix={e2.get_model_suffix()}")
assert e2.profile == 'swing'
assert e2.r['sl_mult'][0] >= 1.0, f"Swing SL lower bound should be 1.0, got {e2.r['sl_mult'][0]}"
assert e2.get_model_suffix() == '_swing'

# Test 3: Feedback trade with MAE/MFE (backtest mode)
e1.feedback_trade(pnl_pct=0.003, mae_pct=-0.001, mfe_pct=0.004)
e1.feedback_trade(pnl_pct=-0.002, mae_pct=-0.003, mfe_pct=0.001)
print(f"  After 2 trades: sl_mult={e1.get('sl_mult'):.4f}, tp_mult={e1.get('tp_mult'):.4f}")
assert e1.r['sl_mult'][0] <= e1.params['sl_mult'] <= e1.r['sl_mult'][1], "SL out of bounds!"

# Test 4: profile_override
e3 = AdaptiveMLParameterEngine(horizon_days=1, profile_override={'ml_lookahead': 5, 'ml_retrain': 200})
assert e3.get('lookahead') == 5
assert e3.get('retrain_interval') == 200

print("\n✅ ALL TESTS PASSED — Single Source of Truth verified!")
print(f"   Scalping SL: {e1.r['sl_mult']} (CORRECTED from 1.2-3.0)")
print(f"   Swing SL: {e2.r['sl_mult']}")
