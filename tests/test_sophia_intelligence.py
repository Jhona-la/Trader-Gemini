"""SOPHIA-INTELLIGENCE Protocol: Verification Test Suite"""
import sys, os
# Ensure project root is on path (parent of tests/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import numpy as np

# 1. Verify all sophia modules parse
for f in ['sophia/__init__.py', 'sophia/intelligence.py', 'sophia/narrative.py', 'sophia/post_mortem.py']:
    ast.parse(open(f, encoding='utf-8').read())
print('✅ All sophia modules parse OK')

# 2. Verify integration files parse
ast.parse(open('strategies/technical.py', encoding='utf-8').read())
print('✅ technical.py parses OK')
ast.parse(open('core/portfolio.py', encoding='utf-8').read())
print('✅ portfolio.py parses OK')

# 3. Import tests
from sophia.intelligence import (
    BayesianCalibrator, FeatureAttributor, SurvivalEstimator,
    AlphaDecayFunction, EntropyAnalyzer, TailRiskAnalyzer,
    SophiaReport, SophiaIntelligence
)
from sophia.narrative import NarrativeGenerator
from sophia.post_mortem import PostMortemComparator, TradeIntent
print('✅ All imports OK')

# --- Test BayesianCalibrator ---
cal = BayesianCalibrator(prior_alpha=10, prior_beta=10)
p = cal.compute_posterior(0.7, 0.5, 1.0)
assert 0.0 < p < 1.0, f'P out of range: {p}'
print(f'✅ BayesianCalibrator: P(Win)={p:.4f} (in [0,1])')

# Monotonicity
p_low = cal.compute_posterior(0.2, 0.0, 0.0)
p_high = cal.compute_posterior(0.9, 0.0, 0.0)
assert p_high > p_low, f'Non-monotonic: {p_low} vs {p_high}'
print(f'✅ Monotonic: P(0.2)={p_low:.4f} < P(0.9)={p_high:.4f}')

# Prior update
cal.sync_from_risk_manager(60, 40)
p_synced = cal.get_prior_win_rate()
assert abs(p_synced - 0.5833) < 0.01  # (10+60)/(10+10+60+40) = 70/120
print(f'✅ Prior Sync: P_prior={p_synced:.4f} (expected ~0.5833)')

# --- Test FeatureAttributor ---
attrib = FeatureAttributor(cal)
features = {
    'rsi': 25.0, 'bb_position': 0.1, 'adx': 30.0, 
    'volume_ratio': 2.5, 'confluence': 0.8, 'macd_hist': 0.5, 
    'trend_aligned': 1.0, 'atr_pct': 0.008
}
attrs = attrib.compute_attributions(features)
assert len(attrs) <= 5
for a in attrs:
    d = a.to_dict()
    assert 'feature' in d and 'contribution' in d
print(f'✅ FeatureAttributor: Top-{len(attrs)} features computed')

# --- Test SurvivalEstimator ---
surv = SurvivalEstimator(bar_minutes=5.0)
est = surv.estimate(50000.0, 0.01, 0.005, returns=np.random.randn(100) * 0.001)
assert est.time_to_tp_mins > 0
assert est.expected_exit_mins > 0
print(f'✅ SurvivalEstimator: E[T]={est.expected_exit_mins:.1f}min, GARCH_vol={est.garch_volatility}')

# --- Test AlphaDecayFunction ---
decay = AlphaDecayFunction(min_threshold=0.30)
exp_mins = decay.get_expiration_time_mins(0.8, ttl_seconds=180)
assert exp_mins > 0
assert decay.is_thesis_expired(0.8, 1000, 180)
assert not decay.is_thesis_expired(0.8, 10, 180)
print(f'✅ AlphaDecayFunction: Expires at {exp_mins:.1f}min')

# --- Test EntropyAnalyzer ---
h, label = EntropyAnalyzer.from_signal(0.85, 'LONG')
assert h >= 0
assert label in ('Alta Convicción', 'Moderada', 'Indeciso')
print(f'✅ EntropyAnalyzer: H={h:.4f} ({label})')

# Low confidence → higher entropy
h_low, label_low = EntropyAnalyzer.from_signal(0.55, 'LONG')
assert h_low > h, f'Entropy should increase with uncertainty: {h_low} vs {h}'
print(f'✅ Entropy monotonic: H(0.85)={h:.4f} < H(0.55)={h_low:.4f}')

# --- Test TailRiskAnalyzer ---
normal_rets = np.random.randn(1000) * 0.01
tail = TailRiskAnalyzer.analyze(normal_rets)
assert hasattr(tail, 'excess_kurtosis')
print(f'✅ TailRiskAnalyzer (normal): Kurt={tail.excess_kurtosis:.4f}, FatTails={tail.has_fat_tails}')

# Fat tail synthetic data
fat_rets = np.concatenate([np.random.randn(900) * 0.01, np.random.randn(100) * 0.1])
tail_fat = TailRiskAnalyzer.analyze(fat_rets)
assert tail_fat.excess_kurtosis > tail.excess_kurtosis
print(f'✅ TailRiskAnalyzer (fat): Kurt={tail_fat.excess_kurtosis:.4f}, FatTails={tail_fat.has_fat_tails}')

# --- Test NarrativeGenerator ---
narrative = NarrativeGenerator.generate_intention(
    'BTC/USDT', 'LONG', 0.78, 12.0,
    [{'feature': 'RSI', 'contribution': 0.12}],
    {'long_mean_rev': True}, 'Alta Convicción', False, 50000.0
)
assert len(narrative) > 50
print(f'✅ NarrativeGenerator: "{narrative[:80]}..."')

pm_narr = NarrativeGenerator.generate_post_mortem_narrative(
    'BTC/USDT', 'LONG', 0.78, 'WIN', 0.0484, 5.23, 11.5, 12.0
)
assert 'WIN' in pm_narr
print(f'✅ PostMortem Narrative: "{pm_narr[:80]}..."')

# --- Test PostMortemComparator ---
pm = PostMortemComparator(rolling_window=100)
pm.store_intent('test-id-001', 'BTC/USDT', 'LONG',
    {'win_probability': 0.75, 'expected_exit_mins': 10.0, 'signal_strength': 0.8},
    'Test narrative')
result = pm.compute_post_mortem('test-id-001', actual_pnl=5.0, duration_seconds=600)
assert result is not None
assert 0 <= result.brier_score <= 1
expected_brier = (0.75 - 1.0) ** 2  # = 0.0625
assert abs(result.brier_score - expected_brier) < 0.001
print(f'✅ PostMortem WIN: Brier={result.brier_score:.4f} (expected {expected_brier:.4f})')

# Test LOSS
pm.store_intent('test-id-002', 'ETH/USDT', 'SHORT',
    {'win_probability': 0.65, 'expected_exit_mins': 8.0, 'signal_strength': 0.7},
    'Test loss')
result2 = pm.compute_post_mortem('test-id-002', actual_pnl=-3.0, duration_seconds=900)
assert result2 is not None
expected_brier2 = (0.65 - 0.0) ** 2  # = 0.4225
assert abs(result2.brier_score - expected_brier2) < 0.001
print(f'✅ PostMortem LOSS: Brier={result2.brier_score:.4f} (expected {expected_brier2:.4f})')

# Calibration status
cal_status = pm.get_calibration_status()
assert 'rolling_brier' in cal_status
assert 'status' in cal_status
print(f'✅ Calibration Status: {cal_status}')

# --- Test SophiaReport ---
report = SophiaReport(
    win_probability=0.78, prior_win_rate=0.55,
    top_features=[{'feature': 'RSI', 'value': 28.3, 'contribution': 0.12}],
    expected_exit_mins=12.0, time_to_tp_mins=15.0, time_to_sl_mins=8.0,
    alpha_decay_threshold_mins=5.2,
    decision_entropy=0.35, entropy_label='Alta Convicción',
    excess_kurtosis=1.2, skewness=-0.3, tail_risk_warning=False,
    symbol='BTC/USDT', direction='LONG', signal_strength=0.75,
)
d = report.to_dict()
assert d['win_probability'] == 0.78
log = report.to_log_line()
assert 'SOPHIA' in log
print(f'✅ SophiaReport.to_log_line(): {log}')
print(f'✅ SophiaReport.to_dict(): {len(d)} fields')

print()
print('=' * 60)
print('🎉 ALL 16 SOPHIA-INTELLIGENCE TESTS PASSED!')
print('=' * 60)
