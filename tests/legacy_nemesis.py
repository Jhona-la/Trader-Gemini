"""NÉMESIS-RETROSPECCIÓN Protocol: Verification Test Suite"""
import sys, os
# Ensure project root is on path (parent of tests/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import traceback

# ============================================================
# 1. PARSE CHECK - All modules parse correctly
# ============================================================
MODULES_TO_CHECK = [
    os.path.join('sophia', 'nemesis.py'),
    os.path.join('sophia', 'post_mortem.py'),
    os.path.join('core', 'portfolio.py'),
]

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for mod in MODULES_TO_CHECK:
    fpath = os.path.join(root, mod)
    with open(fpath, 'r', encoding='utf-8') as f:
        try:
            ast.parse(f.read())
        except SyntaxError as e:
            print(f"❌ PARSE FAIL: {mod} → {e}")
            sys.exit(1)
print("✅ All modules parse OK")

# ============================================================
# 2. IMPORT CHECK
# ============================================================
from sophia.nemesis import (
    NemesisReport,
    BrierBucketAnalyzer,
    OverconfidencePenalizer,
    FalsePositiveAnalyzer,
    TimeDeviationAnalyzer,
    EfficiencyCalculator,
    DispositionBiasDetector,
    PostTradeSHAPComparator,
    SlippageForensics,
    GenePenalizer,
    ManifestWriter,
    NemesisEngine,
)
from sophia.post_mortem import TradeIntent
print("✅ All imports OK")

# ============================================================
# 3. TEST: BrierBucketAnalyzer
# ============================================================
bb = BrierBucketAnalyzer()
bb.record(0.45, 0.30)  # 0-50% bucket
bb.record(0.60, 0.15)  # 50-70% bucket
bb.record(0.75, 0.10)  # 70-85% bucket
bb.record(0.90, 0.05)  # 85-100% bucket

analysis = bb.get_bucket_analysis()
assert analysis["0-50%"]["count"] == 1
assert analysis["50-70%"]["count"] == 1
assert analysis["70-85%"]["count"] == 1
assert analysis["85-100%"]["count"] == 1
assert abs(analysis["85-100%"]["mean_brier"] - 0.05) < 0.001
print(f"✅ BrierBucketAnalyzer: 4 buckets populated correctly")

# ============================================================
# 4. TEST: OverconfidencePenalizer
# ============================================================
oc = OverconfidencePenalizer(lookback=5, brier_threshold=0.20)

# Feed 5 good trades
for _ in range(5):
    oc.record_brier(0.05)
assert not oc.is_active()
assert oc.get_penalty_factor() == 1.0
print(f"✅ OverconfidencePenalizer: no penalty for good Brier (avg=0.05)")

# Feed 5 poor trades → should activate penalty
oc2 = OverconfidencePenalizer(lookback=5, brier_threshold=0.20)
for _ in range(5):
    oc2.record_brier(0.35)
assert oc2.is_active()
penalty = oc2.get_penalty_factor()
expected_penalty = 1.0 + max(0, (0.35 - 0.15) * 3.0)  # 1.0 + 0.6 = 1.6
assert abs(penalty - expected_penalty) < 0.01
print(f"✅ OverconfidencePenalizer: penalty={penalty:.2f}x (expected {expected_penalty:.2f}x)")

# Test probability adjustment
p_adj = oc2.adjust_probability(0.80)
assert abs(p_adj - 0.80 / expected_penalty) < 0.01
print(f"✅ OverconfidencePenalizer: P(Win)=0.80 → adjusted {p_adj:.3f}")

# ============================================================
# 5. TEST: FalsePositiveAnalyzer
# ============================================================
fpa = FalsePositiveAnalyzer(fp_window=10)

# High confidence win → not FP
is_fp, reason = fpa.analyze(0.90, 5.0, {}, 600)
assert not is_fp
print(f"✅ FalsePositiveAnalyzer: P=90% + WIN → not FP")

# High confidence loss with tail event
is_fp, reason = fpa.analyze(0.90, -3.0, {'excess_kurtosis': 8.0}, 600)
assert is_fp
assert reason == "TAIL_EVENT"
print(f"✅ FalsePositiveAnalyzer: P=90% + LOSS + Kurt=8 → TAIL_EVENT")

# High confidence loss with signal decay
is_fp, reason = fpa.analyze(0.88, -2.0, {'alpha_decay_threshold_mins': 3.0}, 1200)
assert is_fp
assert reason == "SIGNAL_DECAY"
print(f"✅ FalsePositiveAnalyzer: P=88% + LOSS + decay → SIGNAL_DECAY")

# Low confidence loss → not FP
is_fp, reason = fpa.analyze(0.60, -1.0, {}, 300)
assert not is_fp
print(f"✅ FalsePositiveAnalyzer: P=60% + LOSS → not FP (below threshold)")

# Swing horizon loss at 78% confidence (Normally < 85% is not FP, but for Swing Threshold is 0.75 so this IS an FP)
is_fp, reason = fpa.analyze(0.78, -1.0, {'horizon_profile': 'MID_TERM', 'excess_kurtosis': 0.0}, 300)
assert is_fp
print(f"✅ FalsePositiveAnalyzer: P=78% + LOSS (MID_TERM) → FP (Threshold dynamically lowered to 75%)")

# ============================================================
# 6. TEST: TimeDeviationAnalyzer
# ============================================================
tda = TimeDeviationAnalyzer()

# Precise
ratio, cls = tda.analyze(12.0, 10.0, 1.0)
assert cls == "PRECISE"
assert abs(ratio - 1.2) < 0.01
print(f"✅ TimeDeviationAnalyzer: 12min/10min = {ratio}x → {cls}")

# Alpha leak (slow win)
ratio, cls = tda.analyze(30.0, 10.0, 2.0)
assert cls == "ALPHA_LEAK"
print(f"✅ TimeDeviationAnalyzer: 30min/10min = {ratio}x → {cls}")

# Volatility stall (slow loss)
ratio, cls = tda.analyze(25.0, 8.0, -1.5)
assert cls == "VOLATILITY_STALL"
print(f"✅ TimeDeviationAnalyzer: 25min/8min = {ratio}x → {cls}")

# Premature exit
ratio, cls = tda.analyze(2.0, 15.0, -0.5)
assert cls == "PREMATURE_EXIT"
print(f"✅ TimeDeviationAnalyzer: 2min/15min = {ratio}x → {cls}")

# Alpha Strike (Swing premature exit with profit)
ratio, cls = tda.analyze(10.0, 120.0, 5.0, horizon_profile='MID_TERM')
assert cls == "ALPHA_STRIKE"
print(f"✅ TimeDeviationAnalyzer: 10min/120min = {ratio}x (MID_TERM WIN) → {cls}")

# ============================================================
# 7. TEST: EfficiencyCalculator
# ============================================================
ec = EfficiencyCalculator(rolling_window=10)
# First trade sets the baseline
e, e_norm, cls = ec.compute(0.50, 10.0)  # $0.50 in 10 min = 0.05 $/min
assert abs(e - 0.05) < 0.001
print(f"✅ EfficiencyCalculator: E={e}$/min, E_norm={e_norm}, class={cls}")

# ============================================================
# 8. TEST: DispositionBiasDetector
# ============================================================
dbd = DispositionBiasDetector(rolling_window=10)

# Premature profit taking: won in 3 min, expected 15 min to TP
bias, score = dbd.analyze(actual_pnl=1.0, actual_duration_mins=3.0, predicted_tp_mins=15.0, predicted_sl_mins=5.0)
assert bias == "PREMATURE_PROFIT"
print(f"✅ DispositionBiasDetector: WIN in 3min (expected 15) → {bias}")

# Loss holding: lost in 20 min, expected 5 min to SL
bias, score = dbd.analyze(actual_pnl=-1.0, actual_duration_mins=20.0, predicted_tp_mins=15.0, predicted_sl_mins=5.0)
assert bias == "LOSS_HOLDING"
print(f"✅ DispositionBiasDetector: LOSS in 20min (expected 5) → {bias}")

# ============================================================
# 9. TEST: PostTradeSHAPComparator
# ============================================================
shap = PostTradeSHAPComparator()

# Features that predicted correctly
features = [
    {'feature': 'RSI', 'contribution': +0.15, 'value': 28.0},
    {'feature': 'BB Position', 'contribution': +0.10, 'value': 0.1},
    {'feature': 'Volume Ratio', 'contribution': -0.05, 'value': 0.8},
]
# Trade won → RSI(+) and BB(+) are hits, Volume(-) is also a hit (correctly predicted negative)
acc, misses = shap.analyze(features, actual_pnl=1.0, direction="LONG")
# RSI=+0.15 predicted success, trade won → HIT
# BB=+0.10 predicted success, trade won → HIT
# Vol=-0.05 predicted failure, trade won → MISS
assert 'Volume Ratio' in misses
print(f"✅ PostTradeSHAPComparator: accuracy={acc}, misses={misses}")

# ============================================================
# 10. TEST: SlippageForensics
# ============================================================
sf = SlippageForensics(alert_threshold_pct=0.05)

slip, alert = sf.compute(trigger_price=50000.0, fill_price=50003.0)
expected_slip = 3.0 / 50000.0 * 100.0  # 0.006%
assert abs(slip - expected_slip) < 0.001
print(f"✅ SlippageForensics: trigger=$50000, fill=$50003 → slip={slip:.4f}%")

# ============================================================
# 11. TEST: GenePenalizer
# ============================================================
gp = GenePenalizer()
from sophia.axioma import AxiomDiagnoser, FallaBase
class MockAxioma:
    def __init__(self, tipo_falla=FallaBase.NO_FALLA):
        self.tipo_falla = tipo_falla
mock_axioma = MockAxioma()

# Good trade → no penalty (actually fitness bonus)
pen, flag = gp.evaluate(0.10, mock_axioma, genotype_id="gene_a")
assert pen == -0.5 and not flag
print(f"✅ GenePenalizer: Brier=0.10 → bonus penalty={pen}")

# 3 consecutive poor trades → flag
mock_axioma_poor = MockAxioma(tipo_falla=FallaBase.TESIS_DECAY)
gp.evaluate(0.40, mock_axioma_poor, "gene_b")
pen, flag = gp.evaluate(0.60, mock_axioma_poor, "gene_b")
assert flag  # 2 consecutive for TESIS_DECAY
assert abs(pen - 0.60) < 0.001  
print(f"✅ GenePenalizer: 2 consecutive TESIS_DECAY → flagged, penalty={pen}")

# ============================================================
# 12. TEST: ManifestWriter
# ============================================================
manifest = ManifestWriter.generate_manifest(
    trade_id="test-123",
    symbol="BTC/USDT",
    direction="LONG",
    predicted_prob=0.85,
    actual_pnl=-2.50,
    brier_score=0.7225,
    time_deviation_class="VOLATILITY_STALL",
    efficiency_class="POOR",
    bias_detected="LOSS_HOLDING",
    false_positive_reason="TAIL_EVENT",
    shap_mismatches=["RSI", "Volume Ratio"],
    overconfidence_active=True,
    penalty_factor=1.6,
    axioma=mock_axioma,
    gene_penalty=0.0
)
assert "perdí" in manifest.lower() or "perd" in manifest.lower()
assert "BTC/USDT" in manifest
assert "TAIL_EVENT" in manifest or "cola gruesa" in manifest or "cisne negro" in manifest
print(f"✅ ManifestWriter: \"{manifest[:100]}...\"")

# ============================================================
# 13. TEST: NemesisEngine.full_autopsy() (Integration)
# ============================================================
engine = NemesisEngine()

report = engine.full_autopsy(
    trade_id="test-autopsy-001",
    symbol="ETH/USDT",
    direction="SHORT",
    predicted_prob=0.72,
    predicted_exit_mins=8.0,
    predicted_tp_mins=12.0,
    predicted_sl_mins=6.0,
    actual_pnl=-1.50,
    actual_duration_mins=15.0,
    brier_score=0.5184,
    sophia_report={'excess_kurtosis': 1.2, 'tail_risk_warning': False, 'alpha_decay_threshold_mins': 5.0},
    top_features=[
        {'feature': 'RSI', 'contribution': +0.08, 'value': 72.0},
        {'feature': 'ADX', 'contribution': +0.05, 'value': 35.0},
    ],
    trigger_price=3200.0,
    fill_price=3201.50,
    genotype_id="gen_eth_1",
    persist_manifest=False,  # Don't write to disk in test
)

# Verify report structure
assert isinstance(report, NemesisReport)
assert report.trade_id == "test-autopsy-001"
assert report.symbol == "ETH/USDT"
assert report.direction == "SHORT"
assert report.brier_score == 0.5184
assert report.brier_bucket == "70-85%"
assert report.time_deviation_ratio > 0
assert report.time_deviation_class == "PRECISE"  # 15/8=1.875 → within [0.5, 2.0]
assert report.efficiency_factor < 0  # negative PnL
assert report.slippage_pct > 0  # 3201.5 vs 3200
assert len(report.manifest) > 50
print(f"✅ NemesisEngine.full_autopsy(): {report.to_log_line()}")

# Verify to_dict
rd = report.to_dict()
assert len(rd) == 26  # 26 fields including timestamp
assert rd['symbol'] == "ETH/USDT"
print(f"✅ NemesisReport.to_dict(): {len(rd)} fields")

# ============================================================
# 14. TEST: Calibration Health
# ============================================================
health = engine.get_calibration_health()
assert 'brier_buckets' in health
assert 'overconfidence' in health
assert 'false_positive_rate' in health
assert 'disposition_bias' in health
assert 'avg_slippage_pct' in health
print(f"✅ NemesisEngine.get_calibration_health(): {len(health)} keys")

# ============================================================
# 15. TEST: TradeIntent now has trigger_price
# ============================================================
intent = TradeIntent(
    trade_id="ti-001",
    symbol="SOL/USDT",
    direction="LONG",
    timestamp="2026-01-01T00:00:00Z",
    win_probability=0.65,
    expected_exit_mins=5.0,
    signal_strength=0.7,
    top_features=[],
    entropy=0.5,
    entropy_label="Moderada",
    narrative="test",
    trigger_price=120.55,
)
assert intent.trigger_price == 120.55
print(f"✅ TradeIntent with trigger_price={intent.trigger_price}")

# ============================================================
# FINAL RESULT
# ============================================================
print()
print("=" * 60)
print("🎉 ALL 15 NÉMESIS-RETROSPECCIÓN TESTS PASSED!")
print("=" * 60)
