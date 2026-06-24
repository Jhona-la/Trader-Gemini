"""Unit test for SwingDCAEngine"""
from datetime import datetime, timezone
from core.swing_dca_engine import SwingDCAEngine

engine = SwingDCAEngine()
now = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)

# Test position template
pos = {'quantity': 0.01, 'avg_price': 50000.0, 'horizon': 'SWING', 'tp_pct': 0.045, 'leverage': 10}

# Test 1: Position not in drawdown -> No DCA
result = engine.evaluate('BTC/USDT_SWING', pos, 'BTC/USDT', 51000.0, 3.0, 'RANGING', False, now=now)
assert result is None, f"Test 1 FAILED: Expected None, got {result}"
print("✅ Test 1 PASSED: No DCA when position is profitable")

# Test 2: Drawdown -2.5% -> DCA Layer 1
result = engine.evaluate('BTC/USDT_SWING', pos, 'BTC/USDT', 48750.0, 3.0, 'RANGING', False, now=now)
assert result is not None, "Test 2 FAILED: Expected DCA signal"
assert result.metadata['dca_layer'] == 1, f"Test 2 FAILED: Expected layer 1, got {result.metadata['dca_layer']}"
assert result.signal_type.name == 'LONG', f"Test 2 FAILED: Expected LONG, got {result.signal_type.name}"
print(f"✅ Test 2 PASSED: DCA Layer 1 triggered at -2.5% | Projected Avg: ${result.metadata['projected_avg_price']:.2f} | New TP: {result.tp_pct*100:.2f}%")

# Test 3: Kill switch active -> No DCA
result = engine.evaluate('ETH/USDT_SWING', pos, 'ETH/USDT', 48750.0, 3.0, 'RANGING', True, now=now)
assert result is None, f"Test 3 FAILED: Expected None with kill switch"
print("✅ Test 3 PASSED: No DCA when kill switch active")

# Test 4: Bear regime -> No DCA for LONG
result = engine.evaluate('SOL/USDT_SWING', pos, 'SOL/USDT', 48750.0, 3.0, 'TRENDING_BEAR', False, now=now)
assert result is None, f"Test 4 FAILED: Expected None in Bear regime"
print("✅ Test 4 PASSED: No DCA LONG in Bear regime")

# Test 5: Insufficient margin -> No DCA
result = engine.evaluate('AVAX/USDT_SWING', pos, 'AVAX/USDT', 48750.0, 0.30, 'RANGING', False, now=now)
assert result is None, f"Test 5 FAILED: Expected None with low margin"
print("✅ Test 5 PASSED: No DCA with insufficient margin ($0.30 < $0.50)")

# Test 6: Cooldown test - same position shortly after
now2 = datetime(2026, 4, 28, 12, 10, 0, tzinfo=timezone.utc)  # 10 min later
result = engine.evaluate('BTC/USDT_SWING', pos, 'BTC/USDT', 48000.0, 3.0, 'RANGING', False, now=now2)
assert result is None, f"Test 6 FAILED: Expected None during cooldown"
print("✅ Test 6 PASSED: DCA blocked during 30-min cooldown")

# Test 7: After cooldown, deeper drawdown -> DCA Layer 2
now3 = datetime(2026, 4, 28, 12, 35, 0, tzinfo=timezone.utc)  # 35 min later
result = engine.evaluate('BTC/USDT_SWING', pos, 'BTC/USDT', 48000.0, 3.0, 'RANGING', False, now=now3)
assert result is not None, "Test 7 FAILED: Expected DCA Layer 2"
assert result.metadata['dca_layer'] == 2, f"Test 7 FAILED: Expected layer 2, got {result.metadata['dca_layer']}"
print(f"✅ Test 7 PASSED: DCA Layer 2 triggered after cooldown | Layer: {result.metadata['dca_layer']}")

# Test 8: ATR safety blocks DCA
result = engine.evaluate('DOGE/USDT_SWING', pos, 'DOGE/USDT', 48750.0, 3.0, 'RANGING', False,
                          atr_current=0.05, atr_average=0.015, now=now)
assert result is None, f"Test 8 FAILED: Expected None with ATR spike"
print("✅ Test 8 PASSED: No DCA during ATR spike (3.3x > 2.5x safety)")

# Test 9: SHORT position DCA
pos_short = {'quantity': -0.01, 'avg_price': 50000.0, 'horizon': 'SWING', 'tp_pct': 0.045, 'leverage': 10}
result = engine.evaluate('XRP/USDT_SWING', pos_short, 'XRP/USDT', 51250.0, 3.0, 'RANGING', False, now=now)
assert result is not None, "Test 9 FAILED: Expected DCA for SHORT"
assert result.signal_type.name == 'SHORT', f"Test 9 FAILED: Expected SHORT, got {result.signal_type.name}"
print(f"✅ Test 9 PASSED: DCA SHORT triggered | Layer: {result.metadata['dca_layer']}")

# Test 10: Reset position
engine.reset_position('BTC/USDT_SWING')
state = engine.get_dca_state('BTC/USDT_SWING')
assert state['layers'] == 0, f"Test 10 FAILED: Expected 0 layers after reset, got {state['layers']}"
print("✅ Test 10 PASSED: Position state reset correctly")

# Test 11: Scalping position should NEVER get DCA
pos_scalp = {'quantity': 0.01, 'avg_price': 50000.0, 'horizon': 'SCALPING', 'tp_pct': 0.006, 'leverage': 10}
result = engine.evaluate('BTC/USDT_SCALPING', pos_scalp, 'BTC/USDT', 48000.0, 3.0, 'RANGING', False, now=now)
assert result is None, "Test 11 FAILED: SCALPING should never get DCA"
print("✅ Test 11 PASSED: SCALPING positions never receive DCA")

print("✅ Test 11 PASSED: SCALPING positions never receive DCA")

# Test 12: Sophia AI Veto & Approval
class MockSophiaReport:
    def __init__(self, win_probability):
        self.win_probability = win_probability
        self.entropy_label = "LOW"
    def to_dict(self):
        return {'win_probability': self.win_probability}

class MockSophia:
    def __init__(self, prob):
        self.prob = prob
    def analyze(self, symbol, direction, signal_strength, setups, confluence_score, tp_pct, sl_pct, returns, ttl_seconds, regime):
        return MockSophiaReport(self.prob)

# 12a: Vetoed (prob = 0.40 < 0.50)
mock_sophia_veto = MockSophia(0.40)
engine.reset_position('LINK/USDT_SWING')
result = engine.evaluate('LINK/USDT_SWING', pos, 'LINK/USDT', 48750.0, 3.0, 'RANGING', False, now=now, sophia_intelligence=mock_sophia_veto)
assert result is None, "Test 12a FAILED: Expected Sophia to VETO the DCA"
print("✅ Test 12a PASSED: Sophia VETOED DCA with P(Win)=40%")

# 12b: Approved (prob = 0.60 > 0.50)
mock_sophia_approve = MockSophia(0.60)
result = engine.evaluate('LINK/USDT_SWING', pos, 'LINK/USDT', 48750.0, 3.0, 'RANGING', False, now=now, sophia_intelligence=mock_sophia_approve)
assert result is not None, "Test 12b FAILED: Expected Sophia to APPROVE the DCA"
assert 'sophia' in result.metadata, "Test 12b FAILED: Expected sophia report in metadata"
print("✅ Test 12b PASSED: Sophia APPROVED DCA with P(Win)=60%")

print("\n" + "=" * 60)
print("🏆 ALL 12 TESTS PASSED — SwingDCAEngine validated!")
print("=" * 60)
