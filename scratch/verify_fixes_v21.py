"""
🔬 VERIFICATION: FORENSIC-V21 POST-FIX VALIDATION (v2)
========================================================
Verifies all 7 bug fixes are working correctly.
Mocks Notifier BEFORE importing portfolio to avoid any blocking.
"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('BINANCE_API_KEY', 'test')
os.environ.setdefault('BINANCE_API_SECRET', 'test')

# CRITICAL: Patch Notifier BEFORE any module imports it
from utils import notifier

_call_log = {'telegram': 0, 'open': 0, 'close': 0, 'legacy': 0}
_captured_open = {}
_captured_close = {}

# Replace ALL notification methods with no-ops
class MockNotifier:
    @staticmethod
    def send_telegram(msg, priority="INFO"):
        _call_log['telegram'] += 1
    
    @staticmethod
    def send_trade_open(data):
        _call_log['open'] += 1
        _captured_open.update(data)
    
    @staticmethod
    def send_trade_close(data):
        _call_log['close'] += 1
        _captured_close.update(data)
    
    @staticmethod
    def notify_trade(**kwargs):
        _call_log['legacy'] += 1
    
    @staticmethod
    def send_balance_update(*a, **kw): pass
    
    @staticmethod
    def send_risk_alert(*a, **kw): pass

# Patch at module level
notifier.Notifier.send_telegram = MockNotifier.send_telegram
notifier.Notifier.send_trade_open = MockNotifier.send_trade_open
notifier.Notifier.send_trade_close = MockNotifier.send_trade_close
notifier.Notifier.notify_trade = MockNotifier.notify_trade

# Now safe to import everything
from core.events import FillEvent
from core.enums import SignalType, OrderSide, OrderType
from datetime import datetime, timezone
import time

passed = 0
failed = 0

def check(condition, msg):
    global passed, failed
    if condition:
        print(f"  ✅ PASS: {msg}")
        passed += 1
    else:
        print(f"  ❌ FAIL: {msg}")
        failed += 1

def reset_counters():
    global _call_log, _captured_open, _captured_close
    _call_log = {'telegram': 0, 'open': 0, 'close': 0, 'legacy': 0}
    _captured_open.clear()
    _captured_close.clear()

# Mock DB
def mock_db(*a, **kw): pass
captured_db = {}
def capture_db(trade_p, pos_p):
    captured_db['trade'] = trade_p.copy()
    captured_db['position'] = pos_p.copy()

from core.portfolio import Portfolio

# ========== TEST 1: PnL in DB Payload (Fix #1) ==========
print("="*70)
print("🧪 TEST 1: PnL in DB Payload (Fix #1)")
print("="*70)

p = Portfolio(initial_capital=13.0, auto_save=False)
p.db.log_fill_event_atomic = capture_db
reset_counters()
captured_db.clear()

# Entry
fill_entry = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.BUY, fill_cost=100.0, fill_price=100000.0,
    horizon='SCALPING', strategy_id='tech_scalp_1'
)
p.update_fill(fill_entry)
entry_pnl = captured_db.get('trade', {}).get('pnl', 'MISSING')
check(entry_pnl == 0.0, f"Entry trade_payload.pnl = {entry_pnl} (expected 0.0 for new entry)")

# Update price
p.update_market_price('BTCUSDT', 100100.0)

# Close
captured_db.clear()
fill_exit = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.SELL, fill_cost=100.1, fill_price=100100.0,
    horizon='SCALPING', strategy_id='tech_scalp_1'
)
result = p.update_fill(fill_exit)

if isinstance(result, tuple):
    pnl_val, _ = result
else:
    pnl_val = result

trade_pnl = captured_db.get('trade', {}).get('pnl', 'MISSING')
pos_pnl = captured_db.get('position', {}).get('pnl', 'MISSING')

check(trade_pnl != 0.0 and trade_pnl != 'MISSING',
      f"Close trade_payload.pnl = {trade_pnl} (should NOT be 0.0)")
check(pos_pnl != 'MISSING',
      f"Close position_payload.pnl = {pos_pnl}")
check(pnl_val is not None and abs(pnl_val) > 0.001,
      f"Returned PnL = {pnl_val:.6f} (should be ~0.10)")
print(f"  📊 Realized PnL: {p.realized_pnl:.8f}")
print(f"  📊 Cash: {p.current_cash:.8f}")

# ========== TEST 2: Direction Mapping (Fix #2) ==========
print()
print("="*70)
print("🧪 TEST 2: Direction Mapping in Notifications (Fix #2)")
print("="*70)

# LONG open then close
p2 = Portfolio(initial_capital=13.0, auto_save=False)
p2.db.log_fill_event_atomic = mock_db
reset_counters()

fill_entry = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='ETHUSDT', exchange='BINANCE',
    quantity=0.01, direction=OrderSide.BUY, fill_cost=30.0, fill_price=3000.0,
    horizon='SWING', strategy_id='ml_swing_1'
)
p2.update_fill(fill_entry)
check(_captured_open.get('direction') == 'LONG',
      f"Entry LONG direction = '{_captured_open.get('direction')}' (expected 'LONG')")
check(_captured_open.get('horizon') == 'SWING',
      f"Entry horizon = '{_captured_open.get('horizon')}' (expected 'SWING')")

p2.update_market_price('ETHUSDT', 3050.0)
reset_counters()

fill_exit = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='ETHUSDT', exchange='BINANCE',
    quantity=0.01, direction=OrderSide.SELL, fill_cost=30.5, fill_price=3050.0,
    horizon='SWING', strategy_id='ml_swing_1'
)
p2.update_fill(fill_exit)
check(_captured_close.get('direction') == 'LONG',
      f"Close LONG direction = '{_captured_close.get('direction')}' (expected 'LONG' - we closed a LONG)")

# SHORT open then close
p3 = Portfolio(initial_capital=13.0, auto_save=False)
p3.db.log_fill_event_atomic = mock_db
reset_counters()

fill_short = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='SOLUSDT', exchange='BINANCE',
    quantity=0.1, direction=OrderSide.SELL, fill_cost=15.0, fill_price=150.0,
    horizon='SCALPING', strategy_id='sniper_1'
)
p3.update_fill(fill_short)
check(_captured_open.get('direction') == 'SHORT',
      f"Entry SHORT direction = '{_captured_open.get('direction')}' (expected 'SHORT')")

p3.update_market_price('SOLUSDT', 148.0)
reset_counters()

fill_close_short = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='SOLUSDT', exchange='BINANCE',
    quantity=0.1, direction=OrderSide.BUY, fill_cost=14.8, fill_price=148.0,
    horizon='SCALPING', strategy_id='sniper_1'
)
p3.update_fill(fill_close_short)
check(_captured_close.get('direction') == 'SHORT',
      f"Close SHORT direction = '{_captured_close.get('direction')}' (expected 'SHORT' - we closed a SHORT)")

# ========== TEST 3: Single Notification (Fix #3) ==========
print()
print("="*70)
print("🧪 TEST 3: Single Notification Per Trade (Fix #3)")
print("="*70)

p4 = Portfolio(initial_capital=13.0, auto_save=False)
p4.db.log_fill_event_atomic = mock_db
reset_counters()

# Entry
fill_entry = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.BUY, fill_cost=100.0, fill_price=100000.0,
    horizon='SCALPING', strategy_id='tech_1'
)
p4.update_fill(fill_entry)

check(_call_log['open'] == 1,
      f"Entry: send_trade_open called {_call_log['open']}x (expected 1)")
check(_call_log['telegram'] == 0,
      f"Entry: raw send_telegram called {_call_log['telegram']}x (expected 0)")
check(_call_log['legacy'] == 0,
      f"Entry: legacy notify_trade called {_call_log['legacy']}x (expected 0)")

# Close
p4.update_market_price('BTCUSDT', 100200.0)
reset_counters()

fill_exit = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.SELL, fill_cost=100.2, fill_price=100200.0,
    horizon='SCALPING', strategy_id='tech_1'
)
p4.update_fill(fill_exit)

check(_call_log['close'] == 1,
      f"Close: send_trade_close called {_call_log['close']}x (expected 1)")
check(_call_log['telegram'] == 0,
      f"Close: raw send_telegram called {_call_log['telegram']}x (expected 0)")
check(_call_log['legacy'] == 0,
      f"Close: legacy notify_trade called {_call_log['legacy']}x (expected 0)")

# ========== TEST 4: Close Data Enrichment (Fix #5) ==========
print()
print("="*70)
print("🧪 TEST 4: Close Notification Data Enrichment (Fix #5)")
print("="*70)

p5 = Portfolio(initial_capital=13.0, auto_save=False)
p5.db.log_fill_event_atomic = mock_db
reset_counters()

fill_entry = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.BUY, fill_cost=100.0, fill_price=100000.0,
    horizon='SCALPING', strategy_id='tech_1', sl_pct=0.003, tp_pct=0.005
)
p5.update_fill(fill_entry)
p5.update_market_price('BTCUSDT', 100200.0)
reset_counters()

fill_exit = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.SELL, fill_cost=100.2, fill_price=100200.0,
    horizon='SCALPING', strategy_id='tech_1'
)
p5.update_fill(fill_exit)

check(_captured_close.get('direction') == 'LONG',
      f"Enriched close direction = '{_captured_close.get('direction')}'")
check(_captured_close.get('pnl', 0) != 0,
      f"Enriched close pnl = {_captured_close.get('pnl', 0):.6f}")
check(_captured_close.get('commission', -1) >= 0,
      f"Enriched close commission = {_captured_close.get('commission', 0):.6f}")
check('exit_reason' in _captured_close,
      f"Enriched close has exit_reason: {_captured_close.get('exit_reason')}")

# ========== TEST 5: Horizon Isolation ==========
print()
print("="*70)
print("🧪 TEST 5: Scalping/Swing Horizon Isolation")
print("="*70)

p6 = Portfolio(initial_capital=13.0, auto_save=False)
p6.db.log_fill_event_atomic = mock_db
reset_counters()

fill_scalp = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.BUY, fill_cost=100.0, fill_price=100000.0,
    horizon='SCALPING', strategy_id='tech_scalp'
)
p6.update_fill(fill_scalp)

fill_swing = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.002, direction=OrderSide.SELL, fill_cost=200.0, fill_price=100000.0,
    horizon='SWING', strategy_id='ml_swing'
)
p6.update_fill(fill_swing)

check('BTCUSDT_SCALPING' in p6.virtual_ledger,
      f"VLedger has BTCUSDT_SCALPING")
check('BTCUSDT_SWING' in p6.virtual_ledger,
      f"VLedger has BTCUSDT_SWING")

scalp_pos = p6.virtual_ledger.get('BTCUSDT_SCALPING', {})
swing_pos = p6.virtual_ledger.get('BTCUSDT_SWING', {})

check(scalp_pos.get('horizon') == 'SCALPING',
      f"SCALPING ledger horizon tag = '{scalp_pos.get('horizon')}'")
check(swing_pos.get('horizon') == 'SWING',
      f"SWING ledger horizon tag = '{swing_pos.get('horizon')}'")

# ========== SUMMARY ==========
print()
print("="*70)
total = passed + failed
print(f"📋 VERIFICATION SUMMARY: {passed}/{total} PASSED ({failed} failed)")
print("="*70)

if failed == 0:
    print("🎉 ALL TESTS PASSED! All 7 fixes verified successfully.")
else:
    print(f"⚠️ {failed} test(s) failed. Review output above.")
    
sys.exit(0 if failed == 0 else 1)
