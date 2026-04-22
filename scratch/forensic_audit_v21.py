"""
🔬 FORENSIC AUDIT V21 - COMPREHENSIVE SYSTEM ANALYSIS
=====================================================
Targets:
1. Scalping/Swing ID identification
2. PnL = 0.000 bug
3. Long/Short direction confusion
4. Telegram notification quality
5. Duplicate notifications
6. DB payload hardcoded PnL
"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('BINANCE_API_KEY', 'test')
os.environ.setdefault('BINANCE_API_SECRET', 'test')

from core.events import SignalEvent, OrderEvent, FillEvent
from core.enums import SignalType, OrderSide, OrderType
from datetime import datetime, timezone

bugs_found = []
bugs_fixed = []

# ========== AUDIT #1: SIGNAL EVENT HORIZON VALIDATION ==========
print('='*80)
print('🔍 AUDIT #1: SIGNAL EVENT HORIZON VALIDATION')
print('='*80)

# Test 1: SignalEvent requires horizon (no default)
try:
    sig = SignalEvent(strategy_id='tech_1', symbol='BTCUSDT', 
                      datetime=datetime.now(timezone.utc), 
                      signal_type=SignalType.LONG, strength=0.8)
    bugs_found.append("SignalEvent can be created WITHOUT horizon - lost horizon tracking")
    print('❌ BUG: SignalEvent created WITHOUT horizon!')
except TypeError as e:
    print(f'✅ SignalEvent correctly requires horizon')

# Test 2: FillEvent has default horizon
fill = FillEvent(timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
                 quantity=0.001, direction=OrderSide.BUY, fill_cost=100.0)
if fill.horizon == "SCALPING":
    bugs_found.append("FillEvent defaults to SCALPING instead of requiring explicit horizon from OrderEvent")
    print(f'⚠️ BUG: FillEvent defaults to "SCALPING" - should propagate from Order, not default!')
else:
    print(f'✅ FillEvent horizon: {fill.horizon}')

# ========== AUDIT #2: PNL CALCULATION PATH ==========
print()
print('='*80)
print('🔍 AUDIT #2: PNL CALCULATION PATH (SIMULATED TRADE)')
print('='*80)

from core.portfolio import Portfolio
p = Portfolio(initial_capital=13.0, auto_save=False)

# Simulate entry
fill_entry = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.BUY, fill_cost=100.0, fill_price=100000.0,
    horizon='SCALPING', strategy_id='tech_scalp_1'
)
result = p.update_fill(fill_entry)
print(f'Entry result: {result} (expected None)')
pos = p.positions.get("BTCUSDT", {})
print(f'  Position qty: {pos.get("quantity", "MISSING")}')
print(f'  Position avg: {pos.get("avg_price", "MISSING")}')
vl_keys = list(p.virtual_ledger.keys())
print(f'  VLedger keys: {vl_keys}')

# Update price
p.update_market_price('BTCUSDT', 100100.0)

# Simulate exit
fill_exit = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.SELL, fill_cost=100.1, fill_price=100100.0,
    horizon='SCALPING', strategy_id='tech_scalp_1'
)
result = p.update_fill(fill_exit)
if result is not None:
    if isinstance(result, tuple):
        pnl_val, outcome = result
    else:
        pnl_val = result
        outcome = None
    print(f'✅ PnL on close: {pnl_val:.6f}')
    if abs(pnl_val) < 0.001:
        bugs_found.append(f"PnL close value suspiciously near zero: {pnl_val}")
        print(f'  ⚠️ WARNING: PnL very close to zero!')
else:
    bugs_found.append("PnL returned None on trade close")
    print('❌ CRITICAL BUG: PnL returned None on close!')

print(f'  Realized PnL:  {p.realized_pnl:.8f}')
print(f'  Total Fees:    {p.total_fees_paid:.8f}')
print(f'  Cash:          {p.current_cash:.8f}')

# ========== AUDIT #3: DIRECTION MAPPING BUG ==========
print()
print('='*80)
print('🔍 AUDIT #3: DIRECTION MAPPING IN NOTIFICATIONS')
print('='*80)

# In log_trade_report line ~1957:
# 'direction': 'LONG' if event.direction == OrderSide.BUY else 'SHORT'
# This is WRONG for close notifications!
# When closing a LONG, we SELL → direction becomes "SHORT" in notification
# When closing a SHORT, we BUY → direction becomes "LONG" in notification

print('Line 1957 in log_trade_report():')
print('  direction = "LONG" if event.direction == BUY else "SHORT"')
print()
print('  Scenario: CLOSE LONG (we SELL to close)')
print('    event.direction = SELL → notification says "SHORT"')
print('    But position was LONG! Telegram shows wrong direction!')
print()
print('  Scenario: CLOSE SHORT (we BUY to close)')
print('    event.direction = BUY → notification says "LONG"')
print('    But position was SHORT! Telegram shows wrong direction!')
bugs_found.append("log_trade_report direction mapping is inverted for close notifications")

# ========== AUDIT #4: DUPLICATE TELEGRAM NOTIFICATIONS ==========
print()
print('='*80)
print('🔍 AUDIT #4: DUPLICATE TELEGRAM NOTIFICATIONS')
print('='*80)

print('ENTRY PATH:')
print('  1. _update_virtual_ledger() L829-835: Notifier.send_telegram() [raw, basic]')
print('  2. log_trade_report() L1986: Notifier.send_trade_open() [enhanced]')
print('  3. log_trade_report() L1989: Notifier.notify_trade()  [legacy duplicate]')
print('  → TRIPLE notification for every entry!')
print()
print('CLOSE PATH:')
print('  1. _record_closed_trade() L1044: Notifier.send_telegram() [raw close]')
print('  2. log_trade_report() L1984: Notifier.send_trade_close() [enhanced]')
print('  3. log_trade_report() L1989: Notifier.notify_trade()  [legacy duplicate]')
print('  → TRIPLE notification for every close!')
bugs_found.append("Triple Telegram notifications per trade (raw + enhanced + legacy)")

# ========== AUDIT #5: DB PAYLOAD HARDCODED PNL ==========
print()
print('='*80)
print('🔍 AUDIT #5: DB PAYLOAD HARDCODED PNL = 0.0')
print('='*80)

print('In update_fill() at line 1464:')
print("  trade_payload = { ... 'pnl': 0.0, ... }")
print('In update_fill() at line 1473:')
print("  position_payload = { ... 'pnl': 0.0, ... }")
print()
print('These ALWAYS write pnl=0.0 to the database, regardless of actual PnL!')
print('This is why PnL shows 0.000 in logs and dashboard!')
bugs_found.append("DB payload hardcodes pnl=0.0 at lines 1464 and 1473")

# ========== AUDIT #6: TELEGRAM MESSAGE CONTENT ANALYSIS ==========
print()
print('='*80)
print('🔍 AUDIT #6: TELEGRAM MESSAGE CONTENT GAPS')
print('='*80)

print('MISSING from raw entry Telegram (L829-835):')
print('  - SL/TP percentages and prices')
print('  - Fee estimation and breakeven')
print('  - Risk/Reward ratio')
print('  - Market regime/volatility')
print('  - Available balance')
print('  - Position size as % of capital')
print()
print('MISSING from raw close Telegram (L1044-1051):')
print('  - Gross PnL vs Net PnL breakdown')
print('  - Fee breakdown (entry vs exit)')
print('  - MAE/MFE metrics')
print('  - R-multiple')
print('  - Balance before/after')
print('  - Win rate and streak info')
print('  - Strategy performance context')
bugs_found.append("Raw Telegram messages missing critical decision-making data")

# ========== AUDIT #7: VIRTUAL LEDGER OPEN/SHORT DIRECTION CHECK ==========
print()
print('='*80)
print('🔍 AUDIT #7: SHORT POSITION OPENING TEST')
print('='*80)

p2 = Portfolio(initial_capital=13.0, auto_save=False)

# Open SHORT
fill_short = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='ETHUSDT', exchange='BINANCE',
    quantity=0.01, direction=OrderSide.SELL, fill_cost=30.0, fill_price=3000.0,
    horizon='SWING', strategy_id='ml_swing_1'
)
result = p2.update_fill(fill_short)
print(f'SHORT entry result: {result} (expected None)')
pos = p2.positions.get("ETHUSDT", {})
vl = p2.virtual_ledger.get("ETHUSDT_SWING", {})
print(f'  Physical qty: {pos.get("quantity", "MISSING")}')
print(f'  VLedger qty:  {vl.get("quantity", "MISSING")}')
print(f'  VLedger horizon: {vl.get("horizon", "MISSING")}')

if pos.get('quantity', 0) > 0:
    bugs_found.append("SHORT entry creates positive quantity in physical ledger")
    print('  ❌ BUG: SHORT should have negative quantity!')
elif pos.get('quantity', 0) < 0:
    print('  ✅ SHORT has negative quantity correctly')
    
# Check if the virtual ledger key matches  
if "ETHUSDT_SWING" in p2.virtual_ledger:
    print('  ✅ VLedger key ETHUSDT_SWING exists')
else:
    bugs_found.append("SWING short position not found in virtual ledger with correct key")
    print(f'  ❌ BUG: Expected key ETHUSDT_SWING, found: {list(p2.virtual_ledger.keys())}')

# Close SHORT  
p2.update_market_price('ETHUSDT', 2950.0)
fill_close_short = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='ETHUSDT', exchange='BINANCE',
    quantity=0.01, direction=OrderSide.BUY, fill_cost=29.5, fill_price=2950.0,
    horizon='SWING', strategy_id='ml_swing_1'
)
result = p2.update_fill(fill_close_short)
if result is not None:
    if isinstance(result, tuple):
        pnl_val, _ = result
    else:
        pnl_val = result
    print(f'  ✅ SHORT close PnL: {pnl_val:.6f} (expected +0.50 = (3000-2950)*0.01)')
    if pnl_val <= 0:
        bugs_found.append(f"SHORT close PnL is non-positive: {pnl_val}")
else:
    bugs_found.append("SHORT close returned None PnL")
    print('  ❌ BUG: SHORT close returned None!')

# ========== SUMMARY ==========
print()
print('='*80)
print('📋 FORENSIC AUDIT SUMMARY')
print('='*80)
print(f'Total bugs found: {len(bugs_found)}')
for i, bug in enumerate(bugs_found, 1):
    print(f'  {i}. ❌ {bug}')
print()
print('Priority fixes needed:')
print('  P0: DB pnl=0.0 hardcoded → causes PnL=0.000 display')
print('  P0: Direction mapping inverted for close notifications')
print('  P1: Triple Telegram notifications (spam)')
print('  P1: Raw Telegram missing decision context')
print('  P2: FillEvent defaults horizon to SCALPING')
