"""Quick diagnostic: Find where portfolio.update_fill hangs"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('BINANCE_API_KEY', 'test')
os.environ.setdefault('BINANCE_API_SECRET', 'test')

# Patch Notifier BEFORE import
from utils import notifier
notifier.Notifier.send_telegram = staticmethod(lambda *a, **kw: None)
notifier.Notifier.send_trade_open = staticmethod(lambda *a, **kw: None)
notifier.Notifier.send_trade_close = staticmethod(lambda *a, **kw: None)
notifier.Notifier.notify_trade = staticmethod(lambda **kw: None)

print("STEP 1: Import Portfolio", flush=True)
from core.portfolio import Portfolio
from core.events import FillEvent
from core.enums import OrderSide
from datetime import datetime, timezone

print("STEP 2: Create Portfolio", flush=True)
p = Portfolio(initial_capital=13.0, auto_save=False)
p.db.log_fill_event_atomic = lambda *a: None

print("STEP 3: Create FillEvent", flush=True)
fill_entry = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.BUY, fill_cost=100.0, fill_price=100000.0,
    horizon='SCALPING', strategy_id='tech_1'
)

print("STEP 4: Calling update_fill (ENTRY)...", flush=True)
result = p.update_fill(fill_entry)
print(f"STEP 5: Entry done. Result: {result}", flush=True)

p.update_market_price('BTCUSDT', 100100.0)
print("STEP 6: Price updated", flush=True)

fill_exit = FillEvent(
    timeindex=datetime.now(timezone.utc), symbol='BTCUSDT', exchange='BINANCE',
    quantity=0.001, direction=OrderSide.SELL, fill_cost=100.1, fill_price=100100.0,
    horizon='SCALPING', strategy_id='tech_1'
)

print("STEP 7: Calling update_fill (EXIT)...", flush=True)
result = p.update_fill(fill_exit)
print(f"STEP 8: Exit done. Result: {result}", flush=True)
print("ALL DONE", flush=True)
