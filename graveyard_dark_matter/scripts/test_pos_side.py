import sys
import os
sys.path.append(os.getcwd())
from core.events import FillEvent
from core.portfolio import Portfolio
from core.enums import OrderSide, OrderType

p = Portfolio(initial_capital=13.0)
from datetime import datetime, timezone

event = FillEvent(
    timeindex=datetime.now(timezone.utc),
    symbol="BTC/USDT",
    exchange="BINANCE",
    quantity=0.01,
    direction=OrderSide.BUY,
    fill_cost=100.0,
    commission=0.0,
    strategy_id="TEST",
    fill_price=10000.0,
    horizon="SCALPING",
    metadata={"is_close": True}
)
try:
    p.update_fill(event)
except Exception as e:
    import traceback
    traceback.print_exc()
