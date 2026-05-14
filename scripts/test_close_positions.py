import sys
import os
sys.path.append(os.getcwd())
from core.events import FillEvent
from core.portfolio import Portfolio
from core.enums import OrderSide, OrderType
from datetime import datetime, timezone

p = Portfolio(initial_capital=13.0)
# Open a position
event_open = FillEvent(
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
    metadata={"is_close": False}
)
p.update_fill(event_open)

# Mock run_god_mode_backtest.py step 4
v_key = "BTC/USDT_SCALPING_LONG"
qty = 0.01
horizon = "SCALPING"
symbol = "BTC/USDT"
current_price = 10100.0
direction = OrderSide.SELL

try:
    close_fill = FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol=symbol,
        exchange="BINANCE_BACKTEST",
        quantity=abs(qty),
        direction=direction,
        fill_cost=abs(qty) * current_price,
        commission=abs(qty) * current_price * 0.0001,
        strategy_id="BACKTEST_CLOSE",
        fill_price=current_price,
        horizon=horizon,
        metadata={'is_close': True, 'reason': 'BACKTEST_CLOSE'}
    )
    p.update_fill(close_fill)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
