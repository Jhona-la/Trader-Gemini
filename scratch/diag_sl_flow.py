import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from core.events import OrderEvent, FillEvent, SignalType, SignalEvent
from core.enums import OrderSide
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from config import Config
from datetime import datetime, timezone

def test_sl_pct_flow():
    portfolio = Portfolio()
    rm = RiskManager()
    rm.portfolio = portfolio

    signal = SignalEvent(
        strategy_id="TEST_SCALP",
        symbol="SOL/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.SHORT,
        strength=1.0,
        horizon="SWING"
    )

    current_price = 100.0
    
    order = rm.generate_order(signal, current_price)
    
    print("OrderEvent SL:", order.sl_pct)
    
    fill = FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol=order.symbol,
        exchange="BINANCE",
        quantity=order.quantity,
        direction=order.direction,
        fill_cost=order.quantity * order.price,
        commission=0.0,
        strategy_id=order.strategy_id,
        fill_price=order.price,
        order_id="123",
        sl_pct=order.sl_pct,
        tp_pct=order.tp_pct,
        horizon=order.horizon,
        leverage=order.leverage
    )
    
    print("FillEvent SL:", fill.sl_pct)
    
    portfolio.update_fill(fill)
    
    pos = portfolio.virtual_ledger.get("SOL/USDT_SWING_SHORT")
    print("Ledger SL:", pos.get("sl_pct"))

if __name__ == "__main__":
    test_sl_pct_flow()
