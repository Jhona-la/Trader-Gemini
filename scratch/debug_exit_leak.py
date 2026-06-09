import os, sys
from datetime import datetime, timezone

# Project setup
_project_root = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini"
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
Config.IS_BACKTEST = True

from core.events import SignalEvent, FillEvent
from core.enums import SignalType, OrderSide, OrderType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from scripts.run_god_mode_backtest import BacktestExecutor

def test_isolated_exit():
    print("--- Diagnostic isolated exit run ---")
    
    # 1. Initialize Portfolio
    portfolio = Portfolio(initial_capital=13.0, auto_save=False)
    portfolio.positions = {}
    portfolio.virtual_ledger = {}
    portfolio.used_margin = 0.0
    portfolio.current_cash = 13.0
    
    # 2. Mock a SHORT microscalping position on ETH/USDT
    v_key = "ETH/USDT_MICROSCALPING_SHORT"
    portfolio.virtual_ledger[v_key] = {
        'quantity': -0.01483886,
        'avg_price': 2247.1399,
        'horizon': 'MICROSCALPING',
        'pos_side': 'SHORT',
        'current_price': 2247.1399,
        'high_water_mark': 2247.1399,
        'low_water_mark': 2247.1399,
        'entry_time': datetime.now(timezone.utc),
        'sl_pct': 0.0015,
        'tp_pct': 0.0036,
        'opener_strategy_id': '[MSC]_Technical Momentum_MICROSCALPING.RSI_MEAN_REVERSION',
        'tp_limit_placed': False,
        'exit_pending_time': 0,
    }
    print(f"Initial virtual ledger state for {v_key}:")
    print(portfolio.virtual_ledger[v_key])
    
    # 3. Initialize RiskManager
    risk_manager = RiskManager(portfolio=portfolio)
    
    # 4. Create SignalEvent for EXIT
    exit_signal = SignalEvent(
        strategy_id="HARD_SL",
        symbol="ETH/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.EXIT,
        strength=1.0,
        horizon="MICROSCALPING"
    )
    
    print("\n1. Generating exit order...")
    order = risk_manager.generate_order(exit_signal, current_price=2250.6699)
    if order is None:
        print("❌ generate_order returned None!")
        return
    
    print(f"✅ Order generated: Type={order.order_type}, Side={order.direction}, Qty={order.quantity}, Price={order.price}, Metadata={order.metadata}")
    
    # 5. Initialize BacktestExecutor
    executor = BacktestExecutor()
    
    print("\n2. Executing order via BacktestExecutor...")
    fill = executor.execute_order(order, current_price=2250.6699)
    if fill is None:
        print("❌ execute_order returned None!")
        return
        
    print(f"✅ Fill returned: Side={fill.direction}, Qty={fill.quantity}, Cost={fill.fill_cost}, Price={fill.fill_price}")
    
    print("\n3. Processing fill via portfolio.update_fill...")
    result = portfolio.update_fill(fill)
    print(f"update_fill result: {result}")
    
    print("\nFinal virtual ledger state:")
    if v_key in portfolio.virtual_ledger:
        print(portfolio.virtual_ledger[v_key])
    else:
        print("Key removed from virtual ledger.")

if __name__ == "__main__":
    test_isolated_exit()
