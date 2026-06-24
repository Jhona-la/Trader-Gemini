import asyncio
import os
import sys
import datetime
from datetime import timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from core.enums import SignalType, TradeDirection
from core.events import SignalEvent
from data.binance_loader import BinanceData
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from execution.binance_executor import BinanceExecutor
from core.engine import Engine
from strategies.omni_strategy import OmniStrategy
import utils.logger as logger

async def async_main():
    print("🚀 Iniciando Simulación del Pipeline End-to-End (TESTNET)")
    
    # 1. Configuration (Force Testnet)
    Config.USE_TESTNET = True
    
    # 2. Data Handlers
    from queue import PriorityQueue
    events_queue = PriorityQueue()
    
    # Mock BinanceData to prevent parallel history fetch heap corruption in test
    class MockBinanceData:
        def __init__(self, q, symbols):
            self.events_queue = q
            self.symbols = symbols
            self.is_live = False
        def get_latest_bars(self, symbol, n=1): return None
        def get_order_flow_metrics(self, symbol): return {}
        def get_derivatives_metrics(self, symbol): return {}
        def get_orderbook(self, symbol): return None

    data_handler = MockBinanceData(events_queue, ["BTCUSDT", "BTC/USDT"])

    # 3. Core Components
    portfolio = Portfolio(initial_capital=15.0)
    risk_manager = RiskManager(portfolio=portfolio)
    executor = BinanceExecutor(events_queue, portfolio, data_handler)
    
    # 4. Engine Assembly
    strategy = OmniStrategy(events_queue, data_handler)
    engine = Engine(events_queue)
    # Inject dependencies manually for simulation
    engine.data_handlers = [data_handler]
    engine.strategy = strategy
    engine.executor = executor
    engine.portfolio = portfolio
    engine.risk_manager = risk_manager
    
    # Fake current price
    portfolio.positions["BTC/USDT"] = {"quantity": 0, "avg_price": 0, "current_price": 60000.0}
    portfolio.current_cash = 15.0
    
    signal = SignalEvent(
        strategy_id="SIMULATED_TEST",
        symbol="BTC/USDT",
        datetime=datetime.datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=0.95,
        horizon="SWING",
        ml_confidence=0.98,
        metadata={
            "bypass_executed": False, 
            "bypass_source": "TESTNET_SIMULATOR",
            "atr_pct": 0.05,
            "close_price": 0.50,
            "ml_confidence": 0.85,
            "system_latency_ms": 1.2,
            "leverage": 50,
            "sophia": {
                "is_anomalous": False,
                "vortex_pulse": 0.0
            },
            "bollinger_squeeze": False,
            "gamma_expansion": False,
            "resonance_multiplier": 1.0,
            "is_paired": False,
            "is_pyramid": False,
            "momentum": 0.0,
            "metrics": {
                "order_flow": {
                    "tick_volatility": 0.0,
                    "toxicity_index": 0.0,
                    "delta": 0,
                    "is_spoofing": False,
                    "spoofing_side": None,
                    "gamma_expansion_risk": False,
                    "magnetic_pull_up": 0.0,
                    "magnetic_pull_down": 0.0,
                    "high_micro_entropy": False,
                    "entropy": 0.0,
                    "spread_pct": 0.0001
                }
            }
        }
    )
    
    print("📡 Inyectando SignalEvent en el Motor...")
    # Instead of engine._process_signal_event which doesn't exist in V2 Metal Engine:
    print("📡 Inyectando SignalEvent en el Motor...")
    # Mocking order execution
    order_event = risk_manager.generate_order(signal, current_price=60000.0)
    if order_event:
        print("✅ RiskManager validó la señal y generó OrderEvent.")
        if isinstance(order_event, list):
            for oe in order_event:
                await executor.execute_order(oe)
        else:
            await executor.execute_order(order_event)
        print("✅ Pipeline E2E completado en Testnet.")
    else:
        print("❌ RiskManager rechazó la señal.")


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except Exception as e:
        import traceback
        traceback.print_exc()
