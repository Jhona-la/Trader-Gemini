import asyncio
import time
import os
import sys
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.getcwd())

from core.engine import Engine
from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import SignalType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from utils.cooldown_manager import cooldown_manager
from utils.logger import setup_logger
from config import Config

logger = setup_logger("ThroughputBenchmark")

class ThroughputBenchmark:
    def __init__(self, iterations=10000):
        self.iterations = iterations
        self.portfolio = Portfolio(initial_capital=1000000)
        self.risk = RiskManager(portfolio=self.portfolio)
        
        # Bypass blocking logic for pure throughput
        self.risk._check_expectancy_viability = lambda symbol: True
        self.risk.record_cooldown = lambda symbol: None
        self.risk.is_in_cooldown = lambda symbol: False
        self.risk._check_correlations = lambda symbol, signal: True
        self.risk._check_market_regime = lambda symbol, signal: True
        self.risk.size_position = lambda sig, price: 1000.0 # Fast size
        self.risk.MAX_TRADES_PER_SYMBOL = 999999
        self.risk.MAX_TRADES_TOTAL = 999999
        
        # PATCH GLOBAL COOLDOWN MANAGER
        cooldown_manager.can_trade = lambda *args, **kwargs: (True, "OK")
        
        # Mock Engine to avoid network/WS
        # Use UNBOUNDED queue to hold all 10k events for stress test
        import queue
        self.engine = Engine(events_queue=queue.Queue())
        self.engine.portfolio = self.portfolio
        self.engine.risk_manager = self.risk
        
        # MOCK PRICE VALIDATION
        self.engine._get_validated_price = lambda symbol: 50000.0
        
        # MOCK EXECUTION HANDLER
        class MockExecution:
            def execute_order(self, order): pass
        self.engine.execution_handler = MockExecution()

    async def run(self):
        logger.info(f"⚡ Starting Throughput Stress Test ({self.iterations} events)...")
        
        # 1. Warm up JIT
        warmup_signal = SignalEvent(
            strategy_id="WARMUP", 
            symbol="BTC/USDT", 
            datetime=datetime.now(timezone.utc), 
            signal_type=SignalType.LONG,
            atr=0.02
        )
        self.engine.events.put(warmup_signal)
        
        # Run engine logic synchronously for maximum speed measurement
        def mock_processor():
            processed = 0
            # +1 for warmup
            target = self.iterations + 1
            get_event = self.engine.events.get
            process = self.engine.process_event
            
            while processed < target:
                event = get_event() # Blocking wait is fine now (no drops)
                process(event)
                processed += 1
        
        # Fill queue
        logger.info("Filling queue...")
        for i in range(self.iterations):
            sig = SignalEvent(
                strategy_id="STRESS",
                symbol="BTC/USDT",
                datetime=datetime.now(timezone.utc),
                signal_type=SignalType.LONG,
                atr=0.02
            )
            self.engine.events.put(sig)
        
        logger.info("Executing processing burst...")
        # AGGRESSIVE SILENCING
        import utils.logger as ul
        original_level = ul.logger.level
        ul.logger.setLevel(100) # Disable all logs
        
        t0 = time.perf_counter()
        
        # Start processing
        mock_processor()
        
        t1 = time.perf_counter()
        ul.logger.setLevel(original_level) # Restore
        
        total_time = t1 - t0
        eps = self.iterations / total_time
        
        print("\n" + "="*50)
        print("🚀 OMEGA THROUGHPUT REPORT")
        print("="*50)
        print(f"Total Events:   {self.iterations}")
        print(f"Total Time:     {total_time:.4f} s")
        print(f"Throughput:     {eps:.2f} EPS")
        
        if eps > 5000:
            print("✅ PERFORMANCE: GOD MODE (High Frequency Ready)")
        elif eps > 1000:
            print("✅ PERFORMANCE: INSTITUTIONAL")
        else:
            print("⚠️ PERFORMANCE: NEEDS OPTIMIZATION")
        print("="*50 + "\n")

if __name__ == "__main__":
    benchmark = ThroughputBenchmark(iterations=10000)
    asyncio.run(benchmark.run())
