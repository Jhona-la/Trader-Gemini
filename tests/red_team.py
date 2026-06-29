import time
import random
import asyncio
import numpy as np
from datetime import datetime, timezone
from core.events import MarketEvent, EventType
from utils.logger import logger
from config import Config

class PanicGenerator:
    """
    🚨 RED TEAM: Synthetic Disaster Injector.
    Simulates Flash Crashes, Latency Spikes, and Spoofing to test System Resilience.
    
    [PHASE 19] Adversarial Simulation
    """
    def __init__(self, engine):
        self.engine = engine
        self._running = False
        self._chaos_thread = None
        logger.info("😈 [RED TEAM] Panic Generator Initialized. Waiting for triggers...")

    async def inject_flash_crash(self, symbol="BTC/USDT", drop_pct=0.05, duration_s=1.0):
        """
        Simulates a violent Flash Crash (-5% in 1s).
        Target: Trigger RiskManager GARCH Circuit Breaker.
        """
        logger.critical(f"🚨 [RED TEAM] 📉 INJECTING FLASH CRASH on {symbol} (Target: -{drop_pct:.1%}) in {duration_s}s")
        
        # Get start price (mock or try to get from DataProvider)
        start_price = 50000.0
        try:
            # Try to get real price if available
            dp = self.engine.data_handlers[0] if self.engine.data_handlers else None
            if dp:
                # Mock access, actual implementation depends on DP interface
                pass
        except:
            pass

        steps = int(duration_s * 20) # 20 ticks per second (50ms interval)
        total_drop = start_price * drop_pct
        step_drop = total_drop / steps
        
        volatility_noise = step_drop * 0.2
        
        current_price = start_price
        
        for i in range(steps):
            # Nonlinear collapse (Panic accelerates)
            acceleration = (i / steps) ** 2 
            this_drop = step_drop * (1 + acceleration)
            
            # Add noise
            noise = random.uniform(-volatility_noise, volatility_noise)
            current_price -= (this_drop + noise)
            
            # Create & Inject Event
            event = MarketEvent(
                symbol=symbol,
                close_price=current_price,
                order_flow={'imbalance': -0.8} # Simulate heavy sell pressure
            )
            # Bypass queue limit if needed, or just put
            try:
                self.engine.events.put(event)
            except:
                pass
                
            await asyncio.sleep(duration_s / steps)
            
        logger.critical(f"🚨 [RED TEAM] 📉 Crash Injection Complete. Final Price: {current_price:.2f}")

    async def inject_latency_spike(self, duration_s=5.0, lag_ms=2000):
        """
        Simulates network congestion or CPU overload.
        Target: Trigger Watchdog or Latency Throttling.
        """
        logger.critical(f"🚨 [RED TEAM] 🐌 INJECTING LATENCY SPIKE ({lag_ms}ms lag for {duration_s}s)")
        
        end_time = time.time() + duration_s
        while time.time() < end_time:
            # Block the loop properly? No, we are async.
            # We want to delay event processing.
            # In a real scenario, we might flood the queue.
            
            # Flood Queue with Noise events
            for _ in range(100):
                self.engine.events.put(MarketEvent(symbol="NOISE", close_price=0.0))
            
            await asyncio.sleep(0.1)
            
        logger.info("🚨 [RED TEAM] Latency Injection Stopped.")
