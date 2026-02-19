
import numpy as np
import time
import random
from utils.logger import logger

class ChaosMonkey:
    """
    [PHASE 14] Chaos Engineering Module.
    Injects synthetic faults into the system to verify resilience.
    Modes:
    - LATENCY: Adds random sleep delays.
    - NOISE: Adds Gaussian noise to prices.
    - BLACK_SWAN: Simulates -10% flash crash.
    - DATA_CORRUPTION: Injects NaNs or Infs.
    """
    def __init__(self, probability=0.01):
        self.probability = probability
        self.active = False
        
    def activate(self):
        self.active = True
        logger.warning("🐵 CHAOS MONKEY RELEASED! System Integrity at Risk.")
        
    def deactivate(self):
        self.active = False
        logger.info("🙈 Chaos Monkey Caged.")

    def inspect_data(self, bars):
        """Intercepts data bars and potentially corrupts them."""
        if not self.active or random.random() > self.probability:
            return bars
            
        fault_type = random.choice(['LATENCY', 'NOISE', 'BLACK_SWAN', 'NULLS'])
        
        if fault_type == 'LATENCY':
            delay = random.uniform(0.1, 2.0)
            logger.warning(f"🐵 Chaos: Injecting {delay:.2f}s Latency")
            time.sleep(delay)
            return bars
            
        elif fault_type == 'NOISE':
            logger.warning("🐵 Chaos: Injecting Price Noise")
            # Apply 0.5% noise to close prices
            noise_factor = 1.0 + np.random.normal(0, 0.005, len(bars))
            # Numpy structured array modification is tricky if immutable.
            # Assuming bars is a structured array or dict list.
            # If standard list of dicts:
            if isinstance(bars, list):
                new_bars = []
                for b in bars:
                    nb = b.copy()
                    nb['close'] = float(nb['close']) * (1.0 + random.normalvariate(0, 0.005))
                    new_bars.append(nb)
                return new_bars
            
        elif fault_type == 'BLACK_SWAN':
            logger.critical("🐵 Chaos: SIMULATING FLASH CRASH (-10%)")
            # Drop last candle by 10%
            if isinstance(bars, list) and len(bars) > 0:
                bars[-1]['close'] = float(bars[-1]['close']) * 0.90
                bars[-1]['low'] = float(bars[-1]['low']) * 0.90
                return bars
                
        elif fault_type == 'NULLS':
            logger.warning("🐵 Chaos: Injecting NaNs")
            if isinstance(bars, list) and len(bars) > 0:
                bars[-1]['close'] = np.nan
                return bars
                
        return bars

chaos_monkey = ChaosMonkey(probability=0.0) # Disabled by default
