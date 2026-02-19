
import time
import random
from utils.logger import setup_logger

logger = setup_logger("ChaosMonkey")

class ChaosEngine:
    """
    🛡️ PHASE 18: CHAOS MONKEY (LATENCY SIMULATION)
    Injects artificial failures and latency to prove system resilience.
    Capabilities:
    1. Lag Spikes (Sleep)
    2. Exception Injection (Crashing functions)
    3. Data Corruption (Modifying values)
    """
    ENABLED = False # Safety first, disabled by default
    
    @staticmethod
    def maybe_lag(component_name, probability=0.05, max_seconds=2.0):
        """
        Injects random latency with given probability.
        """
        if not ChaosEngine.ENABLED: return
        
        if random.random() < probability:
            delay = random.uniform(0.1, max_seconds)
            logger.warning(f"🐒 CHAOS: Injecting {delay:.2f}s lag into {component_name}")
            time.sleep(delay)

    @staticmethod
    def maybe_crash(component_name, probability=0.01):
        """
        Raises a localized exception to test try/except blocks.
        """
        if not ChaosEngine.ENABLED: return
        
        if random.random() < probability:
            logger.critical(f"🐒 CHAOS: Crashing {component_name} intentionally!")
            raise RuntimeError(f"Simulated Crash in {component_name}")

    def activate():
        ChaosEngine.ENABLED = True
        logger.warning("🐒 CHAOS ENGINE ACTIVATED - SYSTEM INTEGRITY AT RISK")

    def deactivate():
        ChaosEngine.ENABLED = False
        logger.info("🐒 Chaos Engine Deactivated")
