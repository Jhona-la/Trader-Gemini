
import gc
import time
from contextlib import contextmanager
from utils.logger import setup_logger

logger = setup_logger("GCTuner")

class GCTuner:
    """
    🔬 PHASE 22: GARBAGE COLLECTION TUNING
    Manages Python's GC to prevent "Stop-the-World" pauses during critical execution.
    
    Usage:
        with GCTuner.critical_section():
            # Critical HFT code (Order execution, Signal filtering)
            pass
    """
    _disabled_count = 0
    
    @staticmethod
    def disable():
        """Disables GC manually."""
        if not gc.isenabled():
            return
            
        gc.disable()
        # logger.debug("🗑️ GC Disabled (Critical Section)")

    @staticmethod
    def enable(force_collect=False):
        """Re-enables GC and optionally collects."""
        if gc.isenabled():
            return
            
        gc.enable()
        if force_collect:
            start = time.perf_counter()
            gc.collect()
            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > 1.0: # Only log slow collections
                logger.debug(f"🗑️ GC Collected in {elapsed:.2f}ms")

    @staticmethod
    @contextmanager
    def critical_section():
        """
        Context manager for atomic critical sections.
        Ensures GC is disabled on entry and re-enabled on exit.
        """
        # Optimized: checking internal flag is faster than calling gc functions
        was_enabled = gc.isenabled()
        
        if was_enabled:
            gc.disable()
            
        try:
            yield
        finally:
            if was_enabled:
                gc.enable()
                # We do NOT force collect here to avoid lag spike immediately after trade.
                # Let Python schedule it naturally or use explicit maintenance window.

    # PHASE 48: Maintenance
    _last_collect = time.time()
    
    @classmethod
    def check_maintenance(cls, interval=60.0):
        """
        Triggers GC if interval has passed since last collection.
        Call this during IDLE times (e.g. Empty Queue).
        """
        now = time.time()
        if now - cls._last_collect > interval:
            logger.debug("🧹 Running Maintenance GC...")
            start = time.perf_counter()
            gc.collect()
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"🧹 GC Maintenance Complete in {elapsed:.2f}ms")
            cls._last_collect = now

