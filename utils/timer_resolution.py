
import os
import ctypes
import contextlib
import logging

logger = logging.getLogger("TimerResolution")

def set_high_resolution_timer():
    """
    🔬 PHASE 32: KERNEL TIMER RESOLUTION
    Forces Windows System Timer to 1ms resolution (default is 15.6ms).
    This drastically improves time.sleep() precision and AsyncIO event loop reactivity.
    
    Equivalent to 'timeBeginPeriod(1)' in C++.
    """
    if os.name != 'nt':
        return

    try:
        # Load winmm.dll
        winmm = ctypes.windll.winmm
        
        # Set resolution to 1ms
        # timeBeginPeriod(UINT uPeriod)
        # Returns TIMERR_NOERROR (0) if successful
        ret = winmm.timeBeginPeriod(1)
        
        if ret == 0:
            logger.info("🔬 [PHASE 32] Windows Timer Resolution forced to 1ms (High Precision).")
        else:
            logger.warning(f"⚠️ Failed to set Windows Timer Resolution (Code: {ret})")
            
    except Exception as e:
        logger.error(f"❌ Failed to access winmm.dll for Timer Resolution: {e}")

@contextlib.contextmanager
def high_resolution_scope():
    """
    Context manager to enable/disable high resolution timer temporarily.
    """
    if os.name == 'nt':
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)
        try:
            yield
        finally:
            winmm.timeEndPeriod(1)
    else:
        yield
