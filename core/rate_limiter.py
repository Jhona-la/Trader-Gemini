
import time
from utils.logger import setup_logger
from threading import Lock

logger = setup_logger("RateLimiter")

class PredictiveRateLimiter:
    """
    🛡️ PHASE 47: PREDICTIVE RATE LIMITER (Metal-Core)
    Uses Numba Token Bucket for thread-safe, nanosecond precision limiting.
    """
    def __init__(self, limit_1m=1200, buffer=100):
        self.limit_1m = limit_1m
        self.safe_buffer = buffer
        
        # Rate: Tokens per second
        # Capacity: Max Tokens (Burst)
        rate = limit_1m / 60.0 # e.g., 20 tokens/sec
        capacity = limit_1m
        
        from utils.token_bucket import NumbaTokenBucket
        self.bucket = NumbaTokenBucket(rate, capacity)
        
        # For backward compat / logging tracking only
        self.used_weight_1m = 0 
        
    def check_limit(self, weight_cost=1):
        """
        Check and consume tokens.
        Returns: (allowed, wait_time)
        """
        success, wait_time = self.bucket.consume(float(weight_cost))
        
        if not success:
            logger.warning(f"🛑 RATE LIMIT: Predicting ban. Wait {wait_time:.3f}s")
            return False, wait_time
            
        return True, 0.0
        
    def update_from_headers(self, headers):
        """
        [DF-C8 FIX] Sync with server headers.
        """
        if not headers:
            return
            
        try:
            # Parse 'x-mbx-used-weight-1m' (case-insensitive usually, but requests headers are compliant)
            # CCXT usually normalizes headers, but let's be safe
            used = None
            for k, v in headers.items():
                if k.lower() == 'x-mbx-used-weight-1m':
                    used = float(v)
                    break
            
            if used is not None:
                self.bucket.sync(used)
                
                # Optional: Log if drift was significant
                # (Requires exposing state getter from bucket, skipping for latency)
        except Exception as e:
            # Don't crash on header parsing
            pass
