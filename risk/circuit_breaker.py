import time
from utils.logger import logger

class CircuitBreaker:
    """
    [PHASE VI] HFT Circuit Breaker & Flash Crash Guard
    Protects the system from catastrophic anomalies like Flash Crashes
    or API degradation by cutting off signal execution entirely.
    """
    
    def __init__(self, flash_crash_threshold=0.15, timeout_limit_ms=50):
        self.flash_crash_threshold = flash_crash_threshold # >15% drop/spike in < 5 mins
        self.timeout_limit_ms = timeout_limit_ms
        self.is_tripped = False
        self.tripped_until = 0
        self.price_history = {} # {symbol: [(timestamp, price)]}
        self.consecutive_timeouts = 0
        
    def check_flash_crash(self, symbol: str, current_price: float) -> bool:
        """
        Returns True if a flash crash is detected, tripping the breaker.
        """
        now = time.time()
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        history = self.price_history[symbol]
        history.append((now, current_price))
        
        # Clean up history older than 5 minutes
        self.price_history[symbol] = [x for x in history if now - x[0] <= 300]
        
        if len(self.price_history[symbol]) > 1:
            oldest_price = self.price_history[symbol][0][1]
            deviation = abs(current_price - oldest_price) / oldest_price
            
            if deviation >= self.flash_crash_threshold:
                logger.critical(f"🛑 [CIRCUIT BREAKER] Flash Crash Detectado en {symbol}: Desviación {deviation*100:.2f}% en < 5 min.")
                self.trip(duration_seconds=900) # Trip for 15 minutes
                return True
                
        return False
        
    def report_api_timeout(self, latency_ms: float):
        if latency_ms > self.timeout_limit_ms:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts >= 3:
                logger.critical(f"🛑 [CIRCUIT BREAKER] Degradación de API: 3+ Timeouts > {self.timeout_limit_ms}ms detectados.")
                self.trip(duration_seconds=300) # Trip for 5 minutes
        else:
            self.consecutive_timeouts = 0
            
    def trip(self, duration_seconds: int = 300):
        self.is_tripped = True
        self.tripped_until = time.time() + duration_seconds
        
    def is_active(self) -> bool:
        if self.is_tripped:
            if time.time() >= self.tripped_until:
                self.is_tripped = False
                logger.info("🟢 [CIRCUIT BREAKER] Sistema restaurado tras enfriamiento.")
                return False
            return True
        return False
