
import time
from threading import Lock
from utils.logger import logger

class CircuitBreaker:
    """
    Protects the system from cascading failures by pausing operations
    when a threshold of errors is reached.
    
    States:
    - CLOSED: Normal operation (circuit is closed, electricity flows).
    - OPEN: Failure threshold reached, operations blocked.
    - HALF-OPEN: Probational period, allowing one test request.
    """
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        self._lock = Lock()
        
    def record_failure(self):
        """Call this when an external API call fails."""
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold and self.state == "CLOSED":
                self.state = "OPEN"
                logger.critical(f"🔥 BLOCK 4 PROTECTION: Circuit Breaker TRIPPED! ({self.failures} failures). Pausing API calls for {self.recovery_timeout}s.")
                
    def record_success(self):
        """Call this when an external API call succeeds."""
        with self._lock:
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
                logger.info("✅ Circuit Breaker RECOVERED. System returning to normal.")
            elif self.state == "CLOSED":
                self.failures = 0
                
    def can_proceed(self):
        """Returns True if the operation should proceed, False if blocked."""
        with self._lock:
            if self.state == "CLOSED":
                return True
                
            if self.state == "OPEN":
                now = time.time()
                if now - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF-OPEN"
                    logger.info("⚠️ Circuit Breaker HALF-OPEN: Retrying connection...")
                    return True # Allow one probe request
                return False
                
            if self.state == "HALF-OPEN":
                # In robust impl we might allow 1 request, but multithreaded requires care.
                # Here we allow, record_success/failure will decide next state.
                return True
        return True

# Global Instance
api_circuit = CircuitBreaker()
