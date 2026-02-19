
import time
import threading
from typing import Dict, Callable
from utils.logger import setup_logger

logger = setup_logger("Watchdog")

class SystemWatchdog:
    """
    🛡️ PHASE 16: SELF-HEALING STATE WATCHDOG
    Monitors heartbeats of critical subsystems. 
    If a subsystem hangs (no heartbeat > timeout), triggers recovery callback.
    """
    
    def __init__(self, check_interval=1.0):
        self._monitors: Dict[str, Dict] = {} # {component_id: {'last_beat': ts, 'timeout': sec, 'callback': fn}}
        self._running = False
        self._check_interval = check_interval
        self._lock = threading.Lock()
        
    def register_component(self, name: str, timeout: float, recovery_callback: Callable):
        """
        Registers a component to be monitored.
        Args:
            name: Component ID (e.g., 'MainEngine', 'BinanceWebsocket')
            timeout: Max seconds allowed without heartbeat before recovery.
            recovery_callback: Function to execute if watchdog bites.
        """
        with self._lock:
            self._monitors[name] = {
                'last_beat': time.time(),
                'timeout': timeout,
                'callback': recovery_callback,
                'restarts': 0
            }
        logger.info(f"🐕 Watchdog monitoring: {name} (Timeout: {timeout}s)")
        
    def heartbeat(self, name: str):
        """
        Component calls this to prove it's alive.
        """
        with self._lock:
            if name in self._monitors:
                self._monitors[name]['last_beat'] = time.time()
                
    def start(self):
        """Starts the monitoring thread."""
        if self._running: return
        self._running = True
        self.thread = threading.Thread(target=self._monitor_loop, name="SystemWatchdog", daemon=True)
        self.thread.start()
        logger.info("🐕 System Watchdog STARTED")
        
    def stop(self):
        self._running = False
        
    def _monitor_loop(self):
        while self._running:
            now = time.time()
            # Iterate copy to avoid locking for too long
            with self._lock:
                items = list(self._monitors.items())
                
            for name, data in items:
                elapsed = now - data['last_beat']
                
                if elapsed > data['timeout']:
                    logger.critical(f"🚨 WATCHDOG BITE! {name} froze for {elapsed:.1f}s (Timeout: {data['timeout']}s)")
                    
                    try:
                        # Increment restart count
                        with self._lock:
                            self._monitors[name]['restarts'] += 1
                            self._monitors[name]['last_beat'] = time.time() + 5.0 # Grace period after restart
                        
                        # Execute Recovery (Catch errors to keep watchdog alive)
                        data['callback']()
                        
                    except Exception as e:
                        logger.error(f"❌ Recovery failed for {name}: {e}")
                        
            time.sleep(self._check_interval)
