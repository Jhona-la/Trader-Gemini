
import time
import logging
import requests
import threading

logger = logging.getLogger("NTPMonitor")

class NTPSync:
    """
    🔬 PHASE 37: NTP SYNC CHECKER
    Monitors the drift between Local System Time and Binance Server Time.
    If drift > 500ms, it warns the user.
    If drift > 1000ms, orders WILL fail (recvWindow default is often 5000ms, but safe margin is 1000).
    """
    
    _drift_ms = 0
    _last_check = 0
    
    @staticmethod
    def get_drift() -> float:
        """Returns cached drift in ms."""
        return NTPSync._drift_ms

    @staticmethod
    def sync_time():
        """
        Calculates time drift against Binance API.
        This is a blocking HTTP call, so run it in startup or async thread.
        """
        try:
            # We use a raw request to avoid huge dependency chains just for time
            t0 = time.time()
            response = requests.get("https://api.binance.com/api/v3/time", timeout=2)
            t1 = time.time()
            
            if response.status_code == 200:
                server_time = response.json()['serverTime']
                # Latency adjustment (Round Trip Time / 2)
                rtt = (t1 - t0) * 1000
                local_time = int(t1 * 1000)
                
                # Estimated server time at moment t1 = server_time + (rtt / 2) ??? 
                # Actually server_time in payload is the time when server PROCESSED it.
                # So it is roughly t1 - (rtt/2).
                # Drift = Local - (Server + RTT/2)
                
                estimated_server_time = server_time + (rtt / 2)
                drift = local_time - estimated_server_time
                
                NTPSync._drift_ms = drift
                NTPSync._last_check = time.time()
                
                status = "✅ OK" if abs(drift) < 500 else "⚠️ HIGH DRIFT"
                logger.info(f"🔬 [PHASE 37] Time Sync: Drift = {drift:.1f}ms (RTT: {rtt:.1f}ms) [{status}]")
                return drift
            else:
                logger.warning(f"Failed to check time sync (HTTP {response.status_code})")
                return 0
                
        except Exception as e:
            logger.error(f"NTP Check Failed: {e}")
            return 0
            
    @staticmethod
    def start_background_monitor(interval_seconds=3600):
        """Starts a background thread to re-sync every hour."""
        def monitor_loop():
            while True:
                time.sleep(interval_seconds)
                NTPSync.sync_time()
                
        t = threading.Thread(target=monitor_loop, daemon=True, name="NTPMonitor")
        t.start()
        logger.info("🔬 [PHASE 37] Background NTP Monitor started.")
