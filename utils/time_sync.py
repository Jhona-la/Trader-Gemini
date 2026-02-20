import time
import ntplib
from utils.logger import logger
import statistics

class TimeSynchronizer:
    """
    🧬 COMPONENT: Stochastic Purity (Time)
    QUÉ: Verifica la precisión del reloj del sistema.
    POR QUÉ: High Frequency Trading requiere <100ms de error.
             Si el reloj está mal, las timestamps de las órdenes y el Order Flow son inútiles.
    """
    
    NTP_SERVERS = ['time.google.com', 'pool.ntp.org', 'time.cloudflare.com']
    
    @staticmethod
    def check_drift(max_drift_ms=100):
        """
        Checks local time against NTP servers.
        Returns drift in milliseconds.
        """
        client = ntplib.NTPClient()
        drifts = []
        
        for server in TimeSynchronizer.NTP_SERVERS:
            try:
                response = client.request(server, version=3, timeout=2)
                # offset = (server_time - client_time)
                drifts.append(response.offset * 1000) # Convert to ms
            except Exception:
                continue
                
        if not drifts:
            logger.warning("⚠️ [TimeSync] Could not reach any NTP server.")
            return 0.0
            
        # Robust drift (Median)
        avg_drift = statistics.median(drifts)
        
        logger.info(f"🕰️ [TimeSync] Drift: {avg_drift:.2f}ms")
        
        if abs(avg_drift) > max_drift_ms:
            logger.critical(f"❌ [TimeSync] SYSTEM CLOCK UNSYNCED! Drift {avg_drift:.2f}ms > {max_drift_ms}ms")
            # In Phase 99, we might force exit or switch to 'server_time' offset mode.
            return avg_drift
            
        return avg_drift
    
    @staticmethod
    def sync():
        """Run sync check."""
        try:
            return TimeSynchronizer.check_drift()
        except ImportError:
            logger.warning("⚠️ [TimeSync] ntplib not installed. Skipping check.")
            return 0.0
