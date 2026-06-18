import time
import ntplib
from utils.logger import logger
import statistics

class TimeSynchronizer:
    """
    🧬 COMPONENT: Stochastic Purity (Time) — Probabilistic Sync
    QUÉ: Verifica la precisión del reloj del sistema contra NTP.
    POR QUÉ: Binance usa recvWindow=5000ms. Un drift <500ms es operacionalmente
             seguro. Drifts mayores degradan la operativa, no la matan.
    CUÁNDO: Al arranque y cada 5 minutos en background.
    EVOLUCIÓN P0: De kill-switch binario a degradación suave.
    """
    
    NTP_SERVERS = ['time.google.com', 'pool.ntp.org', 'time.cloudflare.com']
    
    # Umbrales de degradación (ms) — basados en Binance recvWindow=5000ms
    DRIFT_OK = 500       # Normal para PC doméstica sin NTP dedicado
    DRIFT_WARNING = 1000  # Degradar sizing 30%
    DRIFT_DANGER = 2000   # Degradar sizing 50%
    DRIFT_CRITICAL = 4000 # Bloquear operaciones (cerca del recvWindow)
    
    # Último drift conocido (para background re-sync)
    _last_drift_ms = 0.0
    _last_sync_time = 0.0
    
    @staticmethod
    def check_drift(max_drift_ms=500):
        """
        Checks local time against NTP servers.
        Returns drift in milliseconds.
        
        P0 EVOLUCIÓN: No logea CRITICAL por drift normal (<500ms).
        Un drift de 141ms es completamente esperado en Windows.
        """
        client = ntplib.NTPClient()
        drifts = []
        
        for server in TimeSynchronizer.NTP_SERVERS:
            try:
                response = client.request(server, version=3, timeout=2)
                drifts.append(response.offset * 1000)
            except Exception:
                continue
                
        if not drifts:
            logger.warning("⚠️ [TimeSync] Could not reach any NTP server. Using last known drift.")
            return TimeSynchronizer._last_drift_ms
            
        avg_drift = statistics.median(drifts)
        TimeSynchronizer._last_drift_ms = avg_drift
        TimeSynchronizer._last_sync_time = time.time()
        
        abs_drift = abs(avg_drift)
        
        if abs_drift <= TimeSynchronizer.DRIFT_OK:
            logger.info(f"🕰️ [TimeSync] Drift: {avg_drift:.1f}ms [✅ OK]")
        elif abs_drift <= TimeSynchronizer.DRIFT_WARNING:
            logger.warning(f"🕰️ [TimeSync] Drift: {avg_drift:.1f}ms [⚠️ Elevated — sizing reduced 30%]")
        elif abs_drift <= TimeSynchronizer.DRIFT_DANGER:
            logger.warning(f"🕰️ [TimeSync] Drift: {avg_drift:.1f}ms [🔶 Danger — sizing reduced 50%]")
        else:
            logger.critical(f"🕰️ [TimeSync] Drift: {avg_drift:.1f}ms [🔴 CRITICAL — approaching recvWindow limit]")
            
        return avg_drift
    
    @staticmethod
    def get_degradation_level() -> int:
        """
        Retorna el nivel de degradación basado en el último drift conocido.
        0 = Normal, 1 = Warning, 2 = Danger, 3 = Critical
        """
        abs_drift = abs(TimeSynchronizer._last_drift_ms)
        if abs_drift <= TimeSynchronizer.DRIFT_OK:
            return 0
        elif abs_drift <= TimeSynchronizer.DRIFT_WARNING:
            return 1
        elif abs_drift <= TimeSynchronizer.DRIFT_DANGER:
            return 2
        return 3
    
    @staticmethod
    def sync():
        """Run sync check."""
        try:
            return TimeSynchronizer.check_drift()
        except ImportError:
            logger.warning("⚠️ [TimeSync] ntplib not installed. Skipping check.")
            return 0.0
        except Exception as e:
            logger.warning(f"⚠️ [TimeSync] Sync failed: {e}. Motor continues with last known drift.")
            return TimeSynchronizer._last_drift_ms
