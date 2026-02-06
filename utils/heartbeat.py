
import os
import time
import json
from datetime import datetime, timezone
from utils.logger import logger
from utils.notifier import Notifier

class Heartbeat:
    """
    💓 HEARTBEAT (Latido de Supervivencia)
    
    PARA QUÉ: Detectar si el bot ha dejado de procesar el loop principal.
    CÓMO: Actualiza un archivo local y puede enviar pings externos.
    CUÁNDO: Cada N iteraciones del loop principal.
    """
    
    def __init__(self, interval_seconds=300): # 5 minutos por defecto
        self.interval = interval_seconds
        self.last_pulse = time.time()
        self.pulse_file = "logs/heartbeat.json"
        self._ensure_dir()
        
    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.pulse_file), exist_ok=True)
        
    def pulse(self, metadata=None):
        """Registra un latido de vida."""
        now = time.time()
        
        # Guardar latido localmente
        heartbeat_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_epoch": now,
            "status": "ALIVE",
            "metadata": metadata or {}
        }
        
        try:
            with open(self.pulse_file, "w") as f:
                json.dump(heartbeat_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write heartbeat: {e}")
            
        # Alerta si el tiempo entre latidos es excesivo (opcional)
        if now - self.last_pulse > self.interval * 2:
            logger.warning(f"⚠️ Heartbeat latency high: {now - self.last_pulse:.1f}s")
            
        self.last_pulse = now

    def check_survival(self):
        """
        Puede ser usado por un proceso externo para verificar si el bot vive.
        Retorna (is_alive, seconds_since_last_pulse)
        """
        if not os.path.exists(self.pulse_file):
            return False, 999999
            
        try:
            with open(self.pulse_file, "r") as f:
                data = json.load(f)
                last_ts = data.get("timestamp_epoch", 0)
                diff = time.time() - last_ts
                return diff < self.interval, diff
        except:
            return False, 999999

# Singleton instance
_heartbeat = Heartbeat()

def get_heartbeat():
    return _heartbeat
