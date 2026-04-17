import os
from datetime import datetime
from collections import deque
import threading
from utils.logger import logger
from utils.alert_system import get_alert_system

class InteractionMonitor:
    """
    SISTEMA DE AUTO-DIAGNÓSTICO: Monitor de Interacciones y Conflictos.
    Detecta si dos procesos del bot se están pisando (ej. Inteligencia manda LONG infinito mientras RiskManager cancela).
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(InteractionMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            # Queue to prevent memory leak, only tracks last 500 interactions
            self.interaction_log = deque(maxlen=500)
            self.alert_sys = get_alert_system()
            self.initialized = True
            
    def log_interaction(self, source: str, target: str, action: str, result: str, metadata: dict = None):
        entry = {
            "timestamp": datetime.now(),
            "source": source,
            "target": target,
            "action": action,
            "result": result,
            "metadata": metadata or {},
            "conflict_detected": False
        }
        
        # Async analysis to prevent blocking event loop
        threading.Thread(target=self._analyze_conflict, args=(entry,), daemon=True).start()
        
        self.interaction_log.append(entry)
        
    def _analyze_conflict(self, entry: dict):
        conflict_type = self._detect_conflict(entry)
        if conflict_type:
            entry["conflict_detected"] = True
            entry["conflict_type"] = conflict_type
            
            logger.warning(f"⚠️ [DIAGNOSTIC] Conflicto Interacción: {conflict_type} ({entry['source']} vs {entry['target']})")
            
            # Subir alarma si es severo
            if conflict_type == "PERSISTENT_REJECTION":
                self.alert_sys.raise_alert("strategy_conflict", component=f"{entry['source']}->{entry['target']}")
                
    def _detect_conflict(self, entry: dict) -> str:
        # Detectar conflictos comunes revisando el historial reciente
        if entry["action"] == "place_order" and entry["result"] == "rejected":
            return self._check_persistent_rejection(entry)
            
        return None
        
    def _check_persistent_rejection(self, current_entry: dict):
        """Si en las últimas 10 entradas hay 5 rechazos del target al source."""
        rejects = 0
        recent = list(self.interaction_log)[-10:]
        for e in recent:
            if e["source"] == current_entry["source"] and e["target"] == current_entry["target"] \
               and e["action"] == current_entry["action"] and e["result"] == "rejected":
                rejects += 1
                
        if rejects >= 5:
            return "PERSISTENT_REJECTION"
        return None

_interaction_sys = None
def get_interaction_monitor() -> InteractionMonitor:
    global _interaction_sys
    if not _interaction_sys:
        _interaction_sys = InteractionMonitor()
    return _interaction_sys
