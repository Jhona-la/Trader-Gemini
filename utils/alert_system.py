import os
from datetime import datetime, timedelta
import threading
from typing import Dict, Any

CRITICAL_ALERTS = {
    "process_not_activated": {
        "message": "Proceso crítico no activado o en dead-lock: {process_name}",
        "action": "Revisar inicialización, dependencias o reiniciar hilo de {process_name}.",
        "timeout": 300  
    },
    "consistent_losses": {
        "message": "Pérdidas consistentes detectadas: {loss_streak} trades perdidos.",
        "action": "Revisar estrategias, aplicar engine cooldown.",
        "timeout": 600  
    },
    "fee_death_spiral": {
        "message": "La comisión borra las ganancias (Fees > 40%): Ratio {fee_ratio:.2f}% de PnL bruto.",
        "action": "Aumentar TP mínimo dinámicamente y reducir frecuencia de trades.",
        "timeout": 300
    },
    "strategy_conflict": {
        "message": "Conflictos persistentes (Orders vs Risk rejects) detectado.",
        "action": "Investigar logic loop en Interaction Monitor.",
        "timeout": 300
    },
    "slippage_erosion": {
        "message": "Slippage destruyendo bordes de rentabilidad. Delta: {slippage_diff}%",
        "action": "Optimizar market limits, endurecer tolerance limits.",
        "timeout": 600
    }
}

class AlertSystem:
    """
    SISTEMA DE AUTO-DIAGNÓSTICO: Núcleo Emisor de Alertas Centrales.
    Singleton seguro entre hilos para trackear y emitir alertas.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(AlertSystem, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.active_alerts = {}
            self.alert_history = []
            # Fallback direct telegram check
            self.telegram_ready = bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))
            self.initialized = True
    
    def raise_alert(self, alert_type: str, **kwargs):
        if alert_type not in CRITICAL_ALERTS:
            return
            
        alert_config = CRITICAL_ALERTS[alert_type]
        
        # Rate Limiting: No duplicar alertas activas del mismo tipo
        for ex_id, ex_alert in self.active_alerts.items():
            if ex_alert["type"] == alert_type and ex_alert["status"] == "ACTIVE":
                # Check si ha pasado el timeout de la alerta previa para poder re-enviarla
                if datetime.now() < ex_alert['raised_at'] + timedelta(seconds=ex_alert['timeout']):
                    return # Aún estamos dentro de la ventana de espera, silenciamos la alerta duplicada
                else:
                    # Lo marcamos como TIMEOUT
                    ex_alert["status"] = "TIMEOUT"
                    
        alert_id = f"{alert_type}_{int(datetime.now().timestamp())}"
        
        alert_data = {
            "id": alert_id,
            "type": alert_type,
            "message": alert_config["message"].format(**kwargs),
            "action": alert_config["action"],
            "severity": "CRITICAL",
            "raised_at": datetime.now(),
            "timeout": alert_config["timeout"],
            "status": "ACTIVE"
        }
        
        self.active_alerts[alert_id] = alert_data
        self.alert_history.append(alert_data)
        
        # Disparo Asíncrono para no bloquear Event Loop Trading
        threading.Thread(target=self.send_notification, args=(alert_data,), daemon=True).start()
        
    def resolve_alert(self, alert_type: str):
        """Marca como RESUELTA las alertas activas de este tipo."""
        for ex_id, ex_alert in self.active_alerts.items():
            if ex_alert["type"] == alert_type and ex_alert["status"] == "ACTIVE":
                ex_alert["status"] = "RESOLVED"
                ex_alert["resolved_at"] = datetime.now()
        
    def send_notification(self, alert_data):
        from utils.logger import logger
        deadline = alert_data['raised_at'] + timedelta(seconds=alert_data['timeout'])
        
        msg = (f"🚨 ALERTA CRÍTICA (Auto-Diagnóstico) 🚨\n\n"
               f"{alert_data['message']}\n"
               f"Acción requerida: {alert_data['action']}\n"
               f"Severidad: {alert_data['severity']}\n"
               f"Time: {alert_data['raised_at'].strftime('%H:%M:%S')}\n"
               f"⚠️ Resolver antes de: {deadline.strftime('%H:%M:%S')}")
        
        logger.error(msg.replace('\n', ' | '))
        
        if self.telegram_ready:
            from config import Config
            if getattr(Config.Observability, "TELEGRAM_ENABLED", True):
                try:
                    import requests
                    token = os.getenv("TELEGRAM_BOT_TOKEN")
                    chat_id = os.getenv("TELEGRAM_CHAT_ID")
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": chat_id,
                        "text": msg
                    }
                    requests.post(url, json=payload, timeout=5)
                except Exception as e:
                    logger.error(f"Fallo enviando alerta Telegram Diagnóstico: {e}")

_alert_sys = None
def get_alert_system() -> AlertSystem:
    global _alert_sys
    if not _alert_sys:
        _alert_sys = AlertSystem()
    return _alert_sys
