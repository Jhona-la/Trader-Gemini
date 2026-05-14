import logging
import queue
from typing import Callable, Dict, List, Any
from core.clock import global_clock
from enum import Enum

logger = logging.getLogger(__name__)

class EventChannel(str, Enum):
    MARKET_DATA = "MARKET_DATA"   # Ticks/OHLCV
    FEATURE_UPDATE = "FEATURE_UPDATE" # Nuevas features calculadas
    SIGNALS = "SIGNALS"           # Señales crudas
    INTENTS = "INTENTS"           # TradeIntents desde Sophia
    EXECUTION = "EXECUTION"       # ExecutionPlans emitidos por MetaCoordinator
    FILLS = "FILLS"               # Fills desde Exchange
    RISK_ALERTS = "RISK_ALERTS"   # Alertas de drawdowns/invariants
    MUTATION = "MUTATION"         # Live genetic mutations

class EventBus:
    """
    Message Broker asíncrono y en memoria (Event-Driven Architecture).
    Desacopla a los productores de los consumidores, eliminando llamadas directas entre módulos.
    Garantiza consistencia secuencial.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            # { channel: [callbacks] }
            cls._instance.subscribers: Dict[str, List[Callable[[Any], None]]] = {
                channel.value: [] for channel in EventChannel
            }
            # Queue for event loop
            cls._instance.event_queue = queue.Queue()
        return cls._instance

    def subscribe(self, channel: EventChannel, callback: Callable[[Any], None]):
        """Registra un listener a un canal específico."""
        if callback not in self.subscribers[channel.value]:
            self.subscribers[channel.value].append(callback)
            logger.debug(f"[EventBus] Subscribed {callback.__name__} to {channel.value}")

    def publish(self, channel: EventChannel, payload: Any):
        """Publica un evento a la cola de procesamiento. Es Non-blocking."""
        self.event_queue.put((channel, payload))
        
    def process_queue(self, max_items: int = 100):
        """
        Procesa los eventos encolados y los despacha a los suscriptores sincrónicamente.
        Se llama desde el Event Loop central (engine.py).
        """
        items_processed = 0
        while not self.event_queue.empty() and items_processed < max_items:
            channel, payload = self.event_queue.get()
            self._dispatch(channel, payload)
            self.event_queue.task_done()
            items_processed += 1

    def _dispatch(self, channel: EventChannel, payload: Any):
        """Llama a todos los listeners del canal. Catch the errors to prevent chain collapse."""
        subs = self.subscribers.get(channel.value, [])
        for sub in subs:
            try:
                sub(payload)
            except Exception as e:
                logger.error(f"[EventBus] Error dispatching to {sub.__name__} on {channel.value}: {e}", exc_info=True)

# Global instance
event_bus = EventBus()
