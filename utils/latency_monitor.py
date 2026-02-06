import time
from utils.logger import logger
from collections import deque
import threading

class LatencyMonitor:
    """
    👨‍🏫 MODO PROFESOR:
    QUÉ: Sistema de telemetría para medir retrasos en milisegundos.
    POR QUÉ: En HFT, la 'latencia de ejecución' determina si un modelo rentable en teoría lo es en la práctica.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LatencyMonitor, cls).__new__(cls)
                cls._instance._init_monitor()
        return cls._instance

    def _init_monitor(self):
        self.metrics = {
            'signal_to_order': deque(maxlen=100),
            'order_to_send': deque(maxlen=100),
            'network_roundtrip': deque(maxlen=100),
            'e2e_signal_to_fill': deque(maxlen=100)
        }
        self.last_signal_ns = {}

    def track(self, metric_name: str, duration_ms: float):
        if metric_name in self.metrics:
            self.metrics[metric_name].append(duration_ms)
            
            # Alerta si la latencia supera los 300ms
            if duration_ms > 300:
                logger.warning(f"⚠️ [LATENCY] Metric '{metric_name}' spike: {duration_ms:.2f}ms")

    def report_stats(self):
        logger.info("📊 --- LATENCY REPORT (Last 100 events) ---")
        for name, values in self.metrics.items():
            if values:
                avg = sum(values) / len(values)
                max_v = max(values)
                logger.info(f"  {name:20}: Avg {avg:>7.2f}ms | Max {max_v:>7.2f}ms")
        logger.info("------------------------------------------")

latency_monitor = LatencyMonitor()
