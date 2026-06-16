from utils.logger import logger
from collections import deque
import threading

class LatencyMonitor:
    """
    👨‍🏫 MODO PROFESOR:
    QUÉ: Sistema de telemetría avanzado para medir retrasos en milisegundos.
    POR QUÉ: En HFT, la 'latencia de ejecución' determina si un modelo rentable
      en teoría lo es en la práctica. Sin percentiles (p50/p95/p99) es imposible
      distinguir entre latencia estable y picos esporádicos que destruyen PnL.
    PARA QUÉ: Medir el impacto real de optimizaciones y detectar regresiones.
    CÓMO: Deques acotadas con cálculo de percentiles O(n log n) amortizado.
    CUÁNDO: En cada evento procesado por el Engine.
    DÓNDE: utils/latency_monitor.py
    QUIÉN: SRE/DevOps + QA Engineer

    LOW-LATENCY PHASE: Added p50/p95/p99, jitter tracking, snapshot() for dashboard.
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
            'signal_to_order': deque(maxlen=200),
            'order_to_send': deque(maxlen=200),
            'network_roundtrip': deque(maxlen=200),
            'e2e_signal_to_fill': deque(maxlen=200),
            'engine_jitter_warning': deque(maxlen=200),
        }
        self.last_signal_ns = {}
        # Jitter tracking: store deltas between consecutive measurements
        self._prev_values = {}

    def track(self, metric_name: str, duration_ms: float):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=200)

        self.metrics[metric_name].append(duration_ms)

        # Jitter tracking: measure variance between consecutive values
        prev = self._prev_values.get(metric_name)
        if prev is not None:
            jitter = abs(duration_ms - prev)
            jitter_key = f"{metric_name}_jitter"
            if jitter_key not in self.metrics:
                self.metrics[jitter_key] = deque(maxlen=200)
            self.metrics[jitter_key].append(jitter)
        self._prev_values[metric_name] = duration_ms

        # Alerta si la latencia supera los 300ms
        if duration_ms > 300:
            logger.warning(f"⚠️ [LATENCY] Metric '{metric_name}' spike: {duration_ms:.2f}ms")

    def track_hotpath(self, duration_ns: int):
        """
        FORENSIC FIX: Quantum Telemetry (E2E Latency)
        QUÉ: Recibe latencia en nanosegundos y la registra en ms con alta precisión.
        POR QUÉ: Permite rastrear la fase 3 (Trazado End-to-End).
        """
        duration_ms = duration_ns / 1_000_000.0
        self.track('hotpath_e2e', duration_ms)

    @staticmethod
    def _percentile(sorted_data, pct):
        """Calculate percentile from sorted list. O(1) after sort."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[f]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def get_percentiles(self, metric_name: str) -> dict:
        """
        Returns p50, p95, p99, avg, max for a given metric.
        QUÉ: Cálculo de percentiles para análisis estadístico de latencia.
        POR QUÉ: avg/max solos ocultan la distribución real. p99 revela
          los peores casos que impactan PnL en trades críticos.
        """
        values = self.metrics.get(metric_name)
        if not values:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'avg': 0.0, 'max': 0.0, 'count': 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        avg = sum(sorted_vals) / n

        return {
            'p50': self._percentile(sorted_vals, 50),
            'p95': self._percentile(sorted_vals, 95),
            'p99': self._percentile(sorted_vals, 99),
            'avg': avg,
            'max': sorted_vals[-1],
            'min': sorted_vals[0],
            'count': n,
        }

    def snapshot(self) -> dict:
        """
        Returns a full snapshot of all metrics with percentiles.
        QUÉ: Exporta todas las métricas de latencia en un solo dict.
        POR QUÉ: El dashboard necesita un snapshot atómico para renderizar.
        PARA QUÉ: Integración con dashboard/app.py y Prometheus.
        """
        result = {}
        for name in list(self.metrics.keys()):
            if name.endswith('_jitter'):
                continue  # Jitter metrics are included via their parent
            result[name] = self.get_percentiles(name)
            # Include jitter stats if available
            jitter_key = f"{name}_jitter"
            if jitter_key in self.metrics and self.metrics[jitter_key]:
                jitter_vals = sorted(self.metrics[jitter_key])
                result[name]['jitter_avg'] = sum(jitter_vals) / len(jitter_vals)
                result[name]['jitter_p99'] = self._percentile(jitter_vals, 99)
        return result

    def report_stats(self):
        logger.info("📊 --- LATENCY REPORT (Last 200 events) ---")
        for name, values in self.metrics.items():
            if values and not name.endswith('_jitter'):
                stats = self.get_percentiles(name)
                jitter_avg = stats.get('jitter_avg', 0.0)
                logger.info(
                    f"  {name:25}: "
                    f"p50={stats['p50']:>7.2f}ms | "
                    f"p95={stats['p95']:>7.2f}ms | "
                    f"p99={stats['p99']:>7.2f}ms | "
                    f"Avg={stats['avg']:>7.2f}ms | "
                    f"Max={stats['max']:>7.2f}ms | "
                    f"Jitter={jitter_avg:>5.2f}ms"
                )
        logger.info("------------------------------------------")

latency_monitor = LatencyMonitor()

