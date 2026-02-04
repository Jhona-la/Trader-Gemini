from dataclasses import dataclass, field
from typing import List
import time
from utils.logger import logger

@dataclass
class PerformanceMetrics:
    """
    Métricas de performance del sistema.
    Reference: Master Bible v2.0.1 Phase 7.3
    """
    execution_latencies: List[float] = field(default_factory=list)
    slippages: List[float] = field(default_factory=list)
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    api_errors: int = 0
    signals_generated: int = 0
    signals_executed: int = 0
    
    def add_execution(self, latency_ms: float, slippage_pct: float):
        self.execution_latencies.append(latency_ms)
        self.slippages.append(slippage_pct)
    
    def add_trade_result(self, pnl: float):
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.total_pnl += pnl
    
    def get_win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0
    
    def get_avg_latency(self) -> float:
        return sum(self.execution_latencies) / len(self.execution_latencies) if self.execution_latencies else 0.0
    
    def get_avg_slippage(self) -> float:
        return sum(self.slippages) / len(self.slippages) if self.slippages else 0.0
    
    def to_dict(self) -> dict:
        return {
            'win_rate': self.get_win_rate(),
            'avg_latency_ms': self.get_avg_latency(),
            'avg_slippage_pct': self.get_avg_slippage(),
            'total_trades': self.win_count + self.loss_count,
            'total_pnl': self.total_pnl,
            'api_errors': self.api_errors,
            'signal_execution_rate': self.signals_executed / self.signals_generated if self.signals_generated > 0 else 0
        }

def export_metrics_to_prometheus(metrics: PerformanceMetrics, port: int = 8000):
    """
    Exporta métricas en formato Prometheus.
    Reference: Master Bible v2.0.1 Phase 7.4
    """
    try:
        from prometheus_client import start_http_server, Gauge
        
        # Singleton check (naive) or just try/except if already started
        # In a real app we'd manage the registry better.
        # But per Bible snippet:
        
        # Definir gauges
        # Note: Gauges should be global or checking registry to avoid duplicates on re-calls
        # For now implementing simple version
        
        # We need to only start server ONCE
        if not hasattr(export_metrics_to_prometheus, 'server_started'):
             start_http_server(port)
             export_metrics_to_prometheus.server_started = True
             logger.info(f"Prometheus metrics server started on port {port}")
             
        # Definition (creates or gets if using default registry?)
        # prometheus_client registers globally by default.
        # We should define these outside or handle existing.
        # Impl:
        
        # Using global registry
        from prometheus_client import REGISTRY
        
        def get_or_create_gauge(name, doc):
             if name in REGISTRY._names_to_collectors:
                 return REGISTRY._names_to_collectors[name]
             return Gauge(name, doc)

        win_rate_gauge = get_or_create_gauge('trading_win_rate', 'Win rate percentage')
        latency_gauge = get_or_create_gauge('trading_avg_latency_ms', 'Average execution latency')
        pnl_gauge = get_or_create_gauge('trading_total_pnl', 'Total PnL in USD')
        
        # Actualizar gauges
        win_rate_gauge.set(metrics.get_win_rate())
        latency_gauge.set(metrics.get_avg_latency())
        pnl_gauge.set(metrics.total_pnl)
        
    except ImportError:
        logger.warning("prometheus_client not installed. Metrics export disabled.")
    except Exception as e:
        logger.error(f"Failed to export Prometheus metrics: {e}")
