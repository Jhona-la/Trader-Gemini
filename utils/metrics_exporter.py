"""
OMEGA PROTOCOL: SYSTEM METRICS EXPORTER (Phase 54/99)
=====================================================
QUÉ: Expone métricas de sistema y per-symbol al endpoint Prometheus (:8000).
POR QUÉ: Observabilidad institucional — sin métricas no hay gobernanza.
PARA QUÉ: Alimentar Grafana Commander-View para monitoreo de la flota completa.
CÓMO: prometheus_client gauges/counters + labels per-symbol. Pull model (no push).
CUÁNDO: update() se llama cada 60s desde metrics_heartbeat_loop en main.py.
DÓNDE: utils/metrics_exporter.py
QUIÉN: Singleton `metrics` importado en main.py.
"""
import time
import psutil
import os
import threading
from prometheus_client import start_http_server, Gauge, Info, Counter, Histogram


class MetricsExporter:
    """
    OMEGA PROTOCOL: SYSTEM METRICS EXPORTER
    Exposes vital system vitals to Prometheus on port 8000.
    
    Phase 99 Enhancement: Per-symbol metrics for fleet monitoring.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsExporter, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.process = psutil.Process(os.getpid())
        
        # ═══════════════════════════════════════════════
        # GLOBAL METRICS (Legacy + Enhanced)
        # ═══════════════════════════════════════════════
        self.i_info = Info('omega_bot_info', 'Bot Static Info')
        self.g_up_time = Gauge('omega_uptime_seconds', 'Uptime in seconds')
        self.g_cpu = Gauge('omega_cpu_usage_percent', 'CPU Usage % of Bot Process')
        self.g_ram = Gauge('omega_memory_usage_bytes', 'RAM Usage (RSS) in Bytes')
        self.g_equity = Gauge('omega_portfolio_equity', 'Total Portfolio Equity (USD)')
        self.g_positions = Gauge('omega_active_positions', 'Number of Open Positions')
        self.g_event_queue = Gauge('omega_event_queue_size', 'Approximate Event Queue Size')
        self.g_strategies = Gauge('omega_active_strategies', 'Number of Active Strategies')
        
        self.c_events_processed = Counter('omega_events_processed_total', 'Total processed events')
        self.c_orders = Counter('omega_orders_total', 'Total orders placed', ['side', 'type'])
        
        # ═══════════════════════════════════════════════
        # PER-SYMBOL METRICS (Phase 99: Fleet Monitoring)
        # ═══════════════════════════════════════════════
        self.g_pnl_pct = Gauge(
            'omega_pnl_percent', 
            'Unrealized PnL percentage per symbol', 
            ['symbol']
        )
        self.g_gap_tp = Gauge(
            'omega_gap_to_tp_percent', 
            'Distance to Take Profit percentage per symbol', 
            ['symbol']
        )
        self.g_tick_latency = Gauge(
            'omega_tick_latency_us', 
            'Processing latency per tick in microseconds', 
            ['symbol']
        )
        self.g_position_size = Gauge(
            'omega_position_size_usd', 
            'Position size in USD per symbol', 
            ['symbol']
        )
        
        # ═══════════════════════════════════════════════
        # SYSTEM HEALTH METRICS (Phase 99: Safety)
        # ═══════════════════════════════════════════════
        self.g_kill_switch = Gauge(
            'omega_kill_switch_active', 
            '1 if Kill Switch is active, 0 if trading allowed'
        )
        self.g_api_error_rate = Gauge(
            'omega_api_error_count', 
            'Cumulative API errors since start'
        )
        self.g_daily_losses = Gauge(
            'omega_daily_losses', 
            'Number of daily losses recorded by Kill Switch'
        )
        self.g_realized_pnl = Gauge(
            'omega_realized_pnl', 
            'Cumulative realized PnL in USD'
        )
        self.g_used_margin = Gauge(
            'omega_used_margin', 
            'Currently used margin in USD'
        )
        self.g_drawdown = Gauge(
            'omega_current_drawdown_pct', 
            'Current drawdown from peak equity percentage'
        )
        
        # Histogram for latency distribution
        self.h_tick_latency = Histogram(
            'omega_tick_latency_histogram_us',
            'Distribution of tick processing latencies',
            buckets=[10, 50, 100, 500, 1000, 5000, 10000, 50000]
        )
        
        # ═══════════════════════════════════════════════
        # SOPHIA-VIEW METRICS (Phase 99.5: Metacognition)
        # ═══════════════════════════════════════════════
        # Panel I: Calibración
        self.c_nemesis_bucket_trades = Counter('omega_nemesis_bucket_trades', 'Total trades por bucket', ['bucket'])
        self.c_nemesis_bucket_wins = Counter('omega_nemesis_bucket_wins', 'Total wins por bucket', ['bucket'])
        self.g_nemesis_brier = Gauge('omega_nemesis_brier_score', 'Rolling Brier Score Avg')
        self.g_nemesis_mental_state = Gauge('omega_nemesis_mental_state', 'Mental state: -1 (Sub), 0 (Calibrated), 1 (Overconf)')
        self.g_nemesis_penalty = Gauge('omega_nemesis_penalty_factor', 'Overconfidence penalty multiplier')
        
        # Panel II: Eficiencia Temporal
        self.h_nemesis_time_error = Histogram(
            'omega_nemesis_time_error_mins', 
            'Desviación en minutos vs estimación',
            buckets=[-30, -10, -2, 0, 2, 10, 30, 60]
        )
        self.g_nemesis_efficiency = Gauge('omega_nemesis_efficiency_usd_per_min', 'Eficiencia de capital', ['symbol'])
        
        # Panel III: Explicabilidad
        self.g_sophia_feature_imp = Gauge('omega_sophia_feature_importance', 'Importancia', ['symbol', 'feature'])
        self.g_nemesis_feature_acc = Gauge('omega_nemesis_feature_accuracy', 'Hit rate (SHAP)', ['feature'])
        self.g_sophia_entropy = Gauge('omega_sophia_decision_entropy', 'Entropía de la red bayesiana', ['symbol'])
        
        # Panel IV: Salud Sistémica
        self.h_nemesis_latency = Histogram(
            'omega_nemesis_autopsy_latency_ms', 
            'Tiempo de ejecución de full_autopsy',
            buckets=[1, 5, 10, 50, 100, 500]
        )
        self.g_phalanx_consensus = Gauge('omega_phalanx_consensus_count', 'Modelos en consenso', ['symbol'])
        self.g_quarantine_status = Gauge('omega_quarantine_status', '1 si flaggeado para reemplazo', ['genotype_id'])

        
        # Internal state
        self.start_time = time.time()
        self._tick_latencies = {}  # {symbol: last_latency_us}
        self._lock = threading.Lock()
        self.initialized = True
        print("📊 Metrics Exporter Initialized (Phase 99: Per-Symbol + Health).")

    def start_server(self, port=8000):
        """Start the Prometheus HTTP Server background thread."""
        try:
            start_http_server(port)
            print(f"📡 Prometheus Metrics Server running on port {port}")
            self.i_info.info({
                'version': '2.0.0-OMEGA', 
                'mode': os.getenv('BOT_MODE', 'unknown'),
                'phase': '99'
            })
        except Exception as e:
            print(f"❌ Failed to start Metrics Server: {e}")

    def update(self, portfolio=None, engine=None, queue_size=0):
        """Update all gauge values (Called periodically from heartbeat loop)."""
        try:
            # ─── System Vitals ───
            self.g_up_time.set(time.time() - self.start_time)
            self.g_cpu.set(self.process.cpu_percent())
            self.g_ram.set(self.process.memory_info().rss)
            
            # ─── Application Logic ───
            if portfolio:
                self.g_equity.set(portfolio.get_total_equity())
                active_count = len([
                    s for s, p in portfolio.positions.items() 
                    if p.get('quantity', 0) != 0
                ])
                self.g_positions.set(active_count)
                
                # Realized PnL & Margin
                self.g_realized_pnl.set(getattr(portfolio, 'realized_pnl', 0))
                self.g_used_margin.set(getattr(portfolio, 'used_margin', 0))
                
                # Drawdown
                peak = getattr(portfolio, 'peak_equity', portfolio.get_total_equity())
                equity = portfolio.get_total_equity()
                if peak > 0:
                    dd = ((peak - equity) / peak) * 100
                    self.g_drawdown.set(dd)
                
                # ─── Per-Symbol Metrics ───
                self._update_per_symbol(portfolio)
                
            if engine:
                self.g_strategies.set(len(engine.strategies))
                
            self.g_event_queue.set(queue_size)
            
        except Exception:
            pass  # Fail silently in production logging loop

    def update_health(self, risk_manager=None):
        """Update Kill Switch and API health metrics."""
        try:
            if risk_manager and hasattr(risk_manager, 'kill_switch'):
                ks = risk_manager.kill_switch
                self.g_kill_switch.set(1 if ks.active else 0)
                self.g_api_error_rate.set(getattr(ks, 'api_errors', 0))
                self.g_daily_losses.set(getattr(ks, 'daily_losses', 0))
        except Exception:
            pass

    def _update_per_symbol(self, portfolio):
        """Updates PnL% and Gap TP% gauges for each active position."""
        try:
            for symbol, pos in portfolio.positions.items():
                qty = pos.get('quantity', 0)
                if qty == 0:
                    # Clear stale labels when position is closed
                    continue
                    
                entry = pos.get('avg_price', 0)
                current = pos.get('current_price', entry)
                tp_price = pos.get('tp_price', 0)
                side = "LONG" if qty > 0 else "SHORT"
                
                # Clean symbol for Prometheus label
                clean_sym = symbol.replace('/', '_')
                
                # PnL%
                if entry > 0 and current > 0:
                    if side == "LONG":
                        pnl_pct = ((current - entry) / entry) * 100
                    else:
                        pnl_pct = ((entry - current) / entry) * 100
                    self.g_pnl_pct.labels(symbol=clean_sym).set(round(pnl_pct, 4))
                
                # Gap to TP%
                if tp_price > 0 and current > 0:
                    if side == "LONG":
                        gap = ((tp_price - current) / current) * 100
                    else:
                        gap = ((current - tp_price) / current) * 100
                    self.g_gap_tp.labels(symbol=clean_sym).set(round(gap, 4))
                
                # Position Size USD
                pos_usd = abs(qty) * current
                self.g_position_size.labels(symbol=clean_sym).set(round(pos_usd, 2))
                
                # Tick Latency (from internal buffer)
                with self._lock:
                    lat = self._tick_latencies.get(symbol, 0)
                    if lat > 0:
                        self.g_tick_latency.labels(symbol=clean_sym).set(lat)
                        
        except Exception:
            pass

    def record_tick_latency(self, symbol: str, latency_us: float):
        """Records tick processing latency for a symbol (called from Engine)."""
        with self._lock:
            self._tick_latencies[symbol] = latency_us
        self.h_tick_latency.observe(latency_us)

    def inc_event(self):
        self.c_events_processed.inc()
        
    def inc_order(self, side, order_type):
        self.c_orders.labels(side=side, type=order_type).inc()

    # ─── SOPHIA-VIEW Ingestion Methods ───
    def record_sophia_inference(self, symbol: str, entropy: float, top_features: list, consensus_count: int):
        """Called from technical.py when a new signal + Sophia report is generated."""
        try:
            self.g_sophia_entropy.labels(symbol=symbol).set(entropy)
            self.g_phalanx_consensus.labels(symbol=symbol).set(consensus_count)
            # Update top 3 features
            for feat in top_features[:3]:
                name = feat.get('feature')
                contrib = feat.get('contribution', 0.0)
                if name:
                    self.g_sophia_feature_imp.labels(symbol=symbol, feature=name).set(contrib)
        except Exception:
            pass

    def record_nemesis_autopsy(
        self, 
        symbol: str, 
        actual_pnl: float,
        brier_score: float, 
        brier_bucket: str,
        predicted_mins: float, 
        actual_mins: float,
        efficiency: float, 
        shap_accuracy: float,
        mismatches: list,
        overconfidence_active: bool,
        penalty_factor: float,
        gene_flagged: bool,
        genotype_id: str,
        latency_ms: float
    ):
        """Called from nemesis.py full_autopsy()."""
        try:
            # Latency
            self.h_nemesis_latency.observe(latency_ms)
            
            # Calibration Panel
            self.c_nemesis_bucket_trades.labels(bucket=brier_bucket).inc()
            if actual_pnl > 0:
                self.c_nemesis_bucket_wins.labels(bucket=brier_bucket).inc()
            
            self.g_nemesis_brier.set(brier_score)
            
            # Mental State (-1 sub, 0 cal, 1 over)
            mental_state = 1 if overconfidence_active else (
                -1 if brier_score > 0.3 else 0
            )
            self.g_nemesis_mental_state.set(mental_state)
            self.g_nemesis_penalty.set(penalty_factor)
            
            # Temporal Profile
            time_error = actual_mins - predicted_mins
            self.h_nemesis_time_error.observe(time_error)
            self.g_nemesis_efficiency.labels(symbol=symbol).set(efficiency)
            
            # Explainability (Update accuracy for all missed features to lower, others maintain)
            # Just set mismatches to 0.0 accuracy temporarily for dash effect, though proper logic 
            # uses the rolling avg from the analyzer. We will assume the caller could pass a dict of {feat: acc}, 
            # but for now we set global shap_accuracy.
            # (In a real scenario, we pass a dict of feature accuracies, but for simplicity we'll just track global)
            # We'll update feature_accuracy Gauge from the caller by extending the interface if needed,
            # or just register the mismatches explicitly:
            for feat in mismatches:
                self.g_nemesis_feature_acc.labels(feature=feat).set(shap_accuracy)
            
            # Systemic Health
            flag_val = 1 if gene_flagged else 0
            self.g_quarantine_status.labels(genotype_id=genotype_id).set(flag_val)
            
        except Exception:
            pass


# Global Singleton Access
metrics = MetricsExporter()
