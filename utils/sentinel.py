"""
🛡️ SENTINEL-24: MISSION AUDIT & SUPERVISION SYSTEM
====================================================

PROFESSOR METHOD:
- QUÉ: Sistema de auditoría y supervisión 24h para validar producción.
- POR QUÉ: Garantiza que el test de 24h sea fuente de verdad absoluta.
- CÓMO: Monitorea señales, latencia, margen, equity y estrategias.
- CUÁNDO: Ejecutar continuamente durante pruebas de 24h pre-producción.
- DÓNDE: utils/sentinel.py

MONITORING STREAMS:
- logs/strategy.log: Decisiones de estrategia
- logs/health_log.json: Integridad de datos  
- trades.csv: Esperanza Matemática real
- live_status.json: Telemetría de cuenta

ALERTS:
- Ghost Signals: Señales > 0.0 pero < 0.60 umbral
- Latency Breach: Pipeline > 500ms
- Margin Deviation: Cambio significativo sin trades
- Equity Drift: Deriva de datos cada 15min
- Strategy Momentum: Win rate estabilidad
"""

import os
import logging
import psutil
import shutil
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


# =============================================================================
# ENUMS & TYPES
# =============================================================================

class HealthStatus(Enum):
    """System health status indicators."""
    HEALTHY = "🟢"
    WARNING = "🟡"
    CRITICAL = "🔴"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class Alert:
    """System alert."""
    timestamp: datetime
    severity: AlertSeverity
    source: str
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class GhostSignal:
    """Signal that was close but didn't meet threshold."""
    timestamp: datetime
    symbol: str
    confidence: float
    threshold: float
    reason: str


@dataclass
class QuantitativeReport:
    """4-hour quantitative summary."""
    period_start: datetime
    period_end: datetime
    trades_executed: int
    trades_ignored: int
    pnl_accumulated: float
    health_status: HealthStatus
    ghost_signals: int
    avg_latency_ms: float
    margin_ratio: float


# =============================================================================
# GHOST SIGNAL AUDITOR
# =============================================================================

class GhostSignalAuditor:
    """
    👻 Audits signals that almost triggered but didn't meet threshold.
    
    PROFESSOR METHOD:
    - QUÉ: Rastrea señales con confianza > 0 pero < threshold (0.60).
    - POR QUÉ: Identifica oportunidades perdidas por umbral actual.
    - CÓMO: Intercepta señales y las registra para análisis.
    """
    
    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold
        self.ghost_signals: List[GhostSignal] = []
        self.logger = logging.getLogger('sentinel.ghost')
    
    def audit_signal(
        self, 
        symbol: str, 
        confidence: float, 
        signal_data: Dict = None
    ) -> Optional[GhostSignal]:
        """
        Audit a signal and record if it's a ghost.
        
        Returns GhostSignal if this was a near-miss, None otherwise.
        """
        if confidence > 0.0 and confidence < self.threshold:
            ghost = GhostSignal(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                confidence=confidence,
                threshold=self.threshold,
                reason=f"Confidence {confidence:.2%} < threshold {self.threshold:.2%}"
            )
            self.ghost_signals.append(ghost)
            self.logger.info(
                f"👻 Ghost signal: {symbol} @ {confidence:.2%} "
                f"(needed {self.threshold:.2%})"
            )
            return ghost
        return None
    
    def get_summary(self) -> Dict:
        """Get summary of ghost signals."""
        if not self.ghost_signals:
            return {
                'total': 0,
                'message': "No ghost signals recorded"
            }
        
        by_symbol = {}
        for gs in self.ghost_signals:
            by_symbol[gs.symbol] = by_symbol.get(gs.symbol, 0) + 1
        
        avg_conf = sum(gs.confidence for gs in self.ghost_signals) / len(self.ghost_signals)
        
        return {
            'total': len(self.ghost_signals),
            'by_symbol': by_symbol,
            'average_confidence': round(avg_conf, 4),
            'message': f"Se perdieron {len(self.ghost_signals)} oportunidades por el umbral actual de {self.threshold:.0%}"
        }
    
    def audit_from_log(self, log_line: str) -> Optional[GhostSignal]:
        """
        Audit a signal from a log line.
        Look for: Verdict -> Direction: ... | Final Conf: 0.XX
        """
        if "Final Conf:" in log_line:
            match = re.search(r"Final Conf: (0\.\d+)", log_line)
            if match:
                conf = float(match.group(1))
                # Extraer símbolo si está en la línea
                sym_match = re.search(r"Verifying ([A-Z/]+)", log_line)
                symbol = sym_match.group(1) if sym_match else "UNKNOWN"
                return self.audit_signal(symbol, conf)
        return None

# =============================================================================
# ZOMBIE SYMBOL AUDITOR
# =============================================================================

class ZombieSymbolAuditor:
    """
    🧟 Detects 'Zombie' symbols with flat price action or stale data.
    
    PROFESSOR METHOD:
    - QUÉ: Monitor de inactividad técnica por símbolo.
    - POR QUÉ: Detecta si un par está estancado (flat) y bloquea entrenamiento inútil.
    - CÓMO: Analiza la varianza del precio y la frecuencia de actualización.
    """
    
    def __init__(self, spread_threshold: float = 0.0001, stale_threshold: int = 300):
        self.spread_threshold = spread_threshold
        self.stale_threshold = stale_threshold
        self.zombies: Dict[str, Dict] = {}
        self.logger = logging.getLogger('sentinel.zombie')
    
    def audit_symbol(self, symbol: str, df_or_price_data: any):
        """Audit a symbol for zombie behavior."""
        import pandas as pd
        if isinstance(df_or_price_data, pd.DataFrame):
            if len(df_or_price_data) < 2: return
            
            # Check price spread (volatility)
            high_max = df_or_price_data['high'].max()
            low_min = df_or_price_data['low'].min()
            avg_price = df_or_price_data['close'].mean()
            
            spread = (high_max - low_min) / avg_price if avg_price > 0 else 0
            
            if spread < self.spread_threshold:
                if symbol not in self.zombies:
                    self.logger.warning(f"🧟 [ZOMBIE DETECTED] {symbol} is flat (Spread: {spread*100:.6f}%).")
                self.zombies[symbol] = {
                    'type': 'FLAT_MARKET',
                    'last_detected': datetime.now(timezone.utc),
                    'spread': spread
                }
            elif symbol in self.zombies and self.zombies[symbol]['type'] == 'FLAT_MARKET':
                self.logger.info(f"🧬 [ZOMBIE REVIVED] {symbol} shows price action again.")
                del self.zombies[symbol]
    
    def get_summary(self) -> Dict:
        return {
            'total': len(self.zombies),
            'symbols': list(self.zombies.keys()),
            'message': f"Detectados {len(self.zombies)} símbolos 'zombis' (sin movimiento)." if self.zombies else "No hay símbolos zombis."
        }
    
    def audit_from_log(self, log_line: str):
        """Parse zombie alerts from log lines."""
        if "[ZOMBIE MARKET]" in log_line:
            # 🚫 [ZOMBIE MARKET] BTC/USDT is flat. Spread: 0.000000%
            match = re.search(r"\[ZOMBIE MARKET\] ([A-Z/]+) is flat", log_line)
            if match:
                symbol = match.group(1)
                if symbol not in self.zombies:
                    self.zombies[symbol] = {
                        'type': 'FLAT_MARKET',
                        'last_detected': datetime.now(timezone.utc)
                    }
                    self.logger.warning(f"🧟 Sentinel confirmed Zombie: {symbol}")


# =============================================================================
# RESOURCE VIGILANCE
# =============================================================================

class ResourceVigilance:
    """
    RESOURCE VIGILANCE: Auditoría profunda de recursos del proyecto.
    Desglosa RAM por componente (Main, Supervisor, Oracle) y Disco por categoría.
    """
    
    def __init__(self, ram_threshold_pct: float = 85.0, disk_threshold_pct: float = 90.0):
        self.ram_threshold = ram_threshold_pct
        self.disk_threshold = disk_threshold_pct
        self.history: List[Dict] = []
        self.logger = logging.getLogger('sentinel.resources')
        self.current_dir = os.getcwd().lower()

    def get_project_metrics(self) -> Dict:
        """
        Realiza un escaneo profundo de los procesos del proyecto y uso de disco.
        """
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_ram_pct': psutil.virtual_memory().percent,
            'system_cpu_pct': psutil.cpu_percent(interval=None),
            'project_ram_mb': 0.0,
            'process_breakdown': {
                'Main': {'mb': 0.0, 'cpu': 0.0, 'pids': []},
                'Supervisor': {'mb': 0.0, 'cpu': 0.0, 'pids': []},
                'Oracle': {'mb': 0.0, 'cpu': 0.0, 'pids': []}
            },
            'disk_breakdown': {}
        }

        # 1. RAM Breakdown by Process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent', 'cwd']):
            try:
                info = proc.info
                cmd = info.get('cmdline')
                if not cmd: continue
                
                cmd_str = " ".join(cmd).lower()
                # Verificar que pertenezca al directorio del proyecto
                proc_cwd = (info.get('cwd') or '').lower()
                if self.current_dir not in proc_cwd and self.current_dir not in cmd_str:
                    continue

                # Identificar Rol
                role = None
                if "main.py" in cmd_str: role = 'Main'
                elif "supervisor_24h.py" in cmd_str: role = 'Supervisor'
                elif "check_oracle.py" in cmd_str: role = 'Oracle'
                
                if role:
                    mb = info['memory_info'].rss / (1024 * 1024)
                    cpu = info['cpu_percent']
                    
                    metrics['process_breakdown'][role]['mb'] += mb
                    metrics['process_breakdown'][role]['cpu'] += cpu
                    metrics['process_breakdown'][role]['pids'].append(info['pid'])
                    metrics['project_ram_mb'] += mb

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 2. Disk Breakdown
        paths_to_audit = {
            'Logs': Path("logs"),
            'Data': Path("dashboard/data"),
            'Models': Path("models")
        }
        
        for name, path in paths_to_audit.items():
            if path.exists():
                size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024*1024)
                metrics['disk_breakdown'][name] = round(size_mb, 2)

        # 3. Overall Disk
        usage = shutil.disk_usage(".")
        metrics['total_disk_pct'] = round((usage.used / usage.total) * 100, 2)

        self.history.append(metrics)
        return metrics

    def check_resources(self) -> Dict:
        """Compatibility method for existing supervisor calls."""
        metrics = self.get_project_metrics()
        
        if metrics['system_ram_pct'] > self.ram_threshold:
            self.logger.warning(f"🚨 SYSTEM RAM CRITICAL: {metrics['system_ram_pct']}%")
        
        return {
            'ram_pct': metrics['system_ram_pct'],
            'disk_pct': metrics['total_disk_pct'],
            'project_mb': metrics['project_ram_mb'],
            'details': metrics # Full report attached
        }


# =============================================================================
# PROCESS MONITOR
# =============================================================================

class ProcessMonitor:
    """
    🔍 Monitors the main bot process and its children.
    """
    
    def __init__(self, target_script: str = "main.py"):
        self.target_script = target_script
        self.bot_process: Optional[psutil.Process] = None
        self.logger = logging.getLogger('sentinel.process')
    
    def find_bot_process(self) -> bool:
        """Find the bot process by command line and directory."""
        self.logger.debug(f"🔍 Searching for bot process: {self.target_script}")
        current_dir = os.getcwd().lower()
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
            try:
                info = proc.info
                cmd = info.get('cmdline')
                # Safety for None cwd (Fixing detection bug)
                proc_cwd = (info.get('cwd') or '').lower()
                
                if cmd:
                    # Specific check: script name in args AND (cwd matches OR path in args matches)
                    cmd_str = " ".join(cmd).lower()
                    if self.target_script.lower() in cmd_str:
                        # Safety 1: Not the supervisor itself
                        if "supervisor_24h.py" in cmd_str:
                            continue
                            
                        # Safety 2: Check Directory (Crucial for the user's request)
                        match_dir = False
                        
                        # Case A: Exact CWD match (most reliable)
                        if proc_cwd and proc_cwd == current_dir:
                            match_dir = True
                        # Case B: Current dir inside CWD (subfolders)
                        elif proc_cwd and current_dir in proc_cwd:
                            match_dir = True
                        # Case C: Current dir in the command line itself (e.g. running absolute path)
                        elif current_dir in cmd_str:
                            match_dir = True
                        # Case D: Signature Match (Fallback for Bat files/Access Denied)
                        # If args contain unique project flags, trust it.
                        elif "--mode" in cmd_str and "futures" in cmd_str:
                            match_dir = True

                    # LAST RESORT: Python process in exact CWD without specific script in args
                    # (Happens on some Windows environments where args are hidden/swallowed)
                    elif "python" in info['name'].lower() and proc_cwd == current_dir:
                        # Exclude known siblings
                        if "supervisor_24h.py" not in cmd_str and "check_oracle.py" not in cmd_str:
                             match_dir = True
                            
                    if match_dir:
                        self.bot_process = proc
                        self.logger.info(f"✅ Found and attached to bot process: PID {info['pid']}")
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception as e:
                self.logger.debug(f"Error checking process: {e}")
                continue
        
        self.bot_process = None
        return False
    
    def is_alive(self) -> bool:
        """Check if bot process is still running and not zombie."""
        if not self.bot_process:
            return self.find_bot_process()
        
        try:
            return self.bot_process.is_running() and self.bot_process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False


# =============================================================================
# LATENCY MONITOR
# =============================================================================

class LatencyMonitor:
    """
    ⏱️ Monitors pipeline latency between strategy decision and Dashboard display.
    
    Target: < 500ms
    """
    
    def __init__(self, target_ms: float = 500.0):
        self.target_ms = target_ms
        self.latency_samples: List[float] = []
        self.breaches: List[Dict] = []
        self.logger = logging.getLogger('sentinel.latency')
    
    def record_latency(
        self, 
        strategy_timestamp: datetime, 
        dashboard_timestamp: datetime,
        trade_id: str = None
    ) -> bool:
        """
        Record latency between strategy and dashboard.
        
        Returns True if within target, False if breached.
        """
        latency_ms = (dashboard_timestamp - strategy_timestamp).total_seconds() * 1000
        self.latency_samples.append(latency_ms)
        
        if latency_ms > self.target_ms:
            breach = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'latency_ms': round(latency_ms, 2),
                'target_ms': self.target_ms,
                'trade_id': trade_id
            }
            self.breaches.append(breach)
            self.logger.warning(
                f"⚠️ Latency breach: {latency_ms:.2f}ms > {self.target_ms}ms"
            )
            return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get latency statistics."""
        if not self.latency_samples:
            return {'message': 'No latency samples recorded'}
        
        import statistics
        return {
            'count': len(self.latency_samples),
            'avg_ms': round(statistics.mean(self.latency_samples), 2),
            'min_ms': round(min(self.latency_samples), 2),
            'max_ms': round(max(self.latency_samples), 2),
            'std_ms': round(statistics.stdev(self.latency_samples), 2) if len(self.latency_samples) > 1 else 0,
            'breaches': len(self.breaches),
            'within_target_pct': round((1 - len(self.breaches) / len(self.latency_samples)) * 100, 2)
        }
    
    def clear(self):
        """Clear samples."""
        self.latency_samples = []
        self.breaches = []


# =============================================================================
# MARGIN VIGILANCE
# =============================================================================

class MarginVigilance:
    """
    📊 Monitors margin ratio for unexpected deviations.
    
    Base margin: 21.52%
    Alert if significant change without open trades.
    """
    
    def __init__(self, base_margin: float = 21.52, deviation_threshold: float = 2.0):
        self.base_margin = base_margin
        self.deviation_threshold = deviation_threshold
        self.readings: List[Dict] = []
        self.alerts: List[Alert] = []
        self.logger = logging.getLogger('sentinel.margin')
    
    def record_margin(
        self, 
        current_margin: float, 
        has_open_positions: bool
    ) -> Optional[Alert]:
        """
        Record margin reading and check for anomalies.
        
        Returns Alert if deviation detected without positions.
        """
        reading = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'margin': current_margin,
            'has_positions': has_open_positions,
            'deviation': current_margin - self.base_margin
        }
        self.readings.append(reading)
        
        deviation = abs(current_margin - self.base_margin)
        
        # MODO PROFESOR: Solo alertamos si el margen es NO-CERO pero no hay posiciones
        # (Indica posiciones fantasma o error de API) o si hay una desviación extrema.
        if not has_open_positions and current_margin > 0 and deviation > self.deviation_threshold:
            alert = Alert(
                timestamp=datetime.now(timezone.utc),
                severity=AlertSeverity.WARNING,
                source='MarginVigilance',
                message=f"Margin deviation {deviation:.2f}% without open positions",
                details={
                    'current': current_margin,
                    'base': self.base_margin,
                    'deviation': deviation
                }
            )
            self.alerts.append(alert)
            self.logger.warning(alert.message)
            return alert
        
        return None
    
    def get_status(self) -> Dict:
        """Get margin monitoring status."""
        if not self.readings:
            return {'message': 'No margin readings'}
        
        latest = self.readings[-1]
        return {
            'current_margin': latest['margin'],
            'base_margin': self.base_margin,
            'deviation': latest['deviation'],
            'readings_count': len(self.readings),
            'alerts_count': len(self.alerts)
        }


# =============================================================================
# EQUITY DRIFT DETECTOR
# =============================================================================

class EquityDriftDetector:
    """
    💰 Detects "equity drift" - unexpected changes in totalMarginBalance.
    
    Checks every 15 minutes for consistency.
    """
    
    def __init__(self, check_interval_minutes: int = 15, drift_threshold_pct: float = 0.5):
        self.check_interval = timedelta(minutes=check_interval_minutes)
        self.drift_threshold = drift_threshold_pct
        self.readings: List[Dict] = []
        self.last_check: Optional[datetime] = None
        self.drifts_detected: List[Dict] = []
        self.logger = logging.getLogger('sentinel.equity')
    
    def record_equity(self, equity: float, expected_equity: Optional[float] = None) -> Optional[Dict]:
        """
        Record equity reading and detect drift.
        
        Args:
            equity: Current totalMarginBalance
            expected_equity: Expected value based on calculations
            
        Returns drift info if detected.
        """
        now = datetime.now(timezone.utc)
        
        reading = {
            'timestamp': now.isoformat(),
            'equity': equity,
            'expected': expected_equity
        }
        self.readings.append(reading)
        
        # Check for drift
        drift = None
        if expected_equity is not None and expected_equity > 0:
            drift_pct = abs((equity - expected_equity) / expected_equity) * 100
            
            if drift_pct > self.drift_threshold:
                drift = {
                    'timestamp': now.isoformat(),
                    'actual': equity,
                    'expected': expected_equity,
                    'drift_pct': round(drift_pct, 4),
                    'message': f"Equity drift detected: {drift_pct:.2f}% deviation"
                }
                self.drifts_detected.append(drift)
                self.logger.warning(drift['message'])
        
        self.last_check = now
        return drift
    
    def get_status(self) -> Dict:
        """Get equity drift status."""
        return {
            'readings_count': len(self.readings),
            'drifts_detected': len(self.drifts_detected),
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'drift_threshold_pct': self.drift_threshold
        }


# =============================================================================
# STRATEGY LEADERBOARD
# =============================================================================

class StrategyLeaderboard:
    """
    🏆 Tracks strategy performance and momentum.
    
    Identifies winning strategies and win rate stability.
    """
    
    def __init__(self):
        self.strategy_stats: Dict[str, Dict] = {}
        self.logger = logging.getLogger('sentinel.leaderboard')
    
    def record_trade(
        self, 
        strategy_id: str, 
        is_win: bool, 
        pnl: float,
        trade_time: datetime = None
    ):
        """Record a trade result for a strategy."""
        if strategy_id not in self.strategy_stats:
            self.strategy_stats[strategy_id] = {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'pnl_history': [],
                'win_rates': []  # Track win rate over time for stability
            }
        
        stats = self.strategy_stats[strategy_id]
        stats['total_trades'] += 1
        stats['total_pnl'] += pnl
        stats['pnl_history'].append({
            'time': (trade_time or datetime.now(timezone.utc)).isoformat(),
            'pnl': pnl
        })
        
        if is_win:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        # Track win rate over time (rolling)
        wr = stats['wins'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        stats['win_rates'].append(wr)
    
    def get_leaderboard(self) -> List[Dict]:
        """Get sorted leaderboard by PnL."""
        leaderboard = []
        
        for strategy_id, stats in self.strategy_stats.items():
            win_rate = stats['wins'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
            
            # Calculate win rate stability (std of recent win rates)
            recent_wr = stats['win_rates'][-20:] if len(stats['win_rates']) >= 5 else stats['win_rates']
            import statistics
            wr_stability = statistics.stdev(recent_wr) if len(recent_wr) > 1 else 0
            
            leaderboard.append({
                'strategy_id': strategy_id,
                'total_trades': stats['total_trades'],
                'win_rate': round(win_rate * 100, 2),
                'total_pnl': round(stats['total_pnl'], 4),
                'win_rate_stability': round(wr_stability * 100, 2),
                'momentum': 'GAINING' if len(stats['pnl_history']) > 5 and sum(p['pnl'] for p in stats['pnl_history'][-5:]) > 0 else 'LOSING'
            })
        
        return sorted(leaderboard, key=lambda x: x['total_pnl'], reverse=True)
    
    def get_top_strategy(self) -> Optional[Dict]:
        """Get the top performing strategy."""
        lb = self.get_leaderboard()
        return lb[0] if lb else None


# =============================================================================
# SENTINEL-24 MAIN ENGINE
# =============================================================================

class Sentinel24:
    """
    🛡️ Main Sentinel-24 Audit & Supervision Engine.
    
    Coordinates all monitoring components and generates reports.
    """
    
    def __init__(
        self,
        data_dir: str = "dashboard/data",
        logs_dir: str = "logs",
        signal_threshold: float = 0.60,
        latency_target_ms: float = 500.0,
        base_margin: float = 21.52
    ):
        self.data_dir = Path(data_dir)
        self.logs_dir = Path(logs_dir)
        
        # Initialize components
        self.ghost_auditor = GhostSignalAuditor(threshold=signal_threshold)
        self.latency_monitor = LatencyMonitor(target_ms=latency_target_ms)
        self.margin_vigilance = MarginVigilance(base_margin=base_margin)
        self.equity_detector = EquityDriftDetector()
        self.leaderboard = StrategyLeaderboard()
        self.resource_vigilance = ResourceVigilance()
        self.process_monitor = ProcessMonitor()
        self.zombie_auditor = ZombieSymbolAuditor()
        
        # System state
        self.alerts: List[Alert] = []
        self.reports: List[QuantitativeReport] = []
        self.last_report_time: Optional[datetime] = None
        self.report_interval = timedelta(hours=4)
        
        # Health tracking
        self.consecutive_errors = 0
        self.max_errors_before_critical = 5
        
        self.logger = logging.getLogger('sentinel.main')
    
    def get_health_status(self) -> HealthStatus:
        """Determine overall system health."""
        critical_conditions = [
            len(self.margin_vigilance.alerts) > 3,
            len(self.equity_detector.drifts_detected) > 2,
            len(self.latency_monitor.breaches) > 10,
            self.consecutive_errors >= self.max_errors_before_critical
        ]
        
        warning_conditions = [
            len(self.ghost_auditor.ghost_signals) > 50,
            len(self.latency_monitor.breaches) > 5,
            self.consecutive_errors >= 2
        ]
        
        if any(critical_conditions):
            return HealthStatus.CRITICAL
        elif any(warning_conditions):
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY
    
    def raise_alert(
        self, 
        severity: AlertSeverity, 
        source: str, 
        message: str,
        details: Dict = None
    ):
        """Raise a system alert."""
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            source=source,
            message=message,
            details=details or {}
        )
        self.alerts.append(alert)
        
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            self.logger.critical(f"🚨 {source}: {message}")
        elif severity == AlertSeverity.WARNING:
            self.logger.warning(f"⚠️ {source}: {message}")
        else:
            self.logger.info(f"ℹ️ {source}: {message}")
        
        return alert
    
    def handle_stale_data(self, source: str, staleness_seconds: float):
        """Handle STALE DATA condition."""
        self.raise_alert(
            AlertSeverity.CRITICAL,
            source,
            f"STALE DATA detected - no updates for {staleness_seconds:.1f}s",
            {'staleness_seconds': staleness_seconds, 'probable_cause': 'Network disconnection or data provider failure'}
        )
    
    def handle_disconnection(self, source: str, error_msg: str):
        """Handle DISCONNECTED condition."""
        self.raise_alert(
            AlertSeverity.EMERGENCY,
            source,
            f"DISCONNECTED - {error_msg}",
            {'error': error_msg, 'probable_cause': 'API connection failure or network issue'}
        )
    
    def generate_4h_report(self) -> QuantitativeReport:
        """Generate 4-hour quantitative summary report."""
        now = datetime.now(timezone.utc)
        period_start = self.last_report_time or (now - self.report_interval)
        
        # Get leaderboard for trade counts
        lb = self.leaderboard.get_leaderboard()
        total_trades = sum(s['total_trades'] for s in lb)
        total_pnl = sum(s['total_pnl'] for s in lb)
        
        # Get latency stats
        latency_stats = self.latency_monitor.get_stats()
        avg_latency = latency_stats.get('avg_ms', 0)
        
        # Get margin status
        margin_status = self.margin_vigilance.get_status()
        current_margin = margin_status.get('current_margin', 0)
        
        report = QuantitativeReport(
            period_start=period_start,
            period_end=now,
            trades_executed=total_trades,
            trades_ignored=len(self.ghost_auditor.ghost_signals),
            pnl_accumulated=total_pnl,
            health_status=self.get_health_status(),
            ghost_signals=len(self.ghost_auditor.ghost_signals),
            avg_latency_ms=avg_latency,
            margin_ratio=current_margin
        )
        
        self.reports.append(report)
        self.last_report_time = now
        
        self.logger.info(self._format_report(report))
        
        return report
    
    def _format_report(self, report: QuantitativeReport) -> str:
        """Format report for display."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    SENTINEL-24 REPORT                        ║
╠══════════════════════════════════════════════════════════════╣
║ Period: {report.period_start.strftime('%H:%M')} - {report.period_end.strftime('%H:%M')} UTC
║ Health: {report.health_status.value}
╠══════════════════════════════════════════════════════════════╣
║ Trades Executed:  {report.trades_executed:>6}
║ Trades Ignored:   {report.trades_ignored:>6}  (ghost signals)
║ PnL Accumulated:  ${report.pnl_accumulated:>10.4f}
╠══════════════════════════════════════════════════════════════╣
║ Avg Latency:      {report.avg_latency_ms:>6.2f}ms
║ Margin Ratio:     {report.margin_ratio:>6.2f}%
╚══════════════════════════════════════════════════════════════╝
"""
    
    def get_full_status(self) -> Dict:
        """Get complete Sentinel status."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'health_status': self.get_health_status().value,
            'ghost_signals': self.ghost_auditor.get_summary(),
            'latency': self.latency_monitor.get_stats(),
            'margin': self.margin_vigilance.get_status(),
            'equity': self.equity_detector.get_status(),
            'leaderboard': self.leaderboard.get_leaderboard(),
            'alerts_count': len(self.alerts),
            'reports_generated': len(self.reports)
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_sentinel_instance: Optional[Sentinel24] = None

def get_sentinel() -> Sentinel24:
    """Get or create global Sentinel-24 instance."""
    global _sentinel_instance
    if _sentinel_instance is None:
        _sentinel_instance = Sentinel24()
    return _sentinel_instance

def reset_sentinel():
    """Reset Sentinel instance (for testing)."""
    global _sentinel_instance
    _sentinel_instance = None


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Quick demo
    sentinel = Sentinel24()
    
    # Simulate some activity
    sentinel.ghost_auditor.audit_signal('XRP/USDT', 0.45)
    sentinel.ghost_auditor.audit_signal('DOGE/USDT', 0.55)
    
    sentinel.leaderboard.record_trade('TECHNICAL', True, 1.5)
    sentinel.leaderboard.record_trade('TECHNICAL', False, -0.5)
    sentinel.leaderboard.record_trade('ML_STRATEGY', True, 2.0)
    
    sentinel.margin_vigilance.record_margin(21.52, False)
    sentinel.equity_detector.record_equity(5000.0, 5000.0)
    
    # Generate report
    report = sentinel.generate_4h_report()
    
    print(sentinel.get_full_status())
