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
import json
import csv
import time
import logging
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
    
    def clear(self):
        """Clear recorded ghost signals."""
        self.ghost_signals = []


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
        
        if not has_open_positions and deviation > self.deviation_threshold:
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
