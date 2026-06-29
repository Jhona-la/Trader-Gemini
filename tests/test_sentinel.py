"""
🛡️ SENTINEL-24 TESTS - AUDIT SYSTEM VALIDATION
================================================

PROFESSOR METHOD:
- QUÉ: Tests del sistema Sentinel-24 de auditoría.
- POR QUÉ: Valida que el monitoreo 24h funcione correctamente.
- CÓMO: Tests unitarios de cada componente del Sentinel.
- CUÁNDO: Pre-producción.

TESTS:
- Ghost Signal Auditor
- Latency Monitor
- Margin Vigilance
- Equity Drift Detector
- Strategy Leaderboard
- Full Report Generation
"""

import os
import sys
import pytest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TRADER_GEMINI_ENV'] = 'TEST'


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def ghost_auditor():
    """Create ghost signal auditor with 0.60 threshold."""
    from utils.sentinel import GhostSignalAuditor
    return GhostSignalAuditor(threshold=0.60)


@pytest.fixture
def latency_monitor():
    """Create latency monitor with 500ms target."""
    from utils.sentinel import LatencyMonitor
    return LatencyMonitor(target_ms=500.0)


@pytest.fixture
def margin_vigilance():
    """Create margin vigilance with 21.52% base."""
    from utils.sentinel import MarginVigilance
    return MarginVigilance(base_margin=21.52, deviation_threshold=2.0)


@pytest.fixture
def equity_detector():
    """Create equity drift detector."""
    from utils.sentinel import EquityDriftDetector
    return EquityDriftDetector(check_interval_minutes=15, drift_threshold_pct=0.5)


@pytest.fixture
def leaderboard():
    """Create strategy leaderboard."""
    from utils.sentinel import StrategyLeaderboard
    return StrategyLeaderboard()


@pytest.fixture
def sentinel():
    """Create full Sentinel-24 instance."""
    from utils.sentinel import Sentinel24, reset_sentinel
    reset_sentinel()
    return Sentinel24()


# =============================================================================
# TEST: GHOST SIGNAL AUDITOR
# =============================================================================

class TestGhostSignalAuditor:
    """
    👻 Test Ghost Signal Auditor.
    
    Validates tracking of near-miss signals.
    """
    
    def test_ghost_signal_detected(self, ghost_auditor):
        """Test that signals < threshold are recorded as ghosts."""
        ghost = ghost_auditor.audit_signal('XRP/USDT', 0.45)
        
        assert ghost is not None
        assert ghost.symbol == 'XRP/USDT'
        assert ghost.confidence == 0.45
        assert ghost.threshold == 0.60
        
        print("✅ Ghost signal detected: 0.45 < 0.60 threshold")
    
    def test_passing_signal_not_ghost(self, ghost_auditor):
        """Test that signals >= threshold are NOT ghosts."""
        ghost = ghost_auditor.audit_signal('XRP/USDT', 0.75)
        
        assert ghost is None
        assert len(ghost_auditor.ghost_signals) == 0
        
        print("✅ Passing signal (0.75) correctly not recorded as ghost")
    
    def test_zero_confidence_not_ghost(self, ghost_auditor):
        """Test that zero confidence signals are NOT ghosts."""
        ghost = ghost_auditor.audit_signal('XRP/USDT', 0.0)
        
        assert ghost is None
        
        print("✅ Zero confidence signal not recorded as ghost")
    
    def test_ghost_summary_message(self, ghost_auditor):
        """Test summary message format."""
        # Record several ghosts
        ghost_auditor.audit_signal('XRP/USDT', 0.45)
        ghost_auditor.audit_signal('DOGE/USDT', 0.55)
        ghost_auditor.audit_signal('XRP/USDT', 0.30)
        
        summary = ghost_auditor.get_summary()
        
        assert summary['total'] == 3
        assert 'Se perdieron 3 oportunidades' in summary['message']
        assert summary['by_symbol']['XRP/USDT'] == 2
        assert summary['by_symbol']['DOGE/USDT'] == 1
        
        print(f"✅ Ghost summary: {summary['message']}")


# =============================================================================
# TEST: LATENCY MONITOR
# =============================================================================

class TestLatencyMonitor:
    """
    ⏱️ Test Latency Monitor.
    
    Validates pipeline latency tracking < 500ms.
    """
    
    def test_latency_within_target(self, latency_monitor):
        """Test that latency within target returns True."""
        strategy_time = datetime.now(timezone.utc)
        dashboard_time = strategy_time + timedelta(milliseconds=200)
        
        result = latency_monitor.record_latency(strategy_time, dashboard_time)
        
        assert result == True
        assert len(latency_monitor.breaches) == 0
        
        print("✅ Latency 200ms within 500ms target")
    
    def test_latency_breach_detected(self, latency_monitor):
        """Test that latency > 500ms triggers breach."""
        strategy_time = datetime.now(timezone.utc)
        dashboard_time = strategy_time + timedelta(milliseconds=750)
        
        result = latency_monitor.record_latency(strategy_time, dashboard_time, 'TRADE_001')
        
        assert result == False
        assert len(latency_monitor.breaches) == 1
        assert latency_monitor.breaches[0]['trade_id'] == 'TRADE_001'
        
        print("✅ Latency breach detected: 750ms > 500ms")
    
    def test_latency_stats(self, latency_monitor):
        """Test latency statistics calculation."""
        base = datetime.now(timezone.utc)
        
        # Record various latencies
        for ms in [100, 200, 300, 400, 500, 600]:
            latency_monitor.record_latency(base, base + timedelta(milliseconds=ms))
        
        stats = latency_monitor.get_stats()
        
        assert stats['count'] == 6
        assert 300 < stats['avg_ms'] < 400  # Average should be ~350ms
        assert stats['min_ms'] == 100
        assert stats['max_ms'] == 600
        assert stats['breaches'] == 1  # Only 600ms breached
        
        print(f"✅ Latency stats: avg={stats['avg_ms']}ms, breaches={stats['breaches']}")


# =============================================================================
# TEST: MARGIN VIGILANCE
# =============================================================================

class TestMarginVigilance:
    """
    📊 Test Margin Vigilance.
    
    Validates margin ratio monitoring and alerts.
    """
    
    def test_normal_margin_no_alert(self, margin_vigilance):
        """Test that normal margin doesn't trigger alert."""
        alert = margin_vigilance.record_margin(21.50, False)
        
        assert alert is None
        
        print("✅ Normal margin (21.50%) no alert")
    
    def test_deviation_without_positions_alerts(self, margin_vigilance):
        """Test that deviation without positions triggers alert."""
        alert = margin_vigilance.record_margin(25.0, False)  # 3.48% deviation
        
        assert alert is not None
        assert alert.severity.value == 'WARNING'
        assert 'deviation' in alert.message.lower()
        
        print("✅ Margin deviation 25% without positions triggered alert")
    
    def test_deviation_with_positions_ok(self, margin_vigilance):
        """Test that deviation WITH positions is acceptable."""
        alert = margin_vigilance.record_margin(25.0, True)  # Has positions
        
        assert alert is None  # No alert when positions explain the deviation
        
        print("✅ Margin deviation with open positions - no alert")


# =============================================================================
# TEST: EQUITY DRIFT DETECTOR
# =============================================================================

class TestEquityDriftDetector:
    """
    💰 Test Equity Drift Detector.
    
    Validates 15-minute consistency checks.
    """
    
    def test_no_drift_when_matching(self, equity_detector):
        """Test no drift when actual matches expected."""
        drift = equity_detector.record_equity(5000.0, 5000.0)
        
        assert drift is None
        
        print("✅ No drift: actual matches expected")
    
    def test_drift_detected_when_mismatch(self, equity_detector):
        """Test drift detected when significant mismatch."""
        # 1% drift (> 0.5% threshold)
        drift = equity_detector.record_equity(5050.0, 5000.0)
        
        assert drift is not None
        assert drift['drift_pct'] == 1.0
        
        print(f"✅ Drift detected: {drift['drift_pct']}%")
    
    def test_small_drift_acceptable(self, equity_detector):
        """Test that small drift (< threshold) is acceptable."""
        # 0.2% drift (< 0.5% threshold)
        drift = equity_detector.record_equity(5010.0, 5000.0)
        
        assert drift is None
        
        print("✅ Small drift 0.2% acceptable")


# =============================================================================
# TEST: STRATEGY LEADERBOARD
# =============================================================================

class TestStrategyLeaderboard:
    """
    🏆 Test Strategy Leaderboard.
    
    Validates strategy performance tracking.
    """
    
    def test_trade_recording(self, leaderboard):
        """Test basic trade recording."""
        leaderboard.record_trade('TECHNICAL', True, 1.5)
        leaderboard.record_trade('TECHNICAL', False, -0.5)
        
        lb = leaderboard.get_leaderboard()
        
        assert len(lb) == 1
        assert lb[0]['strategy_id'] == 'TECHNICAL'
        assert lb[0]['total_trades'] == 2
        assert lb[0]['win_rate'] == 50.0
        assert lb[0]['total_pnl'] == 1.0
        
        print("✅ Trade recording and stats working")
    
    def test_leaderboard_sorting(self, leaderboard):
        """Test leaderboard sorted by PnL."""
        leaderboard.record_trade('STRATEGY_A', True, 2.0)
        leaderboard.record_trade('STRATEGY_B', True, 5.0)
        leaderboard.record_trade('STRATEGY_C', True, 1.0)
        
        lb = leaderboard.get_leaderboard()
        
        assert lb[0]['strategy_id'] == 'STRATEGY_B'  # Highest PnL
        assert lb[1]['strategy_id'] == 'STRATEGY_A'
        assert lb[2]['strategy_id'] == 'STRATEGY_C'
        
        print("✅ Leaderboard sorted by PnL (descending)")
    
    def test_momentum_tracking(self, leaderboard):
        """Test momentum tracking."""
        # Strategy with recent wins
        for _ in range(6):
            leaderboard.record_trade('WINNER', True, 1.0)
        
        # Strategy with recent losses
        for _ in range(6):
            leaderboard.record_trade('LOSER', False, -1.0)
        
        lb = leaderboard.get_leaderboard()
        winner = next(s for s in lb if s['strategy_id'] == 'WINNER')
        loser = next(s for s in lb if s['strategy_id'] == 'LOSER')
        
        assert winner['momentum'] == 'GAINING'
        assert loser['momentum'] == 'LOSING'
        
        print("✅ Momentum tracking: GAINING/LOSING correctly identified")


# =============================================================================
# TEST: FULL SENTINEL-24
# =============================================================================

class TestSentinel24:
    """
    🛡️ Test Full Sentinel-24 System.
    """
    
    def test_health_status_healthy(self, sentinel):
        """Test healthy status when no issues."""
        from utils.sentinel import HealthStatus
        
        status = sentinel.get_health_status()
        
        assert status == HealthStatus.HEALTHY
        
        print("✅ Fresh Sentinel status: 🟢 HEALTHY")
    
    def test_4h_report_generation(self, sentinel):
        """Test 4-hour report generation."""
        # Add some activity
        sentinel.ghost_auditor.audit_signal('XRP/USDT', 0.45)
        sentinel.leaderboard.record_trade('TECHNICAL', True, 1.5)
        sentinel.margin_vigilance.record_margin(21.52, False)
        
        report = sentinel.generate_4h_report()
        
        assert report.trades_executed == 1
        assert report.ghost_signals == 1
        assert report.health_status.value == '🟢'
        
        print("✅ 4-hour report generated successfully")
    
    def test_alert_raising(self, sentinel):
        """Test alert system."""
        from utils.sentinel import AlertSeverity
        
        alert = sentinel.raise_alert(
            AlertSeverity.WARNING,
            'TEST',
            'Test alert message'
        )
        
        assert len(sentinel.alerts) == 1
        assert sentinel.alerts[0].severity == AlertSeverity.WARNING
        
        print("✅ Alert system working")
    
    def test_stale_data_handling(self, sentinel):
        """Test STALE DATA alert generation."""
        sentinel.handle_stale_data('DataProvider', 45.0)
        
        assert len(sentinel.alerts) == 1
        assert 'STALE DATA' in sentinel.alerts[0].message
        
        print("✅ STALE DATA alert generated")
    
    def test_disconnection_handling(self, sentinel):
        """Test DISCONNECTED alert generation."""
        sentinel.handle_disconnection('WebSocket', 'Connection refused')
        
        assert len(sentinel.alerts) == 1
        assert 'DISCONNECTED' in sentinel.alerts[0].message
        
        print("✅ DISCONNECTED alert generated")
    
    def test_full_status_report(self, sentinel):
        """Test complete status report."""
        # Add various activity
        sentinel.ghost_auditor.audit_signal('XRP/USDT', 0.45)
        sentinel.leaderboard.record_trade('TECH', True, 1.0)
        sentinel.margin_vigilance.record_margin(21.52, False)
        
        status = sentinel.get_full_status()
        
        assert 'health_status' in status
        assert 'ghost_signals' in status
        assert 'latency' in status
        assert 'margin' in status
        assert 'leaderboard' in status
        
        print("✅ Full status report generated with all components")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
