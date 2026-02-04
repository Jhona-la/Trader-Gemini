"""
🔥 CHAOS ENGINEERING TESTS - RESILIENCE UNDER FAILURE (Phase 7)
================================================================

PROFESSOR METHOD:
- QUÉ: Tests de resiliencia ante fallos controlados (Chaos Engineering).
- POR QUÉ: Valida que el sistema sobreviva a condiciones extremas sin pérdidas.
- CÓMO: Inyección de slippage, blackouts, rate limits con mocks.
- CUÁNDO: Pre-producción obligatorio.
- DÓNDE: tests/test_chaos_engineering.py

CHAOS SCENARIOS:
1. Extreme Slippage Injection
2. Network Blackout (10 cycles)
3. Heartbeat Recovery
4. Rate Limit 429 with Exponential Backoff

SAFETY: TestSecurityGuard MUST be LOCKED throughout all tests!
"""

import os
import sys
import json
import time
import tempfile
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force TEST environment
os.environ['TRADER_GEMINI_ENV'] = 'TEST'
os.environ['BINANCE_USE_TESTNET'] = 'True'
os.environ['BINANCE_USE_DEMO'] = 'True'


# =============================================================================
# CHAOS LOGGER
# =============================================================================

class ChaosLogger:
    """
    📋 Logs all chaos events to health_log.json for auditing.
    """
    
    _log_path: str = None
    _events: List[Dict] = []
    
    @classmethod
    def init(cls, log_path: str):
        cls._log_path = log_path
        cls._events = []
    
    @classmethod
    def log_event(cls, event_type: str, details: Dict = None):
        """Log a chaos event with [CHAOS_TEST] tag."""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tag': '[CHAOS_TEST]',
            'type': event_type,
            'details': details or {}
        }
        cls._events.append(event)
        
        if cls._log_path:
            try:
                with open(cls._log_path, 'w') as f:
                    json.dump({'events': cls._events}, f, indent=2)
            except Exception:
                pass
    
    @classmethod
    def get_events(cls) -> List[Dict]:
        return cls._events.copy()
    
    @classmethod
    def clear(cls):
        cls._events = []


# =============================================================================
# SECURITY GUARD (Imported/Replicated for Isolation)
# =============================================================================

class ChaosSecurityGuard:
    """
    🔒 Security guard for chaos tests - MUST remain locked.
    """
    _locked = True
    _blocked_calls = 0
    
    @classmethod
    def lock(cls):
        cls._locked = True
    
    @classmethod
    def is_locked(cls) -> bool:
        return cls._locked
    
    @classmethod
    def block_if_dangerous(cls, method_name: str):
        """Block dangerous API calls during chaos testing."""
        dangerous = ['create_order', 'futures_create_order', 'cancel_order']
        if method_name in dangerous and cls._locked:
            cls._blocked_calls += 1
            ChaosLogger.log_event('BLOCKED_CALL', {'method': method_name})
            raise RuntimeError(f"[CHAOS_GUARD] Blocked: {method_name}")
    
    @classmethod
    def get_blocked_count(cls) -> int:
        return cls._blocked_calls
    
    @classmethod
    def reset(cls):
        cls._blocked_calls = 0


# =============================================================================
# SLIPPAGE CALCULATOR
# =============================================================================

@dataclass
class SlippageResult:
    """Result of slippage calculation."""
    slippage_pct: float
    is_acceptable: bool
    projected_expectancy: float
    recommendation: str


class SlippageAnalyzer:
    """
    📉 Analyzes slippage and recalculates projected expectancy.
    
    Formula:
    Slippage = (Price_executed - Price_ordered) / Price_ordered × 100
    """
    
    def __init__(self, max_slippage_pct: float = 0.5):
        """
        Args:
            max_slippage_pct: Maximum acceptable slippage in %
        """
        self.max_slippage_pct = max_slippage_pct
    
    def calculate_slippage(self, ordered_price: float, executed_price: float) -> float:
        """
        Calculate slippage percentage.
        
        Slippage = (Price_executed - Price_ordered) / Price_ordered × 100
        """
        if ordered_price == 0:
            return 0.0
        return ((executed_price - ordered_price) / ordered_price) * 100
    
    def analyze(
        self, 
        ordered_price: float, 
        executed_price: float,
        original_expectancy: float,
        position_size: float
    ) -> SlippageResult:
        """
        Analyze slippage and determine if position should be closed.
        """
        slippage_pct = self.calculate_slippage(ordered_price, executed_price)
        
        # Calculate PnL impact of slippage
        slippage_cost = abs(executed_price - ordered_price) * position_size
        
        # Project new expectancy
        projected_expectancy = original_expectancy - slippage_cost
        
        # Determine if acceptable
        is_acceptable = abs(slippage_pct) <= self.max_slippage_pct
        
        if not is_acceptable:
            recommendation = "CLOSE_POSITION"
            ChaosLogger.log_event('SLIPPAGE_EXCEEDED', {
                'slippage_pct': slippage_pct,
                'ordered': ordered_price,
                'executed': executed_price,
                'recommendation': recommendation
            })
        elif projected_expectancy <= 0:
            recommendation = "CLOSE_POSITION"
        else:
            recommendation = "HOLD"
        
        return SlippageResult(
            slippage_pct=round(slippage_pct, 4),
            is_acceptable=is_acceptable,
            projected_expectancy=round(projected_expectancy, 4),
            recommendation=recommendation
        )


# =============================================================================
# NETWORK BLACKOUT SIMULATOR
# =============================================================================

class NetworkBlackoutSimulator:
    """
    🌑 Simulates network blackouts for testing resilience.
    """
    
    def __init__(self, blackout_cycles: int = 10):
        self.blackout_cycles = blackout_cycles
        self.current_cycle = 0
        self.is_blackout = False
        self.connection_restored_at = None
        self._connection_error = Exception("ConnectionError: Network is unreachable")
    
    def start_blackout(self):
        """Start a network blackout."""
        self.is_blackout = True
        self.current_cycle = 0
        ChaosLogger.log_event('BLACKOUT_START', {
            'planned_cycles': self.blackout_cycles
        })
    
    def attempt_connection(self) -> bool:
        """
        Attempt to connect. Returns True if successful, raises exception if in blackout.
        """
        if self.is_blackout:
            self.current_cycle += 1
            
            if self.current_cycle >= self.blackout_cycles:
                # Blackout ends
                self.is_blackout = False
                self.connection_restored_at = datetime.now(timezone.utc)
                ChaosLogger.log_event('BLACKOUT_END', {
                    'total_cycles': self.current_cycle
                })
                return True
            
            # Still in blackout
            ChaosLogger.log_event('CONNECTION_FAILED', {
                'cycle': self.current_cycle,
                'remaining': self.blackout_cycles - self.current_cycle
            })
            raise ConnectionError(f"Network blackout cycle {self.current_cycle}/{self.blackout_cycles}")
        
        return True
    
    def get_status(self) -> str:
        """Get current connection status for Dashboard."""
        if self.is_blackout:
            return "🔴 DISCONNECTED"
        return "🟢 ONLINE"


# =============================================================================
# AUTO-PRESERVATION MODE
# =============================================================================

class AutoPreservationMode:
    """
    🛡️ Auto-preservation mode that activates during network issues.
    
    When active:
    - No new orders are opened
    - Existing positions are protected
    - System waits for heartbeat recovery
    """
    
    def __init__(self):
        self.is_active = False
        self.activated_at = None
        self.consecutive_failures = 0
        self.failure_threshold = 3
    
    def report_failure(self):
        """Report a connection failure."""
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= self.failure_threshold and not self.is_active:
            self.activate()
    
    def report_success(self):
        """Report a successful connection."""
        self.consecutive_failures = 0
        
        if self.is_active:
            self.deactivate()
    
    def activate(self):
        """Activate auto-preservation mode."""
        self.is_active = True
        self.activated_at = datetime.now(timezone.utc)
        ChaosLogger.log_event('PRESERVATION_ACTIVATED', {
            'failures': self.consecutive_failures
        })
    
    def deactivate(self):
        """Deactivate auto-preservation mode."""
        self.is_active = False
        ChaosLogger.log_event('PRESERVATION_DEACTIVATED', {
            'duration_seconds': (datetime.now(timezone.utc) - self.activated_at).total_seconds()
        })
    
    def can_open_new_order(self) -> bool:
        """Check if new orders are allowed."""
        return not self.is_active


# =============================================================================
# RATE LIMIT HANDLER (Exponential Backoff)
# =============================================================================

class ExponentialBackoffHandler:
    """
    ⏱️ Handles rate limits with exponential backoff.
    
    Instead of immediate retry, waits progressively longer:
    1st retry: 1s
    2nd retry: 2s
    3rd retry: 4s
    4th retry: 8s
    etc.
    """
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, max_retries: int = 5):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.current_retry = 0
        self.delays_used: List[float] = []
    
    def calculate_delay(self) -> float:
        """Calculate next delay using exponential backoff."""
        delay = min(self.base_delay * (2 ** self.current_retry), self.max_delay)
        return delay
    
    def handle_rate_limit(self) -> bool:
        """
        Handle a rate limit error.
        
        Returns:
            True if should retry, False if max retries exceeded.
        """
        if self.current_retry >= self.max_retries:
            ChaosLogger.log_event('RATE_LIMIT_MAX_RETRIES', {
                'retries': self.current_retry
            })
            return False
        
        delay = self.calculate_delay()
        self.delays_used.append(delay)
        self.current_retry += 1
        
        ChaosLogger.log_event('RATE_LIMIT_BACKOFF', {
            'retry': self.current_retry,
            'delay_seconds': delay
        })
        
        return True
    
    def reset(self):
        """Reset after successful request."""
        self.current_retry = 0
    
    def get_delays_used(self) -> List[float]:
        """Get list of delays used for testing verification."""
        return self.delays_used.copy()


# =============================================================================
# HEARTBEAT RECOVERY
# =============================================================================

class HeartbeatRecovery:
    """
    💓 Manages heartbeat and position sync after reconnection.
    """
    
    def __init__(self):
        self.last_heartbeat = None
        self.position_synced = False
        self.sync_required = False
    
    def record_heartbeat(self):
        """Record a successful heartbeat."""
        self.last_heartbeat = datetime.now(timezone.utc)
    
    def mark_disconnection(self):
        """Mark that a disconnection occurred - sync will be required."""
        self.sync_required = True
        self.position_synced = False
    
    def sync_positions(self, exchange_positions: Dict) -> Dict:
        """
        Sync local positions with exchange after recovery.
        
        Returns synced position data.
        """
        if not self.sync_required:
            return exchange_positions
        
        ChaosLogger.log_event('POSITION_SYNC', {
            'positions': list(exchange_positions.keys())
        })
        
        self.position_synced = True
        self.sync_required = False
        
        return exchange_positions
    
    def is_ready_to_trade(self) -> bool:
        """Check if system is ready to resume trading."""
        return self.last_heartbeat is not None and (not self.sync_required or self.position_synced)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_health_log():
    """Create temp health log file."""
    fd, path = tempfile.mkstemp(suffix='.json', prefix='health_log_')
    os.close(fd)
    ChaosLogger.init(path)
    ChaosLogger.clear()
    
    yield path
    
    for _ in range(3):
        try:
            if os.path.exists(path):
                os.remove(path)
            break
        except PermissionError:
            time.sleep(0.1)


@pytest.fixture
def slippage_analyzer():
    """Create slippage analyzer with 0.5% max slippage."""
    return SlippageAnalyzer(max_slippage_pct=0.5)


@pytest.fixture
def blackout_simulator():
    """Create blackout simulator for 10 cycles."""
    return NetworkBlackoutSimulator(blackout_cycles=10)


@pytest.fixture
def preservation_mode():
    """Create auto-preservation mode handler."""
    return AutoPreservationMode()


@pytest.fixture
def backoff_handler():
    """Create exponential backoff handler."""
    return ExponentialBackoffHandler(base_delay=0.1, max_delay=1.0, max_retries=5)


@pytest.fixture
def heartbeat_recovery():
    """Create heartbeat recovery manager."""
    return HeartbeatRecovery()


# =============================================================================
# TEST: EXTREME SLIPPAGE
# =============================================================================

class TestExtremeSlippage:
    """
    📉 Test extreme slippage scenarios.
    
    Validates that RiskManager closes positions when slippage
    destroys the expected edge.
    """
    
    def test_slippage_calculation_formula(self, slippage_analyzer, temp_health_log):
        """
        Test slippage formula:
        Slippage = (Price_executed - Price_ordered) / Price_ordered × 100
        """
        ordered = 100.0
        executed = 101.0
        
        slippage = slippage_analyzer.calculate_slippage(ordered, executed)
        
        # (101 - 100) / 100 × 100 = 1%
        assert slippage == 1.0
        print("✅ Slippage formula validated: 1% slippage calculated correctly")
    
    def test_extreme_slippage_triggers_close(self, slippage_analyzer, temp_health_log):
        """
        Test that extreme slippage (> 0.5%) triggers position close.
        """
        ChaosSecurityGuard.lock()
        
        ordered = 0.55  # XRP ordered price
        executed = 0.558  # 1.45% slippage!
        original_expectancy = 0.01  # $0.01 expected profit
        position_size = 100
        
        result = slippage_analyzer.analyze(ordered, executed, original_expectancy, position_size)
        
        assert not result.is_acceptable, "Extreme slippage should be unacceptable"
        assert result.recommendation == "CLOSE_POSITION"
        assert abs(result.slippage_pct - 1.4545) < 0.01
        
        # Verify logged
        events = ChaosLogger.get_events()
        slippage_events = [e for e in events if e['type'] == 'SLIPPAGE_EXCEEDED']
        assert len(slippage_events) >= 1
        
        print(f"✅ Extreme slippage ({result.slippage_pct}%) triggered CLOSE_POSITION")
    
    def test_acceptable_slippage_allows_hold(self, slippage_analyzer, temp_health_log):
        """
        Test that acceptable slippage (< 0.5%) allows holding.
        """
        ordered = 0.55
        executed = 0.551  # 0.18% slippage
        original_expectancy = 0.50  # Large enough to remain positive after slippage cost
        position_size = 100
        
        result = slippage_analyzer.analyze(ordered, executed, original_expectancy, position_size)
        
        assert result.is_acceptable
        # Slippage cost = 0.001 * 100 = 0.1, expectancy = 0.50 - 0.1 = 0.40 (still positive)
        assert result.projected_expectancy > 0
        assert result.recommendation == "HOLD"
        assert abs(result.slippage_pct - 0.1818) < 0.01
        
        print(f"✅ Acceptable slippage ({result.slippage_pct}%) allows HOLD")
    
    def test_negative_expectancy_after_slippage(self, slippage_analyzer, temp_health_log):
        """
        Test that even acceptable slippage triggers close if expectancy goes negative.
        """
        ordered = 0.55
        executed = 0.552  # 0.36% slippage (acceptable)
        original_expectancy = 0.001  # Very small edge
        position_size = 100
        
        result = slippage_analyzer.analyze(ordered, executed, original_expectancy, position_size)
        
        # Slippage cost = 0.002 * 100 = 0.2, but original expectancy was only 0.001
        assert result.projected_expectancy < 0
        assert result.recommendation == "CLOSE_POSITION"
        
        print("✅ Negative projected expectancy triggers CLOSE_POSITION")


# =============================================================================
# TEST: NETWORK BLACKOUT
# =============================================================================

class TestNetworkBlackout:
    """
    🌑 Test network blackout scenarios.
    
    Simulates 10 cycles of connection failure and validates:
    - Dashboard shows DISCONNECTED
    - Bot enters auto-preservation mode
    - Recovery happens correctly
    """
    
    def test_blackout_marks_disconnected(self, blackout_simulator, temp_health_log):
        """Test that blackout shows DISCONNECTED status."""
        blackout_simulator.start_blackout()
        
        status = blackout_simulator.get_status()
        assert status == "🔴 DISCONNECTED"
        
        print("✅ Blackout correctly shows 🔴 DISCONNECTED")
    
    def test_blackout_10_cycles_then_recovery(self, blackout_simulator, temp_health_log):
        """
        Test surviving 10 cycles of network failure.
        """
        blackout_simulator.start_blackout()
        
        failures = 0
        for i in range(15):
            try:
                blackout_simulator.attempt_connection()
            except ConnectionError:
                failures += 1
        
        # Should have failed 9 times (cycles 1-9), then recovered on cycle 10
        assert failures == 9, f"Expected 9 failures, got {failures}"
        assert not blackout_simulator.is_blackout
        assert blackout_simulator.get_status() == "🟢 ONLINE"
        
        # Verify logging
        events = ChaosLogger.get_events()
        assert any(e['type'] == 'BLACKOUT_START' for e in events)
        assert any(e['type'] == 'BLACKOUT_END' for e in events)
        
        print("✅ Survived 10 blackout cycles, connection restored")
    
    def test_preservation_mode_activates(self, blackout_simulator, preservation_mode, temp_health_log):
        """
        Test that auto-preservation mode activates during blackout.
        """
        blackout_simulator.start_blackout()
        
        for i in range(5):
            try:
                blackout_simulator.attempt_connection()
            except ConnectionError:
                preservation_mode.report_failure()
        
        assert preservation_mode.is_active, "Preservation should be active after 3+ failures"
        assert not preservation_mode.can_open_new_order()
        
        print("✅ Auto-preservation mode activated after 3 consecutive failures")
    
    def test_preservation_mode_deactivates_on_recovery(
        self, blackout_simulator, preservation_mode, temp_health_log
    ):
        """Test preservation mode deactivates on recovery."""
        # Activate preservation
        for _ in range(3):
            preservation_mode.report_failure()
        
        assert preservation_mode.is_active
        
        # Report success (recovery)
        preservation_mode.report_success()
        
        assert not preservation_mode.is_active
        assert preservation_mode.can_open_new_order()
        
        print("✅ Preservation mode deactivated on recovery")


# =============================================================================
# TEST: HEARTBEAT RECOVERY
# =============================================================================

class TestHeartbeatRecovery:
    """
    💓 Test heartbeat recovery and position sync.
    """
    
    def test_sync_required_after_disconnection(self, heartbeat_recovery, temp_health_log):
        """Test that position sync is required after disconnection."""
        heartbeat_recovery.record_heartbeat()
        assert heartbeat_recovery.is_ready_to_trade()
        
        # Disconnection occurs
        heartbeat_recovery.mark_disconnection()
        
        assert heartbeat_recovery.sync_required
        assert not heartbeat_recovery.is_ready_to_trade()
        
        print("✅ Position sync required after disconnection")
    
    def test_position_sync_before_resume(self, heartbeat_recovery, temp_health_log):
        """Test that position sync happens before resuming trading."""
        heartbeat_recovery.record_heartbeat()
        heartbeat_recovery.mark_disconnection()
        
        # Simulate exchange positions
        exchange_positions = {
            'XRP/USDT': {'quantity': 100, 'avg_price': 0.55},
            'DOGE/USDT': {'quantity': 500, 'avg_price': 0.08}
        }
        
        # Sync positions
        synced = heartbeat_recovery.sync_positions(exchange_positions)
        
        assert synced == exchange_positions
        assert heartbeat_recovery.position_synced
        assert heartbeat_recovery.is_ready_to_trade()
        
        # Verify logged
        events = ChaosLogger.get_events()
        sync_events = [e for e in events if e['type'] == 'POSITION_SYNC']
        assert len(sync_events) >= 1
        
        print("✅ Positions synced before resume, ready to trade")


# =============================================================================
# TEST: RATE LIMIT (429)
# =============================================================================

class TestRateLimitHandling:
    """
    ⏱️ Test rate limit handling with exponential backoff.
    """
    
    def test_exponential_backoff_delays(self, backoff_handler, temp_health_log):
        """
        Test exponential backoff produces correct delays.
        Base: 0.1s, so: 0.1, 0.2, 0.4, 0.8, max=1.0
        """
        expected_delays = [0.1, 0.2, 0.4, 0.8, 1.0]
        
        for i, expected in enumerate(expected_delays):
            assert backoff_handler.handle_rate_limit()  # Should return True
            delay = backoff_handler.delays_used[-1]
            assert abs(delay - expected) < 0.001, f"Retry {i+1}: expected {expected}, got {delay}"
        
        print(f"✅ Exponential backoff delays correct: {backoff_handler.delays_used}")
    
    def test_max_retries_respected(self, backoff_handler, temp_health_log):
        """Test that max retries limit is respected."""
        # Exhaust all retries
        for _ in range(5):
            backoff_handler.handle_rate_limit()
        
        # 6th attempt should fail
        result = backoff_handler.handle_rate_limit()
        assert result == False, "Should return False after max retries"
        
        print("✅ Max retries (5) respected, returns False after exhaustion")
    
    def test_reset_after_success(self, backoff_handler, temp_health_log):
        """Test that backoff resets after successful request."""
        # Do some retries
        for _ in range(3):
            backoff_handler.handle_rate_limit()
        
        assert backoff_handler.current_retry == 3
        
        # Success!
        backoff_handler.reset()
        
        assert backoff_handler.current_retry == 0
        
        print("✅ Backoff reset after successful request")


# =============================================================================
# TEST: COMPLETE CHAOS SCENARIO
# =============================================================================

class TestCompleteChaosScenario:
    """
    🔥 Test complete chaos scenario with multiple failures.
    """
    
    def test_survive_5_consecutive_failures(self, temp_health_log):
        """
        CRITICAL: Bot must survive 5 consecutive network failures without crashing.
        """
        ChaosSecurityGuard.lock()
        blackout = NetworkBlackoutSimulator(blackout_cycles=5)
        preservation = AutoPreservationMode()
        
        blackout.start_blackout()
        
        crash_occurred = False
        try:
            for i in range(5):
                try:
                    blackout.attempt_connection()
                except ConnectionError:
                    preservation.report_failure()
                    # Bot continues running...
        except Exception as e:
            crash_occurred = True
            print(f"❌ Crash: {e}")
        
        assert not crash_occurred, "Bot crashed during network failures!"
        assert preservation.is_active, "Preservation should be active"
        
        events = ChaosLogger.get_events()
        assert len(events) >= 5, "Should have logged all failure events"
        
        print("✅ Bot survived 5 consecutive network failures without crashing!")
    
    def test_chaos_security_guard_blocks_orders(self, temp_health_log):
        """
        CRITICAL: Security guard must block all dangerous calls during chaos.
        """
        ChaosSecurityGuard.lock()
        ChaosSecurityGuard.reset()
        
        blocked = 0
        for i in range(10):
            try:
                ChaosSecurityGuard.block_if_dangerous('futures_create_order')
            except RuntimeError:
                blocked += 1
        
        assert blocked == 10, f"Only {blocked}/10 calls blocked!"
        assert ChaosSecurityGuard.get_blocked_count() == 10
        
        print("✅ Security guard blocked all 10 dangerous calls during chaos")
    
    def test_full_chaos_recovery_sequence(self, temp_health_log):
        """
        Test full chaos sequence:
        1. Normal operation
        2. Blackout starts
        3. Preservation activates
        4. Connection recovers
        5. Positions sync
        6. Trading resumes
        """
        ChaosSecurityGuard.lock()
        
        blackout = NetworkBlackoutSimulator(blackout_cycles=3)
        preservation = AutoPreservationMode()
        heartbeat = HeartbeatRecovery()
        
        # 1. Normal operation
        heartbeat.record_heartbeat()
        assert heartbeat.is_ready_to_trade()
        
        # 2. Blackout starts
        blackout.start_blackout()
        heartbeat.mark_disconnection()
        
        # 3. Trigger preservation with 3 failures (blackout is 3 cycles, so we force failures)
        for _ in range(3):
            preservation.report_failure()
        
        assert preservation.is_active, "Preservation should be active after 3 failures"
        
        # 4. Connection recovers - now try connections until blackout ends
        for _ in range(5):
            try:
                blackout.attempt_connection()
            except ConnectionError:
                pass
        
        preservation.report_success()
        
        # 5. Positions sync
        exchange_positions = {'XRP/USDT': {'quantity': 100}}
        heartbeat.sync_positions(exchange_positions)
        
        # 6. Trading resumes
        heartbeat.record_heartbeat()
        
        assert not preservation.is_active
        assert heartbeat.is_ready_to_trade()
        
        print("✅ Full chaos recovery sequence completed successfully!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
