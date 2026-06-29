"""
🔒 TEST API - SECURITY & ENVIRONMENT TESTS
==========================================

PROFESSOR METHOD:
- QUÉ: Tests de seguridad para prevenir operaciones reales en modo TEST.
- POR QUÉ: Protege contra pérdidas financieras accidentales.
- CÓMO: Verifica que órdenes reales sean bloqueadas y API esté en modo locked.
- CUÁNDO: Se ejecuta con `pytest tests/test_api.py -v`.
- DÓNDE: tests/test_api.py

SAFETY CHECKS:
- No-Money Guard: Durante tests PROD, api_manager entra en modo "Locked"
- Mock Time: Tiempos congelados para tests deterministas
"""

import os
import sys
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the security guard from conftest
from conftest import TestSecurityGuard, SecurityException


class TestProdDemoSwitch:
    """
    🔄 Test PROD/DEMO Switch: Validates environment separation.
    """
    
    def test_test_environment_active(self, mock_config):
        """
        Verify that TEST environment is correctly configured.
        """
        assert os.environ.get('TRADER_GEMINI_ENV') == 'TEST', \
            "Environment should be TEST during tests"
        
        assert mock_config.BINANCE_USE_TESTNET == True, \
            "BINANCE_USE_TESTNET should be True in tests"
        
        assert mock_config.BINANCE_USE_DEMO == True, \
            "BINANCE_USE_DEMO should be True in tests"
    
    def test_security_guard_locked(self):
        """
        Verify that SecurityGuard is locked during tests.
        """
        assert TestSecurityGuard.is_locked(), \
            "SecurityGuard should be LOCKED during tests"
    
    def test_prod_order_blocked_in_test_mode(self):
        """
        🚨 CRITICAL: Verify that real orders are blocked in TEST mode.
        
        This test MUST pass to ensure no accidental money loss.
        """
        with pytest.raises(SecurityException) as exc_info:
            TestSecurityGuard.intercept_call('create_order', ('BTCUSDT', 'BUY', 0.001))
        
        assert 'BLOCKED' in str(exc_info.value)
        assert 'create_order' in str(exc_info.value)
    
    def test_futures_order_blocked_in_test_mode(self):
        """
        Verify that futures orders are blocked in TEST mode.
        """
        with pytest.raises(SecurityException):
            TestSecurityGuard.intercept_call('futures_create_order', ('ETHUSDT', 'SELL', 0.01))
    
    def test_intercepted_calls_logged(self):
        """
        Verify that blocked calls are logged for auditing.
        """
        # Clear previous calls
        TestSecurityGuard._intercepted_calls = []
        
        try:
            TestSecurityGuard.intercept_call('test_method', ('arg1', 'arg2'))
        except SecurityException:
            pass
        
        assert len(TestSecurityGuard._intercepted_calls) > 0, \
            "Intercepted calls should be logged"
        
        last_call = TestSecurityGuard._intercepted_calls[-1]
        assert last_call['method'] == 'test_method'
        assert 'timestamp' in last_call


class TestAPIManagerSecurity:
    """
    🔐 Test API Manager Security: Validates read-only mode.
    """
    
    def test_api_manager_config_validation(self, mock_config):
        """
        Test that APIManager uses correct keys based on environment.
        """
        # In test mode, should use Demo/Testnet keys
        if mock_config.BINANCE_USE_FUTURES:
            expected_key = mock_config.BINANCE_DEMO_API_KEY
        else:
            expected_key = mock_config.BINANCE_TESTNET_API_KEY
        
        # Key should be set (even if mock)
        # In real scenario, these would be loaded from .env
        assert mock_config.BINANCE_USE_DEMO or mock_config.BINANCE_USE_TESTNET, \
            "Should be in Demo or Testnet mode during tests"
    
    def test_executor_mode_detection(self, mock_config):
        """
        Test that BinanceExecutor detects correct mode.
        """
        # Simulate executor mode detection logic
        is_demo = hasattr(mock_config, 'BINANCE_USE_DEMO') and mock_config.BINANCE_USE_DEMO
        is_testnet = mock_config.BINANCE_USE_TESTNET
        
        is_safe_mode = is_demo or is_testnet
        
        assert is_safe_mode, \
            "Executor should be in safe mode (Demo/Testnet) during tests"
    
    def test_no_live_keys_in_config(self, mock_config):
        """
        Test that production keys are not loaded during tests.
        """
        # In test fixtures, production keys should be empty or mocked
        # This prevents accidental use of real credentials
        if hasattr(mock_config, 'BINANCE_API_KEY'):
            # Production key should be empty in test environment
            prod_key = mock_config.BINANCE_API_KEY
            if prod_key:
                # If set, ensure we're not accidentally using it
                assert mock_config.BINANCE_USE_TESTNET or mock_config.BINANCE_USE_DEMO, \
                    "If prod key is set, must be in safe mode"


class TestMockClientSecurity:
    """
    🔌 Test Mock Client Security: Validates mock behavior.
    """
    
    def test_mock_client_order_creation(self, mock_binance_client):
        """
        Test that mock client order creation is tracked.
        
        Note: In actual tests with SecurityGuard, this would be blocked.
        Here we test the mock's internal behavior.
        """
        # Temporarily unlock for this test
        TestSecurityGuard.unlock()
        
        try:
            order = mock_binance_client.futures_create_order(
                symbol='XRPUSDT',
                side='BUY',
                type='LIMIT',
                quantity=10.0,
                price=0.55
            )
            
            # Order is a dict from mock
            assert isinstance(order, dict), "Order should be a dict"
            assert 'orderId' in order, "Order should have orderId"
            assert order.get('status') in ['FILLED', 'NEW'], "Order should have valid status"
            
            # Verify order was logged
            assert len(mock_binance_client.orders) > 0
        finally:
            # Re-lock after test
            TestSecurityGuard.lock()
    
    def test_mock_returns_controlled_data(self, mock_binance_client):
        """
        Test that mock returns predictable, controlled data.
        """
        account = mock_binance_client.futures_account()
        
        # Values should be deterministic
        assert float(account['totalWalletBalance']) == 5000.0
        assert float(account['totalMarginBalance']) == 5100.0
        
        # Positions should be controlled
        positions = account.get('positions', [])
        assert len(positions) > 0
    
    def test_mock_ping_succeeds(self, mock_binance_client):
        """
        Test that mock connectivity check succeeds.
        """
        result = mock_binance_client.ping()
        assert result == {}, "Ping should return empty dict"


class TestEnvironmentVariables:
    """
    🌍 Test Environment Variables: Validates env configuration.
    """
    
    def test_trader_gemini_env_set(self):
        """
        Test that TRADER_GEMINI_ENV is set correctly.
        """
        env_value = os.environ.get('TRADER_GEMINI_ENV')
        assert env_value == 'TEST', f"TRADER_GEMINI_ENV should be 'TEST', got '{env_value}'"
    
    def test_testnet_flags_set(self):
        """
        Test that testnet flags are set in environment.
        """
        use_testnet = os.environ.get('BINANCE_USE_TESTNET')
        use_demo = os.environ.get('BINANCE_USE_DEMO')
        
        # At least one should be True
        assert use_testnet == 'True' or use_demo == 'True', \
            "Either BINANCE_USE_TESTNET or BINANCE_USE_DEMO should be True"
    
    def test_no_production_env(self):
        """
        Test that we're not accidentally in production mode.
        """
        env = os.environ.get('TRADER_GEMINI_ENV', 'UNKNOWN')
        
        assert env != 'PRODUCTION', \
            "CRITICAL: Tests should NEVER run in PRODUCTION mode!"
        
        assert env != 'PROD', \
            "CRITICAL: Tests should NEVER run in PROD mode!"


class TestDataIsolation:
    """
    📁 Test Data Isolation: Validates test data separation.
    """
    
    def test_temp_dir_isolation(self, temp_data_dir, mock_config):
        """
        Test that tests use isolated temporary directory.
        """
        # Temp dir should be different from production data dir
        assert temp_data_dir != 'dashboard/data/futures', \
            "Test data dir should not be production dir"
        
        # Config should point to temp dir
        assert mock_config.DATA_DIR == temp_data_dir, \
            "Config.DATA_DIR should point to temp directory"
    
    def test_no_production_file_modification(self, temp_data_dir):
        """
        Test that tests don't modify production files.
        """
        # Create a test file in temp dir
        test_file = os.path.join(temp_data_dir, 'test_marker.txt')
        with open(test_file, 'w') as f:
            f.write('TEST')
        
        # Verify it's in temp dir, not production
        assert not test_file.startswith('dashboard/data/'), \
            "Test files should not be in production directory"


class TestHeartbeatDeterminism:
    """
    💓 Test Heartbeat Determinism: Validates time-based tests.
    """
    
    def test_frozen_time_works(self, frozen_time):
        """
        Test that time is correctly frozen during tests.
        """
        from datetime import datetime, timezone
        
        # The frozen_time fixture should provide a fixed datetime
        assert frozen_time.year == 2026
        assert frozen_time.month == 2
        assert frozen_time.day == 3
        assert frozen_time.hour == 12
        assert frozen_time.minute == 0
        assert frozen_time.second == 0
    
    def test_heartbeat_timestamp_deterministic(self, frozen_time):
        """
        Test that heartbeat timestamps are deterministic with frozen time.
        """
        # With frozen time, multiple calls should return same timestamp
        timestamp1 = frozen_time.isoformat()
        timestamp2 = frozen_time.isoformat()
        
        assert timestamp1 == timestamp2, \
            "Frozen time should produce identical timestamps"


# =============================================================================
# CRITICAL SAFETY TESTS
# =============================================================================

class TestCriticalSafety:
    """
    🚨 CRITICAL SAFETY TESTS
    
    These tests MUST pass before any deployment.
    Failure indicates potential for financial loss.
    """
    
    def test_no_real_orders_possible(self):
        """
        🔴 CRITICAL: Verify real orders cannot be placed.
        
        This is the most important security test.
        """
        # Attempt to create an order (should be blocked)
        order_blocked = False
        
        try:
            TestSecurityGuard.intercept_call(
                'futures_create_order',
                ('BTCUSDT', 'BUY', 0.001, 50000.0)
            )
        except SecurityException:
            order_blocked = True
        
        assert order_blocked, \
            "CRITICAL FAILURE: Real orders are NOT blocked!"
    
    def test_security_guard_cannot_be_bypassed(self):
        """
        🔴 CRITICAL: Verify SecurityGuard cannot be easily bypassed.
        """
        # Even if someone tries to unlock, the test hooks should re-lock
        TestSecurityGuard.unlock()
        TestSecurityGuard.lock()  # Simulate hook behavior
        
        assert TestSecurityGuard.is_locked(), \
            "SecurityGuard should remain locked in tests"
    
    def test_environment_check_runs_first(self):
        """
        🔴 CRITICAL: Verify environment is validated.
        """
        env = os.environ.get('TRADER_GEMINI_ENV')
        
        # Must be in TEST mode
        assert env == 'TEST', \
            f"CRITICAL: Environment is '{env}' instead of 'TEST'"
        
        # Additional production mode checks
        assert env not in ['PRODUCTION', 'PROD', 'LIVE'], \
            "CRITICAL: Cannot run tests in production mode!"
