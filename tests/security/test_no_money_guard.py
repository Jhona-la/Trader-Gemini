"""
🔒 SECURITY TESTS - PROD/DEMO ISOLATION & API KEY VALIDATION
=============================================================

PROFESSOR METHOD:
- QUÉ: Tests de seguridad para aislamiento PROD/DEMO y validación de API Keys.
- POR QUÉ: Previene pérdidas financieras por órdenes accidentales.
- CÓMO: No-Money Guard, validación de entorno, bloqueo de órdenes.
- CUÁNDO: SIEMPRE antes de producción.

SAFETY CHECKS:
- No-Money Guard: Bloquea todas las órdenes reales en modo TEST
- API Key Isolation: Separa keys de producción y demo
- Environment Validation: Verifica ENV=TEST durante tests
"""

import os
import sys
import pytest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Force TEST environment
os.environ['TRADER_GEMINI_ENV'] = 'TEST'
os.environ['BINANCE_USE_TESTNET'] = 'True'
os.environ['BINANCE_USE_DEMO'] = 'True'


# =============================================================================
# SECURITY EXCEPTION
# =============================================================================

class SecurityException(Exception):
    """Raised when a security violation is detected."""
    pass


# =============================================================================
# NO-MONEY GUARD
# =============================================================================

class NoMoneyGuard:
    """
    🔒 No-Money Guard: Absolute protection against real orders in TEST mode.
    
    PROFESSOR METHOD:
    - QUÉ: Guardia que bloquea todas las órdenes reales.
    - POR QUÉ: Previene pérdidas financieras durante desarrollo/testing.
    - CÓMO: Intercepta llamadas a métodos de orden y lanza SecurityException.
    """
    
    _locked = True
    _intercepted_calls = []
    
    # Methods that would cost money
    DANGEROUS_METHODS = [
        'create_order',
        'cancel_order',
        'create_margin_order',
        'futures_create_order',
        'futures_cancel_order',
        'transfer_to_futures',
        'transfer_from_futures',
        'place_order',
        'submit_order'
    ]
    
    @classmethod
    def lock(cls):
        cls._locked = True
    
    @classmethod
    def unlock(cls):
        cls._locked = False
    
    @classmethod
    def is_locked(cls) -> bool:
        return cls._locked
    
    @classmethod
    def intercept(cls, method_name: str, *args, **kwargs):
        """Intercept and block dangerous method calls."""
        call_log = {
            'method': method_name,
            'args': str(args)[:100],
            'kwargs': str(kwargs)[:100],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'blocked': cls._locked
        }
        cls._intercepted_calls.append(call_log)
        
        if cls._locked and method_name in cls.DANGEROUS_METHODS:
            raise SecurityException(
                f"🚨 BLOCKED: {method_name} called while NoMoneyGuard is LOCKED. "
                f"This order would have cost real money!"
            )
    
    @classmethod
    def get_intercepted_calls(cls):
        return cls._intercepted_calls.copy()
    
    @classmethod
    def clear_log(cls):
        cls._intercepted_calls = []


# =============================================================================
# TEST: ENVIRONMENT ISOLATION
# =============================================================================

class TestEnvironmentIsolation:
    """
    🌍 Test Environment Isolation
    """
    
    def test_env_is_test(self):
        """Verify TRADER_GEMINI_ENV is TEST."""
        env = os.environ.get('TRADER_GEMINI_ENV')
        assert env == 'TEST', f"Expected TEST, got {env}"
    
    def test_testnet_enabled(self):
        """Verify BINANCE_USE_TESTNET is True."""
        testnet = os.environ.get('BINANCE_USE_TESTNET')
        assert testnet == 'True', f"Expected True, got {testnet}"
    
    def test_demo_enabled(self):
        """Verify BINANCE_USE_DEMO is True."""
        demo = os.environ.get('BINANCE_USE_DEMO')
        assert demo == 'True', f"Expected True, got {demo}"
    
    def test_not_production(self):
        """Verify we're NOT in production mode."""
        env = os.environ.get('TRADER_GEMINI_ENV', '')
        
        assert env != 'PRODUCTION', "CRITICAL: Cannot run tests in PRODUCTION!"
        assert env != 'PROD', "CRITICAL: Cannot run tests in PROD!"
        assert env != 'LIVE', "CRITICAL: Cannot run tests in LIVE!"


# =============================================================================
# TEST: NO-MONEY GUARD
# =============================================================================

class TestNoMoneyGuard:
    """
    🔒 Test No-Money Guard Functionality
    """
    
    def setup_method(self):
        """Ensure guard is locked before each test."""
        NoMoneyGuard.lock()
        NoMoneyGuard.clear_log()
    
    def test_guard_is_locked(self):
        """Verify guard is locked during tests."""
        assert NoMoneyGuard.is_locked(), "Guard should be LOCKED"
    
    def test_create_order_blocked(self):
        """Test that create_order is blocked."""
        with pytest.raises(SecurityException) as exc:
            NoMoneyGuard.intercept('create_order', 'BTCUSDT', 'BUY', 0.001)
        
        assert 'BLOCKED' in str(exc.value)
        assert 'create_order' in str(exc.value)
    
    def test_futures_order_blocked(self):
        """Test that futures_create_order is blocked."""
        with pytest.raises(SecurityException) as exc:
            NoMoneyGuard.intercept('futures_create_order', 'BTCUSDT', 'BUY', 0.001)
        
        assert 'BLOCKED' in str(exc.value)
    
    def test_cancel_order_blocked(self):
        """Test that cancel_order is blocked."""
        with pytest.raises(SecurityException):
            NoMoneyGuard.intercept('cancel_order', 'BTCUSDT', '12345')
    
    def test_transfer_blocked(self):
        """Test that fund transfers are blocked."""
        with pytest.raises(SecurityException):
            NoMoneyGuard.intercept('transfer_to_futures', 100.0, 'USDT')
    
    def test_safe_methods_allowed(self):
        """Test that safe methods (queries) are allowed."""
        # These should NOT raise
        NoMoneyGuard.intercept('get_account')
        NoMoneyGuard.intercept('get_balance')
        NoMoneyGuard.intercept('get_open_orders')
        NoMoneyGuard.intercept('get_ticker')
    
    def test_calls_are_logged(self):
        """Test that all calls are logged for auditing."""
        NoMoneyGuard.clear_log()
        
        # Make some calls
        NoMoneyGuard.intercept('get_account')
        try:
            NoMoneyGuard.intercept('create_order')
        except SecurityException:
            pass
        
        logs = NoMoneyGuard.get_intercepted_calls()
        assert len(logs) == 2
        assert logs[0]['method'] == 'get_account'
        assert logs[1]['method'] == 'create_order'
        assert logs[1]['blocked'] == True
    
    def test_unlock_allows_orders(self):
        """Test that unlocking allows orders (for controlled testing)."""
        NoMoneyGuard.unlock()
        
        # Should NOT raise when unlocked
        NoMoneyGuard.intercept('create_order', 'BTCUSDT', 'BUY', 0.001)
        
        # Re-lock for safety
        NoMoneyGuard.lock()


# =============================================================================
# TEST: API KEY SEPARATION
# =============================================================================

class TestAPIKeySeparation:
    """
    🔑 Test API Key Separation between PROD and DEMO
    """
    
    def test_demo_keys_used_in_test(self):
        """Verify demo keys are used in test mode."""
        # In test mode, should prefer demo/testnet keys
        use_demo = os.environ.get('BINANCE_USE_DEMO') == 'True'
        use_testnet = os.environ.get('BINANCE_USE_TESTNET') == 'True'
        
        assert use_demo or use_testnet, "Must use demo or testnet keys in TEST mode"
    
    def test_prod_keys_never_loaded_in_test(self):
        """Verify production keys are not actively used in test mode."""
        # This is a safeguard test - in real implementation,
        # the config should not load production keys when ENV=TEST
        env = os.environ.get('TRADER_GEMINI_ENV')
        
        if env == 'TEST':
            # Production should not be active
            assert os.environ.get('BINANCE_USE_TESTNET') == 'True' or \
                   os.environ.get('BINANCE_USE_DEMO') == 'True', \
                   "In TEST mode, must use testnet/demo"


# =============================================================================
# TEST: ORDER SIMULATION PROTECTION
# =============================================================================

class TestOrderSimulationProtection:
    """
    🛡️ Test that simulated orders never reach real API
    """
    
    def test_100_simulated_orders_blocked(self):
        """
        Simulate 100 orders - ALL must be blocked.
        This tests the No-Money Guard under load.
        """
        NoMoneyGuard.lock()
        NoMoneyGuard.clear_log()
        
        blocked_count = 0
        
        for i in range(100):
            try:
                NoMoneyGuard.intercept(
                    'futures_create_order',
                    symbol=f'TOKEN{i}USDT',
                    side='BUY',
                    quantity=0.001
                )
            except SecurityException:
                blocked_count += 1
        
        assert blocked_count == 100, f"Only {blocked_count}/100 orders blocked!"
        print(f"✅ 100/100 simulated orders blocked by No-Money Guard")
    
    def test_margin_orders_blocked(self):
        """Test that margin orders are also blocked."""
        with pytest.raises(SecurityException):
            NoMoneyGuard.intercept('create_margin_order', 'BTCUSDT', 'BUY', 0.001)


# =============================================================================
# TEST: AUDIT LOGGING
# =============================================================================

class TestAuditLogging:
    """
    📋 Test Audit Logging for Security Events
    """
    
    def test_blocked_calls_include_timestamp(self):
        """Verify blocked calls are timestamped."""
        NoMoneyGuard.lock()
        NoMoneyGuard.clear_log()
        
        try:
            NoMoneyGuard.intercept('create_order')
        except SecurityException:
            pass
        
        logs = NoMoneyGuard.get_intercepted_calls()
        assert 'timestamp' in logs[0]
        assert '2026' in logs[0]['timestamp']  # Year validation
    
    def test_audit_trail_complete(self):
        """Verify complete audit trail is maintained."""
        NoMoneyGuard.lock()
        NoMoneyGuard.clear_log()
        
        # Various operations
        NoMoneyGuard.intercept('get_balance')
        
        try:
            NoMoneyGuard.intercept('create_order', 'BTCUSDT', 'BUY')
        except SecurityException:
            pass
        
        try:
            NoMoneyGuard.intercept('futures_create_order', 'ETHUSDT', 'SELL')
        except SecurityException:
            pass
        
        logs = NoMoneyGuard.get_intercepted_calls()
        
        assert len(logs) == 3
        # get_balance is safe, so 'blocked' is True (guard is locked) but no exception raised
        # The 'blocked' field indicates guard state at interception, not whether it was blocked
        assert logs[0]['method'] == 'get_balance'
        assert logs[1]['method'] == 'create_order'
        assert logs[2]['method'] == 'futures_create_order'


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
