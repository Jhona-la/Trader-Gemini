"""
🧪 CONFTEST.PY - PYTEST GLOBAL CONFIGURATION
=============================================

PROFESSOR METHOD:
- QUÉ: Configuración global de pytest con fixtures reutilizables.
- POR QUÉ: Centraliza la inicialización de mocks, fixtures y configuración de entorno.
- CÓMO: Define fixtures que se inyectan automáticamente en todos los tests.
- CUÁNDO: Se carga automáticamente por pytest al inicio de cada sesión de tests.
- DÓNDE: Raíz del proyecto para ser descubierto por pytest.
"""

import os
import sys
import json
import time
import pytest
import tempfile
import shutil
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from freezegun import freeze_time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force TEST environment
os.environ['TRADER_GEMINI_ENV'] = 'TEST'
os.environ['BINANCE_USE_TESTNET'] = 'True'
os.environ['BINANCE_USE_DEMO'] = 'True'


# =============================================================================
# WINDOWS FILE CLEANUP HELPER
# =============================================================================

def safe_remove(path: str, retries: int = 5) -> bool:
    """
    🧹 Safely remove a file with retries for Windows file locking (WinError 32).
    
    PROFESSOR METHOD:
    - QUÉ: Eliminación de archivos con reintentos para evitar WinError 32.
    - POR QUÉ: Windows mantiene file locks que impiden borrado inmediato.
    - CÓMO: Intenta eliminar hasta 5 veces con espera progresiva.
    
    Args:
        path: Path to file to remove
        retries: Number of retry attempts
        
    Returns:
        True if removed successfully, False otherwise
    """
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))  # Progressive delay
        except Exception:
            return False
    return False


def safe_rmtree(path: str, retries: int = 5) -> bool:
    """
    🧹 Safely remove a directory tree with retries.
    """
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
            return True
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
        except Exception:
            return False
    return False


# =============================================================================
# SAFETY GUARDS
# =============================================================================

class TestSecurityGuard:
    """
    🔒 No-Money Guard: Prevents any real API calls during tests.
    
    PROFESSOR METHOD:
    - QUÉ: Bloquea llamadas a APIs de producción durante tests.
    - POR QUÉ: Evita pérdidas financieras accidentales.
    - CÓMO: Intercepta y lanza excepción si se detecta llamada real.
    """
    
    _locked = False
    _intercepted_calls = []
    
    @classmethod
    def lock(cls):
        """Activate read-only mode."""
        cls._locked = True
        cls._intercepted_calls = []
    
    @classmethod
    def unlock(cls):
        """Deactivate read-only mode."""
        cls._locked = False
    
    @classmethod
    def is_locked(cls):
        return cls._locked
    
    @classmethod
    def intercept_call(cls, method_name: str, args: tuple):
        """Log intercepted call for verification."""
        cls._intercepted_calls.append({
            'method': method_name,
            'args': args,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        if cls._locked:
            raise SecurityException(
                f"🚨 BLOCKED: Attempted to call '{method_name}' in TEST mode. "
                "Real API calls are prohibited during tests."
            )


class SecurityException(Exception):
    """Raised when a test attempts to make a real API call."""
    pass


# =============================================================================
# FIXTURES - CORE
# =============================================================================

@pytest.fixture
def frozen_time():
    """
    ❄️ Mock Time: Freezes time for deterministic tests.
    
    Usage:
        def test_something(frozen_time):
            assert datetime.now() == frozen_time
    """
    fixed_dt = datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc)
    with freeze_time(fixed_dt):
        yield fixed_dt


@pytest.fixture
def temp_data_dir():
    """
    📁 Temporary Data Directory: Isolated folder for test outputs.
    
    Automatically cleaned up after each test.
    """
    temp_dir = tempfile.mkdtemp(prefix="trader_gemini_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config(temp_data_dir, monkeypatch):
    """
    ⚙️ Mock Config: Isolated configuration for tests.
    
    Sets:
    - DATA_DIR to temp directory
    - BINANCE_USE_TESTNET = True
    - BINANCE_USE_DEMO = True
    - INITIAL_CAPITAL = 1000.0 (test capital)
    """
    from config import Config
    
    monkeypatch.setattr(Config, 'DATA_DIR', temp_data_dir)
    monkeypatch.setattr(Config, 'BINANCE_USE_TESTNET', True)
    monkeypatch.setattr(Config, 'BINANCE_USE_DEMO', True)
    monkeypatch.setattr(Config, 'INITIAL_CAPITAL', 1000.0)
    monkeypatch.setattr(Config, 'BINANCE_LEVERAGE', 3)
    
    # Create required subdirectories
    os.makedirs(os.path.join(temp_data_dir, 'sessions'), exist_ok=True)
    
    yield Config


@pytest.fixture
def mock_portfolio(mock_config, temp_data_dir):
    """
    💼 Mock Portfolio: Isolated portfolio for tests.
    """
    from core.portfolio import Portfolio
    
    csv_path = os.path.join(temp_data_dir, 'trades.csv')
    status_path = os.path.join(temp_data_dir, 'status.csv')
    
    portfolio = Portfolio(
        initial_capital=1000.0,
        csv_path=csv_path,
        status_path=status_path,
        auto_save=False  # Disable auto-save in tests
    )
    
    yield portfolio


@pytest.fixture
def mock_risk_manager(mock_portfolio):
    """
    ⚠️ Mock RiskManager: Isolated risk manager for tests.
    """
    from risk.risk_manager import RiskManager
    
    risk_manager = RiskManager(mock_portfolio)
    yield risk_manager


@pytest.fixture
def mock_data_handler(temp_data_dir, monkeypatch):
    """
    📊 Mock DataHandler: Isolated data handler for tests.
    """
    from core.data_handler import DataHandler, get_data_handler
    
    # Reset singleton
    DataHandler._instance = None
    DataHandler._initialized = False
    
    handler = get_data_handler()
    yield handler
    
    # Cleanup singleton after test
    DataHandler._instance = None
    DataHandler._initialized = False


# =============================================================================
# FIXTURES - BINANCE MOCKS
# =============================================================================

@pytest.fixture
def mock_binance_account():
    """
    🏦 Mock Binance Account: Simulated account response.
    
    Returns a realistic futures account structure.
    """
    return {
        'totalWalletBalance': '5000.00',
        'totalMarginBalance': '5100.00',
        'totalUnrealizedProfit': '100.00',
        'availableBalance': '4500.00',
        'totalInitialMargin': '500.00',
        'totalMaintMargin': '50.00',
        'totalPositionInitialMargin': '400.00',
        'totalOpenOrderInitialMargin': '100.00',
        'maxWithdrawAmount': '4400.00',
        'positions': [
            {
                'symbol': 'XRPUSDT',
                'positionAmt': '100.0',
                'entryPrice': '0.55',
                'unrealizedProfit': '5.00',
                'notional': '60.00',
                'leverage': '3',
                'isolated': True
            },
            {
                'symbol': 'BTCUSDT',
                'positionAmt': '0.01',
                'entryPrice': '45000.00',
                'unrealizedProfit': '50.00',
                'notional': '500.00',
                'leverage': '3',
                'isolated': True
            }
        ]
    }


@pytest.fixture
def mock_binance_client(mock_binance_account):
    """
    🔌 Mock Binance Client: Simulates python-binance Client.
    
    All API methods return controlled, predictable responses.
    Uses the actual MockBinanceClient from mocks module.
    """
    from tests.mocks.binance_mock import create_mock_client_with_xrp_position
    
    client = create_mock_client_with_xrp_position()
    yield client


@pytest.fixture
def mock_live_status():
    """
    📋 Mock Live Status: Expected structure of live_status.json.
    
    Used to validate that generated files match expected schema.
    """
    return {
        'timestamp': '2026-02-03T12:00:00+00:00',
        'total_equity': 5100.00,
        'wallet_balance': 5000.00,
        'available_balance': 4500.00,
        'unrealized_pnl': 100.00,
        'margin_ratio': 1.0,
        'positions': {
            'XRP/USDT': {
                'quantity': 100.0,
                'avg_price': 0.55,
                'current_price': 0.60,
                'unrealized_pnl': 5.00
            }
        },
        'last_heartbeat': '2026-02-03T12:00:00+00:00',
        'source': 'TEST_MOCK'
    }


# =============================================================================
# FIXTURES - EVENTS
# =============================================================================

@pytest.fixture
def sample_signal_event():
    """
    📡 Sample Signal Event: A realistic signal for testing.
    """
    from core.events import SignalEvent
    from core.enums import SignalType
    from datetime import datetime, timezone
    
    return SignalEvent(
        symbol='XRP/USDT',
        signal_type=SignalType.LONG,
        datetime=datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc),
        strength=0.85,
        strategy_id='TEST_STRATEGY',
        sl_pct=0.05,
        tp_pct=0.10,
        current_price=0.55,
        leverage=3
    )


@pytest.fixture
def sample_fill_event():
    """
    ✅ Sample Fill Event: A realistic fill for testing.
    """
    from core.events import FillEvent
    from core.enums import OrderSide
    from datetime import datetime, timezone
    
    return FillEvent(
        symbol='XRP/USDT',
        exchange='BINANCE',
        quantity=100.0,
        direction=OrderSide.BUY,
        fill_cost=55.0,  # Total cost = price * quantity
        fill_price=0.55,
        commission=0.01,
        order_id='TEST_ORDER_001',
        timeindex=datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc)
    )


# =============================================================================
# HOOKS
# =============================================================================

def pytest_configure(config):
    """
    🚀 Pytest Configure Hook: Runs before test collection.
    """
    # Ensure we're in TEST mode
    os.environ['TRADER_GEMINI_ENV'] = 'TEST'
    
    # Lock security guard
    TestSecurityGuard.lock()
    print("\n🔒 TestSecurityGuard: LOCKED (No real API calls allowed)")


def pytest_unconfigure(config):
    """
    🏁 Pytest Unconfigure Hook: Runs after all tests.
    """
    TestSecurityGuard.unlock()
    print("\n🔓 TestSecurityGuard: UNLOCKED")


def pytest_runtest_setup(item):
    """
    ⚙️ Test Setup Hook: Runs before each test.
    """
    # Ensure fresh environment for each test
    if hasattr(item, 'fixturenames'):
        if 'mock_config' in item.fixturenames:
            os.environ['TRADER_GEMINI_ENV'] = 'TEST'
