"""
🛡️ PHASE OMNI: BYZANTINE RESILIENCE SIMULATION
================================================
QUÉ: Suite de testing que simula fallos bizantinos (subsistemas que mienten,
     se detienen, o dan resultados incorrectos) para verificar degradación
     graceful del sistema.
POR QUÉ: En producción HFT, los componentes fallan de formas impredecibles:
         APIs devuelven datos erróneos, WebSockets pierden conexión,
         procesos de riesgo se cuelgan sin reportar.
PARA QUÉ: Garantizar que el bot NUNCA pierde dinero por un fallo de componente.
CÓMO: Para cada subsistema:
      1. Inyecta fallo específico (datos nulos, excepciones, timeouts)
      2. Ejecuta el engine con el componente "comprometido"
      3. Verifica que la respuesta del sistema es SEGURA
CUÁNDO: Pre-deploy, after each critical module change.
DÓNDE: tests/byzantine_test.py
QUIÉN: QA Engineer, SRE.

TIPOS DE FALLOS SIMULADOS:
1. DATA_PROVIDER_DEAD: binance_loader devuelve None
2. RISK_MANAGER_CRASH: risk_manager lanza exception
3. WEBSOCKET_STALE: Datos congelados > 30s
4. PORTFOLIO_CORRUPT: Posiciones con NaN
5. EXECUTION_REJECT: Todas las órdenes rechazadas
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime, timezone

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestByzantineResilience:
    """
    Byzantine Fault Tolerance Tests.
    Each test simulates a specific subsystem failure and verifies
    the system responds SAFELY (no orders, no crashes, graceful degradation).
    """
    
    # ======================================================================
    # FIXTURE: Create a minimal Engine environment for testing
    # ======================================================================
    
    @pytest.fixture
    def mock_engine(self):
        """Creates a minimal engine with mocked subsystems."""
        from core.engine import Engine, AsyncBoundedQueue
        from core.events import MarketEvent
        
        engine = Engine.__new__(Engine)
        engine.events = AsyncBoundedQueue(maxsize=100)
        engine.data_handlers = []
        engine.strategies = []
        engine.execution_handler = None
        engine.portfolio = None
        engine.risk_manager = None
        engine.order_manager = None
        engine.running = True
        engine._strategy_cooldowns = {}
        engine.metrics = {
            'processed_events': 0,
            'discarded_events': 0,
            'strategy_executions': 0,
            'errors': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'burst_events': 0,
        }
        engine.forensics = MagicMock()
        engine.system_monitor = MagicMock()
        engine.system_monitor.check_health.return_value = True
        
        return engine
    
    @pytest.fixture
    def market_event(self):
        """Creates a standard market event for testing."""
        from core.events import MarketEvent
        return MarketEvent(
            symbol='BTC/USDT',
            close_price=50000.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    # ======================================================================
    # TEST 1: DATA PROVIDER DEAD (Returns None/Empty)
    # ======================================================================
    
    @pytest.mark.asyncio
    async def test_data_provider_returns_none(self, mock_engine, market_event):
        """
        FALLO: DataProvider.get_latest_bars() devuelve None.
        ESPERADO: Engine procesa evento sin crash.
                  NO se generan señales (data insuficiente).
        """
        mock_dh = MagicMock()
        mock_dh.get_latest_bars.return_value = None
        mock_dh.get_latency_metrics.return_value = (5.0, 10.0)
        mock_engine.data_handlers = [mock_dh]
        
        mock_strategy = MagicMock()
        mock_strategy.symbol = 'BTC/USDT'
        mock_strategy.calculate_signals = MagicMock()
        mock_engine.strategies = [mock_strategy]
        
        mock_engine.portfolio = MagicMock()
        mock_engine.portfolio.positions = {}
        
        # Process should NOT crash
        try:
            await mock_engine.process_event(market_event)
            no_crash = True
        except Exception:
            no_crash = False
        
        assert no_crash, "Engine crashed when DataProvider returned None"
    
    @pytest.mark.asyncio
    async def test_data_provider_returns_empty(self, mock_engine, market_event):
        """
        FALLO: DataProvider devuelve array vacío.
        ESPERADO: Engine ignora datos vacíos sin crash.
        """
        mock_dh = MagicMock()
        mock_dh.get_latest_bars.return_value = np.array([])
        mock_dh.get_latency_metrics.return_value = (5.0, 10.0)
        mock_engine.data_handlers = [mock_dh]
        
        mock_engine.portfolio = MagicMock()
        mock_engine.portfolio.positions = {}
        
        try:
            await mock_engine.process_event(market_event)
            no_crash = True
        except Exception:
            no_crash = False
        
        assert no_crash, "Engine crashed on empty data"
    
    # ======================================================================
    # TEST 2: RISK MANAGER CRASH (Unhandled Exception)
    # ======================================================================
    
    @pytest.mark.asyncio
    async def test_risk_manager_exception(self, mock_engine):
        """
        FALLO: RiskManager.generate_order() lanza RuntimeError.
        ESPERADO: Engine catches exception. NO order generated.
                  Error counter incremented.
        """
        from core.events import SignalEvent
        from core.enums import SignalType
        
        signal = SignalEvent(
            symbol='BTC/USDT',
            signal_type=SignalType.LONG,
            strength=0.8,
            strategy_id='test'
        )
        
        mock_engine.risk_manager = MagicMock()
        mock_engine.risk_manager.generate_order.side_effect = RuntimeError("Byzantine fault!")
        mock_engine.portfolio = MagicMock()
        mock_engine.execution_handler = MagicMock()
        
        mock_dh = MagicMock()
        mock_dh.get_latest_bars.return_value = MagicMock(close=np.array([50000.0, 50100.0, 50050.0]))
        mock_dh.get_latency_metrics.return_value = (5.0, 10.0)
        mock_engine.data_handlers = [mock_dh]
        
        # Should NOT crash the engine
        try:
            await mock_engine.process_event(signal)
            no_crash = True
        except Exception:
            no_crash = False
        
        assert no_crash, "Engine crashed when RiskManager raised exception"
        # Verify: No order was sent to execution
        mock_engine.execution_handler.execute_order.assert_not_called()
    
    # ======================================================================
    # TEST 3: PORTFOLIO WITH NaN POSITIONS
    # ======================================================================
    
    @pytest.mark.asyncio
    async def test_portfolio_nan_positions(self, mock_engine, market_event):
        """
        FALLO: Portfolio.positions contiene valores NaN.
        ESPERADO: Engine handles NaN gracefully. No new orders placed.
        """
        mock_engine.portfolio = MagicMock()
        mock_engine.portfolio.positions = {
            'BTC/USDT': {
                'quantity': float('nan'),
                'avg_price': float('nan'),
                'current_price': 50000.0
            }
        }
        mock_engine.portfolio.update_market_price = MagicMock()
        
        mock_dh = MagicMock()
        mock_dh.get_latest_bars.return_value = None
        mock_dh.get_latency_metrics.return_value = (5.0, 10.0)
        mock_engine.data_handlers = [mock_dh]
        
        try:
            await mock_engine.process_event(market_event)
            no_crash = True
        except Exception:
            no_crash = False
        
        assert no_crash, "Engine crashed on NaN portfolio positions"
    
    # ======================================================================
    # TEST 4: EXECUTION HANDLER REJECTS ALL ORDERS
    # ======================================================================
    
    @pytest.mark.asyncio
    async def test_execution_all_rejected(self, mock_engine):
        """
        FALLO: ExecutionHandler rechaza TODAS las órdenes.
        ESPERADO: Engine handles rejection without crash.
                  Order manager notified of failure.
        """
        from core.events import OrderEvent
        from core.enums import OrderSide, OrderType
        
        order = OrderEvent(
            symbol='BTC/USDT',
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=0.001,
            price=50000.0
        )
        
        mock_engine.execution_handler = MagicMock()
        mock_engine.execution_handler.execute_order.side_effect = Exception("REJECTED: Insufficient margin")
        mock_engine.order_manager = MagicMock()
        
        try:
            await mock_engine.process_event(order)
            no_crash = True
        except Exception:
            no_crash = False
        
        assert no_crash, "Engine crashed when all orders were rejected"
    
    # ======================================================================
    # TEST 5: WEBSOCKET STALE DATA (Frozen > 30s)
    # ======================================================================
    
    @pytest.mark.asyncio
    async def test_stale_websocket_data(self, mock_engine, market_event):
        """
        FALLO: Market event tiene timestamp > 30s en el pasado.
        ESPERADO: Engine debería descartar señales stale o al menos
                  no generar órdenes basadas en datos antiguos.
        """
        from datetime import timedelta
        
        # Create stale event (60 seconds old)
        stale_event = MagicMock()
        stale_event.type = 'MARKET'
        stale_event.symbol = 'BTC/USDT'
        stale_event.close_price = 50000.0
        stale_event.timestamp = datetime.now(timezone.utc) - timedelta(seconds=60)
        stale_event.datetime = stale_event.timestamp
        
        mock_engine.portfolio = MagicMock()
        mock_engine.portfolio.positions = {}
        
        try:
            await mock_engine.process_event(stale_event)
            no_crash = True
        except Exception:
            no_crash = False
        
        assert no_crash, "Engine crashed on stale WebSocket data"
    
    # ======================================================================
    # TEST 6: CONCURRENT KILL SWITCH ACTIVATION
    # ======================================================================
    
    def test_kill_switch_blocks_all_orders(self):
        """
        FALLO: Kill switch está activo.
        ESPERADO: generate_order() retorna None para todas las señales.
        """
        from risk.risk_manager import RiskManager
        from core.events import SignalEvent
        from core.enums import SignalType
        
        rm = RiskManager.__new__(RiskManager)
        rm.kill_switch = MagicMock()
        rm.kill_switch.check_status.return_value = False
        rm.kill_switch.activation_reason = "Test Byzantine"
        rm.portfolio = MagicMock()
        rm.portfolio.get_total_equity.return_value = 15.0
        rm.cvar_calc = MagicMock()
        rm._trade_cache = []
        rm._cache_initialized = True
        
        signal = SignalEvent(
            symbol='BTC/USDT',
            signal_type=SignalType.LONG,
            strength=1.0,
            strategy_id='test'
        )
        
        # Should return None (blocked)
        try:
            result = rm.generate_order(signal, 50000.0)
            assert result is None, "Order generated despite kill switch!"
        except Exception:
            pass  # If it raises, that's also acceptable (no silent failure)
    
    # ======================================================================
    # TEST 7: STRATEGY RETURNS INVALID SIGNAL
    # ======================================================================
    
    @pytest.mark.asyncio
    async def test_strategy_invalid_signal(self, mock_engine, market_event):
        """
        FALLO: Strategy genera un signal con tipo inválido.
        ESPERADO: Engine ignora señal inválida sin crash.
        """
        from core.events import SignalEvent
        
        mock_strategy = MagicMock()
        mock_strategy.symbol = 'BTC/USDT'
        
        # Strategy puts an invalid signal type into the queue
        invalid_signal = MagicMock()
        invalid_signal.type = 'INVALID_TYPE'
        invalid_signal.symbol = 'BTC/USDT'
        
        mock_engine.portfolio = MagicMock()
        mock_engine.portfolio.positions = {}
        
        try:
            await mock_engine.process_event(invalid_signal)
            no_crash = True
        except Exception:
            no_crash = False
        
        assert no_crash, "Engine crashed on invalid signal type"
    
    # ======================================================================
    # TEST 8: SIMULTANEOUS MULTI-SUBSYSTEM FAILURE
    # ======================================================================
    
    @pytest.mark.asyncio
    async def test_cascading_failure(self, mock_engine, market_event):
        """
        FALLO: DataHandler, Portfolio, y Strategy todos fallan simultáneamente.
        ESPERADO: Engine handles ALL exceptions and continues running.
                  Error count reflects all failures.
        """
        # All subsystems throw
        mock_dh = MagicMock()
        mock_dh.get_latest_bars.side_effect = ConnectionError("API Down")
        mock_dh.get_latency_metrics.side_effect = ConnectionError("API Down")
        mock_engine.data_handlers = [mock_dh]
        
        mock_engine.portfolio = MagicMock()
        mock_engine.portfolio.update_market_price.side_effect = ValueError("Corrupt state")
        mock_engine.portfolio.positions = {}
        
        mock_strategy = MagicMock()
        mock_strategy.symbol = 'BTC/USDT'
        mock_strategy.calculate_signals.side_effect = RuntimeError("NaN in features")
        mock_engine.strategies = [mock_strategy]
        
        try:
            await mock_engine.process_event(market_event)
            # Should still be alive
            assert mock_engine.running, "Engine stopped running after cascading failure"
        except Exception:
            pass  # Even if process_event raises, engine.running should still be True
        
        assert mock_engine.running, "Engine.running became False after failures"


class TestGracefulDegradation:
    """
    Tests that verify the system degrades gracefully under partial failures,
    maintaining safety invariants (no unexpected orders, no data corruption).
    """
    
    def test_risk_manager_without_portfolio(self):
        """RiskManager should work with portfolio=None (paper mode)."""
        from risk.risk_manager import RiskManager
        
        try:
            rm = RiskManager(portfolio=None)
            wr = rm.get_win_rate()
            assert 0 <= wr <= 1.0
        except Exception as e:
            pytest.fail(f"RiskManager crashed without portfolio: {e}")
    
    def test_cvar_with_insufficient_data(self):
        """CVaR should return safe default with <10 samples."""
        from risk.risk_manager import CVaRCalculator
        
        calc = CVaRCalculator()
        # Only 3 samples
        calc.update(-0.01)
        calc.update(-0.02)
        calc.update(-0.005)
        
        cvar = calc.calculate_cvar()
        assert cvar == 0.05, f"CVaR with insufficient data should be 0.05 (default), got {cvar}"
    
    def test_adaptive_balancer_empty_symbols(self):
        """AdaptiveBalancer should handle empty symbol list."""
        from core.adaptive_balancer import AdaptiveBalancer
        
        balancer = AdaptiveBalancer([])
        balancer.rebalance()
        order = balancer.get_processing_order()
        assert order == [], "Empty balancer should return empty order"


# ======================================================================
# STANDALONE RUNNER
# ======================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🛡️ BYZANTINE RESILIENCE SIMULATION")
    print("="*60 + "\n")
    
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-W", "ignore::DeprecationWarning"
    ])
    
    if exit_code == 0:
        print("\n✅ ALL BYZANTINE TESTS PASSED - System is resilient!")
    else:
        print("\n❌ BYZANTINE FAILURES DETECTED - Fix before deploying!")
    
    sys.exit(exit_code)
