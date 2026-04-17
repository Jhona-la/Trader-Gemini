"""
🧪 UNIT TEST: Executor Immutability & SOR Logic
==============================================

👨‍🏫 MODO PROFESOR:
- QUÉ: Verifica que el BinanceExecutor ya no intenta mutar el objeto
  frozen OrderEvent al aplicar Smart Order Routing (SOR) o lógica VWAP.
- POR QUÉ: OrderEvent es un dataclass con frozen=True. Cualquier intento
  de asignar a event.price o event.order_type causaba un crash sistémico.
- PARA QUÉ: Garantizar que el motor puede cambiar de MARKET a LIMIT
  dinámicamente sin romper la inmutabilidad del bus de eventos.
- CÓMO: Fuerza condiciones de VWAP > 0.3% y Rebate Priority, verificando
  que la ejecución final usa los valores locales corregidos mientras el
  evento original permanece intacto.
- DÓNDE: tests/unit/test_executor_immutability.py
- QUIÉN: BinanceExecutor.execute_order()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['TRADER_GEMINI_ENV'] = 'TEST'

import unittest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from core.events import OrderEvent
from core.enums import OrderType, OrderSide
import config

class TestExecutorImmutability(unittest.IsolatedAsyncioTestCase):
    
    async def asyncSetUp(self):
        # Patch Config
        self.config_patcher = patch.object(config.Config, 'BINANCE_USE_FUTURES', True)
        self.config_patcher.start()

        # Mock BinanceExecutor bypass init
        with patch.object(
            __import__('execution.binance_executor', fromlist=['BinanceExecutor']).BinanceExecutor,
            '__init__',
            lambda self_inner, *args, **kwargs: None
        ):
            from execution.binance_executor import BinanceExecutor
            self.executor = BinanceExecutor.__new__(BinanceExecutor)

        self.executor.events_queue = asyncio.Queue()
        self.executor.portfolio = MagicMock()
        self.executor.portfolio.rebate_priority = True # Trigger SOR
        self.executor.data_provider = MagicMock() # Injected DataProvider
        self.executor.order_manager = None # Added missing attribute
        self.executor.latency_violations = 0
        
        # Mock exchange & guardian
        self.executor.exchange = MagicMock()
        self.executor.exchange.markets = {'BTC/USDT': {'id': 'BTCUSDT', 'quote': 'USDT'}}
        self.executor.exchange.market = MagicMock(return_value={'id': 'BTCUSDT', 'quote': 'USDT'})
        self.executor.exchange.price_to_precision = lambda sym, p: f"{float(p):.2f}"
        self.executor.exchange.amount_to_precision = lambda sym, q: f"{float(q):.4f}"
        
        self.executor.async_exchange = AsyncMock()
        self.executor.async_exchange.fapiPrivatePostOrder = AsyncMock(return_value={'orderId': '123', 'status': 'NEW'})
        self.executor.async_exchange.fetch_free_balance = AsyncMock(return_value={'USDT': 1000.0})
        
        self.executor.guardian = MagicMock()
        self.executor.guardian.get_fast_bid_ask = MagicMock(return_value=(50000.0, 50000.1))
        self.executor.guardian.analyze_liquidity = MagicMock(return_value={'is_safe': True})
        
        self.executor.rate_limiter = MagicMock()
        self.executor.rate_limiter.check_limit = MagicMock(return_value=(True, 0))
        self.executor._place_protective_orders = AsyncMock()

    async def asyncTearDown(self):
        self.config_patcher.stop()

    @patch('execution.cost_guard.CostGuard.check_funding_leak', return_value=True)
    async def test_sor_market_to_limit_immutability(self, mock_cost):
        """
        🧪 Test: MARKET -> LIMIT conversion (Rebate Priority).
        Verifica que NO lanza AttributeError por inmutabilidad.
        """
        event = OrderEvent(
            symbol="BTCUSDT",
            order_type=OrderType.MARKET,
            quantity=0.005,
            direction=OrderSide.BUY,
            horizon="SCALPING",
            metadata={'urgent': False}
        )

        print("\n🧪 [IMMUTABILITY] Testing MARKET -> LIMIT (SOR Rebate Priority)...")
        # should NOT raise AttributeError: can't set attribute 'order_type'
        await self.executor.execute_order(event)
        
        # Check if the order was sent as LIMIT
        call_args = self.executor.async_exchange.fapiPrivatePostOrder.call_args[0][0]
        self.assertEqual(call_args['type'], 'LIMIT')
        self.assertEqual(call_args['timeInForce'], 'GTX', "Should be Post-Only for Rebate Priority")
        
        # Check event is UNMODIFIED
        self.assertEqual(event.order_type, OrderType.MARKET)
        print("✅ Correctly converted to LIMIT without mutating frozen OrderEvent")

    @patch('execution.cost_guard.CostGuard.check_funding_leak', return_value=True)
    async def test_vwap_relative_execution_immutability(self, mock_cost):
        """
        🧪 Test: VWAP price push conversion.
        Fuerza la rama de VWAP > 0.3% que antes mutaba event.price.
        """
        # Simular 10 bars donde VWAP es significativamente menor al precio actual
        # VWAP approx 49000, current price 50000
        bars = np.array([
            (49000.0, 49000.0, 49000.0, 49000.0, 1.0, 0)
        ] * 10, dtype=[('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8'), ('datetime', 'i8')])
        
        self.executor.data_provider.get_latest_bars = MagicMock(return_value=bars)
        
        event = OrderEvent(
            symbol="BTCUSDT",
            order_type=OrderType.MARKET,
            quantity=0.005,
            direction=OrderSide.BUY,
            horizon="SCALPING",
            metadata={'urgent': False}
        )
        
        print("🧪 [IMMUTABILITY] Testing VWAP > 0.3% Price Push...")
        # should NOT raise AttributeError: can't set attribute 'price'
        await self.executor.execute_order(event)
        
        # Verify final price is near VWAP adjustment (L460: current_price * 0.9995)
        # Note: Sniper logic adds a 0.0001 (0.01%) bias on top, so we use a wider delta or check logic
        call_args = self.executor.async_exchange.fapiPrivatePostOrder.call_args[0][0]
        self.assertEqual(call_args['type'], 'LIMIT', "VWAP logic should downgrade to LIMIT")
        
        expected_vwap_price = 50000.1 * 0.9995
        # The executor adds spread_adj=0.0001 for BUY Sniper orders
        expected_final = expected_vwap_price * 1.0001 
        
        self.assertAlmostEqual(float(call_args['price']), expected_final, delta=0.1)
        
        # Check event is UNMODIFIED
        self.assertIsNone(event.price)
        print("✅ VWAP Logic applied via locals, original event price remains None")

if __name__ == '__main__':
    unittest.main()
