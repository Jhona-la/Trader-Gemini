"""
🔬 HORIZON ISOLATION — Multi-Horizon Contagion Prevention Test
==============================================================
QUÉ:     Verifica que las órdenes protectoras (SL/TP) usen `reduceOnly=true`
         en lugar de `closePosition=true`.
POR QUÉ: `closePosition` cierra el 100% de la posición neta en Binance.
         Si Scalping tiene 0.1 BTC y Swing tiene 0.5 BTC, un SL de Scalping
         cerraría los 0.6 BTC totales, destruyendo la posición de Swing.
PARA QUÉ: Permitir que ambos horizontes (Scalping + Swing) coexistan sin
          sabotearse mutuamente en modo One-Way de Binance.
CÓMO:    Mockeamos `fapiPrivatePostOrder` y capturamos los parámetros enviados.
         Verificamos que NO contengan `closePosition` y SÍ contengan `reduceOnly`.
CUÁNDO:  Cada vez que se ejecuta la suite de tests antes de producción.
DÓNDE:   execution/binance_executor.py → _place_protective_orders()
QUIÉN:   BinanceExecutor (Layer 3: Exchange-Based Protective Orders)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch, call
import ast


class TestHorizonIsolationCodeAudit(unittest.TestCase):
    """
    STATIC AUDIT: Parse the actual source code AST to guarantee
    'closePosition' is completely absent and 'reduceOnly' is present
    in _place_protective_orders.
    """

    @classmethod
    def setUpClass(cls):
        """Load and parse the executor source code."""
        executor_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'execution', 'binance_executor.py'
        )
        with open(executor_path, 'r', encoding='utf-8') as f:
            cls.source = f.read()
        cls.tree = ast.parse(cls.source)

    def _get_method_source(self, method_name):
        """Extract source lines for a specific method."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                start = node.lineno
                end = node.end_lineno
                lines = self.source.splitlines()
                return '\n'.join(lines[start - 1:end])
        return None

    def test_no_close_position_in_protective_orders(self):
        """CRITICAL: closePosition must NOT exist in _place_protective_orders."""
        method_src = self._get_method_source('_place_protective_orders')
        self.assertIsNotNone(method_src, "_place_protective_orders method not found!")
        self.assertNotIn('closePosition', method_src,
                         "❌ CRITICAL: 'closePosition' found in _place_protective_orders! "
                         "This would cause Horizon Contagion.")
        print("✅ [AUDIT] No 'closePosition' in _place_protective_orders")

    def test_reduce_only_present_in_protective_orders(self):
        """CRITICAL: reduceOnly MUST exist in _place_protective_orders."""
        method_src = self._get_method_source('_place_protective_orders')
        self.assertIsNotNone(method_src, "_place_protective_orders method not found!")
        self.assertIn('reduceOnly', method_src,
                      "❌ CRITICAL: 'reduceOnly' NOT found in _place_protective_orders! "
                      "Protective orders will not be horizon-safe.")
        print("✅ [AUDIT] 'reduceOnly' present in _place_protective_orders")

    def test_no_close_position_anywhere_in_file(self):
        """GLOBAL: closePosition must NOT exist anywhere in binance_executor.py."""
        self.assertNotIn('closePosition', self.source,
                         "❌ 'closePosition' still exists somewhere in binance_executor.py!")
        print("✅ [AUDIT] No 'closePosition' anywhere in binance_executor.py")

    def test_reduce_only_count(self):
        """Verify exactly 2 reduceOnly references (SL + TP)."""
        method_src = self._get_method_source('_place_protective_orders')
        self.assertIsNotNone(method_src)
        count = method_src.count('reduceOnly')
        self.assertEqual(count, 2,
                         f"Expected 2 'reduceOnly' references (SL+TP), found {count}")
        print(f"✅ [AUDIT] Exactly {count} 'reduceOnly' references (SL + TP)")


class TestProtectiveOrderParams(unittest.TestCase):
    """
    BEHAVIORAL TEST: Mock the exchange API and verify the actual
    parameters dispatched by _place_protective_orders.
    """

    def _create_executor_with_mocked_exchange(self):
        """Create a BinanceExecutor-like object with mocked exchange."""
        executor = MagicMock()
        executor.exchange = MagicMock()
        executor.exchange.price_to_precision.side_effect = lambda sym, p: f"{p:.2f}"
        executor.exchange.amount_to_precision.side_effect = lambda sym, q: f"{q:.4f}"
        executor.exchange.fapiPrivatePostOrder.return_value = {'orderId': '12345'}

        # Bind the REAL method to our mock object
        from execution.binance_executor import BinanceExecutor
        executor._place_protective_orders = BinanceExecutor._place_protective_orders.__get__(executor)
        return executor

    def test_stop_loss_uses_reduce_only(self):
        """SL order must use reduceOnly=true, NOT closePosition."""
        executor = self._create_executor_with_mocked_exchange()

        executor._place_protective_orders(
            symbol_id='BTCUSDT',
            side='BUY',
            quantity=0.001,
            entry_price=50000.0,
            sl_pct=0.003,
            tp_pct=0.008
        )

        # Get ALL calls to fapiPrivatePostOrder
        calls = executor.exchange.fapiPrivatePostOrder.call_args_list
        self.assertGreaterEqual(len(calls), 2, "Expected at least 2 orders (SL + TP)")

        # First call = STOP_MARKET (SL)
        sl_params = calls[0][0][0]
        self.assertEqual(sl_params['type'], 'STOP_MARKET')
        self.assertIn('reduceOnly', sl_params,
                      "❌ SL order missing 'reduceOnly'!")
        self.assertEqual(sl_params['reduceOnly'], 'true')
        self.assertNotIn('closePosition', sl_params,
                         "❌ SL order still has 'closePosition'!")
        print(f"✅ [SL] reduceOnly='{sl_params['reduceOnly']}', qty={sl_params['quantity']}")

    def test_take_profit_uses_reduce_only(self):
        """TP order must use reduceOnly=true, NOT closePosition."""
        executor = self._create_executor_with_mocked_exchange()

        executor._place_protective_orders(
            symbol_id='BTCUSDT',
            side='BUY',
            quantity=0.001,
            entry_price=50000.0,
            sl_pct=0.003,
            tp_pct=0.008
        )

        calls = executor.exchange.fapiPrivatePostOrder.call_args_list
        self.assertGreaterEqual(len(calls), 2)

        # Second call = TAKE_PROFIT_MARKET (TP)
        tp_params = calls[1][0][0]
        self.assertEqual(tp_params['type'], 'TAKE_PROFIT_MARKET')
        self.assertIn('reduceOnly', tp_params,
                      "❌ TP order missing 'reduceOnly'!")
        self.assertEqual(tp_params['reduceOnly'], 'true')
        self.assertNotIn('closePosition', tp_params,
                         "❌ TP order still has 'closePosition'!")
        print(f"✅ [TP] reduceOnly='{tp_params['reduceOnly']}', qty={tp_params['quantity']}")

    def test_quantity_is_exact_not_close_all(self):
        """Verify quantity is the exact filled amount, not 'close all'."""
        executor = self._create_executor_with_mocked_exchange()

        test_qty = 0.0037  # Specific fractional qty (Scalping portion only)
        executor._place_protective_orders(
            symbol_id='ETHUSDT',
            side='SELL',
            quantity=test_qty,
            entry_price=3200.0,
            sl_pct=0.005,
            tp_pct=0.010
        )

        calls = executor.exchange.fapiPrivatePostOrder.call_args_list
        for c in calls:
            params = c[0][0]
            # The qty should match our precision-formatted test_qty
            self.assertEqual(params['quantity'], f"{test_qty:.4f}")
            # NO closePosition means Binance will use exactly this qty
            self.assertNotIn('closePosition', params)
        print(f"✅ [QTY] Both SL/TP use exact qty={test_qty} (not close-all)")

    def test_long_and_short_price_directions(self):
        """Verify SL below entry for LONG, above entry for SHORT."""
        executor = self._create_executor_with_mocked_exchange()

        # LONG entry
        executor._place_protective_orders('BTCUSDT', 'BUY', 0.001, 50000.0, 0.003, 0.008)
        calls = executor.exchange.fapiPrivatePostOrder.call_args_list

        sl_params = calls[0][0][0]
        tp_params = calls[1][0][0]

        sl_price = float(sl_params['stopPrice'])
        tp_price = float(tp_params['stopPrice'])

        self.assertLess(sl_price, 50000.0, "LONG SL should be below entry")
        self.assertGreater(tp_price, 50000.0, "LONG TP should be above entry")
        self.assertEqual(sl_params['side'], 'SELL', "LONG SL side should be SELL")
        self.assertEqual(tp_params['side'], 'SELL', "LONG TP side should be SELL")
        print(f"✅ [LONG] SL={sl_price} < 50000 < {tp_price}=TP, sides=SELL")

        # Reset and test SHORT
        executor.exchange.fapiPrivatePostOrder.reset_mock()
        executor._place_protective_orders('BTCUSDT', 'SELL', 0.001, 50000.0, 0.003, 0.008)
        calls = executor.exchange.fapiPrivatePostOrder.call_args_list

        sl_params = calls[0][0][0]
        tp_params = calls[1][0][0]
        sl_price = float(sl_params['stopPrice'])
        tp_price = float(tp_params['stopPrice'])

        self.assertGreater(sl_price, 50000.0, "SHORT SL should be above entry")
        self.assertLess(tp_price, 50000.0, "SHORT TP should be below entry")
        self.assertEqual(sl_params['side'], 'BUY', "SHORT SL side should be BUY")
        print(f"✅ [SHORT] SL={sl_price} > 50000 > {tp_price}=TP, sides=BUY")


class TestMultiHorizonScenario(unittest.TestCase):
    """
    SCENARIO TEST: Simulate exactly the contagion scenario described
    in the implementation plan.
    """

    def test_scalping_sl_does_not_close_swing_position(self):
        """
        Scenario:
        - Scalping holds 0.001 BTC (qty=0.001)
        - Swing holds 0.005 BTC (qty=0.005)
        - Scalping's SL fires → should close ONLY 0.001, not 0.006 total
        
        With closePosition=true: Binance closes ALL 0.006 BTC ❌
        With reduceOnly=true:    Binance closes only 0.001 BTC ✅
        """
        executor = MagicMock()
        executor.exchange = MagicMock()
        executor.exchange.price_to_precision.side_effect = lambda sym, p: f"{p:.2f}"
        executor.exchange.amount_to_precision.side_effect = lambda sym, q: f"{q:.4f}"
        executor.exchange.fapiPrivatePostOrder.return_value = {'orderId': 'SL-001'}

        from execution.binance_executor import BinanceExecutor
        executor._place_protective_orders = BinanceExecutor._place_protective_orders.__get__(executor)

        scalping_qty = 0.001
        swing_qty = 0.005

        # Place Scalping protective orders
        executor._place_protective_orders('BTCUSDT', 'BUY', scalping_qty, 50000.0, 0.003, 0.008)

        sl_call = executor.exchange.fapiPrivatePostOrder.call_args_list[0]
        sl_params = sl_call[0][0]

        # THE CRITICAL ASSERTION
        self.assertNotIn('closePosition', sl_params,
                         "❌ CONTAGION! closePosition would close Swing's 0.005 BTC too!")
        self.assertIn('reduceOnly', sl_params,
                      "reduceOnly must be present for horizon isolation")
        self.assertEqual(sl_params['quantity'], f"{scalping_qty:.4f}",
                         f"SL qty must be {scalping_qty} (Scalping only), not {scalping_qty + swing_qty}")

        print(f"✅ [SCENARIO] Scalping SL closes ONLY {scalping_qty} BTC")
        print(f"   Swing's {swing_qty} BTC remains UNTOUCHED")
        print(f"   Total position on Binance: {scalping_qty + swing_qty} BTC")
        print(f"   After Scalping SL: {swing_qty} BTC (Swing intact) ✅")


if __name__ == '__main__':
    print("=" * 70)
    print("🔬 HORIZON ISOLATION — Multi-Horizon Contagion Prevention")
    print("   Verifying reduceOnly protection for Scalping + Swing coexistence")
    print("=" * 70)
    unittest.main(verbosity=2)
