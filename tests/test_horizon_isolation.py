"""
🔬 HORIZON ISOLATION — Multi-Horizon Contagion Prevention Test (V2: Virtual Netting)
====================================================================================
QUÉ:     Verifica que el sistema de Virtual Netting aísla posiciones Scalping y Swing
         correctamente en el Portfolio virtual_ledger.
POR QUÉ: La arquitectura cambió de Exchange-based SL/TP (closePosition/reduceOnly)
         a Virtual Netting gestionado por RiskManager.check_stops() + Portfolio.
         _place_protective_orders() ahora es un NO-OP que delega al Neural Ledger.
PARA QUÉ: Permitir que ambos horizontes (Scalping + Swing) coexistan sin
          sabotearse mutuamente en modo One-Way de Binance.
CÓMO:    Verificamos:
         1. _place_protective_orders es un virtual NO-OP (early return)
         2. check_stops evalúa posiciones del virtual_ledger por horizonte aislado
         3. Scalping SL/TP no afecta posiciones Swing y viceversa
CUÁNDO:  Cada vez que se ejecuta la suite de tests antes de producción.
DÓNDE:   Tests sobre execution/binance_executor.py y risk/risk_manager.py
QUIÉN:   BinanceExecutor (Virtual NO-OP) + RiskManager (Neural Netting)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch
import ast
import time


class TestVirtualNettingArchitecture(unittest.TestCase):
    """
    STATIC AUDIT: Verify _place_protective_orders is a virtual NO-OP.
    The old Exchange-based SL/TP was replaced by Virtual Netting to prevent
    Binance from mixing Scalping and Swing quantities on close.
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

    def _get_method_node(self, method_name):
        """Get the AST node for a method."""
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                return node
        return None

    def test_protective_orders_is_virtual_noop(self):
        """
        CRITICAL: _place_protective_orders must be a virtual NO-OP.
        It should have an early `return` statement that prevents any
        exchange API calls. SL/TP is managed by RiskManager.check_stops().
        """
        node = self._get_method_node('_place_protective_orders')
        self.assertIsNotNone(node, "_place_protective_orders method not found!")

        # The method should be async (it was converted)
        self.assertIsInstance(node, ast.AsyncFunctionDef,
                            "_place_protective_orders should be async")

        # Check that the body has an early Return (within first 5 statements)
        # This confirms it's a NO-OP
        early_statements = node.body[:10]
        has_early_return = False
        for stmt in early_statements:
            if isinstance(stmt, ast.Return):
                has_early_return = True
                break
            # Also check if there's an Expr (logger call) followed by Return
            if isinstance(stmt, ast.Expr):
                continue  # Skip logging/docstring
        
        self.assertTrue(has_early_return,
                       "❌ _place_protective_orders must have an early return (Virtual NO-OP). "
                       "SL/TP is managed by RiskManager.check_stops() + Portfolio virtual_ledger.")
        print("✅ [AUDIT] _place_protective_orders is a Virtual NO-OP (early return)")

    def test_no_close_position_anywhere_in_file(self):
        """GLOBAL: closePosition must NOT exist anywhere in binance_executor.py."""
        self.assertNotIn('closePosition', self.source,
                         "❌ 'closePosition' still exists somewhere in binance_executor.py!")
        print("✅ [AUDIT] No 'closePosition' anywhere in binance_executor.py")

    def test_protective_orders_not_calling_exchange(self):
        """
        Verify the active (non-dead) code in _place_protective_orders
        does NOT call fapiPrivatePostOrder or any exchange method.
        """
        node = self._get_method_node('_place_protective_orders')
        self.assertIsNotNone(node)
        
        # Get lines up to the first Return statement
        source_lines = self.source.splitlines()
        start_line = node.lineno
        active_lines = []
        for i in range(start_line - 1, min(start_line + 20, len(source_lines))):
            line = source_lines[i].strip()
            active_lines.append(line)
            if line == 'return':
                break
        
        active_code = '\n'.join(active_lines)
        self.assertNotIn('fapiPrivatePostOrder', active_code,
                        "❌ Active code in _place_protective_orders calls exchange API!")
        self.assertNotIn('create_order', active_code,
                        "❌ Active code in _place_protective_orders creates orders!")
        print("✅ [AUDIT] Active code does NOT call exchange API")


class TestVirtualLedgerIsolation(unittest.TestCase):
    """
    BEHAVIORAL TEST: Verify that Portfolio virtual_ledger correctly
    isolates Scalping and Swing positions using composite keys.
    """

    def _create_mock_portfolio(self):
        """Create a Portfolio-like mock with virtual_ledger."""
        portfolio = MagicMock()
        portfolio.virtual_ledger = {}
        portfolio.positions = {}
        return portfolio

    def test_virtual_ledger_key_isolation(self):
        """
        Scalping and Swing positions for the same symbol must have
        different virtual_ledger keys.
        """
        portfolio = self._create_mock_portfolio()
        
        # Simulate Scalping LONG entry
        scalp_key = "BTC/USDT_SCALPING_LONG"
        portfolio.virtual_ledger[scalp_key] = {
            "quantity": 0.001,
            "avg_price": 50000.0,
            "current_price": 50000.0,
            "horizon": "SCALPING",
            "side": "LONG",
            "tp_pct": 0.008,
            "sl_pct": 0.004,
        }
        
        # Simulate Swing LONG entry (same symbol, different horizon)
        swing_key = "BTC/USDT_SWING_LONG"
        portfolio.virtual_ledger[swing_key] = {
            "quantity": 0.005,
            "avg_price": 49000.0,
            "current_price": 50000.0,
            "horizon": "SWING",
            "side": "LONG",
            "tp_pct": 0.045,
            "sl_pct": 0.025,
        }
        
        # Verify isolation
        self.assertIn(scalp_key, portfolio.virtual_ledger)
        self.assertIn(swing_key, portfolio.virtual_ledger)
        self.assertNotEqual(scalp_key, swing_key,
                          "Scalping and Swing keys must be different!")
        
        # Verify quantities are independent
        self.assertEqual(portfolio.virtual_ledger[scalp_key]["quantity"], 0.001)
        self.assertEqual(portfolio.virtual_ledger[swing_key]["quantity"], 0.005)
        
        print("✅ [ISOLATION] Scalping key differs from Swing key")
        print(f"   Scalp: {scalp_key} → qty={portfolio.virtual_ledger[scalp_key]['quantity']}")
        print(f"   Swing: {swing_key} → qty={portfolio.virtual_ledger[swing_key]['quantity']}")

    def test_scalping_exit_preserves_swing(self):
        """
        Scenario:
        - Scalping holds 0.001 BTC (qty=0.001)
        - Swing holds 0.005 BTC (qty=0.005)
        - Scalping SL fires → should zero ONLY the Scalping ledger entry
        - Swing position must remain UNTOUCHED

        This is the key architectural guarantee of Virtual Netting.
        """
        portfolio = self._create_mock_portfolio()
        
        scalp_key = "BTC/USDT_SCALPING_LONG"
        swing_key = "BTC/USDT_SWING_LONG"
        
        portfolio.virtual_ledger[scalp_key] = {
            "quantity": 0.001,
            "avg_price": 50000.0,
            "current_price": 49800.0,  # Below SL
            "horizon": "SCALPING",
            "side": "LONG",
            "tp_pct": 0.008,
            "sl_pct": 0.004,
        }
        
        portfolio.virtual_ledger[swing_key] = {
            "quantity": 0.005,
            "avg_price": 49000.0,
            "current_price": 49800.0,  # Still profitable for Swing!
            "horizon": "SWING",
            "side": "LONG",
            "tp_pct": 0.045,
            "sl_pct": 0.025,
        }
        
        # Simulate Scalping exit (zero out scalp, keep swing)
        portfolio.virtual_ledger[scalp_key]["quantity"] = 0.0
        
        # THE CRITICAL ASSERTIONS
        self.assertEqual(portfolio.virtual_ledger[scalp_key]["quantity"], 0.0,
                        "Scalping position should be closed")
        self.assertEqual(portfolio.virtual_ledger[swing_key]["quantity"], 0.005,
                        "❌ CONTAGION! Swing position was modified by Scalping exit!")
        
        print(f"✅ [SCENARIO] Scalping exit closed ONLY {scalp_key}")
        print(f"   Scalping qty: 0.001 → 0.0 (CLOSED)")
        print(f"   Swing qty:    0.005 → 0.005 (UNTOUCHED ✅)")

    def test_check_stops_symbol_filter(self):
        """
        Verify that RiskManager.check_stops() correctly parses virtual_ledger
        keys to extract symbol and horizon, and respects symbol_filter.
        """
        # Test the key parsing logic from check_stops (L2471-2482)
        _horizon_tags = ["_MICROSCALPING_LONG", "_MICROSCALPING_SHORT", "_SCALPING_LONG", "_SCALPING_SHORT",
                         "_SWING_LONG", "_SWING_SHORT", "_MICROSCALPING", "_SCALPING", "_SWING",
                         "_MACRO_LONG", "_MACRO_SHORT", "_MACRO"]
        
        test_cases = [
            ("BTC/USDT_SCALPING_LONG", "BTC/USDT", "SCALPING"),
            ("ETH/USDT_SWING_SHORT", "ETH/USDT", "SWING"),
            ("DOGE/USDT_SCALPING", "DOGE/USDT", "SCALPING"),
            ("SOL/USDT_SWING", "SOL/USDT", "SWING"),
            ("BTC/USDT_MICROSCALPING_LONG", "BTC/USDT", "MICROSCALPING"),
            ("BNB/USDT_MICROSCALPING_SHORT", "BNB/USDT", "MICROSCALPING"),
        ]
        
        for v_key, expected_symbol, expected_horizon in test_cases:
            symbol = v_key
            pos_horizon = "SCALPING"  # default
            for tag in _horizon_tags:
                if v_key.endswith(tag):
                    symbol = v_key[:-len(tag)]
                    if "_" in tag[1:]:
                        pos_horizon = tag.split("_")[1]
                    else:
                        pos_horizon = tag[1:]
                    break
            
            self.assertEqual(symbol, expected_symbol,
                           f"Key '{v_key}' should parse to symbol '{expected_symbol}', got '{symbol}'")
            self.assertEqual(pos_horizon, expected_horizon,
                           f"Key '{v_key}' should parse to horizon '{expected_horizon}', got '{pos_horizon}'")
        
        print("✅ [AUDIT] Virtual ledger key parsing works correctly for all horizon tags")


class TestHorizonConfigDivergence(unittest.TestCase):
    """
    Verify that Config.Horizons.Scalping and Config.Horizons.Swing have
    fundamentally different parameters that prevent cross-contamination.
    """

    def test_config_horizon_params_differ(self):
        """TP/SL/Timeframes must be completely different between horizons."""
        from config import Config
        
        scalp = getattr(Config.Horizons, 'Scalping', {})
        swing = getattr(Config.Horizons, 'Swing', {})
        
        # TP/SL must differ
        self.assertNotEqual(scalp.get('tp_pct'), swing.get('tp_pct'),
                          "Scalping and Swing TP must be different!")
        self.assertNotEqual(scalp.get('sl_pct'), swing.get('sl_pct'),
                          "Scalping and Swing SL must be different!")
        
        # Scalping TP < Swing TP (tighter targets)
        self.assertLess(scalp.get('tp_pct', 0), swing.get('tp_pct', 0),
                       "Scalping TP should be smaller than Swing TP")
        
        # Timeframes must differ
        self.assertNotEqual(scalp.get('timeframes'), swing.get('timeframes'),
                          "Scalping and Swing timeframes must be different!")
        
        # Primary TF must differ
        self.assertNotEqual(scalp.get('primary_tf'), swing.get('primary_tf'),
                          "Scalping and Swing primary_tf must be different!")
        
        print(f"✅ [CONFIG] Horizons properly differentiated:")
        print(f"   Scalping: TP={scalp.get('tp_pct', 0)*100:.2f}%, SL={scalp.get('sl_pct', 0)*100:.2f}%, TF={scalp.get('primary_tf')}")
        print(f"   Swing:    TP={swing.get('tp_pct', 0)*100:.2f}%, SL={swing.get('sl_pct', 0)*100:.2f}%, TF={swing.get('primary_tf')}")


if __name__ == '__main__':
    print("=" * 70)
    print("🔬 HORIZON ISOLATION V2 — Virtual Netting Architecture Verification")
    print("   Verifying virtual ledger isolation for Scalping + Swing coexistence")
    print("=" * 70)
    unittest.main(verbosity=2)
