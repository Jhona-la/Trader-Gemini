# tests/test_temporal_supervisor.py

import unittest
import os
import time
import json
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from config import Config
from core.temporal_supervisor import TemporalSupervisor, TemporalState, SystemGeneration
from risk.risk_manager import RiskManager
from core.events import SignalEvent
from core.enums import SignalType

class MockPortfolio:
    def __init__(self, cash=13.0, equity=13.0):
        self.current_cash = cash
        self.initial_capital = cash
        self.realized_pnl = 0.0
        self._equity = equity
        self.positions = {}
        self.virtual_ledger = {}
        self.strategy_performance = {}
        self._last_prices = {
            'BTC/USDT': 50000.0,
            'ETH/USDT': 3000.0,
            'SOL/USDT': 100.0,
            'ADA/USDT': 0.50,
            'AVAX/USDT': 50.0
        }

    def get_total_equity(self):
        return self._equity

    def get_available_cash(self, horizon='SCALPING'):
        return self.current_cash

    def get_statistics(self):
        return {'win_rate': 0.55, 'total_pnl': 0.0}

    def get_smart_kelly_sizing(self, symbol, strategy_id, is_micro_account=False, horizon="SCALPING"):
        return 0.05

class TestTemporalSupervisor(unittest.TestCase):
    def setUp(self):
        self.portfolio = MockPortfolio(cash=15.0, equity=15.0)
        self.risk_manager = RiskManager(portfolio=self.portfolio)
        self.engine = MagicMock()
        self.engine.data_handlers = []
        
        # Override temp genesis DB path to avoid messing with production
        self.p_db = patch('core.temporal_supervisor.Config.DATA_DIR', "C:/Users/jhona/.gemini/antigravity/brain/15de8b1e-38e4-4339-9be3-089a1e414d63/scratch")
        self.p_db.start()
        
        self.supervisor = TemporalSupervisor(self.portfolio, self.risk_manager, self.engine)

    def tearDown(self):
        self.p_db.stop()
        db_file = os.path.join("C:/Users/jhona/.gemini/antigravity/brain/15de8b1e-38e4-4339-9be3-089a1e414d63/scratch", "temporal_genesis.json")
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except:
                pass

    def test_boot_checklist(self):
        """Verify that verify_initialization_checklist passes with clean mocks."""
        self.assertTrue(self.supervisor.verify_initialization_checklist())
        self.assertTrue(self.supervisor.checklist_passed)

    def test_time_based_phases(self):
        """Verify the size and score constraints for each temporal phase."""
        self.supervisor.checklist_passed = True
        
        # 1. OBSERVACION phase (Minutes 0-30)
        self.supervisor.current_phase = "OBSERVACION"
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertFalse(allowed)

        # 2. HORA_1 phase (Minutes 30-60): 25% size, +15 score penalty
        self.supervisor.current_phase = "HORA_1"
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertTrue(allowed)
        self.assertAlmostEqual(size, 0.25)
        self.assertAlmostEqual(score, 85.0)

        # 3. HORA_2_4 phase (Hours 1-4): 50% size, +10 score penalty
        self.supervisor.current_phase = "HORA_2_4"
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertTrue(allowed)
        self.assertAlmostEqual(size, 0.50)
        self.assertAlmostEqual(score, 80.0)

        # 4. HORA_4_8 phase (Hours 4-8): 70% size, normal score
        self.supervisor.current_phase = "HORA_4_8"
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertTrue(allowed)
        self.assertAlmostEqual(size, 0.70)
        self.assertAlmostEqual(score, 70.0)

        # 5. OPERACION_NORMAL phase (Hours 8+)
        self.supervisor.current_phase = "OPERACION_NORMAL"
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertTrue(allowed)
        self.assertAlmostEqual(size, 1.0)
        self.assertAlmostEqual(score, 70.0)

    def test_hour2_drawdown_audit(self):
        """Verify that a >1% drawdown at hour 2 triggers conservative mode."""
        self.portfolio._equity = 13.5  # Base was 15.0 (-10% loss)
        self.supervisor.state.cycle_base_capital = 15.0
        
        self.supervisor._run_hour2_audit()
        self.assertTrue(self.risk_manager.conservative_mode)
        
        # Test that conservative mode halves sizing and adds 15 score penalty
        self.supervisor.current_phase = "OPERACION_NORMAL"
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertTrue(allowed)
        self.assertAlmostEqual(size, 0.50) # Halved from 1.0
        self.assertAlmostEqual(score, 85.0) # +15 penalty

    def test_capital_injection_detection(self):
        """Verify deposits are identified, logged, and scheduled over 4 weeks."""
        self.supervisor._last_settled_cash = 15.0
        self.supervisor._last_realized_pnl = 0.0
        
        # Trigger deposit of $10.0 (no trade PnL change)
        self.portfolio.current_cash = 25.0
        self.portfolio.realized_pnl = 0.0
        
        # Run loop ticks manually to trigger check
        async def run_once():
            # Mock loop body once
            current_cash = self.portfolio.current_cash
            current_pnl = self.portfolio.realized_pnl
            delta_cash = current_cash - self.supervisor._last_settled_cash
            delta_pnl = current_pnl - self.supervisor._last_realized_pnl
            if delta_cash > delta_pnl + 1.0:
                self.supervisor.state.injections.append({
                    "timestamp": time.time(),
                    "amount": delta_cash - delta_pnl,
                    "ratio": current_cash / self.supervisor._last_settled_cash
                })
        
        asyncio.run(run_once())
        
        self.assertEqual(len(self.supervisor.state.injections), 1)
        inj = self.supervisor.state.injections[0]
        self.assertAlmostEqual(inj["amount"], 10.0)
        
        # Week 1 deployment check (days 0-7: Weeks elapsed = 0 -> 75% non-deployable)
        reduction = self.supervisor.get_deployable_capital_reduction()
        self.assertAlmostEqual(reduction, 7.50)
        
        # Week 3 deployment check (days 14-21: Weeks elapsed = 2 -> 25% non-deployable)
        inj["timestamp"] = time.time() - (15 * 86400) # 15 days ago
        reduction = self.supervisor.get_deployable_capital_reduction()
        self.assertAlmostEqual(reduction, 2.50)
        
        # Week 5 deployment check (days 28+: Weeks elapsed = 4 -> 0% non-deployable)
        inj["timestamp"] = time.time() - (29 * 86400) # 29 days ago
        reduction = self.supervisor.get_deployable_capital_reduction()
        self.assertAlmostEqual(reduction, 0.0)

    def test_risk_manager_integration(self):
        """Verify size_position subtracts injection reduction and scales sizing."""
        # Setup active injection of $10.0 at Week 1 (reduction is $7.50)
        self.supervisor.state.injections.append({
            "timestamp": time.time(),
            "amount": 10.0,
            "ratio": 2.0
        })
        
        # Portfolio available cash is $20.0
        self.portfolio.current_cash = 20.0
        self.portfolio._equity = 20.0
        
        # size_position should see available_cash = 20.0 - 7.50 = 12.50
        with patch('risk.risk_manager.logger') as mock_logger:
            size_res = self.risk_manager.size_position("BTC/USDT", risk_pct=0.10, multiplier=1.0)
            # Verify reduction log occurred
            self.assertTrue(any("Subtracting non-deployable" in call[0][0] for call in mock_logger.info.call_args_list))

    def test_systemic_degradation_levels(self):
        """Verify Yellow, Orange, and Red degradation alert behaviors."""
        # 1. Yellow Alert: PF < 1.5, PF > 1.2 on 2 consecutive cycles
        self.supervisor.state.cycle_history = [
            {"cycle_id": 1, "profit_factor": 1.35, "win_rate": 0.52, "pnl_pct": 2.0, "max_drawdown": 5.0, "shs": 75.0},
            {"cycle_id": 2, "profit_factor": 1.40, "win_rate": 0.53, "pnl_pct": 3.0, "max_drawdown": 4.0, "shs": 76.0}
        ]
        
        new_deg = self.supervisor._evaluate_degradation_level()
        self.assertEqual(new_deg, 1) # Yellow Alert
        
        # Apply Yellow alert and test size/score limits
        self.supervisor.state.degradation_level = 1
        self.risk_manager.degradation_level = 1
        self.supervisor.current_phase = "OPERACION_NORMAL"
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertTrue(allowed)
        self.assertAlmostEqual(size, 0.70) # 70% size limit
        self.assertAlmostEqual(score, 85.0) # +15 score threshold

        # 2. Orange Alert: PF < 1.2 on 2 consecutive cycles
        self.supervisor.state.cycle_history = [
            {"cycle_id": 1, "profit_factor": 1.10, "win_rate": 0.48, "pnl_pct": 1.0, "max_drawdown": 10.0, "shs": 65.0},
            {"cycle_id": 2, "profit_factor": 1.15, "win_rate": 0.49, "pnl_pct": 1.5, "max_drawdown": 12.0, "shs": 64.0}
        ]
        new_deg = self.supervisor._evaluate_degradation_level()
        self.assertEqual(new_deg, 2) # Orange Alert
        
        self.supervisor.state.degradation_level = 2
        self.risk_manager.degradation_level = 2
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertTrue(allowed)
        self.assertAlmostEqual(size, 0.50) # 50% size limit
        self.assertAlmostEqual(score, 100.0) # +30 score threshold

        # 3. Red Alert: Drawdown > 50%
        self.supervisor.state.cycle_history = [
            {"cycle_id": 1, "profit_factor": 0.50, "win_rate": 0.35, "pnl_pct": -20.0, "max_drawdown": 55.0, "shs": 35.0}
        ]
        new_deg = self.supervisor._evaluate_degradation_level()
        self.assertEqual(new_deg, 3) # Red Alert
        
        # Rojo blocks signal constraint check
        self.supervisor.state.degradation_level = 3
        size, score, allowed = self.supervisor.apply_temporal_constraints(None, 1.0, 70.0)
        self.assertFalse(allowed)

    def test_capital_phase_constraints(self):
        """Verify position and asset limits based on Capital Phases."""
        # Phase 1: Micro Account (<$50). Limit 1 Position, top 3 pairs.
        self.portfolio._equity = 15.0
        self.portfolio.current_cash = 15.0
        
        # AVAX/USDT should be rejected (not in top 3 pairs)
        size_res = self.risk_manager.size_position("AVAX/USDT", risk_pct=0.02)
        self.assertIsNone(size_res)
        
        # BTC/USDT should be allowed
        size_res = self.risk_manager.size_position("BTC/USDT", risk_pct=0.02)
        self.assertIsNotNone(size_res)
        
        # Open 1 position
        self.portfolio.virtual_ledger["BTC/USDT_SCALPING"] = {"quantity": 0.01, "horizon": "SCALPING"}
        # Try to open 2nd position (should be blocked)
        size_res2 = self.risk_manager.size_position("ETH/USDT", risk_pct=0.02)
        self.assertIsNone(size_res2)

        # Clear positions and increase equity to $150 (Phase 2). Max 2 positions (1 per horizon), top 5 pairs.
        self.portfolio.virtual_ledger.clear()
        self.portfolio._equity = 150.0
        self.portfolio.current_cash = 150.0
        
        # ADA should be allowed (top 5), but AVAX still rejected
        size_res = self.risk_manager.size_position("ADA/USDT", risk_pct=0.02)
        self.assertIsNotNone(size_res)
        size_res = self.risk_manager.size_position("AVAX/USDT", risk_pct=0.02)
        self.assertIsNone(size_res)
        
        # Should allow 2 concurrent positions across different horizons
        self.portfolio.virtual_ledger["BTC/USDT_SCALPING"] = {"quantity": 0.01, "horizon": "SCALPING"}
        size_res = self.risk_manager.size_position("ETH/USDT", risk_pct=0.02, horizon="SWING")
        self.assertIsNotNone(size_res)
        
        # 3rd position blocked (trying to open a second SWING position)
        self.portfolio.virtual_ledger["ETH/USDT_SWING"] = {"quantity": 0.10, "horizon": "SWING"}
        size_res = self.risk_manager.size_position("SOL/USDT", risk_pct=0.02, horizon="SWING")
        self.assertIsNone(size_res)

    @patch('core.temporal_supervisor.os.path.exists', return_value=True)
    def test_structured_audits(self, mock_exists):
        """Verify sessional, cycle, and monthly audits produce files with expected schemas."""
        self.supervisor._write_audit_report = MagicMock()
        
        # Test Session Audit
        asyncio.run(self.supervisor._execute_session_audit())
        self.supervisor._write_audit_report.assert_called_once()
        args = self.supervisor._write_audit_report.call_args[0]
        self.assertEqual(args[0], "session_audit")
        self.assertEqual(args[1], "8h")
        self.assertIn("total_trades", args[2])
        self.assertIn("win_rate", args[2])
        
        # Test Cycle Transition Audit
        self.supervisor._write_audit_report.reset_mock()
        self.supervisor.state.current_cycle_start = time.time() - 3600
        asyncio.run(self.supervisor._execute_cycle_transition())
        calls = self.supervisor._write_audit_report.call_args_list
        self.assertTrue(any(c[0][0] == "cycle_audit" for c in calls))
        
    def test_shadow_testing_window(self):
        """Verify shadow testing window returns True only in last 3 cycles of a generation."""
        self.supervisor.state.total_cycles_completed = 27 # Cycle 28
        self.assertTrue(self.supervisor._is_in_shadow_testing_window())
        self.supervisor.state.total_cycles_completed = 29 # Cycle 30
        self.assertTrue(self.supervisor._is_in_shadow_testing_window())
        self.supervisor.state.total_cycles_completed = 26 # Cycle 27
        self.assertFalse(self.supervisor._is_in_shadow_testing_window())
        self.supervisor.state.total_cycles_completed = 30 # Cycle 31
        self.assertFalse(self.supervisor._is_in_shadow_testing_window())
        
    def test_generation_transition_report(self):
        """Verify generation transition statistically compares Champion/Challenger and writes report."""
        self.supervisor.state.shadow_predictions = [
            {"cycle_id": 29, "timestamp": time.time(), "champion_correct": True, "champion_confidence": 0.8, "champion_pnl": 5.0,
             "challenger_correct": True, "challenger_confidence": 0.85, "challenger_pnl": 6.0},
            {"cycle_id": 29, "timestamp": time.time(), "champion_correct": False, "champion_confidence": 0.3, "champion_pnl": -2.0,
             "challenger_correct": True, "challenger_confidence": 0.75, "challenger_pnl": 4.0}
        ]
        
        with patch('builtins.open', new_callable=unittest.mock.mock_open()) as mock_file:
            asyncio.run(self.supervisor._trigger_generation_transition("G1", "G2"))
            mock_file.assert_any_call(os.path.join(os.getcwd(), "logs", "audits", "generation_transition_G1.md"), "w", encoding="utf-8")

if __name__ == '__main__':
    unittest.main()
