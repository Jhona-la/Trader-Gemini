"""
🧪 VALIDATION SCRIPT: AEGIS-ULTRA Telemetry (Phase 17)
QUÉ: Verifies Trade ID tracing and ML Gradient Logging.
POR QUÉ: Ensure forensic auditability and ML health monitoring.
"""

import sys
import os
import unittest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.events import SignalEvent, OrderEvent, FillEvent, SignalType, OrderType, OrderSide
from core.online_learning import OnlineLearner
import utils.wandb_tracker

class TestTelemetry(unittest.TestCase):
    
    def test_trade_id_propagation(self):
        """Test that trade_id exists in all key events."""
        trade_id = "test-uuid-12345"
        
        # 1. Signal
        sig = SignalEvent(
            strategy_id="TEST_STRAT",
            symbol="BTC/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type=SignalType.LONG,
            trade_id=trade_id
        )
        self.assertEqual(sig.trade_id, trade_id)
        
        # 2. Order
        ord = OrderEvent(
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            quantity=1.0,
            direction=OrderSide.BUY,
            trade_id=trade_id
        )
        self.assertEqual(ord.trade_id, trade_id)
        
        # 3. Fill
        fill = FillEvent(
            timeindex=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            exchange="BINANCE",
            quantity=1.0,
            direction=OrderSide.BUY,
            fill_cost=50000.0,
            trade_id=trade_id
        )
        self.assertEqual(fill.trade_id, trade_id)
        print("✅ PASS: Trade ID Propagation Verified")

    @patch('core.online_learning.wandb_tracker')
    def test_gradient_logging(self, mock_tracker):
        """Test that gradients are calculated and logged."""
        learner = OnlineLearner(learning_rate=0.1)
        
        weights = np.array([0.5, 0.5])
        inputs = np.array([1.0, 1.0])
        target = 1.0
        prediction = 0.0 # Error = 1.0
        
        # Update
        new_weights = learner.update_weights(weights, inputs, target, prediction)
        
        # Expected Grad: 0.1 * 1.0 * [1, 1] = [0.1, 0.1]
        # Norm = sqrt(0.01 + 0.01) = 0.1414
        
        # Check calls
        mock_tracker.log_metric.assert_called()
        args = mock_tracker.log_metric.call_args_list[0]
        # args[0] is positional args tuple: ("ml/gradient_norm", val)
        self.assertEqual(args[0][0], "ml/gradient_norm")
        self.assertAlmostEqual(args[0][1], 0.1414, places=2)
        print(f"✅ PASS: Gradient Logged (Norm={args[0][1]:.4f})")

    @patch('core.online_learning.logger')
    @patch('core.online_learning.wandb_tracker')
    def test_exploding_gradient(self, mock_tracker, mock_logger):
        """Test detection of exploding gradients."""
        learner = OnlineLearner(learning_rate=100.0) # Huge LR
        
        weights = np.array([0.5, 0.5])
        inputs = np.array([100.0, 100.0]) # Huge Inputs
        target = 1000.0
        prediction = 0.0 
        
        # Update
        # Error = 1000
        # Delta = 100 * 1000 * 100 = 10,000,000 -> Explosion
        
        learner.update_weights(weights, inputs, target, prediction)
        
        # Check for explosion log
        mock_tracker.log_metric.assert_any_call("ml/gradient_exploded", 1.0)
        # Verify logger warning
        mock_logger.warning.assert_called()
        msg = mock_logger.warning.call_args[0][0]
        self.assertIn("EXPLODING GRADIENT", msg)
        print("✅ PASS: Exploding Gradient Detected")

if __name__ == '__main__':
    unittest.main()
