# tests/test_senior_auditor.py

import unittest
import os
import time
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from core.events import SignalEvent, SignalType
from core.senior_auditor import SeniorAuditor, STRATEGY_DNA
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager

class MockDataProvider:
    def __init__(self, last_bar_time=None):
        self.last_bar_time = last_bar_time or time.time()

    def get_latest_bars(self, symbol, n=1):
        return [{"timestamp": self.last_bar_time, "close": 50000.0}]

class MockPortfolio:
    def __init__(self, cash=13.0, equity=13.0):
        self.current_cash = cash
        self._equity = equity
        self.positions = {}
        self.virtual_ledger = {}
        self.strategy_performance = {}
        self._last_prices = {'BTC/USDT': 50000.0, 'ETH/USDT': 3000.0, 'SOL/USDT': 100.0}

    def get_total_equity(self):
        return self._equity

    def get_available_cash(self, horizon='SCALPING'):
        return self.current_cash

    def get_horizon_position(self, symbol, horizon):
        return self.virtual_ledger.get(f"{symbol}_{horizon}_LONG") or self.virtual_ledger.get(f"{symbol}_{horizon}_SHORT")

class TestSeniorAuditor(unittest.TestCase):
    def setUp(self):
        SeniorAuditor._instance = None
        self.scratch_dir = "C:/Users/jhona/.gemini/antigravity/brain/15de8b1e-38e4-4339-9be3-089a1e414d63/scratch"
        self.p_db = patch('core.senior_auditor.Config.DATA_DIR', self.scratch_dir)
        self.p_db.start()
        
        self.auditor = SeniorAuditor()

    def tearDown(self):
        self.p_db.stop()
        SeniorAuditor._instance = None
        chronicle_file = os.path.join(self.scratch_dir, "audit_chronicle.json")
        if os.path.exists(chronicle_file):
            try:
                os.remove(chronicle_file)
            except:
                pass

    def test_strategy_dna_lookup(self):
        """Verify that all 11 strategies are defined in the DNA registry."""
        strategies = ["TFTF", "OB_RETEST", "LCA", "MRBB", "WYCKOFF", "VBA", "MBV", "FRA", "SC", "STATARB", "OCS"]
        for strat in strategies:
            self.assertIn(strat, STRATEGY_DNA)
            dna = STRATEGY_DNA[strat]
            self.assertIn("nombre", dna)
            self.assertIn("TESIS_DE_APERTURA", dna)
            self.assertIn("INDICADORES_CRITICOS_DE_LA_TESIS", dna)
            self.assertIn("CONDICIONES_DE_INVALIDACION", dna)

    def test_map_strategy_name(self):
        """Test strategy name mapping to DNA keys."""
        self.assertEqual(self.auditor._map_strategy_name("TFTF_V2"), "TFTF")
        self.assertEqual(self.auditor._map_strategy_name("[SCL]_OB_RETEST_SMART"), "OB_RETEST")
        self.assertEqual(self.auditor._map_strategy_name("LCA_CASCADE_HFT"), "LCA")
        self.assertEqual(self.auditor._map_strategy_name("MRBB_BOLLINGER"), "MRBB")
        self.assertEqual(self.auditor._map_strategy_name("WYCKOFF_SPRING"), "WYCKOFF")
        self.assertEqual(self.auditor._map_strategy_name("VOLATILITY_BREAKOUT_ATR"), "VBA")

    @patch('core.global_state.global_state.market_regime', 'TENDENCIAL_ALCISTA')
    def test_opening_audit_approved(self):
        """Verify opening audit approvals for trend following strategies in tendencial regime."""
        portfolio = MockPortfolio()
        event = SignalEvent(
            strategy_id="TFTF_V7",
            symbol="BTC/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type=SignalType.LONG,
            strength=0.85, # Above 0.72 BTC threshold
            horizon="SCALPING",
            metadata={"pullback_volume_ratio": 0.45}
        )
        approved, reason = self.auditor.verify_opening_audit(event, portfolio)
        self.assertTrue(approved)
        self.assertEqual(reason, "APPROVED")

    @patch('core.global_state.global_state.market_regime', 'TENDENCIAL_ALCISTA')
    def test_opening_audit_regime_mismatch(self):
        """Verify opening audit blocks mean reversion during trending regime."""
        portfolio = MockPortfolio()
        event = SignalEvent(
            strategy_id="MRBB_V7",
            symbol="BTC/USDT",
            datetime=datetime.now(timezone.utc),
            signal_type=SignalType.LONG,
            strength=0.90,
            horizon="SCALPING",
            metadata={"adx": 20}
        )
        approved, reason = self.auditor.verify_opening_audit(event, portfolio)
        self.assertFalse(approved)
        self.assertIn("FAIL_ACS: Tesis requires RANGE/LATERAL regime", reason)

    @patch('core.global_state.global_state.market_regime', 'TENDENCIAL_ALCISTA')
    def test_opening_audit_confidence_under_threshold(self):
        """Verify opening audit blocks signals below asset-specific confidence threshold."""
        portfolio = MockPortfolio()
        event = SignalEvent(
            strategy_id="TFTF_V7",
            symbol="BTC/USDT", # BTC has 0.58 threshold
            datetime=datetime.now(timezone.utc),
            signal_type=SignalType.LONG,
            strength=0.55, # Under 0.58
            horizon="SCALPING",
            metadata={"pullback_volume_ratio": 0.45}
        )
        approved, reason = self.auditor.verify_opening_audit(event, portfolio)
        self.assertFalse(approved)
        self.assertIn("FAIL_AEA: Signal confidence", reason)

    def test_tracking_audit_lag_levels(self):
        """Verify ACI tracking lag ceguera checks and degradation levels."""
        now = datetime.now(timezone.utc)
        
        # 1. Healthy state (lag 5 seconds)
        pos = {"symbol": "BTC/USDT", "horizon": "SCALPING", "opener_strategy_id": "TFTF", "last_feed_time": now.timestamp() - 5.0}
        dp = MockDataProvider(last_bar_time=now.timestamp() - 5.0)
        level, reason = self.auditor.verify_tracking_audit(pos, dp, 50000.0, now)
        self.assertEqual(level, 0)
        self.assertEqual(reason, "OK")
        self.assertIn("tracking_heartbeats", pos)
        self.assertEqual(pos["tracking_heartbeats"][-1]["status"], "VALID")

        # 2. Level 1: Stale warning (lag 50 seconds > 45s limit)
        pos = {"symbol": "BTC/USDT", "horizon": "SCALPING", "opener_strategy_id": "TFTF", "last_feed_time": now.timestamp() - 50.0}
        level, reason = self.auditor.verify_tracking_audit(pos, dp, 50000.0, now)
        self.assertEqual(level, 1)
        self.assertIn("CEGUERA_PARCIAL", reason)

        # 3. Level 2: Disconnected (lag 150 seconds > 3 * 45s limit)
        pos = {"symbol": "BTC/USDT", "horizon": "SCALPING", "opener_strategy_id": "TFTF", "last_feed_time": now.timestamp() - 150.0}
        level, reason = self.auditor.verify_tracking_audit(pos, dp, 50000.0, now)
        self.assertEqual(level, 2)
        self.assertIn("CEGUERA_CRÍTICA", reason)

        # 4. Level 3: Outage / panic (lag 500 seconds > 10 * 45s limit)
        pos = {"symbol": "BTC/USDT", "horizon": "SCALPING", "opener_strategy_id": "TFTF", "last_feed_time": now.timestamp() - 500.0}
        level, reason = self.auditor.verify_tracking_audit(pos, dp, 50000.0, now)
        self.assertEqual(level, 3)
        self.assertIn("CEGUERA_TOTAL_EMERGENCY", reason)

    def test_closing_audit_invalidations(self):
        """Verify that strategy-specific invalidation rules trigger closures."""
        now = datetime.now(timezone.utc)
        
        # TFTF invalidates when ADX < 20
        pos = {"symbol": "BTC/USDT", "horizon": "SCALPING", "opener_strategy_id": "TFTF", "quantity": 0.05, "avg_price": 50000.0, "last_adx_value": 18}
        should_close, reason = self.auditor.verify_closing_audit(pos, 50000.0, None, now)
        self.assertTrue(should_close)
        self.assertEqual(reason, "INVALIDATION_TFTF_ADX_DROPPED_BELOW_20")

        # MRBB invalidates when ADX >= 25 (trending breakout)
        pos = {"symbol": "BTC/USDT", "horizon": "SCALPING", "opener_strategy_id": "MRBB", "quantity": 0.05, "avg_price": 50000.0, "last_adx_value": 26}
        should_close, reason = self.auditor.verify_closing_audit(pos, 50000.0, None, now)
        self.assertTrue(should_close)
        self.assertEqual(reason, "INVALIDATION_MRBB_MARKET_TRENDED_ADX_ABOVE_25")

        # LCA invalidates on spike decay (held > 90s without profit)
        pos = {"symbol": "BTC/USDT", "horizon": "SCALPING", "opener_strategy_id": "LCA", "quantity": 0.05, "avg_price": 50000.0, "entry_time": now.timestamp() - 100.0}
        should_close, reason = self.auditor.verify_closing_audit(pos, 50000.0, None, now)
        self.assertTrue(should_close)
        self.assertEqual(reason, "INVALIDATION_LCA_SPIKE_DECAY_EXHAUSTION")

    def test_chronicle_persistence(self):
        """Verify that trade events write cleanly to audit_chronicle.json."""
        trade_id = "TRD-TEST-12345"
        self.auditor.log_trade_lifecycle(trade_id, "ENTRY", {"price": 50000.0, "qty": 0.05})
        self.auditor.log_trade_lifecycle(trade_id, "EXIT", {"price": 50500.0, "gross_pnl": 25.0})
        
        chronicle_file = os.path.join(self.scratch_dir, "audit_chronicle.json")
        self.assertTrue(os.path.exists(chronicle_file))
        
        with open(chronicle_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.assertEqual(len(data), 1)
        entry = data[0]
        self.assertEqual(entry["trade_id"], trade_id)
        self.assertEqual(len(entry["events"]), 2)
        self.assertEqual(entry["events"][0]["action"], "ENTRY")
        self.assertEqual(entry["events"][1]["action"], "EXIT")

if __name__ == '__main__':
    unittest.main()
