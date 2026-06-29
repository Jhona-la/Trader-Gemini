"""
Unit Tests - Telemetry Display & Efficacy Tracker (Phase 99)
============================================================
"""
import pytest
import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════
#  TEST 1: TelemetryDisplay
# ═══════════════════════════════════════════════════════════

class MockPortfolioIdle:
    """Mock portfolio with no active positions."""
    def get_atomic_snapshot(self):
        return {
            'positions': {
                'BTC/USDT': {'quantity': 0, 'avg_price': 0},
                'ETH/USDT': {'quantity': 0, 'avg_price': 0},
            },
            'total_equity': 15.0,
            'cash': 15.0,
        }


class MockPortfolioActive:
    """Mock portfolio with active positions."""
    def get_atomic_snapshot(self):
        return {
            'positions': {
                'BTC/USDT': {
                    'quantity': 0.001,
                    'avg_price': 100000.0,
                    'current_price': 100500.0,
                    'tp_price': 101000.0,
                    'sl_price': 99500.0,
                },
                'ETH/USDT': {
                    'quantity': -0.05,
                    'avg_price': 3000.0,
                    'current_price': 2950.0,
                    'tp_price': 2900.0,
                    'sl_price': 3100.0,
                },
                'SOL/USDT': {
                    'quantity': 0,
                    'avg_price': 0,
                },
            },
            'total_equity': 16.50,
            'cash': 10.00,
        }


class TestTelemetryDisplay:
    """Tests for TelemetryDisplay rendering."""
    
    def test_render_idle(self):
        """Should render compact status when no positions are open."""
        from utils.telemetry import TelemetryDisplay
        td = TelemetryDisplay()
        result = td.render(MockPortfolioIdle())
        
        assert "FLEET TELEMETRY" in result
        assert "No active positions" in result
        assert "15.00" in result
        print(f"✅ Idle render:\n{result}")
    
    def test_render_active_positions(self):
        """Should render table with all active position columns."""
        from utils.telemetry import TelemetryDisplay
        td = TelemetryDisplay()
        result = td.render(MockPortfolioActive())
        
        assert "FLEET TELEMETRY" in result
        assert "BTC/USDT" in result or "BTC/" in result
        assert "ETH/USDT" in result or "ETH/" in result
        assert "LONG" in result
        assert "SHORT" in result
        assert "SOL" not in result  # Should NOT appear (qty == 0)
        assert "PnL%" in result
        assert "Gap TP%" in result
        print(f"✅ Active render:\n{result}")
    
    def test_pnl_calculation(self):
        """PnL% should be positive for profitable LONG."""
        from utils.telemetry import TelemetryDisplay
        td = TelemetryDisplay()
        result = td.render(MockPortfolioActive())
        
        # BTC LONG: (100500 - 100000) / 100000 = +0.50%
        assert "+0.50%" in result
        print("✅ PnL calculation correct")
    
    def test_gap_calculation(self):
        """Gap TP% should show distance to take profit."""
        from utils.telemetry import TelemetryDisplay
        td = TelemetryDisplay()
        result = td.render(MockPortfolioActive())
        
        # BTC LONG: (101000 - 100500) / 100500 ≈ 0.50%
        assert "0.50%" in result
        print("✅ Gap calculation present")
    
    def test_ttt_first_sample(self):
        """TTT should show '—' on first render (only 1 price sample)."""
        from utils.telemetry import TelemetryDisplay
        td = TelemetryDisplay()
        result = td.render(MockPortfolioActive())
        
        # First render = only 1 price sample → TTT = '—'
        assert "—" in result
        print("✅ TTT shows dash on first sample")
    
    def test_clear_symbol(self):
        """clear_symbol should remove price history."""
        from utils.telemetry import TelemetryDisplay
        td = TelemetryDisplay()
        td._price_history['BTC/USDT'] = [(time.time(), 100000)]
        td.clear_symbol('BTC/USDT')
        assert 'BTC/USDT' not in td._price_history
        print("✅ clear_symbol works")


# ═══════════════════════════════════════════════════════════
#  TEST 2: EfficacyTracker
# ═══════════════════════════════════════════════════════════

class TestEfficacyTracker:
    """Tests for EfficacyTracker calculations."""
    
    def test_perfect_close_at_tp(self):
        """Efficacy ratio should be 1.0 when closing at TP."""
        from utils.efficacy_tracker import EfficacyTracker
        et = EfficacyTracker(log_path="dashboard/data/test_efficacy.csv")
        
        result = et.record_manual_close(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=100000.0,
            manual_close_price=101000.0,  # Exactly at TP
            tp_price=101000.0,
            sl_price=99500.0,
            strategy_id="test"
        )
        
        assert result['efficacy_ratio'] == 1.0
        assert result['premature_exit_cost'] == 0.0
        print(f"✅ Perfect close: {result}")
    
    def test_premature_close(self):
        """Efficacy ratio < 1.0 when closing before TP."""
        from utils.efficacy_tracker import EfficacyTracker
        et = EfficacyTracker(log_path="dashboard/data/test_efficacy.csv")
        
        result = et.record_manual_close(
            symbol="ETH/USDT",
            side="LONG",
            entry_price=3000.0,
            manual_close_price=3050.0,  # Halfway to TP
            tp_price=3100.0,
            sl_price=2900.0,
            strategy_id="test"
        )
        
        assert result['efficacy_ratio'] == 0.5
        assert result['premature_exit_cost'] == 50.0
        print(f"✅ Premature close: {result}")
    
    def test_beyond_tp(self):
        """Efficacy ratio > 1.0 when closing beyond TP."""
        from utils.efficacy_tracker import EfficacyTracker
        et = EfficacyTracker(log_path="dashboard/data/test_efficacy.csv")
        
        result = et.record_manual_close(
            symbol="SOL/USDT",
            side="LONG",
            entry_price=200.0,
            manual_close_price=220.0,  # Beyond TP of 210
            tp_price=210.0,
            sl_price=190.0,
            strategy_id="test"
        )
        
        assert result['efficacy_ratio'] == 2.0  # (220-200)/(210-200) = 2.0
        assert result['premature_exit_cost'] == -10.0  # Negative = earned MORE
        print(f"✅ Beyond TP: {result}")
    
    def test_loss_close(self):
        """Efficacy ratio < 0 when closing at a loss."""
        from utils.efficacy_tracker import EfficacyTracker
        et = EfficacyTracker(log_path="dashboard/data/test_efficacy.csv")
        
        result = et.record_manual_close(
            symbol="BTC/USDT",
            side="LONG",
            entry_price=100000.0,
            manual_close_price=99000.0,  # Loss $1000
            tp_price=101000.0,
            sl_price=99500.0,
            strategy_id="test"
        )
        
        assert result['efficacy_ratio'] == -1.0  # (-1000)/(1000) = -1.0
        assert result['pnl_pct'] < 0
        print(f"✅ Loss close: {result}")
    
    def test_short_side(self):
        """Efficacy should calculate correctly for SHORT positions."""
        from utils.efficacy_tracker import EfficacyTracker
        et = EfficacyTracker(log_path="dashboard/data/test_efficacy.csv")
        
        result = et.record_manual_close(
            symbol="ETH/USDT",
            side="SHORT",
            entry_price=3000.0,
            manual_close_price=2950.0,  # Halfway to TP
            tp_price=2900.0,
            sl_price=3100.0,
            strategy_id="test"
        )
        
        # SHORT: tp_distance = 3000 - 2900 = 100
        # actual_distance = 3000 - 2950 = 50
        # ratio = 50/100 = 0.5
        assert result['efficacy_ratio'] == 0.5
        assert result['pnl_pct'] > 0  # Profitable short
        print(f"✅ Short close: {result}")
    
    def test_rl_outcome_mapping(self):
        """RL outcome should map efficacy to [0, 1] correctly."""
        from utils.efficacy_tracker import EfficacyTracker
        et = EfficacyTracker(log_path="dashboard/data/test_efficacy.csv")
        
        assert et.get_rl_outcome(1.0) == 1.0    # Perfect
        assert et.get_rl_outcome(0.9) == 1.0    # >= 0.8 → 1.0
        assert et.get_rl_outcome(0.5) == 0.5    # Proportional
        assert et.get_rl_outcome(-0.5) == 0.0   # Loss → 0.0
        print("✅ RL outcome mapping correct")
    
    def test_csv_persistence(self, tmp_path):
        """Records should be persisted to CSV."""
        from utils.efficacy_tracker import EfficacyTracker
        csv_path = str(tmp_path / "test_eff.csv")
        et = EfficacyTracker(log_path=csv_path)
        
        et.record_manual_close(
            symbol="TEST/USDT", side="LONG",
            entry_price=100, manual_close_price=110,
            tp_price=120, sl_price=90,
        )
        
        assert os.path.exists(csv_path)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 record
            assert "TEST/USDT" in lines[1]
        print(f"✅ CSV persistence works: {csv_path}")
    
    def test_summary(self):
        """Summary should aggregate statistics."""
        from utils.efficacy_tracker import EfficacyTracker
        et = EfficacyTracker(log_path="dashboard/data/test_efficacy.csv")
        
        et.record_manual_close("A", "LONG", 100, 110, 120, 90)
        et.record_manual_close("B", "LONG", 100, 105, 120, 90)
        
        summary = et.get_summary()
        assert summary['total_manual_closes'] == 2
        assert summary['profitable_closes'] == 2
        print(f"✅ Summary: {summary}")


# ═══════════════════════════════════════════════════════════
#  CLEANUP
# ═══════════════════════════════════════════════════════════

def teardown_module():
    """Clean up test CSV files."""
    test_csv = "dashboard/data/test_efficacy.csv"
    if os.path.exists(test_csv):
        os.remove(test_csv)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
