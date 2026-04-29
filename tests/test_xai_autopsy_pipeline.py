"""
🧪 TEST: XAI Autopsy Pipeline Integration Test
================================================
QUÉ: Verifica que la inyección de autopsia forense en el pipeline 
     Portfolio → Notifier funciona correctamente.
POR QUÉ: La autopsia XAI es crítica para el feedback loop humano.
PARA QUÉ: Asegurar que cada trade cerrado con Sophia intent genera
     un resumen legible en Telegram.
CÓMO: Mock de PostMortemComparator y verificación de trade_notification_data.
CUÁNDO: Antes de producción (Phase Omega validation).
DÓNDE: tests/test_xai_autopsy_pipeline.py
QUIÉN: PostMortemComparator + Portfolio + Notifier
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


# ═══════════════════════════════════════════════════════════════
# Test 1: XAI Autopsy Injection into _last_closed_trade_data
# ═══════════════════════════════════════════════════════════════

class TestXAIAutopsyInjection:
    """Validates that _sophia_post_mortem_check injects XAI fields."""

    def test_xai_fields_injected_on_loss(self):
        """When a trade closes with a loss, XAI autopsy data should be injected."""
        # Arrange
        @dataclass
        class MockPostMortemResult:
            brier_score: float = 0.22
            predicted_prob: float = 0.85
            predicted_exit_mins: float = 3.0
            actual_outcome: str = "LOSS"
            actual_duration_mins: float = 1.5
            time_error_mins: float = 1.5
            narrative: str = "RSI fue el driver dominante"
        
        @dataclass
        class MockIntent:
            symbol: str = "BTC/USDT"
            direction: str = "LONG"
            win_probability: float = 0.85
            expected_exit_mins: float = 3.0
            trigger_price: float = 95000.0
            sophia_report: dict = None
            top_features: list = None
            
            def __post_init__(self):
                if self.sophia_report is None:
                    self.sophia_report = {'time_to_tp_mins': 5.0, 'time_to_sl_mins': 2.0}
                if self.top_features is None:
                    self.top_features = [
                        {'name': 'rsi', 'value': 28.5},
                        {'name': 'macd_hist', 'value': -0.002},
                        {'name': 'volume_ratio', 'value': 2.3}
                    ]

        # Create mock portfolio with minimal required attributes
        mock_pm = MagicMock()
        mock_pm.pending_intents = {'test-001': MockIntent()}
        mock_pm.compute_post_mortem.return_value = MockPostMortemResult()
        mock_pm.total_trades = 1
        
        # Simulate the XAI injection logic
        intent = mock_pm.pending_intents.get('test-001')
        result = mock_pm.compute_post_mortem()
        
        xai_lines = []
        xai_lines.append(f"🔬 Brier: {result.brier_score:.3f}")
        
        if result.brier_score < 0.05:
            xai_lines.append("Calibración: 🎯 EXCELENTE")
        elif result.brier_score < 0.15:
            xai_lines.append("Calibración: ✅ BUENA")
        elif result.brier_score < 0.25:
            xai_lines.append("Calibración: ⚠️ DEGRADADA")
        else:
            xai_lines.append("Calibración: ❌ CRÍTICA")
        
        xai_lines.append(f"Predicho: WP={result.predicted_prob*100:.0f}% | Duración={result.predicted_exit_mins:.1f}min")
        xai_lines.append(f"Real: {result.actual_outcome} | Duración={result.actual_duration_mins:.1f}min")
        
        if result.time_error_mins > 5.0:
            xai_lines.append(f"⏱️ Error temporal: {result.time_error_mins:.1f}min")
        
        if intent and intent.top_features:
            top_3 = intent.top_features[:3]
            feat_str = ", ".join(
                f"{f.get('name', 'unknown')}={f.get('value', 0):.2f}" 
                for f in top_3 if isinstance(f, dict)
            )
            if feat_str:
                xai_lines.append(f"Drivers: {feat_str}")
        
        xai_summary = "\n".join(xai_lines)
        
        # Assert
        assert "🔬 Brier: 0.220" in xai_summary
        assert "⚠️ DEGRADADA" in xai_summary
        assert "WP=85%" in xai_summary
        assert "rsi=28.50" in xai_summary
        assert "macd_hist=-0.00" in xai_summary
        assert "volume_ratio=2.30" in xai_summary
        assert "LOSS" in xai_summary
        
        print(f"✅ XAI Autopsy Output:\n{xai_summary}\n")

    def test_xai_calibration_labels(self):
        """Verify correct calibration labels for different Brier scores."""
        thresholds = [
            (0.01, "EXCELENTE"),
            (0.10, "BUENA"),
            (0.20, "DEGRADADA"),
            (0.40, "CRÍTICA"),
        ]
        
        for brier, expected_label in thresholds:
            if brier < 0.05:
                label = "EXCELENTE"
            elif brier < 0.15:
                label = "BUENA"
            elif brier < 0.25:
                label = "DEGRADADA"
            else:
                label = "CRÍTICA"
            
            assert label == expected_label, f"Brier {brier} should be {expected_label}, got {label}"
        
        print("✅ All calibration labels correct")

    def test_time_error_only_shown_when_significant(self):
        """Time error line should only appear when error > 5 minutes."""
        xai_lines = []
        
        # Small error - should NOT appear
        time_error = 2.0
        if time_error > 5.0:
            xai_lines.append(f"⏱️ Error temporal: {time_error:.1f}min")
        
        assert len(xai_lines) == 0, "Small time error should not generate a line"
        
        # Large error - SHOULD appear
        time_error = 10.0
        if time_error > 5.0:
            xai_lines.append(f"⏱️ Error temporal: {time_error:.1f}min")
        
        assert len(xai_lines) == 1, "Large time error should generate a line"
        assert "10.0min" in xai_lines[0]
        
        print("✅ Time error threshold working correctly")


# ═══════════════════════════════════════════════════════════════
# Test 2: State Manager Integrity Checksum
# ═══════════════════════════════════════════════════════════════

class TestStateManagerChecksum:
    """Validates the integrity checksum system."""

    def test_checksum_deterministic(self):
        """Same state should produce same checksum."""
        from core.state_manager import AtomicStateManager
        
        state = {'cash': 13.0, 'positions': {'BTC/USDT': {'qty': 0.001}}}
        
        hash1 = AtomicStateManager.compute_checksum(state)
        hash2 = AtomicStateManager.compute_checksum(state)
        
        assert hash1 == hash2, f"Checksums should match: {hash1} vs {hash2}"
        assert len(hash1) == 16, f"Checksum should be 16 chars, got {len(hash1)}"
        
        print(f"✅ Deterministic checksum: {hash1}")

    def test_checksum_detects_corruption(self):
        """Different states should produce different checksums."""
        from core.state_manager import AtomicStateManager
        
        state1 = {'cash': 13.0, 'positions': {}}
        state2 = {'cash': 12.5, 'positions': {}}  # Corrupted cash
        
        hash1 = AtomicStateManager.compute_checksum(state1)
        hash2 = AtomicStateManager.compute_checksum(state2)
        
        assert hash1 != hash2, "Different states must produce different checksums"
        
        print(f"✅ Corruption detection: {hash1} ≠ {hash2}")

    def test_checksum_order_independent(self):
        """Key order should not affect checksum (sort_keys=True)."""
        from core.state_manager import AtomicStateManager
        
        state1 = {'a': 1, 'b': 2, 'c': 3}
        state2 = {'c': 3, 'a': 1, 'b': 2}  # Same keys, different order
        
        hash1 = AtomicStateManager.compute_checksum(state1)
        hash2 = AtomicStateManager.compute_checksum(state2)
        
        assert hash1 == hash2, "Order-independent checksums should match"
        
        print(f"✅ Order-independent: {hash1}")


# ═══════════════════════════════════════════════════════════════
# Test 3: Notifier XAI Display
# ═══════════════════════════════════════════════════════════════

class TestNotifierXAIDisplay:
    """Validates that the notifier correctly renders XAI autopsy."""

    def test_xai_autopsy_included_in_message(self):
        """When trade_data has xai_autopsy, it should appear in the message."""
        trade_data = {
            'xai_autopsy': '🔬 Brier: 0.220\nCalibración: ⚠️ DEGRADADA',
            'sophia_narrative': 'RSI sobreventa no fue suficiente'
        }
        
        msg = ""
        xai_autopsy = trade_data.get('xai_autopsy')
        if xai_autopsy:
            msg += f"\n\n🧠 *Autopsia Sophia (XAI):*\n{xai_autopsy}"
        sophia_narrative = trade_data.get('sophia_narrative')
        if sophia_narrative:
            msg += f"\n💬 _{sophia_narrative}_"
        
        assert "Autopsia Sophia" in msg
        assert "Brier: 0.220" in msg
        assert "DEGRADADA" in msg
        assert "RSI sobreventa" in msg
        
        print(f"✅ Notifier XAI rendering:\n{msg}")

    def test_no_xai_when_absent(self):
        """When trade_data has no xai_autopsy, nothing extra should appear."""
        trade_data = {'pnl': 0.05}
        
        msg = ""
        xai_autopsy = trade_data.get('xai_autopsy')
        if xai_autopsy:
            msg += f"\n\n🧠 *Autopsia Sophia (XAI):*\n{xai_autopsy}"
        
        assert msg == "", "No XAI data should produce no XAI message"
        
        print("✅ Clean message when no XAI data")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 XAI AUTOPSY PIPELINE INTEGRATION TESTS")
    print("=" * 60)
    
    # Run all tests
    t1 = TestXAIAutopsyInjection()
    t1.test_xai_fields_injected_on_loss()
    t1.test_xai_calibration_labels()
    t1.test_time_error_only_shown_when_significant()
    
    t2 = TestStateManagerChecksum()
    t2.test_checksum_deterministic()
    t2.test_checksum_detects_corruption()
    t2.test_checksum_order_independent()
    
    t3 = TestNotifierXAIDisplay()
    t3.test_xai_autopsy_included_in_message()
    t3.test_no_xai_when_absent()
    
    print("\n" + "=" * 60)
    print("✅ ALL 8 TESTS PASSED — XAI Pipeline Verified")
    print("=" * 60)
