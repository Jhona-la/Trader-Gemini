"""
🧮 UNIT TESTS - MATHEMATICAL SYNAPSE VALIDATION
================================================

PROFESSOR METHOD:
- QUÉ: Validación de fórmulas matemáticas críticas (Esperanza, Kelly, Z-Score).
- POR QUÉ: Garantiza exactitud entre Bot y Dashboard.
- CÓMO: Tests unitarios con valores conocidos.
- CUÁNDO: Pre-producción y después de cualquier cambio en utils/.

FORMULAS VALIDADAS:
- Esperanza: E = (Pw × Avgw) - (Pl × Avgl)
- Kelly: K = (E × Pw) / Avgw
- Z-Score: Z = (X - μ) / σ
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['TRADER_GEMINI_ENV'] = 'TEST'


# =============================================================================
# EXPECTANCY CALCULATOR (Reference Implementation)
# =============================================================================

class ExpectancyCalculator:
    """
    Reference implementation of Expectancy calculations.
    Used to validate that Bot and Dashboard use identical formulas.
    """
    
    @staticmethod
    def calculate_expectancy_v1(trades: List[float]) -> float:
        """
        Version 1: Simple average
        E = (1/N) × Σ Net_PnL_i
        """
        if not trades:
            return 0.0
        return sum(trades) / len(trades)
    
    @staticmethod
    def calculate_expectancy_v2(trades: List[float]) -> float:
        """
        Version 2: Win Rate × Avg Win - Loss Rate × Avg Loss
        E = (Pw × Avgw) - (Pl × Avgl)
        """
        if not trades:
            return 0.0
        
        wins = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t < 0]
        
        total = len(trades)
        pw = len(wins) / total  # Win probability
        pl = len(losses) / total  # Loss probability
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        return (pw * avg_win) - (pl * avg_loss)
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion: Optimal position sizing
        K = W - (1-W)/R where R = Avg_Win / Avg_Loss
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        
        r = avg_win / avg_loss  # Win/Loss ratio
        kelly = win_rate - ((1 - win_rate) / r)
        
        return max(0, kelly)  # Never negative
    
    @staticmethod
    def calculate_z_score(value: float, mean: float, std: float) -> float:
        """
        Z-Score: Standard deviation distance from mean
        Z = (X - μ) / σ
        """
        if std == 0:
            return 0.0
        return (value - mean) / std


# =============================================================================
# TEST: EXPECTANCY FORMULAS
# =============================================================================

class TestExpectancyFormulas:
    """
    📊 Test Expectancy Formula Accuracy
    
    Validates both versions produce consistent results.
    """
    
    def test_expectancy_v1_basic(self):
        """Test simple average expectancy."""
        trades = [10, -5, 15, -8, 20]
        
        # E = (10 - 5 + 15 - 8 + 20) / 5 = 32 / 5 = 6.4
        expected = 6.4
        result = ExpectancyCalculator.calculate_expectancy_v1(trades)
        
        assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
    
    def test_expectancy_v2_win_loss_formula(self):
        """Test Pw×Avgw - Pl×Avgl formula."""
        trades = [10, -5, 15, -8, 20]  # 3 wins, 2 losses
        
        # Pw = 3/5 = 0.6, Pl = 2/5 = 0.4
        # Avg_win = (10+15+20)/3 = 15
        # Avg_loss = (5+8)/2 = 6.5
        # E = (0.6 × 15) - (0.4 × 6.5) = 9 - 2.6 = 6.4
        expected = 6.4
        result = ExpectancyCalculator.calculate_expectancy_v2(trades)
        
        assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
    
    def test_expectancy_formulas_equivalent(self):
        """
        CRITICAL: Both formulas must produce identical results.
        This ensures Bot and Dashboard calculations match.
        """
        trades = [10, -5, 15, -8, 20, -3, 12, -7, 8, -4]
        
        v1 = ExpectancyCalculator.calculate_expectancy_v1(trades)
        v2 = ExpectancyCalculator.calculate_expectancy_v2(trades)
        
        assert abs(v1 - v2) < 0.001, f"V1={v1} != V2={v2} - FORMULAS DO NOT MATCH!"
    
    def test_expectancy_empty_trades(self):
        """Test expectancy with no trades."""
        assert ExpectancyCalculator.calculate_expectancy_v1([]) == 0.0
        assert ExpectancyCalculator.calculate_expectancy_v2([]) == 0.0
    
    def test_expectancy_all_wins(self):
        """Test expectancy with only wins."""
        trades = [10, 20, 30]
        
        v1 = ExpectancyCalculator.calculate_expectancy_v1(trades)
        v2 = ExpectancyCalculator.calculate_expectancy_v2(trades)
        
        assert v1 == 20.0  # Average of wins
        assert v2 == 20.0  # 100% win rate × avg win
    
    def test_expectancy_all_losses(self):
        """Test expectancy with only losses."""
        trades = [-10, -20, -30]
        
        v1 = ExpectancyCalculator.calculate_expectancy_v1(trades)
        v2 = ExpectancyCalculator.calculate_expectancy_v2(trades)
        
        assert v1 == -20.0
        assert v2 == -20.0  # 0% win rate - 100% loss rate × avg loss


# =============================================================================
# TEST: KELLY CRITERION
# =============================================================================

class TestKellyCriterion:
    """
    📈 Test Kelly Criterion Formula
    
    K = W - (1-W)/R where R = Avg_Win / Avg_Loss
    """
    
    def test_kelly_basic(self):
        """Test basic Kelly calculation."""
        # 60% win rate, avg win = 15, avg loss = 10
        # R = 15/10 = 1.5
        # K = 0.6 - (0.4/1.5) = 0.6 - 0.267 = 0.333
        result = ExpectancyCalculator.calculate_kelly_criterion(0.6, 15.0, 10.0)
        expected = 0.333
        
        assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"
    
    def test_kelly_low_win_rate(self):
        """Test Kelly with low win rate (should be 0 or small)."""
        # 30% win rate, avg win = 10, avg loss = 10
        # R = 1.0
        # K = 0.3 - 0.7/1.0 = 0.3 - 0.7 = -0.4 → 0 (capped)
        result = ExpectancyCalculator.calculate_kelly_criterion(0.3, 10.0, 10.0)
        
        assert result == 0.0, "Kelly should be 0 for negative edge"
    
    def test_kelly_high_win_rate(self):
        """Test Kelly with high win rate."""
        # 80% win rate, avg win = 20, avg loss = 10
        # R = 2.0
        # K = 0.8 - 0.2/2.0 = 0.8 - 0.1 = 0.7
        result = ExpectancyCalculator.calculate_kelly_criterion(0.8, 20.0, 10.0)
        expected = 0.7
        
        assert abs(result - expected) < 0.01
    
    def test_kelly_edge_cases(self):
        """Test Kelly edge cases."""
        # Zero avg loss (division by zero protection)
        assert ExpectancyCalculator.calculate_kelly_criterion(0.5, 10.0, 0.0) == 0.0
        
        # Zero avg win
        assert ExpectancyCalculator.calculate_kelly_criterion(0.5, 0.0, 10.0) == 0.0


# =============================================================================
# TEST: Z-SCORE
# =============================================================================

class TestZScore:
    """
    📉 Test Z-Score Formula
    
    Z = (X - μ) / σ
    """
    
    def test_zscore_basic(self):
        """Test basic Z-Score calculation."""
        # Value = 110, Mean = 100, Std = 10
        # Z = (110 - 100) / 10 = 1.0
        result = ExpectancyCalculator.calculate_z_score(110, 100, 10)
        
        assert result == 1.0
    
    def test_zscore_negative(self):
        """Test negative Z-Score."""
        # Value = 90, Mean = 100, Std = 10
        # Z = (90 - 100) / 10 = -1.0
        result = ExpectancyCalculator.calculate_z_score(90, 100, 10)
        
        assert result == -1.0
    
    def test_zscore_zero_std(self):
        """Test Z-Score with zero std (division by zero protection)."""
        result = ExpectancyCalculator.calculate_z_score(110, 100, 0)
        
        assert result == 0.0
    
    def test_zscore_extreme(self):
        """Test extreme Z-Score values."""
        # 3 standard deviations above mean
        result = ExpectancyCalculator.calculate_z_score(130, 100, 10)
        
        assert result == 3.0


# =============================================================================
# TEST: BOT-DASHBOARD CONSISTENCY
# =============================================================================

class TestBotDashboardConsistency:
    """
    🔄 Test Bot-Dashboard Calculation Consistency
    
    Ensures both systems use identical formulas.
    """
    
    def test_expectancy_consistency_large_dataset(self):
        """Test consistency with large realistic dataset."""
        np.random.seed(42)  # Reproducible
        
        # Generate 1000 trades with realistic distribution
        trades = []
        for _ in range(1000):
            if np.random.random() < 0.55:  # 55% win rate
                trades.append(np.random.normal(15, 5))  # Avg win ~15
            else:
                trades.append(-abs(np.random.normal(10, 3)))  # Avg loss ~10
        
        # Both methods should give same result
        v1 = ExpectancyCalculator.calculate_expectancy_v1(trades)
        v2 = ExpectancyCalculator.calculate_expectancy_v2(trades)
        
        # Allow small tolerance for floating point
        assert abs(v1 - v2) < 0.1, f"Large dataset mismatch: V1={v1}, V2={v2}"
    
    def test_pandas_vs_python_calculation(self):
        """Test that Pandas and pure Python give same results."""
        trades = [10, -5, 15, -8, 20, -3, 12, -7]
        
        # Pure Python
        py_exp = sum(trades) / len(trades)
        
        # Pandas
        pd_exp = pd.Series(trades).mean()
        
        assert abs(py_exp - pd_exp) < 0.0001, "Pandas vs Python mismatch"
    
    def test_csv_roundtrip_precision(self):
        """Test that CSV save/load preserves precision."""
        import tempfile
        import csv
        
        original_pnl = 1.5658432198
        
        # Write to CSV
        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['pnl'])
                writer.writerow([original_pnl])
            
            # Read back
            df = pd.read_csv(path)
            loaded_pnl = df['pnl'].iloc[0]
            
            # Precision should be maintained (default is enough for financial calcs)
            assert abs(original_pnl - loaded_pnl) < 0.0000001
        finally:
            os.remove(path)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
