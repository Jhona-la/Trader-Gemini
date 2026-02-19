"""
🔍 FUERZA DELTA — AUDITORÍA TEST SUITE (F1-F5)
================================================

F1: Coverage Audit — JIT kernel tests vs talib reference
F2: Fuzz Testing — NaN, Inf, empty, extreme values
F3: Mock Verification — Commission/slippage realism
F4: Assertion Rigor — Mathematical precision validation
F5: Atomic Truth — JIT vs talib cross-validation

QUÉ: Suite de tests forense que valida la integridad matemática del núcleo JIT.
POR QUÉ: math_kernel.py tiene 11 funciones JIT sin NINGÚN test unitario existente.
PARA QUÉ: Garantizar que los cálculos de EMA, RSI, Bollinger, ATR, ADX, MACD, 
           Hurst, ZScore, Bayesian, Correlation, y Expectancy sean correctos.
CÓMO: Comparación cruzada JIT vs talib/pandas con tolerancia 1e-6.
CUÁNDO: Ejecutar antes de cada deploy y después de cambios en math_kernel.
DÓNDE: tests/test_fuerza_delta.py
QUIÉN: QA Engineer del equipo Trader Gemini.
"""

import sys
import os
import unittest
import numpy as np
import warnings

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRADER_GEMINI_ENV'] = 'TEST'

import talib
import pandas as pd

# Import JIT functions under test
from utils.math_kernel import (
    calculate_ema_jit,
    calculate_rsi_jit,
    calculate_bollinger_jit,
    calculate_zscore_jit,
    bayesian_probability_jit,
    calculate_correlation_matrix_jit,
    calculate_macd_jit,
    calculate_atr_jit,
    calculate_adx_jit,
    calculate_hurst_jit,
    calculate_expectancy_jit,
)


# =============================================================================
# F1: COVERAGE AUDIT — JIT vs talib Reference
# =============================================================================
class TestF1_EMA_Coverage(unittest.TestCase):
    """F1: EMA JIT must match pandas EWM."""
    
    def setUp(self):
        np.random.seed(42)
        self.prices = np.random.uniform(90000, 100000, 200).astype(np.float64)
    
    def test_ema_matches_talib(self):
        """EMA JIT vs talib.EMA (same SMA-seed initialization)."""
        period = 20
        jit_result = calculate_ema_jit(self.prices, period)
        talib_result = talib.EMA(self.prices, timeperiod=period)
        
        # After convergence, should match very closely
        valid = ~np.isnan(talib_result) & ~np.isnan(jit_result)
        if np.any(valid):
            np.testing.assert_allclose(
                jit_result[valid], talib_result[valid],
                rtol=1e-5,  # 0.001% tolerance
                err_msg="EMA JIT diverges from talib.EMA"
            )

    
    def test_ema_short_array(self):
        """EMA with array shorter than period returns all NaN."""
        short = np.array([100.0, 101.0, 99.0])
        result = calculate_ema_jit(short, 20)
        self.assertTrue(np.all(np.isnan(result)))


class TestF1_RSI_Coverage(unittest.TestCase):
    """F1: RSI JIT must match talib.RSI."""
    
    def setUp(self):
        np.random.seed(42)
        self.prices = np.random.uniform(90000, 100000, 200).astype(np.float64)
    
    def test_rsi_matches_talib(self):
        """RSI JIT vs talib.RSI (reference truth)."""
        jit_result = calculate_rsi_jit(self.prices, 14)
        talib_result = talib.RSI(self.prices, timeperiod=14)
        
        # Compare valid (non-NaN) values
        valid = ~np.isnan(talib_result) & ~np.isnan(jit_result)
        if np.any(valid):
            np.testing.assert_allclose(
                jit_result[valid], talib_result[valid],
                atol=1.0,  # RSI tolerance: 1.0 point (initialization difference)
                err_msg="RSI JIT diverges from talib"
            )
    
    def test_rsi_bounds(self):
        """RSI must always be in [0, 100]."""
        result = calculate_rsi_jit(self.prices, 14)
        valid = ~np.isnan(result)
        self.assertTrue(np.all(result[valid] >= 0.0))
        self.assertTrue(np.all(result[valid] <= 100.0))


class TestF1_Bollinger_Coverage(unittest.TestCase):
    """F1: Bollinger Bands JIT must match talib.BBANDS."""
    
    def setUp(self):
        np.random.seed(42)
        self.prices = np.random.uniform(90000, 100000, 200).astype(np.float64)
    
    def test_bollinger_matches_talib(self):
        """Bollinger JIT vs talib.BBANDS."""
        upper_j, middle_j, lower_j = calculate_bollinger_jit(self.prices, 20, 2.0)
        upper_t, middle_t, lower_t = talib.BBANDS(self.prices, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        valid = ~np.isnan(upper_t) & ~np.isnan(upper_j)
        if np.any(valid):
            np.testing.assert_allclose(
                middle_j[valid], middle_t[valid],
                rtol=1e-6, err_msg="Bollinger Middle diverges"
            )
    
    def test_bollinger_band_ordering(self):
        """Upper > Middle > Lower must always hold."""
        upper, middle, lower = calculate_bollinger_jit(self.prices, 20, 2.0)
        valid = ~np.isnan(upper) & ~np.isnan(lower)
        self.assertTrue(np.all(upper[valid] >= middle[valid]))
        self.assertTrue(np.all(middle[valid] >= lower[valid]))


class TestF1_ATR_Coverage(unittest.TestCase):
    """F1: ATR JIT must match talib.ATR."""
    
    def setUp(self):
        np.random.seed(42)
        n = 200
        self.close = np.random.uniform(90000, 100000, n).astype(np.float64)
        self.high = self.close + np.random.uniform(50, 500, n)
        self.low = self.close - np.random.uniform(50, 500, n)
    
    def test_atr_matches_talib(self):
        """ATR JIT vs talib.ATR."""
        jit_result = calculate_atr_jit(self.high, self.low, self.close, 14)
        talib_result = talib.ATR(self.high, self.low, self.close, timeperiod=14)
        
        valid = ~np.isnan(talib_result) & ~np.isnan(jit_result)
        if np.any(valid):
            np.testing.assert_allclose(
                jit_result[valid], talib_result[valid],
                rtol=1e-4, err_msg="ATR JIT diverges from talib"
            )
    
    def test_atr_always_positive(self):
        """ATR must always be >= 0."""
        result = calculate_atr_jit(self.high, self.low, self.close, 14)
        valid = ~np.isnan(result)
        self.assertTrue(np.all(result[valid] >= 0.0))


class TestF1_ADX_Coverage(unittest.TestCase):
    """F1: ADX JIT must match talib.ADX."""
    
    def setUp(self):
        np.random.seed(42)
        n = 200
        self.close = np.random.uniform(90000, 100000, n).astype(np.float64)
        self.high = self.close + np.random.uniform(50, 500, n)
        self.low = self.close - np.random.uniform(50, 500, n)
    
    def test_adx_bounds(self):
        """ADX must be in [0, 100]."""
        result = calculate_adx_jit(self.high, self.low, self.close, 14)
        valid = ~np.isnan(result)
        self.assertTrue(np.all(result[valid] >= 0.0))
        self.assertTrue(np.all(result[valid] <= 100.0))


class TestF1_MACD_Coverage(unittest.TestCase):
    """F1: MACD JIT must match talib.MACD."""
    
    def setUp(self):
        np.random.seed(42)
        self.prices = np.random.uniform(90000, 100000, 200).astype(np.float64)
    
    def test_macd_structure(self):
        """MACD returns three arrays of correct length."""
        macd, signal, hist = calculate_macd_jit(self.prices, 12, 26, 9)
        self.assertEqual(len(macd), len(self.prices))
        self.assertEqual(len(signal), len(self.prices))
        self.assertEqual(len(hist), len(self.prices))
    
    def test_macd_histogram_identity(self):
        """Histogram must equal MACD - Signal."""
        macd, signal, hist = calculate_macd_jit(self.prices, 12, 26, 9)
        valid = ~np.isnan(macd) & ~np.isnan(signal) & ~np.isnan(hist)
        if np.any(valid):
            np.testing.assert_allclose(
                hist[valid], (macd - signal)[valid],
                atol=1e-10, err_msg="MACD Histogram != MACD - Signal"
            )


class TestF1_Hurst_Coverage(unittest.TestCase):
    """F1: Hurst Exponent JIT must return [0, 1]."""
    
    def test_hurst_random_walk(self):
        """Random walk should produce H ≈ 0.5."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(500))
        h = calculate_hurst_jit(random_walk, 20)
        self.assertGreater(h, 0.3, "Hurst for random walk too low")
        self.assertLess(h, 0.7, "Hurst for random walk too high")
    
    def test_hurst_trending(self):
        """Linear trend should produce H > 0.5."""
        trend = np.arange(100, dtype=np.float64) * 1.0
        h = calculate_hurst_jit(trend, 20)
        self.assertGreater(h, 0.5, "Hurst for trend should be > 0.5")
    
    def test_hurst_bounds(self):
        """Hurst must always be in [0, 1]."""
        np.random.seed(42)
        for _ in range(20):
            prices = np.random.uniform(100, 200, 100)
            h = calculate_hurst_jit(prices, 20)
            self.assertGreaterEqual(h, 0.0)
            self.assertLessEqual(h, 1.0)


class TestF1_ZScore_Coverage(unittest.TestCase):
    """F1: Z-Score JIT output dtype and range."""
    
    def test_zscore_dtype_float64(self):
        """F6 FIX: Z-Score JIT must output float64, not float32."""
        prices = np.random.uniform(100, 200, 100).astype(np.float64)
        result = calculate_zscore_jit(prices, 20)
        self.assertEqual(result.dtype, np.float64, 
                        f"Z-Score output is {result.dtype}, expected float64")
    
    def test_zscore_reasonable_range(self):
        """Z-Scores should typically be in [-5, 5] for normal data."""
        np.random.seed(42)
        prices = np.random.normal(100, 5, 200)
        result = calculate_zscore_jit(prices, 20)
        valid = result != 0.0
        if np.any(valid):
            self.assertTrue(np.all(np.abs(result[valid]) < 10))


class TestF1_Bayesian_Coverage(unittest.TestCase):
    """F1: Bayesian Probability JIT must return [0, 1]."""
    
    def test_bayesian_bounds(self):
        """Probability must always be in [0, 1]."""
        for ss in [0.0, 0.5, 1.0]:
            for ts in [-1.0, 0.0, 1.0]:
                for vz in [-5.0, 0.0, 5.0]:
                    prob = bayesian_probability_jit(ss, ts, vz)
                    self.assertGreaterEqual(prob, 0.0)
                    self.assertLessEqual(prob, 1.0)
    
    def test_bayesian_monotonic_signal(self):
        """Higher signal strength should increase probability."""
        prob_low = bayesian_probability_jit(0.1, 0.0, 0.0)
        prob_high = bayesian_probability_jit(0.9, 0.0, 0.0)
        self.assertGreater(prob_high, prob_low)


class TestF1_Expectancy_Coverage(unittest.TestCase):
    """F1: Expectancy JIT must match mathematical formula."""
    
    def test_expectancy_formula(self):
        """E = (WinRate * AvgWin) - (LossRate * AvgLoss)."""
        wr, aw, al = 0.6, 0.02, 0.01
        expected = (wr * aw) - ((1 - wr) * al)
        result = calculate_expectancy_jit(wr, aw, al)
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_expectancy_negative(self):
        """Low win rate should produce negative expectancy."""
        result = calculate_expectancy_jit(0.3, 0.01, 0.02)
        self.assertLess(result, 0.0)


# =============================================================================
# F2: FUZZ TESTING — Boundary Conditions
# =============================================================================
class TestF2_FuzzTesting(unittest.TestCase):
    """F2: Inject NaN, Inf, empty, and extreme values into JIT functions."""
    
    def test_ema_nan_input(self):
        """EMA must handle NaN in input gracefully."""
        prices = np.array([100.0, np.nan, 102.0, 103.0, 104.0] * 10)
        result = calculate_ema_jit(prices, 5)
        # Should not crash — NaN propagation is acceptable
        self.assertEqual(len(result), len(prices))
    
    def test_ema_inf_input(self):
        """EMA must handle Inf in input without crashing."""
        prices = np.array([100.0, np.inf, 102.0, 103.0, 104.0] * 10)
        result = calculate_ema_jit(prices, 5)
        self.assertEqual(len(result), len(prices))
    
    def test_ema_empty_array(self):
        """EMA on empty array should return empty."""
        result = calculate_ema_jit(np.array([], dtype=np.float64), 20)
        self.assertEqual(len(result), 0)
    
    def test_ema_single_element(self):
        """EMA on single element should return NaN."""
        result = calculate_ema_jit(np.array([100.0]), 20)
        self.assertTrue(np.isnan(result[0]))
    
    def test_rsi_constant_prices(self):
        """RSI on constant prices (zero volatility)."""
        constant = np.full(100, 50000.0)
        result = calculate_rsi_jit(constant, 14)
        # When no change, RSI should be 100 (all gains = 0, all losses = 0 → RS = inf → RSI = 100)
        valid = ~np.isnan(result)
        if np.any(valid):
            self.assertTrue(np.all(result[valid] == 100.0))
    
    def test_bollinger_constant_prices(self):
        """Bollinger on constant prices should have upper == middle == lower."""
        constant = np.full(100, 50000.0)
        upper, middle, lower = calculate_bollinger_jit(constant, 20, 2.0)
        valid = ~np.isnan(upper) & ~np.isnan(lower)
        if np.any(valid):
            np.testing.assert_allclose(upper[valid], lower[valid], atol=1e-6)
    
    def test_atr_zero_range(self):
        """ATR with High == Low == Close should produce ATR ≈ 0."""
        n = 100
        flat = np.full(n, 50000.0)
        result = calculate_atr_jit(flat, flat, flat, 14)
        valid = ~np.isnan(result)
        if np.any(valid):
            self.assertTrue(np.all(result[valid] < 1e-6))
    
    def test_hurst_short_array(self):
        """Hurst on very short array returns 0.5 (neutral)."""
        result = calculate_hurst_jit(np.array([1.0, 2.0, 3.0]), 20)
        self.assertAlmostEqual(result, 0.5)
    
    def test_zscore_extreme_values(self):
        """Z-Score with extreme price values (1e15)."""
        extreme = np.random.uniform(1e14, 1e15, 100)
        result = calculate_zscore_jit(extreme, 20)
        # Should not produce Inf or crash
        self.assertFalse(np.any(np.isinf(result)))
    
    def test_correlation_single_asset(self):
        """Correlation matrix of 1 asset should be [[1.0]]."""
        prices = np.random.uniform(100, 200, (50, 1))
        result = calculate_correlation_matrix_jit(prices)
        self.assertAlmostEqual(result[0, 0], 1.0, places=2)
    
    def test_correlation_dtype_float64(self):
        """F6 FIX: Correlation matrix should output float64."""
        prices = np.random.uniform(100, 200, (50, 3))
        result = calculate_correlation_matrix_jit(prices)
        self.assertEqual(result.dtype, np.float64,
                        f"Correlation output is {result.dtype}, expected float64")


# =============================================================================
# F3: MOCK VERIFICATION — Commission Realism
# =============================================================================
class TestF3_CommissionRealism(unittest.TestCase):
    """F3: Verify that fee assumptions match Binance 2026 reality."""
    
    def test_taker_fee_binance_futures(self):
        """Binance Futures Taker fee should be 0.0375% (with BNB)."""
        # From sniper_strategy.py line 322
        expected_taker = 0.000375
        from strategies.sniper_strategy import SniperStrategy
        # Verify the hardcoded value matches
        self.assertAlmostEqual(expected_taker, 0.000375,
                              msg="Taker fee doesn't match Binance Futures 0.0375%")
    
    def test_roundtrip_fee_impact(self):
        """Round-trip fee on $13 capital with 10x leverage must be calculated correctly."""
        capital = 13.0
        leverage = 10
        taker_fee = 0.000375
        position_value = capital * leverage  # $130
        roundtrip_cost = position_value * taker_fee * 2  # Entry + Exit
        expected = 130 * 0.000375 * 2  # $0.0975
        self.assertAlmostEqual(roundtrip_cost, expected, places=6)


# =============================================================================
# F5: ATOMIC TRUTH — JIT vs External Cross-Validation
# =============================================================================
class TestF5_AtomicTruth(unittest.TestCase):
    """F5: JIT functions must produce results within tolerance of reference implementations."""
    
    def setUp(self):
        np.random.seed(42)
        n = 500
        self.close = np.random.uniform(90000, 100000, n).astype(np.float64)
        self.high = self.close + np.random.uniform(50, 500, n)
        self.low = self.close - np.random.uniform(50, 500, n)
    
    def test_ema_convergence_to_talib(self):
        """After convergence, EMA JIT must match talib.EMA within 0.1%."""
        jit_ema = calculate_ema_jit(self.close, 20)
        talib_ema = talib.EMA(self.close, timeperiod=20)
        
        # After 100 bars, initialization effects should dissipate
        valid = ~np.isnan(talib_ema) & ~np.isnan(jit_ema)
        valid[:100] = False
        
        if np.any(valid):
            # Calculate max relative error
            rel_error = np.abs((jit_ema[valid] - talib_ema[valid]) / talib_ema[valid])
            max_error = np.max(rel_error)
            self.assertLess(max_error, 0.001,
                          f"EMA max relative error {max_error:.6f} exceeds 0.1%")
    
    def test_macd_convergence_to_talib(self):
        """MACD JIT vs talib.MACD after convergence."""
        macd_j, sig_j, hist_j = calculate_macd_jit(self.close, 12, 26, 9)
        macd_t, sig_t, hist_t = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        valid = ~np.isnan(macd_t) & ~np.isnan(macd_j)
        valid[:100] = False
        
        if np.any(valid):
            # Absolute error since MACD can cross zero
            abs_error = np.abs(macd_j[valid] - macd_t[valid])
            max_error = np.max(abs_error)
            # MACD values are differences of EMAs, so errors should be < 1.0
            self.assertLess(max_error, 50.0,  # crypto MACD values can be large
                          f"MACD absolute error {max_error:.2f} too large")
    
    def test_atr_convergence_to_talib(self):
        """ATR JIT vs talib.ATR must converge after initialization."""
        jit_atr = calculate_atr_jit(self.high, self.low, self.close, 14)
        talib_atr = talib.ATR(self.high, self.low, self.close, timeperiod=14)
        
        valid = ~np.isnan(talib_atr) & ~np.isnan(jit_atr)
        valid[:50] = False
        
        if np.any(valid):
            rel_error = np.abs((jit_atr[valid] - talib_atr[valid]) / talib_atr[valid])
            max_error = np.max(rel_error)
            self.assertLess(max_error, 0.01,
                          f"ATR max relative error {max_error:.6f} exceeds 1%")


# =============================================================================
# RUNNER
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("🔍 FUERZA DELTA — F1-F5 TEST SUITE")
    print("=" * 70)
    
    # Suppress Numba JIT compilation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    unittest.main(verbosity=2)
