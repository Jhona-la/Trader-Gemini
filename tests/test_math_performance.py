
import unittest
import numpy as np
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.statistics_pro import StatisticsPro

class TestMathValidity(unittest.TestCase):
    def setUp(self):
        self.stats = StatisticsPro()
        
    def test_hurst_exponent_random(self):
        """Random Walk should have Hurst ~0.5"""
        print("\n🧪 Testing Hurst - Random Walk...")
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(1000))
        H, _, _ = self.stats.calculate_hurst(random_walk)
        print(f"   -> Random Walk H: {H:.4f}")
        self.assertTrue(0.4 < H < 0.6, "Hurst for Random Walk should be near 0.5")

    def test_hurst_exponent_trend(self):
        """Trend should have Hurst > 0.5"""
        print("\n🧪 Testing Hurst - Strong Trend...")
        trend = np.linspace(0, 100, 1000) + np.random.randn(1000)*0.5
        H, _, _ = self.stats.calculate_hurst(trend)
        print(f"   -> Trend H: {H:.4f}")
        self.assertTrue(H > 0.6, "Hurst for Trend should be > 0.5")

    def test_ransac_robustness(self):
        """RANSAC should ignore outliers"""
        print("\n🧪 Testing RANSAC Robustness...")
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 # Slope 2
        
        # Add outliers
        y[90:] += 50 # Massive outliers
        
        slope, intercept = self.stats.calculate_robust_beta_ransac(x, y)
        print(f"   -> RANSAC Slope: {slope:.4f} (True: 2.0)")
        
        self.assertTrue(1.9 < slope < 2.1, f"RANSAC failed to ignore outliers (Got {slope})")

    def test_nan_handling(self):
        """Functions should handle NaNs gracefully"""
        print("\n🧪 Testing NaN Handling...")
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # Should not crash
        try:
            self.stats.calculate_hurst(data_with_nan)
            print("   -> Hurst handled NaNs (or ignored them) ✅")
        except Exception as e:
            self.fail(f"Hurst crashed on NaN: {e}")

if __name__ == '__main__':
    unittest.main()
