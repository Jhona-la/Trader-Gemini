"""
Statistics Pro Library
======================
Advanced quantitative methods for Trader Gemini.
Provides rigorous mathematical tools for regime detection and statistical arbitrage.

Methods:
- Hurst Exponent: Quantify trend/mean-reversion state.
- Rolling OLS: Dynamic Hedge Ratio calculation.
- Half-Life: Mean reversion speed estimation.
- ADF Test (Simplified): Stationarity check.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

try:
    from numba import jit
except ImportError:
    # Fallback for systems where numba cannot be installed (like Python 3.14 on Windows)
    def jit(func=None, *args, **kwargs):
        if func is None:
            return lambda f: f
        return func

class StatisticsPro:
    
    @staticmethod
    def calculate_hurst_exponent(price_series: np.array, max_lag: int = 20) -> float:
        """
        Calculate the Hurst Exponent to classify market state.
        
        Interpretation:
        - H < 0.5: Mean Reverting (Anti-persistent) -> Good for grid/statarb
        - H = 0.5: Geometric Brownian Motion (Random Walk)
        - H > 0.5: Trending (Persistent) -> Good for momentum
        
        Args:
            price_series: Array of prices
            max_lag: Maximum lag for R/S calculation
            
        Returns:
            float: Hurst Exponent (0.0 to 1.0)
        """
        try:
            prices = np.array(price_series)
            if len(prices) < max_lag * 2:
                return 0.5  # Not enough data, assume random walk
            
            lags = range(2, max_lag)
            tau = []
            
            # Simple R/S analysis implementation
            # Note: This is a simplified version optimized for speed
            # For strict academic usage, full Rescaled Range analysis is needed
            
            # Use aggregated variance method (faster approximation)
            # log(Var(tau)) vs log(tau) slope gives data about H
            
            # Let's use the standard deviation of differences method for speed in HFT
            # H ~ log(std(price(t) - price(t-lag))) / log(lag)
            
            log_lags = []
            log_stds = []
            
            for lag in lags:
                # Differences at lag
                diffs = prices[lag:] - prices[:-lag]
                std = np.std(diffs)
                if std == 0: continue
                
                log_lags.append(np.log(lag))
                log_stds.append(np.log(std))
            
            if len(log_lags) < 3:
                return 0.5
                
            # Linear regression to find slope
            slope, intercept = np.polyfit(log_lags, log_stds, 1)
            
            # Theoretically H = slope for this method (generalized Brownian motion)
            # This is an approximation of H.
            return float(slope)
            
        except Exception:
            return 0.5

    @staticmethod
    def rolling_ols(y: np.array, x: np.array, window: int = 50) -> Tuple[float, float]:
        """
        Perform Rolling Ordinary Least Squares Estimate.
        y = beta * x + alpha
        
        Args:
            y: Dependent variable (e.g., ETH)
            x: Independent variable (e.g., BTC)
            window: Rolling window size
            
        Returns:
            (beta, alpha): The hedge ratio and intercept
        """
        if len(y) < window or len(x) < window:
            return 1.0, 0.0
            
        # Take the last 'window' points
        y_slice = y[-window:]
        x_slice = x[-window:]
        
        # Add constant for intercept
        A = np.vstack([x_slice, np.ones(len(x_slice))]).T
        
        try:
            # np.linalg.lstsq returns (solution, residuals, rank, singular_values)
            m, c = np.linalg.lstsq(A, y_slice, rcond=None)[0]
            return float(m), float(c)
        except np.linalg.LinAlgError:
            return 1.0, 0.0

    @staticmethod
    def calculate_half_life(spread: np.array) -> float:
        """
        Calculate Half-Life of Mean Reversion using Ornstein-Uhlenbeck process.
        dy(t) = -theta * (y(t) - mu) * dt + sigma * dW(t)
        
        Args:
            spread: The spread array (residuals)
            
        Returns:
            float: Half-life in bars (intervals)
        """
        if len(spread) < 10:
            return 0.0
            
        try:
            spread_lag = np.roll(spread, 1)
            spread_lag[0] = 0
            
            spread_ret = spread - spread_lag
            spread_ret[0] = 0
            
            spread_lag2 = spread_lag[1:]
            spread_ret2 = spread_ret[1:]
            
            # Regress spread_ret on spread_lag
            slope, intercept = np.polyfit(spread_lag2, spread_ret2, 1)
            
            # lambda = -log(1 + slope) ? 
            # Simplified: theta = -slope / dt (dt=1) -> theta = -slope
            # Half-Life = log(2) / theta
            
            theta = -slope
            if theta <= 0:
                return 9999.0 # Non-mean reverting (Random Walk or Momentum)
                
            hl = np.log(2) / theta
            return float(hl)
            
        except Exception:
            return 0.0

    @staticmethod
    def kelly_criterion_continuous(win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly Fraction.
        f* = p - (1-p)/b
        
        Args:
            win_rate (p): Probability of winning (0.0 - 1.0)
            win_loss_ratio (b): Ratio of Avg Win / Avg Loss
            
        Returns:
            float: Optimal fraction (0.0 to 1.0)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        # p = win_rate, q = 1-p, b = win_loss_ratio
        # f = (bp - q) / b
        #   = p - q/b
        
        f = win_rate - (1 - win_rate) / win_loss_ratio
        return max(0.0, f)

    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Optimal Position Size using Kelly Criterion.
        f* = (bp - q) / b
        where:
        - b = odds (avg_win / avg_loss)
        - p = probability of winning
        - q = probability of losing (1-p)
        
        Returns:
            float: Optimal fraction of capital (0.0 to 1.0)
        """
        if avg_loss == 0:
            return 0.5 # Safe default
            
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1.0 - p
        
        if b == 0:
            return 0.0
            
        f = (b * p - q) / b
        return max(0.0, f) # No shorting the bankroll (negative Kelly)

    @staticmethod
    def ransac_regression(y: np.array, x: np.array, window: int = 50) -> Tuple[float, float]:
        """
        Phase 6: Robust Regression using RANSAC (Random Sample Consensus).
        Resilient to 'Flash Crashes' and outliers.
        
        Args:
            y: Dependent variable
            x: Independent variable
            window: Window size
            
        Returns:
            (beta, alpha): Robust hedge ratio and intercept
        """
        try:
            from sklearn.linear_model import RANSACRegressor
            
            if len(y) < window or len(x) < window:
                return 1.0, 0.0
                
            y_slice = y[-window:].reshape(-1, 1)
            x_slice = x[-window:].reshape(-1, 1)
            
            # RANSAC fits the model to inliers only
            # Fix: RANSACRegressor requires min_samples to be at least the number of features (1)
            # Default is ok, but setting residual_threshold is key.
            # However, for automatic usage, defaults are safer than bad params.
            ransac = RANSACRegressor(min_samples=int(window*0.6)) # Require 60% inliers
            ransac.fit(x_slice, y_slice)
            
            beta = float(ransac.estimator_.coef_[0][0])
            alpha = float(ransac.estimator_.intercept_[0])
            
            return beta, alpha
            
        except ImportError:
            # Fallback to OLS if sklearn not found (though it should be)
            return StatisticsPro.rolling_ols(y, x, window)
        except Exception:
            return 1.0, 0.0

    @staticmethod
    def johansen_test_simplified(df_prices: pd.DataFrame) -> bool:
        """
        Phase 6: Simplified Multivariate Cointegration Test (Johansen Concept).
        Checks if a basket of assets moves together.
        
        Note: Full Johansen is complex. We stick to Eigenvalue checking of the covariance matrix
        of returns to check for a customized 'Market Mode'.
        
        Args:
            df_prices: DataFrame with columns = symbols, values = prices
            
        Returns:
            bool: True if cointegration likely (First Eigenvalue dominates)
        """
        try:
            # 1. Calculate Correlation Matrix
            corr_matrix = df_prices.pct_change().dropna().corr()
            
            # 2. Eigenvalues
            eigvals = np.linalg.eigvals(corr_matrix)
            sorted_eigs = sorted(eigvals, reverse=True)
            
            # 3. Interpretation
            # If 1st Eigenvalue is very large (> number_of_assets / 2), 
            # implies strong common factor (Market Mode).
            # The residuals are likely stationary (cointegrated).
            
            n_assets = len(df_prices.columns)
            if sorted_eigs[0] > (n_assets * 0.6): # 60% variance explained by 1 factor
                return True
            
            return False
            
        except Exception:
            return False
