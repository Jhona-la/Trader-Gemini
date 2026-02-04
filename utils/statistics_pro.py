import numpy as np
from scipy import stats
try:
    from hurst import compute_hc
except ImportError:
    compute_hc = None

from utils.logger import logger

class StatisticsPro:
    """
    Advanced Statistical Inference Engine (Phase 6).
    Tools: Hurst Exponent (Memory), Bayesian Inference, Monte Carlo.
    """

    @staticmethod
    def calculate_hurst_exponent(price_series: list) -> float:
        """
        Calculates the Hurst Exponent to determine market memory.
        H < 0.5: Mean Reverting (Anti-persistent)
        H = 0.5: Random Walk (Brownian)
        H > 0.5: Trending (Persistent)
        """
        if not compute_hc:
            return 0.5 # Default to random if lib missing
            
        if len(price_series) < 100:
            return 0.5 # Insufficient data
            
        try:
            # compute_hc returns H, c, data
            H, c, data = compute_hc(price_series, kind='price', simplified=True)
            return H
        except Exception as e:
            logger.error(f"⚠️ Hurst Calc Error: {e}")
            return 0.5

    @staticmethod
    def bayesian_win_rate(wins: int, losses: int, prior_alpha=1, prior_beta=1) -> float:
        """
        Calculates the posterior mean of the Win Rate using a Beta distribution.
        Updates the "Prior" belief with new evidence.
        """
        # Posterior Alpha = Prior Alpha + Wins
        # Posterior Beta = Prior Beta + Losses
        post_alpha = prior_alpha + wins
        post_beta = prior_beta + losses
        
        # Mean of Beta Distribution = Alpha / (Alpha + Beta)
        return post_alpha / (post_alpha + post_beta)

    @staticmethod
    def monte_carlo_ruin_prob(win_rate: float, risk_reward: float, risk_per_trade: float, 
                             simulations=1000, trades_per_sim=100) -> float:
        """
        Estimates Probability of Ruin (Drawdown > 50%) using Monte Carlo.
        """
        ruin_count = 0
        start_equity = 1.0 # Normalized
        
        for _ in range(simulations):
            equity = start_equity
            for _ in range(trades_per_sim):
                if np.random.random() < win_rate:
                    equity += (risk_per_trade * risk_reward)
                else:
                    equity -= risk_per_trade
                
                if equity <= 0.5: # Ruin threshold (50% DD)
                    ruin_count += 1
                    break
                    
        return ruin_count / simulations

    @staticmethod
    def kelly_criterion_continuous(win_rate: float, reward_risk: float) -> float:
        """
        Calculates fraction of bankroll to wager using Kelly Criterion.
        K = p - (1-p)/b
        """
        if reward_risk <= 0: return 0.0
        
        q = 1 - win_rate
        kelly = win_rate - (q / reward_risk)
        return max(0.0, kelly)

    @staticmethod
    def generate_monte_carlo_paths(pnl_returns: list, start_equity=100.0, n_sims=1000, n_period=100) -> np.ndarray:
        """
        Generates N future equity paths using bootstrap resampling of historical PnL returns.
        Returns: Numpy Matrix [n_sims, n_period]
        """
        if not pnl_returns or len(pnl_returns) < 10:
            return np.array([])
            
        returns_array = np.array(pnl_returns)
        # Bootstrap resampling: Randomly select returns from history with replacement
        # Shape: [n_sims, n_period]
        random_returns = np.random.choice(returns_array, size=(n_sims, n_period))
        
        # Calculate cumulative equity paths
        # Equity_t = Equity_{t-1} * (1 + r_t)
        # Cumulative product along the period axis
        cumulative_returns = np.cumprod(1 + random_returns, axis=1)
        paths = start_equity * cumulative_returns
        
        # Prepend start_equity column for t=0
        start_col = np.full((n_sims, 1), start_equity)
        paths = np.hstack((start_col, paths))
        
        return paths

    @staticmethod
    def calculate_stress_metrics(paths: np.ndarray, ruin_threshold=0.5) -> dict:
        """
        Calculates Risk of Ruin (PoR) and other stress metrics from MC paths.
        ruin_threshold: Fraction of starting equity (0.5 = 50% Drawdown)
        """
        if paths.size == 0:
            return {'por': 0.0, 'stress_score': 0.0, 'mcl': 0}
            
        n_sims = paths.shape[0]
        start_equity = paths[0, 0]
        ruin_level = start_equity * ruin_threshold
        
        # 1. Probability of Ruin (PoR)
        # Check if any point in the path drops below ruin_level
        ruined_paths = np.any(paths < ruin_level, axis=1)
        por_pct = (np.sum(ruined_paths) / n_sims) * 100
        
        # 2. Stress Score (0-100, Higher is better)
        # Simple inversion of PoR
        stress_score = max(0.0, 100.0 - por_pct)
        
        # 3. VaR (Value at Risk) - 95% Confidence max loss at end of period
        final_equities = paths[:, -1]
        var_95 = np.percentile(final_equities, 5) 
        var_pct = ((start_equity - var_95) / start_equity) * 100
        
        return {
            'por': por_pct,
            'stress_score': stress_score,
            'var_95_pct': max(0.0, var_pct)
        }
