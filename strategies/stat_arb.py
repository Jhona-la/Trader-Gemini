"""
📊 AEGIS-ULTRA: Statistical Arbitrage Module
QUÉ: Cointegration (Lite-EG) & Dynamic Correlation Matrix using NumPy.
POR QUÉ: Detect pairs trading opportunities and Systemic Risk (Contagion).
PARA QUÉ: HFT StatArb without heavy 'statsmodels' dependency.
"""

import numpy as np
from utils.logger import logger
from dataclasses import dataclass

@dataclass
class CointegrationResult:
    is_cointegrated: bool
    p_value: float
    beta: float # Hedge Ratio
    c: float    # Constant
    z_score: float # Current Z-Score of spread

class StatArbEngine:
    
    @staticmethod
    def calculate_correlation_matrix(returns_matrix: np.ndarray) -> np.ndarray:
        """
        🚀 Fast Pearson Correlation Matrix
        Input: (n_samples, n_assets)
        """
        try:
            # Check dimensions
            if returns_matrix.size == 0 or returns_matrix.shape[0] < 2:
                return np.array([])
                
            # Centering
            means = np.mean(returns_matrix, axis=0)
            centered = returns_matrix - means
            
            # Covariance
            cov = np.dot(centered.T, centered) / (returns_matrix.shape[0] - 1)
            
            # Standard Deviations
            stds = np.std(returns_matrix, axis=0, ddof=1)
            
            # Correlation = Cov / (std_x * std_y)
            # Outer product of stds gives denominator matrix
            denominator = np.outer(stds, stds)
            
            # Handle division by zero
            denominator[denominator == 0] = 1e-9
            
            corr_matrix = cov / denominator
            
            # Clamp to [-1, 1] for numerical stability
            corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
            
            return corr_matrix
        except Exception as e:
            logger.error(f"Correlation Matrix Error: {e}")
            return np.array([])

    @staticmethod
    def get_systemic_risk(corr_matrix: np.ndarray) -> float:
        """
        Calculates average correlation of the fleet.
        If > 0.85, market is in synchronized move (Crash/Rally).
        """
        if corr_matrix.size == 0: return 0.0
        
        # Upper triangle only, excluding diagonal
        n_assets = corr_matrix.shape[0]
        if n_assets < 2: return 0.0
        
        # Mask for upper triangle
        mask = np.triu_indices(n_assets, k=1)
        avg_corr = np.mean(corr_matrix[mask])
        
        return float(avg_corr)

    @staticmethod
    def lite_engle_granger(y: np.ndarray, x: np.ndarray) -> CointegrationResult:
        """
        🚀 Lite-EG: NumPy-based Cointegration Test
        Approximation of Augmented Dickey-Fuller on residuals.
        Model: Y = beta * X + c + epsilon
        """
        try:
            n = len(y)
            if n != len(x) or n < 30:
                return CointegrationResult(False, 1.0, 0.0, 0.0, 0.0)

            # 1. Linear Regression (OLS) to find Residuals
            # Design Matrix [X, 1]
            A = np.vstack([x, np.ones(n)]).T
            
            # Solve A * [beta, c] = y
            # Use lstsq for speed
            result = np.linalg.lstsq(A, y, rcond=None)
            params = result[0]
            beta, c = params[0], params[1]
            
            # Calculate Residuals (Spread)
            residuals = y - (beta * x + c)
            
            # 2. ADF Test on Residuals (Simplified)
            # Delta_Res = gamma * Res_lag + error
            # t-stat of gamma checks mean reversion
            
            res_lag = residuals[:-1]
            res_delta = np.diff(residuals)
            
            # Regress Delta on Lag (No constant, residuals are centered)
            # A_adf = res_lag.reshape(-1, 1)
            # gamma = lstsq(A_adf, res_delta)
            
            # Fast scalar regression for 1 variable
            # gamma = sum(x*y) / sum(x*x)
            numerator = np.dot(res_lag, res_delta)
            denominator = np.dot(res_lag, res_lag)
            
            if denominator == 0:
                return CointegrationResult(False, 1.0, beta, c, 0.0)
                
            gamma = numerator / denominator
            
            # Calculate Standard Error of gamma
            # e_adf = res_delta - gamma * res_lag
            # var_e = sum(e^2) / (n - 2)
            # var_gamma = var_e / sum(lag^2)
            # t_stat = gamma / sqrt(var_gamma)
            
            e_adf = res_delta - (gamma * res_lag)
            sigma2_e = np.dot(e_adf, e_adf) / (len(res_delta) - 1)
            
            if sigma2_e == 0: t_stat = -10.0 # Perfect fit
            else:
                std_gamma = np.sqrt(sigma2_e / denominator)
                t_stat = gamma / std_gamma
            
            # Critical Values for EG (No constant in ADF, N=Large)
            # MacKinnon (1994) approx for N=infinity, no trend:
            # 1%: -3.90, 5%: -3.34, 10%: -3.04 (for 2 variables)
            
            # Previous values were for standard ADF, EG residuals distribution is different.
            is_coint = t_stat < -3.34
            
            # P-value approximation
            if t_stat < -3.90: p_val = 0.01
            elif t_stat < -3.34: p_val = 0.05
            elif t_stat < -3.04: p_val = 0.10
            else: p_val = 1.0 # Clearly not cointegrated
            
            # Current Z-Score of the Spread
            spread_mean = np.mean(residuals)
            spread_std = np.std(residuals)
            z_score = (residuals[-1] - spread_mean) / spread_std if spread_std > 0 else 0.0
            
            return CointegrationResult(is_coint, round(p_val, 3), beta, c, z_score)
            

        except Exception as e:
            # logger.error(f"Lite-EG Error: {e}") # Verbose
            return CointegrationResult(False, 1.0, 0.0, 0.0, 0.0)

from core.events import SignalEvent, SignalType
from datetime import datetime, timezone
from strategies.strategy import Strategy
from config import Config

class StatArbStrategy(Strategy):
    """
    StatArb Strategy Wrapper — HORIZON-AWARE (Forensic Phase 4)
    ═══════════════════════════════════════════════════════════════
    QUÉ: Estrategia de arbitraje estadístico por cointegración.
    POR QUÉ: Arbitraje estadístico en SWING captura mean-reversion
      de spreads a largo plazo, necesita TP/SL más amplios.
    CÓMO: Lee Config.Horizons.Scalping o SWING_PARAMS.
    DÓNDE: strategies/stat_arb.py → StatArbStrategy
    QUIÉN: StatArbStrategy (Quant Developer)
    ═══════════════════════════════════════════════════════════════
    """
    def __init__(self, data_provider, events_queue, horizon="SCALPING", priority=1):
        super().__init__()
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.horizon = horizon
        self.priority = priority
        self.strategy_id = f"STATARB_{horizon}"
        self.engine = StatArbEngine()
        
        # ================================================================
        # PHASE FORENSIC-4: HORIZON-AWARE PARAMETER LOADING
        # ================================================================
        if horizon.upper() == 'SCALPING':
            h_params = getattr(Config.Horizons, 'Scalping', {})
        elif horizon.upper() == 'SWING':
            h_params = getattr(Config.Horizons, 'Swing', {})
        else:
            h_params = {}
        
        from utils.logger import logger
        
        self.TP_PCT = h_params.get('tp_pct', 0.015)
        self.SL_PCT = h_params.get('sl_pct', 0.015)
        self.SOPHIA_TTL = 300.0 if horizon == 'SCALPING' else 3600.0
        
        logger.info(f"📐 STATARB [{horizon}] INITIALIZED | TP={self.TP_PCT*100:.2f}% SL={self.SL_PCT*100:.2f}%")

    def generate_signals(self, event):
        from config import Config
        pairs = Config.TRADING_PAIRS
        if len(pairs) < 2: return
        
        # Simple scan
        try:
            x_sym = "BTC/USDT"
            y_sym = pairs[1] if pairs[0] == "BTC/USDT" else pairs[0]
            
            data_x = self.data_provider.get_data(x_sym)
            data_y = self.data_provider.get_data(y_sym)
            
            if data_x is None or data_y is None or len(data_x) < 500 or len(data_y) < 500:
                return
                
            px = data_x['close'].values[-500:]
            py = data_y['close'].values[-500:]
            
            # FORENSIC-4 FIX: Correct method name and regression order (regress py on px)
            coint = self.engine.lite_engle_granger(py, px)
            
            if coint.is_cointegrated and abs(coint.z_score) > 2.0:
                signal_type = SignalType.SHORT if coint.z_score > 0 else SignalType.LONG
                current_price = data_y['close'].values[-1]
                
                sophia_report_dict = {}
                if hasattr(self, 'sophia') and self.sophia:
                    sophia_report = self.sophia.analyze(
                        symbol=y_sym,
                        direction=signal_type.name,
                        signal_strength=0.85,
                        setups={'z_score': coint.z_score},
                        confluence_score=1.0,
                        tp_pct=self.TP_PCT,
                        sl_pct=self.SL_PCT,
                        returns=None,
                        ttl_seconds=self.SOPHIA_TTL,
                        regime="RANGING"
                    )
                    
                    if sophia_report.win_probability < 0.70:
                        return
                    sophia_report_dict = sophia_report.to_dict()
                    
                signal = SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=y_sym,
                    datetime=datetime.now(timezone.utc),
                    signal_type=signal_type,
                    strength=0.85,
                    ml_confidence=sophia_report.win_probability if 'sophia_report' in locals() and sophia_report else 0.5,
                    atr=0.0,
                    tp_pct=self.TP_PCT,
                    sl_pct=self.SL_PCT,
                    current_price=current_price,
                    horizon=self.horizon,
                    priority=self.priority,
                    metadata={'sophia': sophia_report_dict, 'z_score': coint.z_score}
                )
                self.events_queue.put(signal)
        except Exception as e:
            logger.debug(f"Silent exception caught: {e}")
            
    def calculate_signals(self, event):
        self.generate_signals(event)

    def check_exit(self, position, current_price, data_provider, now=None):
        if now is None:
            now = datetime.now(timezone.utc)
            
        qty = position.get("quantity", 0.0)
        symbol = position.get("symbol")
        pos_horizon = position.get("horizon", self.horizon)
        
        # 🧠 [INTELLIGENT EXIT]: Sophia AI Real-time validation
        if hasattr(self, 'sophia') and self.sophia:
            try:
                df_primary = data_provider.get_data(symbol, "5m")
                if df_primary is not None and not df_primary.empty:
                    sophia_report = self.sophia.get_insight(symbol, df_primary)
                    if sophia_report:
                        current_prob = sophia_report.win_probability
                        if current_prob < 0.45:
                            logger.warning(f"🧠 [SOPHIA EXIT] StatArb {symbol} AI confidence dropped to {current_prob:.2f}")
                            return SignalEvent(
                                strategy_id="SOPHIA_EMERGENCY_EXIT",
                                symbol=symbol,
                                datetime=now,
                                signal_type=SignalType.EXIT,
                                horizon=pos_horizon,
                                metadata={"exit_reason": f"SOPHIA_LOSS_OF_CONFIDENCE:{current_prob:.2f}"}
                            )
            except Exception as e:
                logger.debug(f"⚠️ Sophia exit check failed for {symbol}: {e}")
        return None
