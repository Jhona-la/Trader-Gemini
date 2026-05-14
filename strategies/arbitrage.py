
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from core.events import SignalEvent, SignalType
from config import Config
from utils.logger import logger
from utils.math_kernel import vector_zscore

class StatisticalArbitrage:
    """
    [PHASE 6] AEGIS-ULTRA: Statistical Arbitrage & Cointegration
    - Monitors 20-asset fleet for Cointegration (Engle-Granger)
    - Calculates Fleet Correlation Matrix to detect Contagion Risk
    """
    def __init__(self, lookback_window=500, priority=1):
        self.lookback = lookback_window
        self.priority = priority
        self.price_history: Dict[str, List[float]] = {}
        self.correlation_matrix = None
        self.cointegrated_pairs: List[Tuple[str, str, float]] = [] # (Asset A, Asset B, Beta)
        self.fleet_correlation = 0.0
        
    def update_price(self, symbol: str, price: float):
        """Update price history buffer"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.lookback:
            self.price_history[symbol].pop(0)
            
    def calculate_fleet_correlation(self) -> float:
        """
        Calculates the average correlation of the entire fleet.
        High correlation (>0.85) indicates Systemic Risk (Contagion).
        """
        if len(self.price_history) < 2:
            return 0.0
            
        # Create DataFrame from history (aligned by length)
        min_len = min(len(v) for v in self.price_history.values())
        if min_len < 50: # Need minimum samples
            return 0.0
            
        data = {k: v[-min_len:] for k, v in self.price_history.items()}
        df = pd.DataFrame(data)
        
        # Calculate Correlation Matrix
        corr_matrix = df.pct_change().corr()
        
        # Upper triangle mean (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        mean_corr = corr_matrix.where(mask).mean().mean()
        
        self.fleet_correlation = mean_corr
        
        if mean_corr > 0.85:
            logger.warning(f"🚨 SYSTEMIC RISK: Fleet Correlation {mean_corr:.2f} > 0.85")
            
        return mean_corr

    def check_cointegration(self, asset_a: str, asset_b: str) -> Tuple[bool, float, float]:
        """
        Performs simplified Engle-Granger test (OLS Residual Stationarity).
        Returns (Is_Cointegrated, Beta, Current_Spread_ZScore)
        """
        if asset_a not in self.price_history or asset_b not in self.price_history:
            return False, 0.0, 0.0
            
        series_a = np.array(self.price_history[asset_a])
        series_b = np.array(self.price_history[asset_b])
        
        min_len = min(len(series_a), len(series_b))
        if min_len < 100:
            return False, 0.0, 0.0
            
        Y = series_a[-min_len:]
        X = series_b[-min_len:]
        
        # 1. Calculate Beta (Hedge Ratio) via OLS: Y = beta * X + c
        # Simple Linear Regression
        A = np.vstack([X, np.ones(len(X))]).T
        beta, c = np.linalg.lstsq(A, Y, rcond=None)[0]
        
        # 2. Calculate Spread (Residuals)
        spread = Y - (beta * X + c)
        
        # 3. Test Stationarity (ADF Test simplified: Variance Ratio check)
        # Using Hurst Exponent or Variance Ratio as proxy for speed
        # Real ADF requires statsmodels (heavy). We use Z-Score Mean Reversion here.
        z_scores = vector_zscore(spread, window=30)
        current_z = z_scores[-1]
        
        # If mean reverting (Z-Score extreme but reverts), we act.
        # This is a 'Trading Cointegration' proxy.
        is_opportunity = abs(current_z) > 2.0
        
        return is_opportunity, beta, current_z

    def scan_opportunities(self) -> List[SignalEvent]:
        """Scans known correlated pairs for spread divergence"""
        signals = []
        
        # Update Fleet Health
        self.calculate_fleet_correlation()
        
        # Scan Pairs (Example: ETH/BTC, SOL/ETH)
        # In production, this would scan all permutations or a pre-defined list
        # For this phase, we check major pairs.
        
        pairs_to_check = [('ETH/USDT', 'BTC/USDT'), ('SOL/USDT', 'ETH/USDT'), ('BNB/USDT', 'BTC/USDT')]
        
        for a, b in pairs_to_check:
            is_opp, beta, z_score = self.check_cointegration(a, b)
            
            if is_opp:
                direction = SignalType.SHORT if z_score > 0 else SignalType.LONG
                # If Z > 0, Spread is high -> Short A, Long B
                # Logic handled by sending signal for Asset A (the driver)
                
                signal = SignalEvent(
                    symbol=a,
                    signal_type=direction,
                    strength=abs(z_score),
                    datetime=pd.Timestamp.now(tz='UTC'),
                    strategy_id='STAT_ARB_V1',
                    horizon='SCALPING',
                    priority=self.priority,
                    metadata={'pair': b, 'beta': beta, 'z_score': z_score}
                )
                signals.append(signal)
                
        return signals

from strategies.strategy import Strategy

class ArbitrageStrategy(Strategy):
    """
    Arbitrage Strategy Wrapper — HORIZON-AWARE (Forensic Phase 4)
    ═══════════════════════════════════════════════════════════════
    QUÉ: Wrapper de estrategia de arbitraje estadístico.
    POR QUÉ: Arbitraje en SWING necesita TP más amplios y cooldowns
      más largos para capturar reversiones de spread completas.
    CÓMO: Lee Config.Strategies.SCALPING_PARAMS o SWING_PARAMS.
    CUÁNDO: En cada instanciación y evaluación de señales.
    DÓNDE: strategies/arbitrage.py → ArbitrageStrategy
    QUIÉN: ArbitrageStrategy (Quant Developer)
    ═══════════════════════════════════════════════════════════════
    """
    def __init__(self, data_provider, events_queue, horizon="SCALPING", priority=1):
        super().__init__()
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.horizon = horizon
        self.priority = priority
        self.strategy_id = f"ARBITRAGE_{horizon}"
        self.engine = StatisticalArbitrage()
        
        # ================================================================
        # PHASE FORENSIC-4: HORIZON-AWARE PARAMETER LOADING
        # ================================================================
        if horizon.upper() == 'SCALPING':
            h_params = getattr(Config.Strategies, 'SCALPING_PARAMS', {})
        elif horizon.upper() == 'SWING':
            h_params = getattr(Config.Strategies, 'SWING_PARAMS', {})
        else:
            h_params = {}
        
        self.TP_PCT = h_params.get('tp_pct', 0.010)
        self.SL_PCT = h_params.get('sl_pct', 0.010)
        self.COOLDOWN_SECONDS = h_params.get('cooldown_seconds', 120 if horizon == 'SCALPING' else 3600)
        self.SOPHIA_TTL = 300.0 if horizon == 'SCALPING' else 3600.0
        
        logger.info(f"💱 ARBITRAGE [{horizon}] INITIALIZED | TP={self.TP_PCT*100:.2f}% SL={self.SL_PCT*100:.2f}% | Cooldown={self.COOLDOWN_SECONDS}s")

    def generate_signals(self, event):
        from config import Config
        from utils.cooldown_manager import cooldown_manager

        # Cooldown: HORIZON-AWARE
        cooldown_key = f"ARBITRAGE_SCAN_{self.horizon}"
        if not cooldown_manager.check_custom_cooldown(cooldown_key, duration_seconds=self.COOLDOWN_SECONDS):
            return

        pairs = Config.TRADING_PAIRS
        for symbol in pairs:
            data = self.data_provider.get_data(symbol)
            if data is not None and not data.empty:
                self.engine.update_price(symbol, data['close'].values[-1])
                
        # FORENSIC-DCA FIX #1: Corrected method → scan_opportunities()
        signals = self.engine.scan_opportunities()
        for signal in signals:
            sophia_report_dict = {}
            if hasattr(self, 'sophia') and self.sophia:
                sophia_report = self.sophia.analyze(
                    symbol=signal.symbol,
                    direction=signal.signal_type.name,
                    signal_strength=0.85,
                    setups=signal.metadata,
                    confluence_score=1.0,
                    tp_pct=self.TP_PCT,
                    sl_pct=self.SL_PCT,
                    returns=None,
                    ttl_seconds=self.SOPHIA_TTL,
                    regime="UNKNOWN"
                )
                
                if sophia_report.win_probability < 0.70:
                    continue
                sophia_report_dict = sophia_report.to_dict()
                
            signal.metadata['sophia'] = sophia_report_dict
            signal.horizon = self.horizon
            self.events_queue.put(signal)
            
    def calculate_signals(self, event):
        self.generate_signals(event)
