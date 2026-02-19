import talib
import numpy as np
from utils.math_kernel import calculate_ema_jit
from utils.logger import logger
from typing import Dict
from core.market_regime_hmm import HiddenMarkovModelDetector

class MarketRegimeDetector:
    """
    Detects market regime to help strategies adapt their behavior.
    
    Regimes:
    - TRENDING_BULL: Strong uptrend (ADX>25, EMA50>EMA200, use ML)
    - TRENDING_BEAR: Strong downtrend (ADX>25, EMA50<EMA200, avoid trading)
    - RANGING: Sideways market (ADX<20, use mean reversion)
    - CHOPPY: Uncertain (ADX 20-25, reduce position size)
    """
    
    def __init__(self):
        self.last_regime = {}  # Per-symbol regime cache
        self.global_regime = 'UNKNOWN'
        self.market_breadth = {'sentiment': 'UNKNOWN', 'bull_pct': 0.0, 'bear_pct': 0.0}
        self.regime_history = {}
        self.last_hurst = 0.5
        self.hmm_detector = HiddenMarkovModelDetector()
        self.transition_risk = 0.0
    
    def detect_regime(self, symbol, bars_1m, bars_5m=None, bars_15m=None, bars_1h=None):
        """
        Detect current market regime for a symbol with MTF Confluence.
        """
        try:
            # Validate input
            if len(bars_1m) < 50:
                return self.last_regime.get(symbol, 'RANGING')
            
            # 1. MTF Trend Filter (5m & 15m)
            mtf_bias = 0
            
            if bars_5m is not None and len(bars_5m) >= 20:
                c5m = bars_5m['close'].astype(np.float64)  # F6: float32→float64 for JIT
                # Fast JIT EMA
                ema20_5m_arr = calculate_ema_jit(c5m, 20)
                mtf_bias += 1 if c5m[-1] > ema20_5m_arr[-1] else -1
                
            if bars_15m is not None and len(bars_15m) >= 20:
                c15m = bars_15m['close'].astype(np.float64)  # F6: float32→float64
                ema20_15m_arr = calculate_ema_jit(c15m, 20)
                mtf_bias += 1 if c15m[-1] > ema20_15m_arr[-1] else -1

            # 2. ADX & Metrics (1m) — F6: Cast to float64 for talib compatibility
            closes_1m = bars_1m['close'].astype(np.float64)
            highs_1m = bars_1m['high'].astype(np.float64)
            lows_1m = bars_1m['low'].astype(np.float64)
            
            # Keep talib for ADX if no jit yet, but extraction is now O(1)
            adx = talib.ADX(highs_1m, lows_1m, closes_1m, timeperiod=14)[-1]
            
            # 3. Trend Direction (1h)
            is_bullish = True
            if bars_1h is not None and len(bars_1h) >= 50:
                closes_1h = bars_1h['close'].astype(np.float64)  # F6: float32→float64
                ema50_1h_arr = calculate_ema_jit(closes_1h, 50)
                is_bullish = closes_1h[-1] > ema50_1h_arr[-1]
            
            # 4. Hurst (1m)
            from utils.statistics_pro import StatisticsPro
            hurst = StatisticsPro.calculate_hurst_exponent(closes_1m[-100:]) if len(closes_1m) >= 100 else 0.5
            self.last_hurst = hurst

            # 5. Logic
            raw_regime = 'CHOPPY'
            
            if adx > 25:
                if is_bullish and mtf_bias >= 0:
                    raw_regime = 'TRENDING_BULL'
                elif not is_bullish and mtf_bias <= 0:
                    raw_regime = 'TRENDING_BEAR'
            elif adx < 20:
                raw_regime = 'RANGING'
                if hurst < 0.4: raw_regime = 'MEAN_REVERTING'

            # Hysteresis
            if symbol not in self.regime_history: self.regime_history[symbol] = []
            self.regime_history[symbol].append(raw_regime)
            if len(self.regime_history[symbol]) > 3: self.regime_history[symbol].pop(0)
            
            if len(self.regime_history[symbol]) == 3 and all(x == raw_regime for x in self.regime_history[symbol]):
                final_regime = raw_regime
            else:
                final_regime = self.last_regime.get(symbol, raw_regime)
            
            self.last_regime[symbol] = final_regime
            
            # --- PHASE 14: HMM REINFORCEMENT ---
            # Solo ejecutamos HMM para el líder o si se requiere validación profunda
            if len(bars_1m) >= 100:
                rets = bars_1m['close'].pct_change().fillna(0).values
                hmm_regime, trans_risk, _ = self.hmm_detector.update(rets)
                self.transition_risk = trans_risk
                
                # Reporte cruzado si hay divergencia crítica
                if hmm_regime == 'TREND_BEAR' and final_regime == 'TRENDING_BULL':
                    logger.warning(f"⚠️ [HMM Divergence] Detected BEAR Trend via HMM while TA indicates BULL for {symbol}. Transition Risk: {trans_risk:.2f}")

            return final_regime
            
        except Exception as e:
            logger.error(f"Regime Error {symbol}: {e}")
            return self.last_regime.get(symbol, 'RANGING')

    def calculate_market_context(self, active_symbols_data: Dict[str, Dict]):
        """
        SOVEREIGN MARKET CONTEXT (Swarm Intelligence).
        QUÉ: Calcula el sentimiento agregado de la canasta Elite.
        POR QUÉ: Evita dependencia de un solo símbolo y mide la amplitud real del mercado.
        
        active_symbols_data: {
            'BTC/USDT': {'1m': bars, '5m': bars, '1h': bars},
            ...
        }
        """
        regimes = []
        
        for symbol, data in active_symbols_data.items():
            r = self.detect_regime(
                symbol, 
                data.get('1m', []), 
                data.get('5m', []), 
                data.get('15m', []), 
                data.get('1h', [])
            )
            regimes.append(r)
            
        if not regimes:
            return self.market_breadth
            
        # Stats
        total = len(regimes)
        bulls = regimes.count('TRENDING_BULL')
        bears = regimes.count('TRENDING_BEAR')
        
        bull_pct = (bulls / total)
        bear_pct = (bears / total)
        
        # Determine Aggregate Sentiment
        # Phase 8.1 Rule: Consensus > 60%
        if bear_pct >= 0.60:
            sentiment = 'TRENDING_BEAR'
        elif bull_pct >= 0.60:
            sentiment = 'TRENDING_BULL'
        else:
            sentiment = 'MIXED'
            
        self.global_regime = sentiment # For backwards compatibility
        self.market_breadth = {
            'sentiment': sentiment,
            'bull_pct': bull_pct,
            'bear_pct': bear_pct,
            'regime_count': total,
            'transition_risk': self.transition_risk
        }
        
        # LOGGING INSTITUCIONAL
        if sentiment == 'TRENDING_BEAR':
            logger.warning(f"🚨 [Sovereign Context] MARKET PANIC: {bear_pct:.0%} of assets are Bearish. Veto Active.")
        elif sentiment == 'TRENDING_BULL':
            logger.info(f"🐂 [Sovereign Context] MARKET FRENZY: {bull_pct:.0%} of assets are Bullish.")
            
        return self.market_breadth

    def detect_global_regime(self, btc_bars_1m, btc_bars_5m, btc_bars_1h):
        """
        DEPRECATED: Use calculate_market_context for breadth-based analysis.
        Kept for transition.
        """
        return self.detect_regime('BTC/USDT', btc_bars_1m, btc_bars_5m, None, btc_bars_1h)
    
    def get_regime_advice(self, regime):
        """
        Get trading advice for each regime (DYNAMIC ADAPTATION).
        Returns: dict with recommended actions + dynamic params.
        """
        # Default Safe Advice
        advice = {
            'action': 'NEUTRAL',
            'leverage': 1,
            'threshold_mod': 0.0,
            'scale': 0.0
        }
        
        try:
            from config import Config
            if getattr(Config.Sniper, 'DYNAMIC_ADAPTATION', False):
                # ✅ EVOLUTIONARY ADAPTATION
                regime_map = getattr(Config.Sniper, 'REGIME_MAP', {})
                params = regime_map.get(regime, regime_map.get('RANGING'))
                
                advice.update({
                    'leverage': params.get('leverage', 1),
                    'threshold_mod': params.get('threshold_mod', 0.0),
                    'scale': params.get('scale', 0.0),
                    'action': 'LONG' if regime in ['TRENDING_BULL', 'RANGING'] else 'NEUTRAL'
                })
                
                # Special cases
                if regime == 'TRENDING_BEAR': advice['action'] = 'SHORT_OR_CASH'
                if regime == 'ZOMBIE': advice['action'] = 'HALT'
                
            else:
                # Fallback to Static Logic (Deprecating)
                if regime == 'TRENDING_BULL':
                    advice.update({'leverage': 5, 'threshold_mod': -0.02, 'scale': 1.0, 'action': 'LONG'})
                elif regime == 'RANGING':
                    advice.update({'leverage': 3, 'threshold_mod': 0.0, 'scale': 0.8, 'action': 'LONG'})
                else:
                    advice.update({'leverage': 1, 'threshold_mod': 0.1, 'scale': 0.0, 'action': 'NEUTRAL'})
                    
        except Exception as e:
            logger.error(f"Advice Error: {e}")
            
        return advice

    def get_learning_factor(self, regime: str) -> float:
        """
        Retorna un multiplicador para el Learning Rate basado en el Régimen.
        Phase 47: Modulation of Neuroplasticity.
        """
        factors = {
            'TRENDING_BULL': 1.0,  # Full learning in clear trends
            'TRENDING_BEAR': 1.0,  
            'RANGING': 0.2,        # Slow learning in noise
            'CHOPPY': 0.0,         # Stop learning in chaos
            'ZOMBIE': 0.0,
            'MEAN_REVERTING': 0.5
        }
        return factors.get(regime, 0.0)

    def is_volatility_shock(self, bars: Dict, atr_period: int = 14, threshold: float = 2.5) -> bool:
        """
        Detects sudden volatility expansion (Shock).
        TR > Threshold * ATR
        """
        try:
            highs = bars['high'].astype(np.float64)    # F6: float32→float64 for talib
            lows = bars['low'].astype(np.float64)
            closes = bars['close'].astype(np.float64)
            
            if len(closes) < atr_period + 1:
                return False
                
            # Calculate ATR (can be JIT optimized later)
            atr_arr = talib.ATR(highs, lows, closes, timeperiod=atr_period)
            current_atr = atr_arr[-1]
            
            # Current True Range
            tr = max(highs[-1] - lows[-1], abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2]))
            
            if tr > current_atr * threshold:
                return True
                
            return False
        except Exception:
            return True # Fail safe: Assume shock if error

