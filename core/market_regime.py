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
    
    def __init__(self, events_queue=None):
        self.last_regime = {}  # Per-symbol regime cache
        self.global_regime = 'UNKNOWN'
        self.market_breadth = {'sentiment': 'UNKNOWN', 'bull_pct': 0.0, 'bear_pct': 0.0}
        self.regime_history = {}
        self.last_hurst = 0.5
        self.hmm_detector = HiddenMarkovModelDetector()
        self.transition_risk = 0.0
        # Adaptive Evolution Protocol: Horizon-Aware Hysteresis
        self.hysteresis_window = 3  # Default (backward-compatible)
        self.regime_confidence = {}  # Per-symbol: float 0-1
        self.horizon_profile = 'DEFAULT'
        self.events_queue = events_queue # Phase 6.2: Emergency Exits
    
    def set_horizon_profile(self, horizon_days: int):
        """
        Adaptive Evolution Protocol: Ajusta la histéresis del régimen
        según el horizonte temporal de operación.
        
        QUÉ: Modifica la ventana de confirmación de cambio de régimen.
        POR QUÉ: En scalping (1D), necesitamos MÁS resistencia al ruido
                  (ventana mayor). En macro (30D), necesitamos reaccionar
                  RÁPIDO a verdaderos cambios de tendencia (ventana menor).
        PARA QUÉ: Minimizar Drawdown en 1D (evita whipsaws) y maximizar
                   Sharpe en 30D (captura tendencias antes).
        CÓMO: 1D→5 barras, 15D→3 barras, 30D→2 barras de histéresis.
        CUÁNDO: Se invoca al inicio de cada sesión de backtest o al conectar.
        DÓNDE: core/market_regime.py → MarketRegimeDetector
        QUIÉN: Engine.py al inicializar, o run_backtest.py.
        """
        if horizon_days <= 1:
            self.hysteresis_window = 5
            self.horizon_profile = 'SCALPING'
        elif horizon_days <= 7:
            self.hysteresis_window = 4
            self.horizon_profile = 'SHORT_TERM'
        elif horizon_days <= 15:
            self.hysteresis_window = 3
            self.horizon_profile = 'MID_TERM'
        else:
            self.hysteresis_window = 2
            self.horizon_profile = 'MACRO'
        
        logger.info(
            f"🔧 [REGIME] Horizon Profile set: {self.horizon_profile} "
            f"({horizon_days}D) → Hysteresis={self.hysteresis_window}"
        )
    
    def detect_regime(self, symbol, bars_1m, bars_5m=None, bars_15m=None, bars_1h=None):
        """
        Detect current market regime for a symbol with MTF Consensus (Phase 6).
        
        QUÉ: Evalúa el régimen en 4 escalas (1m, 5m, 15m, 1h) para un consenso robusto.
        POR QUÉ: Evita señales falsas (whipsaws) al confirmar la tendencia en marcos superiores.
        CÓMO: Calcula un régimen 'candidato' por escala y aplica Votación Ponderada.
        """
        try:
            if len(bars_1m) < 50:
                return self.last_regime.get(symbol, 'RANGING')

            # 1. Definir Escalas Disponibles
            scales = {
                '1m': bars_1m,
                '5m': bars_5m,
                '15m': bars_15m,
                '1h': bars_1h
            }
            
            scale_results = {}
            for tf, bars in scales.items():
                if bars is not None and len(bars) >= 20:
                    scale_results[tf] = self._detect_single_scale_regime(bars)
            
            # 2. Consenso de Votación Ponderada
            # Pesos: 1m(0.1), 5m(0.2), 15m(0.3), 1h(0.4) para darle más peso a la estructura macro
            weights = {'1m': 0.1, '5m': 0.2, '15m': 0.3, '1h': 0.4}
            votes = {'TRENDING_BULL': 0.0, 'TRENDING_BEAR': 0.0, 'RANGING': 0.0, 'CHOPPY': 0.0, 'MEAN_REVERTING': 0.0}
            
            for tf, regime in scale_results.items():
                votes[regime] += weights.get(tf, 0.1)
            
            # El ganador por puntos
            raw_regime = max(votes, key=votes.get)
            
            # 3. Cálculo de Confianza del Consenso
            total_voted_weight = sum(weights[tf] for tf in scale_results.keys())
            consensus_score = votes[raw_regime] / total_voted_weight if total_voted_weight > 0 else 0.0
            
            # Si hay mucha divergencia (consenso < 40%), forzar CHOPPY
            if consensus_score < 0.40 and raw_regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
                raw_regime = 'CHOPPY'

            # 4. Histéresis Adaptativa (Misma lógica anterior para suavizado)
            hw = self.hysteresis_window
            if symbol not in self.regime_history: self.regime_history[symbol] = []
            self.regime_history[symbol].append(raw_regime)
            while len(self.regime_history[symbol]) > hw:
                self.regime_history[symbol].pop(0)
            
            self.regime_confidence[symbol] = consensus_score
            
            if len(self.regime_history[symbol]) >= hw and all(x == raw_regime for x in self.regime_history[symbol]):
                final_regime = raw_regime
            else:
                final_regime = self.last_regime.get(symbol, raw_regime)
            
            previous_regime = self.last_regime.get(symbol, 'UNKNOWN')
            self.last_regime[symbol] = final_regime
            
            # --- PHASE 6.2: SOPHIA EMERGENCY EXITS ---
            # Si pasamos de CHOPPY a BEAR brusco, lanzar EXIT para abortar long positions y proteger PnL
            if previous_regime == 'CHOPPY' and final_regime == 'TRENDING_BEAR':
                if self.events_queue is not None:
                    try:
                        from core.events import SignalEvent
                        from core.enums import SignalType
                        from datetime import datetime, timezone
                        logger.warning(f"🚨 [SOPHIA EMERGENCY] Regime Shift {previous_regime} -> {final_regime} for {symbol}. Emitting EXIT!")
                        self.events_queue.put(SignalEvent(
                            symbol=symbol,
                            datetime=datetime.now(timezone.utc),
                            strategy_id="SOPHIA_EMERGENCY_EXIT",
                            signal_type=SignalType.EXIT,
                            strength=1.0
                        ))
                    except Exception as ev_err:
                        logger.error(f"Failed to emit emergency exit for {symbol}: {ev_err}")
            
            # --- PHASE 14: HMM REINFORCEMENT ---
            if len(bars_1m) >= 100:
                close_prices = bars_1m['close']
                rets = np.zeros(len(close_prices), dtype=np.float32)
                if len(close_prices) > 1:
                    rets[1:] = np.diff(close_prices) / close_prices[:-1]
                    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
                
                hmm_regime, trans_risk, _ = self.hmm_detector.update(rets)
                self.transition_risk = trans_risk
                
                if hmm_regime == 'TREND_BEAR' and final_regime == 'TRENDING_BULL':
                    logger.warning(f"⚠️ [HMM Divergence] HMM=BEAR, TA=BULL for {symbol}. Risk: {trans_risk:.2f}")

            return final_regime
            
        except Exception as e:
            logger.error(f"Regime Error {symbol}: {e}")
            return self.last_regime.get(symbol, 'RANGING')

    def _detect_single_scale_regime(self, bars) -> str:
        """
        Helper para detectar el régimen en una sola escala temporal.
        v3: Lógica difusa (Fuzzy Logic) para transiciones suaves de régimen.
        """
        try:
            from utils.math_kernel import calculate_hurst_exponent
            c = bars['close'].astype(np.float64)
            h = bars['high'].astype(np.float64)
            l = bars['low'].astype(np.float64)
            
            # ADX de Talib (se mantiene a petición del user)
            adx = talib.ADX(h, l, c, timeperiod=14)[-1]
            
            # EMA para tendencia
            ema50 = calculate_ema_jit(c, 50)[-1]
            is_bullish = c[-1] > ema50
            
            # Hurst para persistencia
            hurst = calculate_hurst_exponent(c[-100:].copy(), max_lags=min(30, len(c)//4)) if len(c) >= 20 else 0.5
            
            # --- Lógica Difusa (Probabilities) ---
            # 1. P(Trending): Sube linealmente de ADX 20 a 30, y Hurst de 0.5 a 0.65
            p_trend_adx = max(0.0, min(1.0, (adx - 20) / 10.0))
            p_trend_hurst = max(0.0, min(1.0, (hurst - 0.5) / 0.15))
            score_trending = (p_trend_adx * 0.6) + (p_trend_hurst * 0.4)
            
            # 2. P(Mean-Reverting): Hurst muy bajo (< 0.45) y ADX bajo (< 22)
            p_mr_hurst = max(0.0, min(1.0, (0.45 - hurst) / 0.1))
            p_mr_adx = max(0.0, min(1.0, (22 - adx) / 7.0))
            score_mean_reverting = (p_mr_hurst * 0.7) + (p_mr_adx * 0.3)
            
            # 3. P(Ranging): Variables estancadas. ADX bajo, Hurst neutral (0.45-0.55)
            p_range_adx = max(0.0, min(1.0, (22 - adx) / 7.0))
            dist_to_neutral = abs(hurst - 0.5)
            p_range_hurst = max(0.0, min(1.0, (0.1 - dist_to_neutral) / 0.1))
            score_ranging = (p_range_adx * 0.5) + (p_range_hurst * 0.5)
            
            # 4. P(Choppy): Zona de conflicto. Lo que sobra del universo de probabilidad.
            score_choppy = max(0.0, 1.0 - max(score_trending, score_mean_reverting, score_ranging))
            
            scores = {
                'TRENDING_BULL' if is_bullish else 'TRENDING_BEAR': score_trending,
                'MEAN_REVERTING': score_mean_reverting,
                'RANGING': score_ranging,
                'CHOPPY': score_choppy
            }
            
            # Añadir un sesgo inercial para preferir mantenerse en el trend si el score es muy parecido
            # (opcional, por ahora se escoge el máximo matemático)
            best_regime = max(scores, key=scores.get)
            
            # Fallback a CHOPPY si la máxima certeza es demasiado baja (ruido total)
            if scores[best_regime] < 0.35:
                return 'CHOPPY'
                
            return best_regime
            
        except Exception as e:
            logger.error(f"Regime Error interno: {e}")
            return 'RANGING'

    def calculate_market_context(self, active_symbols_data: Dict[str, Dict]):
        """
        SOVEREIGN MARKET CONTEXT (Swarm Intelligence).
        QUÉ: Calcula el sentimiento agregado de la canasta Elite.
        POR QUÉ: Evita dependencia de un solo símbolo y mide la amplitud real del mercado.
        NUEVO (Fuzzy/V3): Pondera el voto de cada activo por su volumen negociado (Quote Vol), dando
                          más peso de voto a los activos con mayor actividad institucional.
        """
        regimes = {}
        volumes = {}
        
        for symbol, data in active_symbols_data.items():
            r = self.detect_regime(
                symbol, 
                data.get('1m', []), 
                data.get('5m', []), 
                data.get('15m', []), 
                data.get('1h', [])
            )
            regimes[symbol] = r
            
            # Calcular volumen en USD para ponderación de voto
            bars_1m = data.get('1m', [])
            if len(bars_1m) > 0:
                last_bar = bars_1m.iloc[-1]
                quote_vol = last_bar['volume'] * last_bar['close'] if 'volume' in last_bar else 1.0
                volumes[symbol] = quote_vol
            else:
                volumes[symbol] = 1.0
            
        if not regimes:
            return self.market_breadth
            
        # Ponderar por volumen para evitar que shitcoins distorsionen el mercado global
        total_volume = sum(volumes.values()) + 1e-9
        
        votes = {'TRENDING_BULL': 0.0, 'TRENDING_BEAR': 0.0, 'RANGING': 0.0, 'CHOPPY': 0.0, 'MEAN_REVERTING': 0.0}
        
        for sym, reg in regimes.items():
            if reg in votes:
                weight = volumes[sym] / total_volume
                votes[reg] += weight
        
        bull_pct = votes['TRENDING_BULL']
        bear_pct = votes['TRENDING_BEAR']
        
        # Determine Aggregate Sentiment
        # Umbral bajado de 60% a 50% ya que al ponderar por volumen, el voto está más concentrado
        if bear_pct >= 0.50:
            sentiment = 'TRENDING_BEAR'
        elif bull_pct >= 0.50:
            sentiment = 'TRENDING_BULL'
        else:
            sentiment = 'MIXED'
            
        self.global_regime = sentiment # For backwards compatibility
        self.market_breadth = {
            'sentiment': sentiment,
            'bull_pct': bull_pct,
            'bear_pct': bear_pct,
            'regime_count': len(regimes),
            'transition_risk': self.transition_risk,
            'symbol_regimes': regimes
        }
        
        # LOGGING INSTITUCIONAL
        if sentiment == 'TRENDING_BEAR':
            logger.warning(f"🚨 [Sovereign Context] MARKET PANIC: {bear_pct:.0%} of vol is Bearish. Veto Active.")
        elif sentiment == 'TRENDING_BULL':
            logger.info(f"🐂 [Sovereign Context] MARKET FRENZY: {bull_pct:.0%} of vol is Bullish.")
            
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

    def predict_regime_shift(self, symbol: str, bars_1m) -> Dict[str, Any]:
        """
        SOPHIA §6.3: Forecasting del próximo estado.
        """
        if len(bars_1m) < 30:
            return {"forecast": "STABLE", "tension": 0.0}
            
        try:
            closes = bars_1m['close'].astype(np.float64)
            highs = bars_1m['high'].astype(np.float64)
            lows = bars_1m['low'].astype(np.float64)
            
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            
            if len(adx) < 5 or len(atr) < 5: 
                return {"forecast": "STABLE", "tension": 0.0}
            
            adx_slope = adx[-1] - adx[-5]
            atr_slope = atr[-1] - atr[-5]
            
            tension = (adx_slope * 0.5) + (atr_slope * 0.5)
            
            forecast = "STABLE"
            if tension > 1.5: forecast = "VOLATILITY_EXPANSION_LIKELY"
            elif tension < -1.5: forecast = "CONSOLIDATION_LIKELY"
            
            return {
                "forecast": forecast,
                "tension": tension,
                "adx_slope": adx_slope
            }
        except:
            return {"forecast": "UNKNOWN", "tension": 0.0}

