import talib
import numpy as np
from utils.math_kernel import calculate_ema_jit, calculate_adx_jit, calculate_atr_jit
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
        self.hmm_detectors = {}  # Isolated HMM instances per symbol
        self.hmm_update_counts = {}  # Track ticks for rolling calibration
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
                            strength=1.0,
                            horizon="SCALPING"
                        ))
                    except Exception as ev_err:
                        logger.error(f"Failed to emit emergency exit for {symbol}: {ev_err}")
            
            # --- PHASE 14: HMM REINFORCEMENT ---
            if len(bars_1m) >= 100:
                # Check if it's a DataFrame, list of dicts, or recarray
                if hasattr(bars_1m, 'iloc'):
                    close_prices = np.array(bars_1m['close'], dtype=np.float64)
                elif hasattr(bars_1m, 'to_pandas'):
                    close_prices = np.array(bars_1m.to_pandas()['close'], dtype=np.float64)
                elif isinstance(bars_1m, list) and len(bars_1m) > 0 and isinstance(bars_1m[0], dict):
                    close_prices = np.array([b['close'] for b in bars_1m], dtype=np.float64)
                else:
                    close_prices = np.array(bars_1m['close'], dtype=np.float64)
                    
                rets = np.zeros(len(close_prices), dtype=np.float64)
                if len(close_prices) > 1:
                    rets[1:] = np.diff(close_prices) / close_prices[:-1]
                    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Instanciar HMM específico por símbolo para evitar contaminación
                if symbol not in self.hmm_detectors:
                    self.hmm_detectors[symbol] = HiddenMarkovModelDetector()
                    self.hmm_detectors[symbol].calibrate(rets)
                    self.hmm_update_counts[symbol] = 0
                
                # Calibración adaptiva rodante cada 1440 ticks (1 día en 1m)
                self.hmm_update_counts[symbol] += 1
                if self.hmm_update_counts[symbol] % 1440 == 0:
                    self.hmm_detectors[symbol].calibrate(rets)
                
                hmm_regime, trans_risk, _ = self.hmm_detectors[symbol].update(rets)
                self.transition_risk = trans_risk
                
                if hmm_regime == 'TRENDING_BEAR' and final_regime == 'TRENDING_BULL':
                    logger.warning(f"⚠️ [HMM Divergence] HMM=BEAR, TA=BULL for {symbol}. Risk: {trans_risk:.2f}")

            return final_regime
            
        except Exception as e:
            logger.error(f"Regime Error {symbol}: {e}")
            return self.last_regime.get(symbol, 'RANGING')

    def get_current_regime(self, symbol=None) -> str:
        """
        Devuelve el régimen actual almacenado.
        Utilizado extensivamente por estrategias.
        """
        if symbol and symbol in self.last_regime:
            return self.last_regime[symbol]
        return self.global_regime

    def get_regime_locks(self, symbol=None) -> Dict[str, bool]:
        """
        🚀 FASE 22: Clasificador Estricto de Regímenes
        Genera candados estrictos para evitar cruces mortales entre horizontes.
        """
        regime = self.get_current_regime(symbol)
        locks = {
            'LOCK_SWING': False,
            'LOCK_SCALP_LONG': False,
            'LOCK_SCALP_SHORT': False
        }
        
        if regime in ['CHOPPY', 'RANGING']:
            # Prohibido Swing en mercados laterales/ruidosos
            locks['LOCK_SWING'] = True
        elif regime == 'TRENDING_BULL':
            # Prohibido short-scalping (contra tendencia) en Bull fuerte
            locks['LOCK_SCALP_SHORT'] = True
        elif regime == 'TRENDING_BEAR':
            # Prohibido long-scalping (contra tendencia) en Bear fuerte
            locks['LOCK_SCALP_LONG'] = True
            
        return locks

    def _detect_single_scale_regime(self, bars) -> str:
        """
        Helper para detectar el régimen en una sola escala temporal.
        v3: Lógica difusa (Fuzzy Logic) para transiciones suaves de régimen.
        """
        try:
            from utils.math_kernel import calculate_hurst_exponent, compute_fuzzy_regime_scores_jit
            
            # --- O(1) ZERO-COPY NATIVE EXTRACTION ---
            if isinstance(bars, np.ndarray) and getattr(bars.dtype, 'names', None):
                # Backtest Mode (Structured Arrays) - Ultra Fast Path
                c = bars['close'].astype(np.float64)
                h = bars['high'].astype(np.float64)
                l = bars['low'].astype(np.float64)
            elif isinstance(bars, dict) and 'close' in bars:
                c = np.array(bars['close'], dtype=np.float64)
                h = np.array(bars['high'], dtype=np.float64)
                l = np.array(bars['low'], dtype=np.float64)
            elif hasattr(bars, 'iloc'):
                # Live Mode (Pandas)
                c = bars['close'].values.astype(np.float64)
                h = bars['high'].values.astype(np.float64)
                l = bars['low'].values.astype(np.float64)
            elif isinstance(bars, list) and len(bars) > 0 and isinstance(bars[0], dict):
                c = np.array([b.get('close', 0.0) for b in bars], dtype=np.float64)
                h = np.array([b.get('high', 0.0) for b in bars], dtype=np.float64)
                l = np.array([b.get('low', 0.0) for b in bars], dtype=np.float64)
            else:
                c = np.array(bars['close'], dtype=np.float64)
                h = np.array(bars['high'], dtype=np.float64)
                l = np.array(bars['low'], dtype=np.float64)
            
            # --- Lógica Difusa (Probabilities) ---
            # Dynamic ADX scaling based on Horizon Profile
            tm = 1.0
            if self.horizon_profile == 'SCALPING': tm = 0.6
            elif self.horizon_profile == 'SHORT_TERM': tm = 0.8
            elif self.horizon_profile == 'MACRO': tm = 1.2
            
            adx_period = max(7, int(14 * tm))
            ema_period = max(20, int(50 * tm))
            
            # ADX and EMA with Dynamic Periods
            adx = calculate_adx_jit(h, l, c, period=adx_period)[-1]
            ema_trend = calculate_ema_jit(c, ema_period)[-1]
            is_bullish = c[-1] > ema_trend
            
            # Hurst para persistencia
            hurst = calculate_hurst_exponent(c[-100:].copy(), max_lags=min(30, len(c)//4)) if len(c) >= 20 else 0.5
            
            # --- [NANO-SPEED] Precompiled Fuzzy Logic ---
            best_idx, best_score = compute_fuzzy_regime_scores_jit(
                adx=float(adx),
                hurst=float(hurst),
                tm_multiplier=float(tm),
                is_bullish=bool(is_bullish)
            )
            
            regime_map = ['TRENDING_BEAR', 'TRENDING_BULL', 'MEAN_REVERTING', 'RANGING', 'CHOPPY']
            best_regime = regime_map[best_idx]
            
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
                try:
                    if hasattr(bars_1m, 'iloc'):
                        last_bar = bars_1m.iloc[-1]
                        vol = last_bar['volume'] if 'volume' in last_bar else 1.0
                        close = last_bar['close'] if 'close' in last_bar else 1.0
                        quote_vol = float(vol) * float(close)
                    elif isinstance(bars_1m, np.ndarray) and getattr(bars_1m.dtype, 'names', None):
                        vol = float(bars_1m['volume'][-1]) if 'volume' in bars_1m.dtype.names else 1.0
                        close = float(bars_1m['close'][-1]) if 'close' in bars_1m.dtype.names else 1.0
                        quote_vol = vol * close
                    elif isinstance(bars_1m, dict):
                        vol = bars_1m.get('volume', [1.0])[-1]
                        close = bars_1m.get('close', [1.0])[-1]
                        quote_vol = float(vol) * float(close)
                    else:
                        quote_vol = 1.0
                except Exception:
                    quote_vol = 1.0
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
        
    def detect_ml_regime(self, df) -> tuple[str, float, dict]:
        """
        Detección avanzada de régimen ML con múltiples capas de validación.
        (Migrated from MLStrategy for centralization)
        """
        from config import Config
        from collections import Counter
        
        if len(df) < 50:
            return "UNKNOWN", 0.0, {}

        try:
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V50 FIX: CAST ALL COLUMNS TO FLOAT64
            # ═══════════════════════════════════════════════════════════════
            float32_cols = df.select_dtypes(include=['float32']).columns
            if len(float32_cols) > 0:
                df = df.copy()
                df[float32_cols] = df[float32_cols].astype(np.float64)

            # Indicadores principales
            current_adx = df["adx"].iloc[-1] if "adx" in df.columns else 20
            current_atr_pct = (
                (df["atr_pct"].iloc[-1] / 100) if "atr_pct" in df.columns else 0.01
            )
            rsi_std = df["rsi_14"].tail(20).std() if "rsi_14" in df.columns else 15

            # Volatilidad y tendencia (Vectorizado numpy para latencia < 5ms)
            close_vals = df["close"].values[-21:]
            if len(close_vals) > 1:
                price_volatility = float(np.std(np.diff(close_vals) / close_vals[:-1]))
            else:
                price_volatility = 0.0
                
            vol_vals = df["volume"].values[-21:]
            if len(vol_vals) > 1:
                volume_volatility = float(np.std(np.diff(vol_vals) / (vol_vals[:-1] + 1e-9)))
            else:
                volume_volatility = 0.0

            # Tendencia EMAs
            closes = df["close"].values.astype(np.float64)
            try:
                ema20 = float(calculate_ema_jit(closes, period=20)[-1])
            except Exception:
                ema20 = closes[-1]
            try:
                ema50 = float(calculate_ema_jit(closes, period=50)[-1])
            except Exception:
                ema50 = closes[-1]
            trend_strength = abs(ema20 - ema50) / ema50 if ema50 > 0 else 0

            # Sistema de scoring mejorado
            regime_scores = {
                "TRENDING": 0.0,
                "RANGING": 0.0,
                "VOLATILE": 0.0,
                "STAGNANT": 0.0,
                "MIXED": 0.0,
            }

            # ✅ TRENDING: ADX alto + tendencia fuerte + volatilidad controlada
            if current_adx > Config.Strategies.ML_THRESHOLDS['regime_adx_trend']:
                regime_scores["TRENDING"] += 0.4
            if trend_strength > Config.Strategies.ML_THRESHOLDS['regime_trend_strength']:
                regime_scores["TRENDING"] += 0.3
            if current_atr_pct < Config.Strategies.ML_THRESHOLDS['regime_atr_trend_max']:
                regime_scores["TRENDING"] += 0.2
            if volume_volatility < Config.Strategies.ML_THRESHOLDS['regime_vol_volatility_max']:
                regime_scores["TRENDING"] += 0.1

            # ✅ VOLATILE: ATR alto + RSI volátil + alta volatilidad precio
            if current_atr_pct > Config.Strategies.ML_THRESHOLDS['regime_atr_volatile_min']:
                regime_scores["VOLATILE"] += 0.5
            if rsi_std > Config.Strategies.ML_THRESHOLDS['regime_rsi_std_volatile']:
                regime_scores["VOLATILE"] += 0.3
            if price_volatility > Config.Strategies.ML_THRESHOLDS['regime_price_vol_volatile']:
                regime_scores["VOLATILE"] += 0.2

            # ✅ RANGING: ADX bajo + RSI estable + baja volatilidad
            if current_adx < Config.Strategies.ML_THRESHOLDS['regime_adx_range_max']:
                regime_scores["RANGING"] += 0.3
            if rsi_std < Config.Strategies.ML_THRESHOLDS['regime_rsi_std_range_max']:
                regime_scores["RANGING"] += 0.3
            if current_atr_pct < Config.Strategies.ML_THRESHOLDS['regime_atr_range_max']:
                regime_scores["RANGING"] += 0.2

            # ✅ STAGNANT (ZOMBIE): Volatilidad nula o insignificante
            price_spread = (df["high"].max() - df["low"].min()) / df["close"].mean()
            identical_bars = (df["high"] == df["low"]).sum() / len(df)

            if (
                current_atr_pct < Config.Strategies.ML_THRESHOLDS['regime_atr_zombie_1']
                or price_spread < Config.Strategies.ML_THRESHOLDS['regime_spread_zombie']
                or identical_bars > Config.Strategies.ML_THRESHOLDS['regime_ident_bars_zombie']
            ):
                regime_scores["STAGNANT"] += 0.8
            elif current_atr_pct < Config.Strategies.ML_THRESHOLDS['regime_atr_zombie_2']:
                regime_scores["STAGNANT"] += 0.5
                regime_scores["RANGING"] += 0.1

            # ✅ MIXED: Sin señales claras o transición
            if max(regime_scores.values()) < Config.Strategies.ML_THRESHOLDS['mixed_regime_max_score']:
                regime_scores["MIXED"] = 1.0

            # Determinar régimen dominante
            best_regime_pair = max(regime_scores.items(), key=lambda x: x[1])
            best_regime = best_regime_pair[0]
            confidence = min(best_regime_pair[1] * 1.2, 1.0)  # Boost de confianza

            # --- MÉTRICAS ESTADÍSTICAS PARA LOGGING ---
            stats = {
                "adx": float(current_adx),
                "atr_pct": float(current_atr_pct) * 100,
                "rsi_std": float(rsi_std),
                "trend_strength": float(trend_strength) * 100,
            }

            return best_regime, confidence, stats

        except Exception as e:
            logger.error(f"Error detecting ML regime: {e}")
            return "UNKNOWN", 0.0, {}
    
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
                    advice.update({'leverage': 3, 'threshold_mod': 0.0, 'scale': 0.8, 'action': 'BOTH' if getattr(Config.Strategies, 'SYMMETRIC_SHORTS_SCALPING', False) else 'LONG'})
                elif regime == 'TRENDING_BEAR':
                    advice.update({'leverage': 3, 'threshold_mod': 0.02, 'scale': 1.0, 'action': 'SHORT'})
                else:
                    advice.update({'leverage': 1, 'threshold_mod': 0.1, 'scale': 0.0, 'action': 'NEUTRAL'})
                    
        except Exception as e:
            logger.error(f"Advice Error: {e}")
            
        return advice

    def get_directional_bias(self, regime: str, horizon: str, direction: str) -> float:
        """
        [CAPA 4: LONG/SHORT INTELLIGENCE]
        Matriz de Neuroplasticidad Direccional consciente del Horizonte.
        Retorna la compatibilidad de 0.0 (Veto Absoluto) a 1.0 (Máxima).
        """
        is_bull = regime == 'TRENDING_BULL'
        is_bear = regime == 'TRENDING_BEAR'
        is_chop = regime in ('CHOPPY', 'ZOMBIE', 'HIGH_VOLATILITY')
        
        horizon = horizon.upper()
        direction = direction.upper()
        
        # --- SWING / MACRO (Seguidores de Tendencia Estrictos) ---
        if horizon in ('SWING', 'MACRO'):
            if is_chop:
                # [QUANTUM EVOLUTION] Machine Learning Regime Oracle Veto
                # Veto absoluto: Swing no opera en chop/zombie. Además, alerta para liquidar.
                from logger import logger
                logger.warning("🔮 [REGIME ORACLE] Régimen lateral detectado. VETO ABSOLUTO a Swing. Cierre recomendado si el funding es adverso.")
                return 0.0
            elif is_bull:
                return 1.0 if direction == 'LONG' else 0.0
            elif is_bear:
                return 1.0 if direction == 'SHORT' else 0.0
            else: # RANGING, MEAN_REVERTING
                return 0.5
                
        # --- SCALPING / MICROSCALPING (Bidireccionales Adaptativos) ---
        elif horizon in ('SCALPING', 'MICROSCALPING'):
            if is_bull:
                # Permitir shorts rápidos de mean-reversion (ej. pullback al EMA)
                return 1.0 if direction == 'LONG' else 0.7
            elif is_bear:
                # Permitir longs rápidos por rebotes de sobreventa extrema
                return 1.0 if direction == 'SHORT' else 0.7
            elif is_chop:
                # Microscalping puede sobrevivir en chop, pero con precaución.
                return 0.5
            else: # RANGING, MEAN_REVERTING
                # Ideal para el scalper bidireccional
                return 1.0
                
        # Fallback de seguridad
        return 0.5

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

    def is_volatility_shock(self, bars: Dict, atr_period: int = 14, threshold: float = 2.5, oi_delta: float = 0.0) -> bool:
        """
        Detects sudden volatility expansion (Shock).
        TR > Threshold * ATR OR (TR > Threshold_Squeeze * ATR AND oi_delta < -0.02)
        
        QUÉ: Detecta si el mercado acaba de sufrir un "Shock" de volatilidad (Vela inusualmente gigante), 
             y cruza esto con el 'oi_delta' para detectar Sorteos/Liquidaciones ("Squeezes" en cascada).
        POR QUÉ: Para poder bloquear preventivamente entradas falsas en un momento de pánico institucional.
        """
        try:
            highs = bars['high'].astype(np.float64)    # F6: float32→float64 for talib
            lows = bars['low'].astype(np.float64)
            closes = bars['close'].astype(np.float64)
            
            if len(closes) < atr_period + 1:
                return False
                
            # Calculate ATR (can be JIT optimized later)
            atr_arr = calculate_atr_jit(highs, lows, closes, period=atr_period)
            current_atr = atr_arr[-1]
            
            # Current True Range
            tr = max(highs[-1] - lows[-1], abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2]))
            
            # 1. Extreme Price Volatility (Classic Shock)
            if tr > current_atr * threshold:
                return True
                
            # 2. Institutional Squeeze (Price spike + OI Drop)
            # Threshold lowered to 1.5x ATR if OI is dropping significantly (Squeeze signature)
            squeeze_threshold = threshold * 0.6
            if tr > current_atr * squeeze_threshold and oi_delta < -0.02:
                logger.warning(f"🚨 [SQUEEZE DETECTED] Volatility Expansion ({tr:.2f} > {current_atr*squeeze_threshold:.2f}) with OI drain ({oi_delta*100:.2f}%).")
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error in is_volatility_shock: {e}")
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
            
            adx = calculate_adx_jit(highs, lows, closes, period=14)
            atr = calculate_atr_jit(highs, lows, closes, period=14)
            
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

    def calculate_isn_and_med(self, symbol: str, df, sl_score: float, sc_score: float, funding_rate: float = 0.0) -> dict:
        """
        [MÓDULO DUAL - Inteligencia Bidireccional]
        Calcula el Índice de Sesgo Neto (ISN) y determina el Mapa de Estado Direccional (MED).
        Retorna un dict con {'isn': int, 'med': str, 'volatility': float}
        """
        if df is None or len(df) < 50:
            return {'isn': 0, 'med': 'MED-4', 'volatility': 0.0}
            
        try:
            # Extract float arrays safely (O(1) memory view)
            if isinstance(df, np.ndarray) and getattr(df.dtype, 'names', None):
                c = df['close'].astype(np.float64)
                h = df['high'].astype(np.float64)
                l = df['low'].astype(np.float64)
                v = df['volume'].astype(np.float64)
            elif hasattr(df, 'iloc'):
                c = df['close'].values.astype(np.float64)
                h = df['high'].values.astype(np.float64)
                l = df['low'].values.astype(np.float64)
                v = df['volume'].values.astype(np.float64)
            elif isinstance(df, list) and len(df) > 0 and isinstance(df[0], dict):
                c = np.array([b.get('close', 0.0) for b in df], dtype=np.float64)
                h = np.array([b.get('high', 0.0) for b in df], dtype=np.float64)
                l = np.array([b.get('low', 0.0) for b in df], dtype=np.float64)
                v = np.array([b.get('volume', 0.0) for b in df], dtype=np.float64)
            else:
                c = np.array(df['close'], dtype=np.float64)
                h = np.array(df['high'], dtype=np.float64)
                l = np.array(df['low'], dtype=np.float64)
                v = np.array(df['volume'], dtype=np.float64)
            
            # EMA Calculations
            ema50 = calculate_ema_jit(c, 50)[-1]
            ema200 = calculate_ema_jit(c, 200)[-1]
            
            # ADX Calculation
            adx_arr = calculate_adx_jit(h, l, c, period=14)
            adx = adx_arr[-1]
            
            # ATR y Volatilidad
            atr_arr = calculate_atr_jit(h, l, c, period=14)
            current_atr_pct = (atr_arr[-1] / c[-1]) * 100
            
            # CVD Proxy (Sign of price change * volume)
            if len(c) > 20:
                price_changes = np.diff(c[-21:])
                signs = np.sign(price_changes)
                cvd_proxy = np.sum(signs * v[-20:])
                cvd_points = 15 if cvd_proxy > 0 else -15
            else:
                cvd_proxy = 0
                cvd_points = 0
            
            # Puntos Funding
            from config import Config
            funding_points = 0
            if funding_rate >= getattr(Config.DualDirectional, 'FUNDING_EXTREME_POS', 0.0005):
                funding_points = -20 # Extreme pos funding favors short
            elif funding_rate <= getattr(Config.DualDirectional, 'FUNDING_EXTREME_NEG', -0.0002):
                funding_points = 20  # Extreme neg funding favors long
                
            # Puntos Tendencia (Macro)
            trend_points = 0
            if c[-1] > ema50 and ema50 > ema200 and adx > 25:
                trend_points = 30
            elif c[-1] < ema50 and ema50 < ema200 and adx > 25:
                trend_points = -30
                
            # Puntos Modelo (Basado en SL y SC)
            model_points = 0
            if sl_score > sc_score and sl_score > 65:
                model_points = 25
            elif sc_score > sl_score and sc_score > 65:
                model_points = -25
                
            # Calcular ISN Final (-100 a +100)
            isn = cvd_points + funding_points + trend_points + model_points
            isn = max(min(isn, 100), -100)
            
            # Mapa de Estado Direccional (MED)
            if isn > 50:
                med = 'MED-1' # Bullish Extremo
            elif 15 <= isn <= 50:
                med = 'MED-2' # Bullish Moderado
            elif -14 <= isn <= 14:
                # Volatility threshold relative to normally expected ATR %
                if current_atr_pct > 0.15: # Arbitrary high-vol threshold
                    med = 'MED-3' # Rango Estructural
                else:
                    med = 'MED-4' # Rango Volátil / Choppy
            elif -50 <= isn <= -15:
                med = 'MED-5' # Bearish Moderado
            else:
                med = 'MED-6' # Bearish Extremo
                
            # Override global regime logic for compatibility
            self.last_regime[symbol] = med
            
            return {
                'isn': isn,
                'med': med,
                'volatility': current_atr_pct,
                'cvd_proxy': cvd_proxy
            }
            
        except Exception as e:
            logger.error(f"Error en calculate_isn_and_med: {e}")
            return {'isn': 0, 'med': 'MED-4', 'volatility': 0.0}

    def get_hmm_risk_multiplier(self, symbol: str, direction: str) -> float:
        """
        [CAPA 14: HMM DIVERGENCE PROTECTION]
        QUÉ: Retorna un multiplicador de riesgo (0.25 a 1.0) basado en la alineación HMM.
        POR QUÉ: Si la tendencia técnica (EMA) dice comprar (LONG) pero la distribución de 
                 retornos de Markov (HMM) es fuertemente bajista, es una trampa mortal de distribución.
        PARA QUÉ: Reducir pérdidas y subir WR a ~100% filtrando trade traps.
        """
        try:
            if symbol not in self.hmm_detectors:
                return 1.0
                
            hmm = self.hmm_detectors[symbol]
            # Determinar el estado más probable y su probabilidad
            state_idx = int(np.argmax(hmm.state_probabilities))
            hmm_regime = hmm.REGIMES.get(state_idx, 'UNKNOWN')
            prob = hmm.state_probabilities[state_idx]
            
            direction = direction.upper()
            
            # Divergencia 1: Queriendo ir LONG cuando HMM es fuertemente bajista (TRENDING_BEAR)
            if direction == 'LONG' and hmm_regime == 'TRENDING_BEAR' and prob > 0.60:
                logger.warning(f"🛡️ [HMM VETO] Long divergence on {symbol}. HMM is TRENDING_BEAR ({prob:.0%}). Risk Multiplier applied.")
                return 0.25
                
            # Divergencia 2: Queriendo ir SHORT cuando HMM es fuertemente alcista (TRENDING_BULL)
            if direction == 'SHORT' and hmm_regime == 'TRENDING_BULL' and prob > 0.60:
                logger.warning(f"🛡️ [HMM VETO] Short divergence on {symbol}. HMM is TRENDING_BULL ({prob:.0%}). Risk Multiplier applied.")
                return 0.25
                
            # Divergencia 3: Queriendo operar en alta volatilidad caótica (CHOPPY)
            if hmm_regime == 'CHOPPY' and prob > 0.70:
                logger.warning(f"🛡️ [HMM VETO] High Choppiness on {symbol} ({prob:.0%}). Reducing risk.")
                return 0.50
                
            return 1.0
        except Exception as e:
            logger.error(f"Error in get_hmm_risk_multiplier: {e}")
            return 1.0


