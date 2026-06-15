"""
SNIPER STRATEGY - Optimized for High-Frequency Scalping
========================================================
Simplified 2-layer system for better signal frequency.

Layer A: Technical Confluence (2/3 required)
Layer B: Volume Confirmation (bonus, not required)
Layer C: Fee & Risk Validation

OPTIMIZATIONS:
- Order book analysis made optional (non-blocking)
- R:R ratio lowered to 1.5:1 for scalping
- Leverage capped at 12x (safer for $12 capital)
- Passes leverage to Risk Manager
- Dynamic capital from portfolio
"""

import numpy as np
from utils.math_kernel import calculate_bollinger_jit, calculate_atr_jit, calculate_rsi_jit, calculate_macd_jit
from datetime import datetime, timezone
from typing import Dict, Optional

from .strategy import Strategy
from core.events import SignalEvent
from core.enums import SignalType
from config import Config
from utils.logger import logger
from utils.common import validate_market_data, performance_timer
from utils.safe_leverage import safe_leverage_calculator
from core.neural_bridge import neural_bridge
from .phalanx import OrderFlowAnalyzer, OnlineGARCH  # PHASE 13 + D1 FIX: Module-level import
from utils.math_kernel import calculate_hurst_jit, bayesian_probability_jit, calculate_zscore_jit


class SniperStrategy(Strategy):
    """
    High-precision scalping strategy with 2-layer confluence.
    Optimized for frequent signals with controlled risk.
    """
    
    def __init__(self, data_provider, events_queue, executor=None, portfolio=None, horizon: str = "SCALPING", priority: int = 1):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.executor = executor
        self.portfolio = portfolio  # NEW: For dynamic capital
        self.horizon = horizon
        self.priority = priority
        lbl = "[SCL]" if horizon == "SCALPING" else "[SWG]"
        self.strategy_id = f"{lbl}_SNIPER_{horizon}"
        
        # ================================================================
        # PHASE FORENSIC-4: HORIZON-AWARE PARAMETER LOADING
        # QUÉ: Carga parámetros especializados según el horizonte.
        # POR QUÉ: Sniper con SWING necesita ATR multipliers más amplios,
        #   RSI periods más largos y TP/SL targets mayores.
        # CÓMO: Lee Config.Horizons.Scalping o SWING_PARAMS.
        # CUÁNDO: En cada instanciación de la estrategia.
        # DÓNDE: strategies/sniper_strategy.py → __init__
        # QUIÉN: SniperStrategy
        # ================================================================
        if horizon.upper() == 'SCALPING':
            h_params = getattr(Config.Horizons, 'Scalping', {})
            self.primary_tf = h_params.get('primary_tf', '1m') if h_params else '1m'
        elif horizon.upper() == 'SWING':
            h_params = getattr(Config.Horizons, 'Swing', {})
            self.primary_tf = h_params.get('primary_tf', '1h') if h_params else '1h'
        else:
            h_params = {}
            self.primary_tf = '5m'
        
        # Indicator periods — HORIZON-AWARE
        self.RSI_PERIOD = h_params.get('rsi_period', Config.Sniper.RSI_PERIOD)
        self.RSI_OVERSOLD = h_params.get('rsi_buy', Config.Sniper.RSI_OVERSOLD)
        self.RSI_OVERBOUGHT = h_params.get('rsi_sell', Config.Sniper.RSI_OVERBOUGHT)
        self.BB_PERIOD = h_params.get('bb_period', Config.Sniper.BB_PERIOD)
        self.BB_STD = h_params.get('bb_std', Config.Sniper.BB_STD)
        self.ATR_PERIOD = h_params.get('atr_period', 14)
        
        # TP/SL — HORIZON-AWARE (critical for sizing)
        self.TP_PCT = h_params.get('tp_pct', 0.0080)
        self.SL_PCT = h_params.get('sl_pct', 0.0040)
        
        # ATR Multipliers — HORIZON-AWARE
        self.ATR_SL_MULT = h_params.get('atr_sl_mult', 2.0 if horizon == 'SCALPING' else 3.0)
        self.ATR_TP_MULT = h_params.get('atr_tp_mult', 3.0 if horizon == 'SCALPING' else 4.5)
        
        # Operational params — HORIZON-AWARE
        self.COOLDOWN_SECONDS = h_params.get('cooldown_seconds', 20 if horizon == 'SCALPING' else 1800)
        self.STRENGTH_THRESHOLD = h_params.get('strength_threshold', 0.57)
        
        # Whitelist filter (Disabled - Now using Dynamic Basket)
        self.whitelist = getattr(Config.Sniper, 'WHITELIST', [])
        
        # Order Flow & Regime (Phase 13/14)
        self.of_analyzer = OrderFlowAnalyzer()
        self.garch_models = {} # {symbol: OnlineGARCH}
        
        self.last_signal_time = {}
        self.signal_count = 0
        self._current_event_time = None

    def _now(self):
        """Forensic Time Fix: Return event timestamp if available, else system time"""
        return self._current_event_time if self._current_event_time else datetime.now(tz=timezone.utc)
        
        logger.info(f"🎯 SNIPER [{horizon}] INITIALIZED | TP={self.TP_PCT*100:.2f}% SL={self.SL_PCT*100:.2f}% | RSI={self.RSI_PERIOD} | ATR_SL={self.ATR_SL_MULT}x ATR_TP={self.ATR_TP_MULT}x")
    
    @validate_market_data
    @performance_timer
    def calculate_signals(self, event):
        """Main signal generation loop."""
        if not Config.Sniper.ENABLED:
            return
        
        # Forensic Fix: Sync strategy clock to market event clock
        self._current_event_time = getattr(event, 'timestamp', self._now())

        # Session filter
        if Config.Sniper.REQUIRE_ACTIVE_SESSION and not self._is_active_session():
            return
        
        # OMNI-ADAPTIVE: Sniper now trades whatever is in the active basket
        # Optimization: If event is for specific symbol, only process that one.
        target_symbols = [event.symbol] if hasattr(event, 'symbol') and event.symbol else self.data_provider.symbol_list
        
        order_flow = getattr(event, 'order_flow', None)
        
        for symbol in target_symbols:
            try:
                # ── PHASE 1: LIQUIDATION SQUEEZE TRIGGER ──
                # If the event is a direct liquidation notification, bypass standard analysis
                # AUDIT FIX #2: Read from order_flow instead of 'data' (which does not exist)
                event_data = getattr(event, 'order_flow', {}) or {}
                if event_data and event_data.get('liquidation'):
                    liq_side = event_data.get('side')
                    liq_usd = event_data.get('usd_value', 0)
                    
                    # If LONGs are liquidated (SELL), we go LONG. If SHORTs are liquidated (BUY), we go SHORT.
                    signal_type = SignalType.LONG if liq_side == 'SELL' else SignalType.SHORT
                    
                    logger.warning(f"⚡ [LIQUIDATION SQUEEZE] {symbol} | Sniping {signal_type.name} after ${liq_usd:,.0f} {liq_side} liquidation cascade!")
                    
                    # Create atomic high-conviction signal
                    signal = SignalEvent(
                        strategy_id="SNIPER_LIQ_SQUEEZE",
                        symbol=symbol,
                        datetime=getattr(event, 'timestamp', self._now()),
                        signal_type=signal_type,
                        strength=1.0,  # Max conviction
                        atr=0.0, # Will be dynamically calculated in risk_manager or executor
                        tp_pct=self.TP_PCT,
                        sl_pct=self.SL_PCT,
                        current_price=event_data.get('price', 0),
                        leverage=Config.BINANCE_LEVERAGE,
                        horizon=self.horizon,
                        priority=10, # Top priority
                        metadata={
                            'is_liquidation_squeeze': True,
                            'liq_value': liq_usd
                        }
                    )
                    self.events_queue.put(signal)
                    self.last_signal_time[symbol] = getattr(event, 'timestamp', self._now())
                    continue # Skip standard analysis
                    
                # D2 FIX: Single data fetch for both GARCH and analysis
                bars = self.data_provider.get_latest_bars(symbol, n=200, timeframe=self.primary_tf)
                if bars is None or len(bars) < 2:
                    continue
                
                # PHASE 14: Update GARCH Volatility (reuse bars from single fetch)
                ret = (bars['close'][-1] - bars['close'][-2]) / bars['close'][-2]
                # D1 FIX: OnlineGARCH imported at module level now
                if symbol not in self.garch_models:
                    self.garch_models[symbol] = OnlineGARCH(0.01, 0.1, 0.85, 0.001)
                
                vol = self.garch_models[symbol].update(ret)
                
                # Update RiskManager centrally if available
                if self.portfolio and hasattr(self.portfolio, 'risk_manager'):
                    self.portfolio.risk_manager.update_leverage_and_params(vol, "ADAPTIVE")

                # 0. Local Cooldown Check — HORIZON-AWARE
                # FORENSIC FIX: Use event.timestamp for backtest parity, fallback to now
                now = getattr(event, 'timestamp', self._now())
                if symbol in self.last_signal_time:
                    if (now - self.last_signal_time[symbol]).total_seconds() < self.COOLDOWN_SECONDS:
                        continue
                
                # D2 FIX: Pass pre-fetched bars to avoid redundant API call
                signal = self._analyze_symbol(symbol, order_flow=order_flow, prefetched_bars=bars)
                if signal:
                    self.events_queue.put(signal)
                    self.last_signal_time[symbol] = now
                    self.signal_count += 1
                    logger.info(f"🎯 SNIPER #{self.signal_count}: {signal.signal_type.name} {symbol} (Strength: {signal.strength:.2f})")
            except Exception as e:
                logger.error(f"Sniper error for {symbol}: {e}")
    
    def _is_active_session(self) -> bool:
        """Check if current time is within London or NY session."""
        now = self._now()
        hour = now.hour
        
        sessions = Config.Sniper.ACTIVE_SESSIONS
        london_active = sessions['london_open'] <= hour < sessions['london_close']
        ny_active = sessions['ny_open'] <= hour < sessions['ny_close']
        
        return london_active or ny_active
    
    def _analyze_symbol(self, symbol: str, order_flow: Optional[dict] = None, prefetched_bars=None) -> Optional[SignalEvent]:
        """Run simplified 2-layer confluence + Phase 13 Order Flow analysis."""
        
        # D2 FIX: Use pre-fetched bars to avoid redundant API call
        bars = prefetched_bars if prefetched_bars is not None else self.data_provider.get_latest_bars(symbol, n=200, timeframe=self.primary_tf)
        if len(bars) < 100:
            return None
        
        # Extract OHLCV — F6: Cast float32→float64 for talib compatibility
        closes = bars['close'].astype(np.float64)
        highs = bars['high'].astype(np.float64)
        lows = bars['low'].astype(np.float64)
        volumes = bars['volume'].astype(np.float64)
        
        current_price = closes[-1]
        
        # =====================================================================
        # QUANTUM MATH LAYER: Random Walk Filter (Hurst) & Z-Score
        # =====================================================================
        hurst = calculate_hurst_jit(closes, period=20)
        
        # HURST NOISE FILTER: 0.45 to 0.55 is pure noise. Block execution.
        if 0.45 < hurst < 0.55:
            logger.debug(f"[{symbol}] Sniper blocked by Hurst {hurst:.2f} (Random Walk Noise)")
            return None
            
        # =====================================================================
        # BOLLINGER BAND SQUEEZE EXPANSION FILTER
        # QUÉ: Exige que la volatilidad esté en expansión.
        # POR QUÉ: Sniper es pésimo en mercados planos. Solo disparar si la banda se abre.
        # =====================================================================
        upper, middle, lower = calculate_bollinger_jit(closes, period=20, std_dev=2.0)
        bbw = (upper - lower) / middle
        # Require BBW to be expanding or at least not in an extreme squeeze
        if len(bbw) >= 2 and bbw[-1] <= bbw[-2]:
            bbw_10pct = np.percentile(bbw, 10)
            if bbw[-1] < bbw_10pct:
                logger.debug(f"[{symbol}] Sniper blocked by BBW Extreme Squeeze (BBW={bbw[-1]:.4f} < 10th pct={bbw_10pct:.4f})")
                return None
            
        z_scores = calculate_zscore_jit(closes, period=20)
        current_z = z_scores[-1]
        
        # =====================================================================
        # LAYER A: Technical Confluence (SIMPLIFIED: 2/3 required)
        # =====================================================================
        layer_a_results = self._analyze_technical(closes, highs, lows)
        layer_a_score = sum(1 for v in layer_a_results.values() if v['signal'] != 'NEUTRAL')
        
        # D3 FIX: ALWAYS compute direction BEFORE using it
        # This prevents UnboundLocalError when layer_a_score >= 2 but is_sniper_v3 is False
        layer_a_direction = self._get_consensus_direction(layer_a_results)
        
        # PHASE 13: Order Flow (Sniper Trigger)
        of_res = self.of_analyzer.analyze_imbalance(order_flow) if order_flow else None
        is_sniper_v3 = of_res and of_res.get('sniper')
        
        # EXIGENTE: 2/3 confluencia requerida or Sniper V3 (High Imbalance + Delta)
        if layer_a_score < 2 and not is_sniper_v3:
            return None
        
        # D3 FIX: If no consensus direction could be determined, skip
        if layer_a_direction == 'NEUTRAL' and not is_sniper_v3:
            return None
        
        if is_sniper_v3:
            layer_a_direction = 'LONG' if of_res['signal'] > 0 else 'SHORT'
            layer_a_score = 3.5 # Maximum technical conviction for Sniper V3
        
        # =====================================================================
        # PHASE 8: NEURAL BRIDGE CROSS-VALIDATION
        # =====================================================================
        ml_insight = neural_bridge.query_insight(symbol, "ML_ENSEMBLE")
        if ml_insight:
            ml_dir = ml_insight.get('direction')
            ml_conf = ml_insight.get('confidence', 0.0)
            
            # Bloquear si la IA detecta dirección opuesta con fuerza (>0.55)
            if ml_dir != layer_a_direction and ml_conf > 0.55:
                # logger.info(f"🧠 [BRIDGE] Sniper Blocked: ML sees {ml_dir} ({ml_conf:.2f})")
                return None
            
            # Bonus de fuerza si la IA está de acuerdo
            if ml_dir == layer_a_direction:
                layer_a_score += 0.5

        # --- STATISTICAL SYNC ---
        stat_insight = neural_bridge.query_insight(symbol, "STAT_SPREAD")
        if stat_insight:
            stat_dir = stat_insight.get('direction')
            stat_z = stat_insight.get('z_score', 0.0)
            
            # Si el motor estadístico ve una reversión fuerte en contra, precaución
            if stat_dir != layer_a_direction and abs(stat_z) > 2.0:
                # logger.warning(f"🧠 [BRIDGE] Sniper Blocked: Stat sees Mean Reversion {stat_dir} (Z={stat_z:.1f})")
                return None
            
            # Si ambos coinciden en dirección con sobre-extensión
            if stat_dir == layer_a_direction and abs(stat_z) > 1.5:
                layer_a_score += 0.5
                # logger.info(f"🧠 [BRIDGE] Sniper Buff: Stat alignment {stat_dir} (Z={stat_z:.1f})")
        
        # =====================================================================
        # BAYESIAN SELECTIVITY ENGINE
        # =====================================================================
        # Normalize signal strength (layer_a_score out of 3.5 max)
        sig_str = min(1.0, layer_a_score / 3.5)
        
        # ═══════════════════════════════════════════════════════
        # FORENSIC-FIX #5: SNIPER BAYESIAN INVERSION
        # QUÉ: Alinear la fuerza de tendencia con MOMENTUM/BREAKOUT.
        # POR QUÉ: Sniper busca rompimientos con flujo de órdenes. Para LONG, 
        #   el precio debe estar rompiendo al alza (Z > 0), no cayendo (Z < 0).
        # ═══════════════════════════════════════════════════════
        trend_str_val = 1.0 if (layer_a_direction == 'LONG' and current_z > 0) or (layer_a_direction == 'SHORT' and current_z < 0) else -1.0
        
        bayes_prob = bayesian_probability_jit(
            signal_strength=sig_str,
            trend_strength=trend_str_val,
            volatility_z=current_z
        )
        
        # RIGUROSO INSTITUCIONAL: Umbral Bayesiano 0.75 (Opción B)
        if bayes_prob < 0.75:
            logger.debug(f"[{symbol}] Sniper blocked: Bayesian Prob {bayes_prob:.2f} < 0.75")
            return None

        # ================= ====================================================
        # LAYER B: Volume Confirmation (BONUS, not blocking)
        # =====================================================================
        volume_result = self._analyze_volume(volumes)
        volume_bonus = 1.2 if volume_result['is_strong'] else 1.0
        
        # =====================================================================
        # LAYER C: Fee & Risk Validation
        # =====================================================================
        atr = calculate_atr_jit(highs, lows, closes, period=self.ATR_PERIOD)[-1]
        
        # Dynamic leverage from SafeLeverageCalculator
        leverage = self._calculate_dynamic_leverage(atr, current_price)
        if leverage == 0:
            return None
        
        # HORIZON-AWARE: Use ATR multipliers from Config params
        if layer_a_direction == 'LONG':
            stop_price = current_price - (atr * self.ATR_SL_MULT)
            target_price = current_price + (atr * self.ATR_TP_MULT)
        else:
            stop_price = current_price + (atr * self.ATR_SL_MULT)
            target_price = current_price - (atr * self.ATR_TP_MULT)
        
        # FIXED: Dynamic capital from portfolio
        capital = self.portfolio.get_total_equity() if self.portfolio else 12.0
        
        validation = self._validate_trade_economics(
            current_price, target_price, stop_price, 
            capital=capital,
            leverage=leverage
        )
        
        if not validation['is_valid']:
            logger.debug(f"[{symbol}] Trade invalid: Net {validation['net_profit_pct']:.2%}, R:R {validation['rr_ratio']:.1f}")
            return None
        
        # =====================================================================
        # SIGNAL GENERATION
        # =====================================================================
        strength = min(1.0, (layer_a_score / 3) * volume_bonus)
        signal_type = SignalType.LONG if layer_a_direction == 'LONG' else SignalType.SHORT
        
        # Phase 8: Neural Bridge Publication (Master conviction)
        neural_bridge.publish_insight(
            strategy_id="SNIPER_CORE",
            symbol=symbol,
            insight={
                'confidence': strength,
                'direction': 'LONG' if layer_a_direction == 'LONG' else 'SHORT',
                'technical_score': layer_a_score,
                'volume_boost': volume_bonus > 1.0
            }
        )

        # FORENSIC FIX #1b: Send as decimal fractions (standard contract)
        # Before: 3.0 meant 3% but risk_manager used heuristic (> 0.5 → /100 = 0.03) 
        # Now: explicit decimal fraction, no ambiguity
        tp_pct_val = abs(target_price - current_price) / current_price  # Already decimal
        sl_pct_val = abs(current_price - stop_price) / current_price    # Already decimal
        
        # ── SOPHIA-INTELLIGENCE: Pre-trade XAI Analysis ──
        sophia_report_dict = {}
        if hasattr(self, 'sophia') and self.sophia:
            _returns = None
            if len(closes) > 1:
                _returns = np.diff(np.log(closes))

            # Obtener régimen del portfolio o por defecto
            market_regime = "UNKNOWN"
            if self.portfolio and hasattr(self.portfolio, 'market_regime') and self.portfolio.market_regime:
                market_regime = self.portfolio.market_regime.get_current_regime()

            # Sniper requires tight TTL
            dynamic_ttl = 120.0 if self.horizon == 'SCALPING' else 900.0

            sophia_report = self.sophia.analyze(
                symbol=symbol,
                direction=signal_type.name,
                signal_strength=strength,
                setups={'is_sniper_v3': is_sniper_v3, 'bayes_prob': bayes_prob},
                confluence_score=1.0,
                tp_pct=tp_pct_val,
                sl_pct=sl_pct_val,
                returns=_returns,
                ttl_seconds=dynamic_ttl,
                regime=market_regime,
            )

            # FORENSIC-31: DYNAMIC EXACTITUDE THRESHOLD
            veto_threshold = 0.65 if market_regime == "TRENDING" else 0.55
            if sophia_report.win_probability < veto_threshold:
                logger.info(f"🛑 [SOPHIA VETO] {symbol} Sniper Signal Blocked. Exactitude ({sophia_report.win_probability*100:.1f}%) < {veto_threshold*100:.0f}% (Regime: {market_regime}).")
                return None
            sophia_report_dict = sophia_report.to_dict()
        
        # FIXED: Pass ALL metadata in constructor (frozen dataclass)
        signal_timestamp = getattr(event, 'timestamp', self._now()) if hasattr(self, 'event') else self._now() # fallback
        # Wait, we don't have event in _analyze_symbol directly, let's just use datetime.now or pass it down.
        # Actually, let's just use datetime.now here. It's only for the SignalEvent creation time. 
        signal = SignalEvent(
            strategy_id="SNIPER",
            symbol=symbol,
            datetime=self._now(),
            signal_type=signal_type,
            strength=strength,
            atr=atr,
            tp_pct=tp_pct_val,
            sl_pct=sl_pct_val,
            current_price=current_price,
            leverage=leverage,
            horizon=self.horizon,
            priority=self.priority,
            metadata={
                'sniper_mode': is_sniper_v3,
                'of_reason': of_res.get('reason') if of_res else 'TECHNICAL',
                'delta': order_flow.get('delta', 0.0) if order_flow else 0.0,
                'bayes_prob': float(bayes_prob),
                'hurst': float(hurst),
                'sophia': sophia_report_dict
            }
        )
        
        return signal
    
    def _analyze_technical(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
        """Layer A: Technical indicator analysis — HORIZON-AWARE."""
        results = {}
        
        # 1. RSI Divergence — uses horizon-loaded period
        rsi = calculate_rsi_jit(closes, period=self.RSI_PERIOD)
        current_rsi = rsi[-1]
        prev_rsi = rsi[-5]
        
        price_lower = closes[-1] < closes[-5]
        rsi_higher = current_rsi > prev_rsi
        bullish_div = price_lower and rsi_higher and current_rsi < self.RSI_OVERSOLD + 10
        
        price_higher = closes[-1] > closes[-5]
        rsi_lower = current_rsi < prev_rsi
        bearish_div = price_higher and rsi_lower and current_rsi > self.RSI_OVERBOUGHT - 10
        
        results['rsi_divergence'] = {
            'signal': 'LONG' if bullish_div else ('SHORT' if bearish_div else 'NEUTRAL'),
            'value': current_rsi
        }
        
        # 2. MACD Crossover
        macd, signal, hist = calculate_macd_jit(closes, 
                                         fast_period=Config.Sniper.MACD_FAST,
                                         slow_period=Config.Sniper.MACD_SLOW,
                                         signal_period=Config.Sniper.MACD_SIGNAL)
        
        bullish_cross = hist[-2] < 0 and hist[-1] > 0
        bearish_cross = hist[-2] > 0 and hist[-1] < 0
        
        results['macd_cross'] = {
            'signal': 'LONG' if bullish_cross else ('SHORT' if bearish_cross else 'NEUTRAL'),
            'value': hist[-1]
        }
        
        # 3. Bollinger Band Rejection — uses horizon-loaded params
        upper, middle, lower = calculate_bollinger_jit(closes, 
                                             period=self.BB_PERIOD,
                                             std_dev=self.BB_STD)
        
        touch_lower = lows[-2] <= lower[-2]
        bounce_up = closes[-1] > closes[-2]
        bullish_bb = touch_lower and bounce_up
        
        touch_upper = highs[-2] >= upper[-2]
        reject_down = closes[-1] < closes[-2]
        bearish_bb = touch_upper and reject_down
        
        results['bb_rejection'] = {
            'signal': 'LONG' if bullish_bb else ('SHORT' if bearish_bb else 'NEUTRAL'),
            'value': (closes[-1] - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) > 0 else 0.5
        }
        
        return results
    
    def _analyze_volume(self, volumes: np.ndarray) -> Dict:
        """
        Layer B: Volume analysis (SIMPLIFIED).
        No longer blocks signals if volume is normal.
        """
        lookback = volumes[-50:] if len(volumes) >= 50 else volumes
        
        mean_vol = np.mean(lookback)
        current_vol = volumes[-1]
        
        ratio = current_vol / mean_vol if mean_vol > 0 else 1.0
        
        # Strong volume = 1.5x average or more
        is_strong = ratio >= 1.5
        
        return {
            'ratio': ratio,
            'is_strong': is_strong,
            'current_vol': current_vol,
            'avg_vol': mean_vol
        }
    
    def _calculate_dynamic_leverage(self, atr: float, price: float) -> int:
        """
        Calculate leverage based on SafeLeverageCalculator.
        """
        result = safe_leverage_calculator.calculate_safe_leverage(atr, price)
        if not result['is_safe']:
            return 0
        return result['leverage']
    
    def _validate_trade_economics(self, entry_price: float, target_price: float, 
                                   stop_price: float, capital: float, leverage: int) -> Dict:
        """
        Layer C: Validate trade profitability after fees — UNIFIED FEE SOURCE.
        """
        position_value = capital * leverage
        
        # [UNIFIED] Use Config.BINANCE_*_FEE_BNB — single source of truth
        maker_fee = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002)
        taker_fee = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375)
        # Round trip: entry (maker via BBO) + exit (worst case taker)
        fee = position_value * (maker_fee + taker_fee)
        
        if entry_price > 0:
            profit_pct = abs(target_price - entry_price) / entry_price
            loss_pct = abs(entry_price - stop_price) / entry_price
        else:
            return {'is_valid': False, 'net_profit_pct': 0, 'rr_ratio': 0}
        
        potential_profit = position_value * profit_pct
        potential_loss = position_value * loss_pct
        
        net_profit = potential_profit - fee
        net_profit_pct = net_profit / position_value if position_value > 0 else 0
        
        rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
        
        # FIXED: Global Config thresholds
        is_valid = (
            net_profit_pct >= Config.MIN_PROFIT_AFTER_FEES and
            rr_ratio >= Config.MIN_RR_RATIO
        )
        
        return {
            'is_valid': is_valid,
            'net_profit_pct': net_profit_pct,
            'rr_ratio': rr_ratio,
            'fee': fee,
            'potential_profit': potential_profit
        }
    
    def _get_consensus_direction(self, results: Dict) -> str:
        """Get consensus direction from technical indicators."""
        long_count = sum(1 for r in results.values() if r['signal'] == 'LONG')
        short_count = sum(1 for r in results.values() if r['signal'] == 'SHORT')
        
        if long_count > short_count:
            return 'LONG'
        elif short_count > long_count:
            return 'SHORT'
        return 'NEUTRAL'