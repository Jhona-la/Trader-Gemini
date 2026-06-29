import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from utils.logger import logger, log_supreme_event
from data.database import DatabaseHandler
from core.enums import TradeDirection
import uuid
import json

@dataclass
class OracleVerdict:
    trade_id: str
    symbol: str
    should_exit: bool
    reason: str
    confidence: float
    proposing_strategies: List[str]
    dynamic_targets: Dict[str, float] = None  # CTOS Phase 8: Dynamic Target Chasing

class ExitOracle:
    """
    ═══════════════════════════════════════════════════════════════
    CTOS PHASE 3: Centralized Exit Coordination Engine.
    ═══════════════════════════════════════════════════════════════
    
    PROFESSOR METHOD:
    - QUÉ: Motor centralizado de decisiones de cierre que coordina
      TODAS las estrategias de salida y toma la decisión final
      utilizando motores de salida especializados por estrategia.
    - POR QUÉ: Sin coordinación, las estrategias se "pisan las patas":
      una cierra cuando la predicción dice que el precio va a subir más.
      Además, un setup técnico no decae de la misma forma que uno de ML.
    - PARA QUÉ: Maximizar PnL por trade, evitando cierres prematuros
      y permitiendo que los trades ganadores corran a su punto óptimo.
    - CÓMO: Identifica la clase de estrategia (ML, Technical, Pattern,
      Statistical) y delega a su motor de salida correspondiente.
    - CUÁNDO: En cada tick de evaluación de posiciones abiertas.
    - DÓNDE: sophia/exit_oracle.py
    - QUIÉN: Llamado desde engine.py y risk_manager.py en el loop de stops.
    """
    def __init__(self, db_handler: DatabaseHandler = None, 
                 sophia_intelligence=None, prediction_tracker=None):
        self.db = db_handler
        self.sophia = sophia_intelligence
        self.prediction_tracker = prediction_tracker  # CTOS Phase 3: PredictionTracker connection
        self.strategies = {} # strategy_id -> Strategy instance
        self.veto_threshold = 0.65 # Need 65% consensus among polled strategies
        self._eval_counts = {}  # {trade_id: int} — tick counter for exit_strategy_log bar_number
        
        # Load optimal profiles to get veto_threshold dynamically
        self.calibrated_profiles = {}
        try:
            import os
            import json
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prof_path = os.path.join(root_dir, 'optimal_profiles.json')
            if os.path.exists(prof_path):
                with open(prof_path, 'r') as f:
                    self.calibrated_profiles = json.load(f)
                logger.info(f"🧠 [ExitOracle] Loaded optimal profiles from {prof_path} for horizons: {list(self.calibrated_profiles.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ [ExitOracle] Could not load optimal_profiles.json: {e}")
        
    def register_strategy(self, strategy_id: str, strategy_obj: Any):
        """Registers a strategy so the Oracle can poll it."""
        self.strategies[strategy_id] = strategy_obj
        logger.debug(f"[ExitOracle] Registered strategy: {strategy_id}")

    def _classify_strategy_class(self, strategy_id: str) -> str:
        """
        - QUÉ: Helper para clasificar el ID de la estrategia en su categoría.
        - POR QUÉ: Permite derivar la lógica al motor especializado correcto.
        - PARA QUÉ: Evitar aplicar decay de ML a estrategias técnicas o de patrón.
        """
        if not strategy_id:
            return 'ML'  # Default fallback
        
        sid_upper = strategy_id.upper()
        if any(kw in sid_upper for kw in ['ML', 'NEURAL', 'ENSEMBLE']):
            return 'ML'
        elif any(kw in sid_upper for kw in ['SNIPER', 'PHALANX']):
            return 'PATTERN'
        elif any(kw in sid_upper for kw in ['STAT', 'STAT_V1', 'STATARB', 'ARBITRAGE']):
            return 'STATISTICAL'
        elif any(kw in sid_upper for kw in ['TECHNICAL', 'MOMENTUM', 'REVERSAL']):
            return 'TECHNICAL'
        else:
            return 'TECHNICAL'

    def _get_data_provider(self) -> Optional[Any]:
        """
        - QUÉ: Obtiene el proveedor de datos dinámicamente.
        - POR QUÉ: Permite a los motores técnicos consultar velas históricas.
        """
        for strat in self.strategies.values():
            if hasattr(strat, 'data_provider') and strat.data_provider:
                return strat.data_provider
        return None

    def _evaluate_ml_exit(self, symbol: str, pos_data: Dict[str, Any], direction: int, avg_price: float,
                          current_price: float, duration_mins: float, pnl_pct: float, horizon: str,
                          prediction_tracker: Any) -> Tuple[str, str]:
        """
        [ML ALPHA DECAY EXIT ENGINE]
        - QUÉ: Lógica de salida dedicada para Machine Learning.
        - POR QUÉ: Las predicciones de ML tienen un tiempo de decaimiento dinámico.
        - PARA QUÉ: Matar trades estancados o evadir drawdowns predichos por la Omnisciencia.
        """
        pred_magnitude = pos_data.get('predicted_magnitude') or pos_data.get('metadata', {}).get('predicted_magnitude', 0)
        pred_duration = pos_data.get('predicted_duration') or pos_data.get('metadata', {}).get('predicted_duration', 0)
        opener_strat = pos_data.get('opener_strategy_id') or pos_data.get('strategy_id', '')

        # --- CTOS OMNISCIENCE: Trajectory Parsing ---
        omni_route = pos_data.get('metadata', {}).get('omni_route')
        omni_dump_warn = False
        omni_peak_passed = False
        if omni_route:
            bar_mins = omni_route.get("bar_minutes", 1)
            is_long = direction == 1
            
            if is_long:
                dump_pct = omni_route.get("macro_dump_pct", 0.0) / 100.0
                dump_mins = omni_route.get("macro_dump_bars", 0) * bar_mins
                peak_mins = omni_route.get("macro_peak_bars", 0) * bar_mins
                
                if dump_pct < -0.015 and dump_mins < peak_mins and abs(duration_mins - dump_mins) < (5 * bar_mins):
                    omni_dump_warn = True
            else:
                adverse_pct = omni_route.get("macro_peak_pct", 0.0) / 100.0
                adverse_mins = omni_route.get("macro_peak_bars", 0) * bar_mins
                dump_mins = omni_route.get("macro_dump_bars", 0) * bar_mins
                
                if adverse_pct > 0.015 and adverse_mins < dump_mins and abs(duration_mins - adverse_mins) < (5 * bar_mins):
                    omni_dump_warn = True
                    
            target_mins = peak_mins if is_long else dump_mins
            if duration_mins > target_mins:
                omni_peak_passed = True

        # ════════════════════════════════════════════════════════════════
        # 📉 ALPHA DECAY ENGINE (SCALPING vs SWING)
        # ════════════════════════════════════════════════════════════════
        import math
        
        # Inyección Dinámica del Perfil Adaptativo
        profile = pos_data.get('metadata', {}).get('adaptive_profile', {})
        max_ttl_scalp = profile.get('alpha_max_ttl', 60.0) if horizon == 'SCALPING' else 60.0
        max_ttl_swing = profile.get('alpha_max_ttl', 360.0) if horizon == 'SWING' else 360.0
        
        # Función de Retención del Alpha (1.0 = Max Edge, 0.0 = Dead Edge)
        alpha_retention = 1.0
        
        if horizon == 'SCALPING':
            # Exponential Decay: e^(-lambda * t)
            lambda_scalp = profile.get('alpha_decay_lambda', 0.05)
            alpha_retention = math.exp(-lambda_scalp * duration_mins)
            
            if duration_mins > max_ttl_scalp:
                if pnl_pct < 0.0:
                    return "CLOSE_ML_ALPHA_DECAY", f"Scalping Hard TTL ({duration_mins:.1f}m > {max_ttl_scalp:.0f}m)"
                elif pnl_pct > 0.0:
                    pos_data['chasing_mode'] = True
                    logger.info(f"⏳ [CHASING MODE] Scalping trade > TTL ({duration_mins:.1f}m) but +{pnl_pct*100:.2f}%. Activating Chasing.")
        else:
            # SWING: Linear Decay: 1 - (t / max_ttl)
            alpha_retention = max(0.0, 1.0 - (duration_mins / max_ttl_swing))
            
            if duration_mins > max_ttl_swing:
                if pnl_pct < 0.0:
                    return "CLOSE_ML_ALPHA_DECAY", f"Swing Hard TTL ({duration_mins:.1f}m > {max_ttl_swing:.0f}m)"
                elif pnl_pct > 0.0:
                    pos_data['chasing_mode'] = True
                    logger.info(f"⏳ [CHASING MODE] Swing trade > TTL ({duration_mins:.1f}m) but +{pnl_pct*100:.2f}%. Activating Chasing.")

        # Trajectory-based decisions with Dynamic Decay
        if pred_magnitude and pred_magnitude > 0 and pred_duration and pred_duration > 0:
            progress_ratio = abs(pnl_pct) / pred_magnitude if pred_magnitude > 0 else 0
            time_ratio = duration_mins / max(1, pred_duration)
            
            if omni_dump_warn and pnl_pct > 0.001:
                return "CLOSE_ML_ALPHA_DECAY", f"Omniscience Evasion (Projected adverse excursion, locking {pnl_pct*100:.2f}% profit)"
            elif omni_peak_passed and pnl_pct > 0.002 and time_ratio > 1.1:
                return "CLOSE_ML_ALPHA_DECAY", f"Omniscience Peak Passed (Duration {duration_mins:.0f}m > projected peak, locking {pnl_pct*100:.2f}%)"
            
            # Decay-aware prediction miss
            threshold_time = 0.8 if horizon == 'SWING' else 0.6
            if time_ratio > threshold_time and progress_ratio < 0.3 and pnl_pct < 0.0006:
                return "CLOSE_ML_ALPHA_DECAY", f"ML Prediction Miss under Alpha Decay (Retention={alpha_retention:.2f}, Progress={progress_ratio*100:.0f}%)"
            elif time_ratio > 1.5 and pnl_pct < 0.0006:
                return "CLOSE_ML_ALPHA_DECAY", f"ML Time Exhaustion (>{int(pred_duration*1.5)}m, PnL={pnl_pct*100:.2f}%)"
        
        elif prediction_tracker and opener_strat:
            edge_prob = prediction_tracker.calculate_realtime_edge(
                strategy_id=opener_strat,
                elapsed_bars=duration_mins,
                horizon=horizon
            )
            # Edge weighted by alpha retention
            dynamic_edge = edge_prob * alpha_retention
            
            # FORENSIC AUDIT: Record dynamic decay metrics
            if 'metrics' not in pos_data:
                pos_data['metrics'] = {}
            pos_data['metrics']['alpha_retention'] = alpha_retention
            pos_data['metrics']['dynamic_edge'] = dynamic_edge
            pos_data['metrics']['alpha_decay'] = 1.0 - alpha_retention
            
            # CORRELACIÓN ADAPTIVE PROFILE ENGINE LOGGING
            if int(duration_mins) > pos_data.get('_last_decay_log_min', -1) and duration_mins > 0:
                pos_data['_last_decay_log_min'] = int(duration_mins)
                logger.debug(f"📉 [ALPHA DECAY AUDIT] {symbol} ({horizon}) | "
                             f"Lambda: {profile.get('alpha_decay_lambda', 0):.4f} | "
                             f"MaxTTL: {profile.get('alpha_max_ttl', 0):.1f} | "
                             f"Retention: {alpha_retention:.2f} | "
                             f"BaseEdge: {edge_prob:.2f} | "
                             f"DynEdge: {dynamic_edge:.2f} | PnL: {pnl_pct*100:.2f}%")
            
            critical_threshold = 0.45
            if pnl_pct >= 0.0006:
                critical_threshold = 0.35
                
            if dynamic_edge < critical_threshold:
                if not (0 < pnl_pct < 0.0015 and dynamic_edge >= 0.25):
                    return "CLOSE_ML_ALPHA_DECAY", f"ML Alpha Decay (DynamicEdge={dynamic_edge:.2f} < {critical_threshold}, PnL={pnl_pct*100:.2f}%)"
        else:
            # Fallback threshold adjusted by Decay
            if alpha_retention < 0.15 and pnl_pct < 0.0006:
                return "CLOSE_ML_ALPHA_DECAY", f"Alpha Depleted (Retention < 0.15, {duration_mins:.1f}m, PnL={pnl_pct*100:.2f}%)"

        return "KEEP_OPEN", ""

    def _evaluate_technical_exit(self, symbol: str, pos_data: Dict[str, Any], direction: int, avg_price: float,
                                 current_price: float, duration_mins: float, pnl_pct: float, horizon: str,
                                 data_handler: Any) -> Tuple[str, str]:
        """
        [TECHNICAL EXIT ENGINE]
        - QUÉ: Lógica de salida dedicada para setups técnicos/híbridos.
        - POR QUÉ: Los setups técnicos no decaen predictivamente de la misma forma que ML.
        - PARA QUÉ: Esperar a zonas de sobrecompra/sobreventa o reversión estructural antes de cerrar.
        """
        if not data_handler:
            return "KEEP_OPEN", ""

        timeframe = '5m' if horizon == 'SCALPING' else '1h'
        bars = data_handler.get_latest_bars(symbol, n=50, timeframe=timeframe)
        if bars is None or len(bars) < 20:
            return "KEEP_OPEN", ""

        closes = bars['close']
        highs = bars['high']
        lows = bars['low']

        from utils.math_kernel import calculate_rsi_jit, calculate_bollinger_jit

        # 1. RSI Extreme Exit
        profile = pos_data.get('metadata', {}).get('adaptive_profile', {})
        # Fallback to Config if metadata not injected (retro-compatibility)
        if not profile:
            from config import Config
            profile = Config.SymbolProfiles.get(symbol, horizon)
            
        rsi_overbought = profile.get('rsi_overbought', 80.0 if horizon == 'SCALPING' else 75.0)
        rsi_oversold = profile.get('rsi_oversold', 20.0 if horizon == 'SCALPING' else 25.0)

        rsi_vals = calculate_rsi_jit(closes, period=14)
        if len(rsi_vals) > 0 and not np.isnan(rsi_vals[-1]):
            rsi = rsi_vals[-1]
            if direction == 1:
                if rsi > rsi_overbought:
                    if horizon != 'SCALPING' or pnl_pct >= 0.0015:
                        return "CLOSE_TECH_REVERSAL", f"Technical Overbought RSI ({rsi:.1f} > {rsi_overbought:.0f})"
            else:
                if rsi < rsi_oversold:
                    if horizon != 'SCALPING' or pnl_pct >= 0.0015:
                        return "CLOSE_TECH_REVERSAL", f"Technical Oversold RSI ({rsi:.1f} < {rsi_oversold:.0f})"

        # 2. Bollinger Band Extreme Exit
        bb_period = 20
        bb_std = 2.0
        if len(closes) >= bb_period:
            upper_band, middle_band, lower_band = calculate_bollinger_jit(closes, period=bb_period, std_dev=bb_std)
            if len(upper_band) > 0 and not np.isnan(upper_band[-1]):
                up = upper_band[-1]
                lo = lower_band[-1]
                if direction == 1 and current_price >= up:
                    # Only exit on Bollinger Band touch if we have at least 0.15% profit to cover fees.
                    if horizon != 'SCALPING' or pnl_pct >= 0.0015:
                        return "CLOSE_TECH_REVERSAL", f"Technical Upper BB Touch ({current_price:.4f} >= {up:.4f})"
                elif direction == -1 and current_price <= lo:
                    # Only exit on Bollinger Band touch if we have at least 0.15% profit to cover fees.
                    if horizon != 'SCALPING' or pnl_pct >= 0.0015:
                        return "CLOSE_TECH_REVERSAL", f"Technical Lower BB Touch ({current_price:.4f} <= {lo:.4f})"

        # 3. Structure Reversal (4 consecutive opposite bars on raw timeframe)
        required_duration = 20.0 if horizon == 'SCALPING' else 240.0
        if duration_mins >= required_duration:
            # FORENSIC-V100: "Zombie" trade detection (Time Stop)
            # If a trade hasn't moved into profit after X minutes, kill it
            if duration_mins > 60 and horizon == "SCALPING":
                if pnl_pct >= 0.0005: 
                    # CHASING MODE: Do not close profitable trades! Let RiskManager chase it with aggressive ATR Trailing.
                    pos_data['chasing_mode'] = True
                    logger.info(f"⏳ [CHASING MODE] {symbol} stagnant but profitable (+{pnl_pct*100:.2f}%) for {duration_mins:.1f}m. Delegating to ATR Trailing.")
            
            if duration_mins > 120 and horizon == "SCALPING":
                if pnl_pct < 0.0:
                    return "CLOSE_TECH_TIMEOUT", f"Trade exceeded 120m technical window in loss (PnL={pnl_pct*100:.2f}%)"
            
            if direction == 1 and all(closes[i] < closes[i-1] for i in range(-4, 0)):
                if pnl_pct < 0: # Only exit on reversal if trade is in negative territory to avoid cutting green trades
                    return "CLOSE_TECH_REVERSAL", "Technical Structure Reversal (4 consecutive red bars)"
            elif direction == -1 and all(closes[i] > closes[i-1] for i in range(-4, 0)):
                if pnl_pct < 0:
                    return "CLOSE_TECH_REVERSAL", "Technical Structure Reversal (4 consecutive green bars)"

        # 4. Technical Stalled Timeout
        tech_ttl = 90.0 if horizon == 'SCALPING' else 480.0
        if duration_mins > tech_ttl and pnl_pct < 0.0:
            return "CLOSE_TECH_TIMEOUT", f"Technical Timeout Stall ({duration_mins:.1f}m > {tech_ttl:.0f}m, PnL={pnl_pct*100:.2f}%)"

        return "KEEP_OPEN", ""

    def _evaluate_pattern_exit(self, symbol: str, pos_data: Dict[str, Any], direction: int, avg_price: float,
                               current_price: float, duration_mins: float, pnl_pct: float, horizon: str,
                               data_handler: Any) -> Tuple[str, str]:
        """
        [PATTERN EXIT ENGINE]
        - QUÉ: Lógica de salida dedicada para Sniper y Phalanx (estrategias de patrón).
        - POR QUÉ: Los patrones operan en desequilibrios de ordenbook y micro-momentum de alta velocidad.
        - PARA QUÉ: Evitar quedar atrapados en reversiones rápidas y cortar pérdidas a nivel de microsegundos.
        """
        pattern_ttl = 20.0 if horizon == 'SCALPING' else 120.0
        if duration_mins > pattern_ttl and pnl_pct < 0.0:
            return "CLOSE_PATTERN_TIMEOUT", f"Pattern Timeout Stall ({duration_mins:.1f}m > {pattern_ttl:.0f}m, PnL={pnl_pct*100:.2f}%)"

        if not data_handler:
            return "KEEP_OPEN", ""

        timeframe = '1m' if horizon == 'SCALPING' else '15m'
        bars = data_handler.get_latest_bars(symbol, n=10, timeframe=timeframe)
        if bars is None or len(bars) < 4:
            return "KEEP_OPEN", ""

        closes = bars['close']
        required_duration = 3.0 if horizon == 'SCALPING' else 45.0
        if duration_mins >= required_duration:
            if direction == 1 and all(closes[i] < closes[i-1] for i in range(-3, 0)):
                if pnl_pct < 0:
                    return "CLOSE_PATTERN_REVERSAL", "Pattern Micro-Reversal (3 consecutive red bars)"
            elif direction == -1 and all(closes[i] > closes[i-1] for i in range(-3, 0)):
                if pnl_pct < 0:
                    return "CLOSE_PATTERN_REVERSAL", "Pattern Micro-Reversal (3 consecutive green bars)"

        return "KEEP_OPEN", ""

    def _evaluate_statistical_exit(self, symbol: str, pos_data: Dict[str, Any], direction: int, avg_price: float,
                                   current_price: float, duration_mins: float, pnl_pct: float, horizon: str,
                                   data_handler: Any) -> Tuple[str, str]:
        """
        [STATISTICAL EXIT ENGINE]
        - QUÉ: Lógica de salida dedicada para arbitraje estadístico y pares.
        - POR QUÉ: Las estrategias estadísticas entran en desvíos extremos de spread (Z-Score > 2) y deben salir
          cuando el spread revierte a su media histórica (Z-Score cercano a 0).
        - PARA QUÉ: Asegurar la captura del premium de reversión y limitar pérdidas si la cointegración se rompe.
        """
        # DELEGATION: If the strategy is STAT_V1 (main pairs trading), it calculates spread Z-scores 
        # and puts correct pair-wide EXIT signals in the queue. The ExitOracle should not override it
        # with individual asset Z-scores (which are mathematically incorrect for pairs trading).
        strategy_id = pos_data.get('opener_strategy_id') or pos_data.get('strategy_id', '') if pos_data else ''
        if "STAT_V1" in strategy_id:
            return "KEEP_OPEN", ""

        stat_ttl = 120.0 if horizon == 'SCALPING' else 720.0
        if duration_mins > stat_ttl and pnl_pct < 0.0:
            return "CLOSE_STAT_TIMEOUT", f"Statistical Timeout Stall ({duration_mins:.1f}m > {stat_ttl:.0f}m, PnL={pnl_pct*100:.2f}%)"

        if not data_handler:
            return "KEEP_OPEN", ""

        timeframe = '5m' if horizon == 'SCALPING' else '1h'
        bars = data_handler.get_latest_bars(symbol, n=50, timeframe=timeframe)
        if bars is None or len(bars) < 20:
            return "KEEP_OPEN", ""

        closes = bars['close']
        pos_dir = "LONG" if direction == 1 else "SHORT"
        
        from utils.math_kernel import calculate_zscore_jit
        z_scores = calculate_zscore_jit(closes, period=20)
        if len(z_scores) > 0 and not np.isnan(z_scores[-1]):
            z = z_scores[-1]
            # FORENSIC-V134: Refine mean reversion exits to avoid fee death and premature exits.
            # QUÉ: Lógica optimizada de salida por reversión a la media.
            # POR QUÉ: Anteriormente se salía en Z >= -0.5 para largos cuando PnL < 0, lo cual
            #   es cortar el trade justo cuando está revirtiendo en la dirección correcta.
            # PARA QUÉ: Evitar realizar pérdidas innecesarias por comisiones y permitir
            #   que el spread complete su movimiento a la media (Z=0) o alcance un beneficio del 0.20%.
            if pos_dir == "LONG":
                # FORENSIC-V139: Require minimum profit OR wait for extreme overshoot
                if pnl_pct >= 0.0015 and z >= 0.0:
                    return "CLOSE_STAT_REVERSION", f"Statistical Mean Reversion reached mean (Z={z:.2f} >= 0.0, PnL={pnl_pct*100:.2f}%)"
                elif z >= 0.8 and pnl_pct >= 0.0005:
                    return "CLOSE_STAT_REVERSION", f"Statistical Mean Reversion overshoot (Z={z:.2f} >= 0.8, PnL={pnl_pct*100:.2f}%)"
                elif pnl_pct >= 0.0020:
                    # High enough profit to cover fees and lock gain early.
                    return "CLOSE_STAT_REVERSION", f"Statistical Mean Reversion profit target met (PnL={pnl_pct*100:.2f}%)"
                elif z <= -3.0:
                    # Cointegration broke. Cut loss.
                    return "CLOSE_STAT_REVERSION", f"Statistical Mean Reversion invalidation (Z={z:.2f} <= -3.0, cutting loss)"
            else:
                # SHORT position: entered when Z was high (e.g. Z >= 2.0). We want it to fall to 0.0.
                if z <= 0.0:
                    # FORENSIC-V139: Require minimum profit OR wait for extreme overshoot
                    if pnl_pct >= 0.0015:
                        return "CLOSE_STAT_REVERSION", f"Statistical Mean Reversion reached mean (Z={z:.2f} <= 0.0, PnL={pnl_pct*100:.2f}%)"
                    elif z <= -0.8:
                        return "CLOSE_STAT_REVERSION", f"Statistical Mean Reversion overshoot (Z={z:.2f} <= -0.8, PnL={pnl_pct*100:.2f}%)"
                elif pnl_pct >= 0.0020:
                    # High enough profit to cover fees and lock gain early.
                    return "CLOSE_STAT_REVERSION", f"Statistical Mean Reversion profit target met (PnL={pnl_pct*100:.2f}%)"
                elif z >= 3.0:
                    # Cointegration broke. Cut loss.
                    return "CLOSE_STAT_REVERSION", f"Statistical Mean Reversion invalidation (Z={z:.2f} >= 3.0, cutting loss)"

        return "KEEP_OPEN", ""

    def evaluate_open_positions(self, open_positions: Dict[str, Dict[str, Any]], market_data: Dict[str, Any]) -> List[OracleVerdict]:
        """
        🔮 CTOS Phase 3: Enhanced evaluation with prediction awareness and specialized engines.
        """
        verdicts = []
        data_provider = self._get_data_provider()
        
        for pos_key, pos_data in open_positions.items():
            if pos_data.get('quantity', 0) == 0:
                continue
                
            symbol = pos_key.split('_')[0]
            trade_id = pos_data.get('trade_id', 'UNKNOWN')
            pnl = pos_data.get('unrealized_pnl', 0.0)
            current_price = pos_data.get('current_price', 0.0)
            
            # Increment evaluation counter for this trade
            self._eval_counts[trade_id] = self._eval_counts.get(trade_id, 0) + 1
            bar_number = self._eval_counts[trade_id]
            
            votes_to_exit = []
            reasons = []
            all_decisions = []  # CTOS Phase 3: Track ALL decisions for DB
            dynamic_targets = None  # CTOS Phase 8: Dynamic Target Tracking
            
            # CTOS Phase 3: Get prediction context from PredictionTracker
            prediction_context = None
            if self.prediction_tracker:
                opener_strat = pos_data.get('opener_strategy_id') or pos_data.get('strategy_id')
                prediction_context = self.prediction_tracker.get_prediction_for_trade(
                    symbol=symbol,
                    strategy_id=opener_strat,
                    trade_id=trade_id
                )
            
            # Inject Context for Self-Awareness (CTOS Phase 2)
            pos_data['_prediction_context'] = prediction_context
            pos_data['_current_votes'] = votes_to_exit
 
            # 1. Poll all strategies
            for strat_id, strat in self.strategies.items():
                if hasattr(strat, 'request_exit_opinion'):
                    opinion = strat.request_exit_opinion(pos_data)
                    
                    if opinion and opinion.get('vote') == 'EXIT':
                        votes_to_exit.append(strat_id)
                        reasons.append(f"{strat_id}: {opinion.get('reason', 'Unknown')}")
                        all_decisions.append({
                            'strategy_id': strat_id,
                            'action': 'EXIT',
                            'reason': opinion.get('reason', 'Unknown'),
                        })
                    else:
                        hold_reason = opinion.get('reason', 'NO_EXIT_SIGNAL') if opinion else 'NO_OPINION'
                        all_decisions.append({
                            'strategy_id': strat_id,
                            'action': 'HOLD',
                            'reason': hold_reason,
                        })
                        
                    # CTOS Phase 8: Extract Dynamic Targets if provided
                    if opinion and 'dynamic_targets' in opinion:
                        if dynamic_targets is None:
                            dynamic_targets = {"tp_mult": 1.0, "sl_mult": 1.0}
                        dt = opinion['dynamic_targets']
                        dynamic_targets["tp_mult"] = max(dynamic_targets["tp_mult"], dt.get("tp_mult", 1.0))
                        dynamic_targets["sl_mult"] = min(dynamic_targets["sl_mult"], dt.get("sl_mult", 1.0))
            
            # 2. Time-based and indicators specialized exit engines evaluation
            entry_time = pos_data.get('entry_time')
            horizon = pos_data.get('horizon', 'SCALPING')
            now = datetime.now(timezone.utc)
            duration_mins = 0.0
            if isinstance(entry_time, datetime):
                duration_mins = (now - entry_time).total_seconds() / 60.0
            elif isinstance(entry_time, (int, float)):
                duration_mins = (time.time() - entry_time) / 60.0
            
            strategy_id = pos_data.get('opener_strategy_id') or pos_data.get('strategy_id', '')
            strat_class = self._classify_strategy_class(strategy_id)
            direction = 1 if pos_data.get('direction', 'LONG').upper() == 'LONG' else -1

            engine_action = "KEEP_OPEN"
            engine_reason = ""
            engine_id = "ALPHA_DECAY"

            if strat_class == 'ML':
                engine_action, engine_reason = self._evaluate_ml_exit(
                    symbol=symbol,
                    pos_data=pos_data,
                    direction=direction,
                    avg_price=pos_data.get('entry_price', current_price),
                    current_price=current_price,
                    duration_mins=duration_mins,
                    pnl_pct=pnl,
                    horizon=horizon,
                    prediction_tracker=self.prediction_tracker
                )
                engine_id = "ML_EXIT"
            elif strat_class == 'TECHNICAL':
                engine_action, engine_reason = self._evaluate_technical_exit(
                    symbol=symbol,
                    pos_data=pos_data,
                    direction=direction,
                    avg_price=pos_data.get('entry_price', current_price),
                    current_price=current_price,
                    duration_mins=duration_mins,
                    pnl_pct=pnl,
                    horizon=horizon,
                    data_handler=data_provider
                )
                engine_id = "TECH_EXIT"
            elif strat_class == 'PATTERN':
                engine_action, engine_reason = self._evaluate_pattern_exit(
                    symbol=symbol,
                    pos_data=pos_data,
                    direction=direction,
                    avg_price=pos_data.get('entry_price', current_price),
                    current_price=current_price,
                    duration_mins=duration_mins,
                    pnl_pct=pnl,
                    horizon=horizon,
                    data_handler=data_provider
                )
                engine_id = "PATTERN_EXIT"
            elif strat_class == 'STATISTICAL':
                engine_action, engine_reason = self._evaluate_statistical_exit(
                    symbol=symbol,
                    pos_data=pos_data,
                    direction=direction,
                    avg_price=pos_data.get('entry_price', current_price),
                    current_price=current_price,
                    duration_mins=duration_mins,
                    pnl_pct=pnl,
                    horizon=horizon,
                    data_handler=data_provider
                )
                engine_id = "STAT_EXIT"

            if engine_action != "KEEP_OPEN":
                votes_to_exit.append(engine_id)
                reasons.append(engine_reason)
                all_decisions.append({
                    'strategy_id': engine_id,
                    'action': 'PROPOSE_EXIT',
                    'reason': engine_reason,
                })
            else:
                all_decisions.append({
                    'strategy_id': engine_id,
                    'action': 'HOLD',
                    'reason': f"Engine {engine_id} says KEEP_OPEN",
                })

            # 4. Consensus Logic
            total_strategies = len(self.strategies) or 1
            bypass_exits = ["RISK_MANAGER", "ALPHA_DECAY", "ML_EXIT", "TECH_EXIT", "PATTERN_EXIT", "STAT_EXIT"]
            
            # CRITICAL FIX: The strategy that opened the position has sovereignty to close it!
            opener_id = pos_data.get("opener_strategy_id", "Unknown")
            if opener_id not in bypass_exits:
                bypass_exits.append(opener_id)
                
            pure_votes = [v for v in votes_to_exit if v not in bypass_exits]
            consensus_ratio = len(pure_votes) / total_strategies if total_strategies > 0 else 0
            
            should_exit = False
            final_reason = "MAINTAIN"
            confidence = 0.0
            
            # Dyn veto_threshold based on calibrated profiles
            veto_threshold = self.veto_threshold
            horizon_key = str(horizon).upper()
            clean_symbol = symbol.replace("/", "").upper()
            if horizon_key in self.calibrated_profiles:
                for k, v in self.calibrated_profiles[horizon_key].items():
                    if k.replace("/", "").upper() == clean_symbol:
                        if 'veto_threshold' in v:
                            veto_threshold = v['veto_threshold']
                            break

            # Check if any bypass exit was proposed (including the opener strategy)
            triggered_bypass = [v for v in votes_to_exit if v in bypass_exits]
            if triggered_bypass:
                should_exit = True
                final_reason = reasons[-1] # The reason from the bypass strategy
                confidence = 1.0
            elif consensus_ratio >= veto_threshold:
                should_exit = True
                final_reason = f"CONSENSUS_EXIT ({', '.join(reasons)})"
                confidence = consensus_ratio
            elif votes_to_exit:
                final_reason = f"VETOED_EXIT (Only {consensus_ratio*100:.1f}% agreed)"
                
            verdict = OracleVerdict(
                trade_id=trade_id,
                symbol=symbol,
                should_exit=should_exit,
                reason=final_reason,
                confidence=confidence,
                proposing_strategies=votes_to_exit,
                dynamic_targets=dynamic_targets
            )
            
            # Log ALL strategy decisions to exit_strategy_log
            if self.db and all_decisions:
                for dec in all_decisions:
                    was_overridden = dec['action'] == 'PROPOSE_EXIT' and not should_exit
                    try:
                        self.db.log_exit_strategy_decision(
                            trade_id=trade_id,
                            symbol=symbol,
                            bar_number=bar_number,
                            strategy_id=dec['strategy_id'],
                            action=dec['action'],
                            reason=dec['reason'],
                            unrealized_pnl=pnl,
                            price_at_decision=current_price,
                            was_overridden=was_overridden,
                            override_reason=final_reason if was_overridden else None
                        )
                    except Exception as e:
                        logger.debug(f"[ExitOracle] Decision log skipped: {e}")
            
            # Legacy exit_decision logging (backward compat)
            if votes_to_exit:
                if self.db:
                    self.db.log_exit_decision(
                        trade_id=trade_id,
                        symbol=symbol,
                        exit_reason=final_reason,
                        proposing_strategy=",".join(votes_to_exit),
                        oracle_verdict="APPROVED" if should_exit else "DENIED",
                        pnl_at_decision=pnl
                    )
                
            if should_exit or dynamic_targets:
                verdicts.append(verdict)
                if should_exit:
                    logger.info(f"🔮 [ExitOracle] EXIT APPROVED for {symbol} ({trade_id}) - Reason: {final_reason}")
                    
                    import logging
                    log_supreme_event(
                        logger_instance=logger,
                        level=logging.INFO,
                        event_id=f"ORACLE_EXIT_{trade_id}",
                        que_ocurrio={
                            "tipo_evento": "EXIT_DECISION",
                            "descripcion": f"Cierre aprobado para {symbol} ({horizon})",
                            "resultado": "EXIT_APPROVED"
                        },
                        por_que_ocurrio={
                            "razon_principal": final_reason,
                            "estrategias_proponentes": votes_to_exit,
                            "ratio_consenso": consensus_ratio
                        },
                        como_ocurrio={
                            "pnl_actual": pnl,
                            "tiempo_abierto_mins": duration_mins,
                            "motor_decisor": engine_id
                        },
                        donde_ocurrio={
                            "modulo": "ExitOracle",
                            "funcion": "evaluate_open_positions"
                        },
                        quien_lo_provoco={
                            "componente": "OracleConsensus",
                            "metadata_trade": pos_data.get('metadata', {})
                        }
                    )
                else:
                    logger.debug(f"🔮 [ExitOracle] UPDATE TARGETS for {symbol} ({trade_id}) - {dynamic_targets}")
            
            # Clean up eval counter on exit
            if should_exit and trade_id in self._eval_counts:
                del self._eval_counts[trade_id]
                
        return verdicts

    def evaluate_position(self, symbol: str, pos: Dict[str, Any], current_price: float, data_handler: Any, prediction_tracker: Any = None, current_time: datetime = None) -> Tuple[str, str]:
        """
        🔮 evaluate_position: Evaluador individual para backward compatibility con el risk_manager.
        """
        qty = pos.get('quantity', 0)
        if abs(qty) < 1e-8:
            return "KEEP_OPEN", ""

        direction = 1 if qty > 0 else -1
        avg_price = pos.get('avg_price', current_price)
        horizon = pos.get('horizon', 'SCALPING')
        entry_time = pos.get('entry_time')

        now = current_time or datetime.now(timezone.utc)
        duration_mins = 0.0
        if isinstance(entry_time, datetime):
            duration_mins = (now - entry_time).total_seconds() / 60.0
        elif isinstance(entry_time, (int, float)):
            now_ts = now.timestamp() if isinstance(now, datetime) else time.time()
            duration_mins = (now_ts - entry_time) / 60.0

        pnl_pct = (current_price - avg_price) / avg_price if direction == 1 else (avg_price - current_price) / avg_price
        
        strategy_id = pos.get('opener_strategy_id') or pos.get('strategy_id', '')
        strat_class = self._classify_strategy_class(strategy_id)
        pt = prediction_tracker or self.prediction_tracker

        if strat_class == 'ML':
            action, reason = self._evaluate_ml_exit(
                symbol=symbol,
                pos_data=pos,
                direction=direction,
                avg_price=avg_price,
                current_price=current_price,
                duration_mins=duration_mins,
                pnl_pct=pnl_pct,
                horizon=horizon,
                prediction_tracker=pt
            )
        elif strat_class == 'TECHNICAL':
            action, reason = self._evaluate_technical_exit(
                symbol=symbol,
                pos_data=pos,
                direction=direction,
                avg_price=avg_price,
                current_price=current_price,
                duration_mins=duration_mins,
                pnl_pct=pnl_pct,
                horizon=horizon,
                data_handler=data_handler
            )
        elif strat_class == 'PATTERN':
            action, reason = self._evaluate_pattern_exit(
                symbol=symbol,
                pos_data=pos,
                direction=direction,
                avg_price=avg_price,
                current_price=current_price,
                duration_mins=duration_mins,
                pnl_pct=pnl_pct,
                horizon=horizon,
                data_handler=data_handler
            )
        elif strat_class == 'STATISTICAL':
            action, reason = self._evaluate_statistical_exit(
                symbol=symbol,
                pos_data=pos,
                direction=direction,
                avg_price=avg_price,
                current_price=current_price,
                duration_mins=duration_mins,
                pnl_pct=pnl_pct,
                horizon=horizon,
                data_handler=data_handler
            )
        else:
            action, reason = "KEEP_OPEN", ""

        return action, reason

    def evaluate_flip_exit(self, symbol: str, current_direction: str, new_signal_direction: str, pnl_pct: float, mfe_pct: float) -> Tuple[bool, str]:
        """
        🔮 CTOS Phase 4: Gestor Universal de FLIP_EXITs.
        """
        if current_direction.lower() == new_signal_direction.lower():
            return False, "Same direction"
            
        is_growing = mfe_pct > 0.0015 and pnl_pct > -0.0015
        
        if is_growing:
            if self.prediction_tracker:
                pred = self.prediction_tracker.get_prediction_for_trade(symbol=symbol)
                if pred:
                    edge_prob = self.prediction_tracker.calculate_realtime_edge(
                        strategy_id=pred.get('strategy_id', ''),
                        elapsed_bars=pred.get('bar_count', 0),
                        horizon=pred.get('horizon', 'SCALPING')
                    )
                    if edge_prob > 0.40:
                        return False, f"Edge sigue vivo ({edge_prob*100:.1f}%)"
            
            return False, "Posición sana y creciendo (MFE)"
            
        return True, "Posición estancada o en pérdida. FLIP AUTORIZADO."
