import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from utils.logger import logger
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
      TODAS las estrategias de salida y toma la decisión final.
    - POR QUÉ: Sin coordinación, las estrategias se "pisan las patas":
      una cierra cuando la predicción dice que el precio va a subir más.
    - PARA QUÉ: Maximizar PnL por trade, evitando cierres prematuros
      y permitiendo que los trades ganadores corran a su punto óptimo.
    - CÓMO: Consulta a TODAS las estrategias + PredictionTracker,
      pondera votos con confianza, y registra CADA decisión en DB
      para auditoría post-mortem.
    - CUÁNDO: En cada tick de evaluación de posiciones abiertas.
    - DÓNDE: sophia/exit_oracle.py
    - QUIÉN: Llamado desde engine.py en el loop de evaluación de posiciones.
    """
    def __init__(self, db_handler: DatabaseHandler = None, 
                 sophia_intelligence=None, prediction_tracker=None):
        self.db = db_handler
        self.sophia = sophia_intelligence
        self.prediction_tracker = prediction_tracker  # CTOS Phase 3: PredictionTracker connection
        self.strategies = {} # strategy_id -> Strategy instance
        self.veto_threshold = 0.65 # Need 65% consensus among polled strategies
        self._eval_counts = {}  # {trade_id: int} — tick counter for exit_strategy_log bar_number
        
    def register_strategy(self, strategy_id: str, strategy_obj: Any):
        """Registers a strategy so the Oracle can poll it."""
        self.strategies[strategy_id] = strategy_obj
        logger.debug(f"[ExitOracle] Registered strategy: {strategy_id}")

    def evaluate_open_positions(self, open_positions: Dict[str, Dict[str, Any]], market_data: Dict[str, Any]) -> List[OracleVerdict]:
        """
        🔮 CTOS Phase 3: Enhanced evaluation with prediction awareness.
        
        Polls all strategies to determine if any open position should be closed.
        NOW: also checks PredictionTracker to see if the position is on track
        to reach its predicted target, and logs EVERY strategy decision.
        
        open_positions: { 'BTC/USDT_SCALPING': { 'trade_id': '...', 'unrealized_pnl': ..., 'quantity': ... } }
        """
        verdicts = []
        
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
            
            # ═══════════════════════════════════════════════════════════════
            # CTOS PHASE 3: Get prediction context from PredictionTracker
            # QUÉ: Antes de pedir opiniones, obtener qué predijo la estrategia
            #   que abrió este trade. Si predijo "+0.35% en 15 barras" y apenas
            #   lleva 3 barras con +0.12%, quizá NO debemos cerrar.
            # ═══════════════════════════════════════════════════════════════
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
                        # We use the most aggressive (highest tp) from all strategies
                        dt = opinion['dynamic_targets']
                        dynamic_targets["tp_mult"] = max(dynamic_targets["tp_mult"], dt.get("tp_mult", 1.0))
                        dynamic_targets["sl_mult"] = min(dynamic_targets["sl_mult"], dt.get("sl_mult", 1.0))
            
            # 2. Hard stops and risk triggers (these bypass consensus)
            hard_stop_triggered = False
            
            # 3. Time-based Alpha Decay (Inherited from original logic but centralized)
            entry_time = pos_data.get('entry_time')
            horizon = pos_data.get('horizon', 'SCALPING')
            now = datetime.now(timezone.utc)
            duration_mins = 0.0
            if isinstance(entry_time, datetime):
                duration_mins = (now - entry_time).total_seconds() / 60.0
            elif isinstance(entry_time, (int, float)):
                duration_mins = (time.time() - entry_time) / 60.0
            
            # ═══════════════════════════════════════════════════════════════
            # CTOS PHASE 3: INTELLIGENT CONTINUOUS ALPHA DECAY
            # QUÉ: NO cerrar por tiempo puro. Se evalúa el "Edge Probability"
            #   en tiempo real usando la función Numba de decaimiento continuo.
            # POR QUÉ: Permite dar un colchón (cushion) a los trades ganadores
            #   y corta rápido los trades estancados que perdieron su inercia.
            # PARA QUÉ: Maximizar PnL dejando correr ganadores y cerrando zombies.
            # ═══════════════════════════════════════════════════════════════
            if self.prediction_tracker and opener_strat:
                # 1. Calculamos el Edge Probability actual (0.0 a 1.0)
                edge_prob = self.prediction_tracker.calculate_realtime_edge(
                    strategy_id=opener_strat,
                    elapsed_bars=duration_mins,  # Asumimos 1 barra por minuto en SCALPING
                    horizon=horizon
                )
                
                # 2. Lógica del "PnL Cushion" (Colchón de ganancias)
                # Si el trade está a favor (ej. > 0.1%), requerimos que el Edge caiga más bajo para cerrarlo.
                # Si el trade no arrancó (PnL <= 0.1%), lo matamos apenas el Edge baja de 0.45.
                critical_threshold = 0.45
                if pnl > 0.001:  # +0.1% PnL cushion
                    critical_threshold = 0.35 # Le damos más oxígeno porque está en ganancias
                    
                if edge_prob < critical_threshold:
                    # 🛡️ MVP CONSTANT: Evitar salir con ganancias minúsculas (Fee Erosion)
                    # Si el trade está positivo pero por debajo del 0.15% de rentabilidad, NO salimos
                    # por Alpha Decay todavía, a menos que el edge esté terriblemente muerto (< 0.25).
                    # Preferimos que caiga a negativo (donde sí lo cerramos) o que suba al MVP.
                    if 0 < pnl < 0.0015 and edge_prob >= 0.25:
                        all_decisions.append({
                            'strategy_id': 'ALPHA_DECAY',
                            'action': 'HOLD',
                            'reason': f"Edge={edge_prob:.2f} < {critical_threshold} BUT PnL={pnl*100:.2f}% < MVP (Waiting)",
                        })
                    else:
                        votes_to_exit.append("ALPHA_DECAY")
                        reasons.append(f"{horizon} Alpha Decay (Edge={edge_prob:.2f} < {critical_threshold}, PnL={pnl*100:.2f}%)")
                        all_decisions.append({
                            'strategy_id': 'ALPHA_DECAY',
                            'action': 'PROPOSE_EXIT',
                            'reason': f"Edge decayed to {edge_prob:.2f} after {duration_mins:.1f}m",
                        })
                else:
                    all_decisions.append({
                        'strategy_id': 'ALPHA_DECAY',
                        'action': 'HOLD',
                        'reason': f"Edge={edge_prob:.2f} > {critical_threshold} (PnL={pnl*100:.2f}%)",
                    })
            else:
                # Fallback: Lógica vieja rígida si no hay PredictionTracker
                base_threshold = 25.0 if horizon == 'SCALPING' else 180.0
                if duration_mins > base_threshold and pnl < 0.001:
                    votes_to_exit.append("ALPHA_DECAY")
                    reasons.append(f"Fallback {horizon} Alpha Decay ({duration_mins:.1f}m, PnL={pnl*100:.2f}%)")
                    all_decisions.append({
                        'strategy_id': 'ALPHA_DECAY',
                        'action': 'PROPOSE_EXIT',
                        'reason': f"Duration {duration_mins:.1f}m > threshold",
                    })


            # 4. Consensus Logic
            total_strategies = len(self.strategies) or 1
            pure_votes = [v for v in votes_to_exit if v not in ["RISK_MANAGER", "ALPHA_DECAY"]]
            consensus_ratio = len(pure_votes) / total_strategies if total_strategies > 0 else 0
            
            should_exit = False
            final_reason = "MAINTAIN"
            confidence = 0.0
            
            if "RISK_MANAGER" in votes_to_exit or "ALPHA_DECAY" in votes_to_exit:
                should_exit = True
                final_reason = reasons[-1]
                confidence = 1.0
            elif consensus_ratio >= self.veto_threshold:
                should_exit = True
                final_reason = f"CONSENSUS_EXIT ({', '.join(reasons)})"
                confidence = consensus_ratio
            elif votes_to_exit:
                # Polled but not enough consensus
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
            
            # ═══════════════════════════════════════════════════════════════
            # CTOS PHASE 3: Log ALL strategy decisions to exit_strategy_log
            # QUÉ: Registra CADA decisión de CADA estrategia para cada trade.
            # POR QUÉ: Para saber post-mortem por qué no se cerró antes.
            # PARA QUÉ: Diagnosticar trades que deberían haberse cerrado.
            # ═══════════════════════════════════════════════════════════════
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
                else:
                    logger.debug(f"🔮 [ExitOracle] UPDATE TARGETS for {symbol} ({trade_id}) - {dynamic_targets}")
            
            # Clean up eval counter on exit
            if should_exit and trade_id in self._eval_counts:
                del self._eval_counts[trade_id]
                
        return verdicts

    # Fallback/Backward Compatibility method
    def evaluate_position(self, symbol: str, pos: Dict[str, Any], current_price: float, data_handler: Any, prediction_tracker: Any = None, current_time: datetime = None) -> Tuple[str, str]:
        # Minimal wrapper to not break existing tests calling this directly
        # Just evaluates Alpha Decay and Reversals like it used to.
        qty = pos.get('quantity', 0)
        if abs(qty) < 1e-8:
            return "KEEP_OPEN", ""

        direction = TradeDirection.LONG if qty > 0 else TradeDirection.SHORT
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

        pnl_pct = (current_price - avg_price) / avg_price if direction == TradeDirection.LONG else (avg_price - current_price) / avg_price
        
        # Use prediction_tracker parameter OR self.prediction_tracker
        pt = prediction_tracker or self.prediction_tracker
        
        if horizon == 'SCALPING':
            decay_threshold = 15.0
            if pt and pos.get('strategy_id'):
                metrics = pt.get_strategy_metrics(pos.get('strategy_id'), horizon)
                if metrics and 'optimal_ttl_bars' in metrics:
                    decay_threshold = max(5.0, min(30.0, metrics['optimal_ttl_bars'] * 1.5))
            if duration_mins > decay_threshold and pnl_pct < 0.001:
                return "CLOSE_ALPHA_DECAY", f"Scalping Alpha Decay ({duration_mins:.1f}m)"
        elif horizon == 'SWING':
            if duration_mins > 180 and pnl_pct < -0.005:
                return "CLOSE_ALPHA_DECAY", f"Swing Alpha Decay ({duration_mins:.1f}m)"

        if data_handler:
            bars = data_handler.get_latest_bars(symbol, n=10)
            if bars is not None and len(bars) >= 5:
                closes = bars['close']
                # FIX-FORENSIC-V82: bars['close'] is a numpy array, NOT a pandas Series.
                # Using .iloc[] on numpy raises AttributeError.
                if direction == TradeDirection.LONG and all(closes[i] < closes[i-1] for i in range(-4, 0)):
                    if pnl_pct < 0:
                        return "CLOSE_REVERSAL", "Structure Reversal (4 red bars)"
                elif direction == TradeDirection.SHORT and all(closes[i] > closes[i-1] for i in range(-4, 0)):
                    if pnl_pct < 0:
                        return "CLOSE_REVERSAL", "Structure Reversal (4 green bars)"

        return "KEEP_OPEN", ""

    def evaluate_flip_exit(self, symbol: str, current_direction: str, new_signal_direction: str, pnl_pct: float, mfe_pct: float) -> Tuple[bool, str]:
        """
        🔮 CTOS Phase 4: Gestor Universal de FLIP_EXITs.
        QUÉ: Evalúa si una señal contraria del ML es una Reversión Real o Ruido de 1 minuto.
        POR QUÉ: Para evitar cerrar posiciones ganadoras por pánico.
        """
        if current_direction.lower() == new_signal_direction.lower():
            return False, "Same direction"
            
        # 1. Regla Base: Adaptive Holding
        is_growing = mfe_pct > 0.0015 and pnl_pct > -0.0015
        
        if is_growing:
            # 2. Si está ganando, le preguntamos al Tracker si nuestro Edge sigue vivo
            if self.prediction_tracker:
                pred = self.prediction_tracker.get_prediction_for_trade(symbol=symbol)
                if pred:
                    edge_prob = self.prediction_tracker.calculate_realtime_edge(
                        strategy_id=pred.get('strategy_id', ''),
                        elapsed_bars=pred.get('bar_count', 0),
                        horizon=pred.get('horizon', 'SCALPING')
                    )
                    # Si el Edge inicial sigue siendo matemáticamente > 40%, ignoramos el ruido
                    if edge_prob > 0.40:
                        return False, f"Edge sigue vivo ({edge_prob*100:.1f}%)"
            
            return False, "Posición sana y creciendo (MFE)"
            
        # 3. Si la posición NO está creciendo (o está perdiendo dinero), el FLIP es legítimo
        return True, "Posición estancada o en pérdida. FLIP AUTORIZADO."
