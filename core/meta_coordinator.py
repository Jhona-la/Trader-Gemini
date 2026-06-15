"""
MetaCoordinator — Unified System Brain (CTOS)
═══════════════════════════════════════════════════════════════
QUÉ: Cerebro central ÚNICO del sistema. Fusiona MetaArbitrator + MetaCoordinator.
POR QUÉ: Antes había DOS cerebros (meta_arbitrator.py y meta_coordinator.py)
  compitiendo por la gobernanza de señales. Esto causaba inconsistencias.
PARA QUÉ: Un solo punto de decisión que:
  1. Intercepta TradeIntents (señales crudas desde estrategias)
  2. Aplica Invariants (reglas axiomáticas inquebrantables)
  3. Ejecuta GraphVetoes (análisis topológico del ecosistema)
  4. Resuelve conflictos intra-horizonte (LONG vs SHORT simultáneo)
  5. Emite ExecutionPlans aprobados al RiskManager
CÓMO: Escucha EventChannel.INTENTS → Procesa → Emite a approved_queue.
CUÁNDO: En cada ciclo del arbitration_loop (cada 10ms).
DÓNDE: core/meta_coordinator.py (este archivo — reemplaza meta_arbitrator.py)
QUIÉN: Arquitecto Senior + Risk Manager + Quant Developer
═══════════════════════════════════════════════════════════════
"""

import time
import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional

from utils.logger import logger
from core.events import SignalEvent, EventType, SignalType
from core.enums import OrderSide
from core.invariants import invariants
from config import Config
from data.database import DatabaseHandler
from core.enums import OrderSide
from core.invariants import invariants
from config import Config

# ════════════════════════════════════════════════════════════════
# SUB-COMPONENTS (Absorbed from meta_arbitrator.py)
# ════════════════════════════════════════════════════════════════

class RegimeHorizonRouter:
    """
    Assigns dynamic weights to strategies based on the current market regime.
    Absorbed from meta_arbitrator.py for unified governance.
    """
    def __init__(self):
        self.regime = "UNKNOWN"

    def get_weights(self, global_regime: str) -> Dict[str, float]:
        if "TREND" in global_regime.upper():
            return {"SWING_TREND": 1.0, "BREAKOUT": 0.8, "SCALP_COUNTER": 0.2}
        elif "CHOP" in global_regime.upper() or "SIDEWAYS" in global_regime.upper():
            return {"SCALP_MEAN_REVERSION": 1.0, "RANGE_SCALP": 0.9, "SWING_BREAKOUT": 0.15}
        return {"SCALP": 1.0, "SWING": 1.0}


class SymbolRanker:
    """
    Ranks symbols by tradability.
    Absorbed from meta_arbitrator.py for unified governance.
    """
    def rank_symbols(self, states: Dict[str, Dict[str, Any]]) -> List[str]:
        def score(state):
            return state.get("liquidity_score", 0.5) * 0.4 + abs(state.get("trend_score", 0)) * 0.6
        ranked = sorted(states.items(), key=lambda x: score(x[1]), reverse=True)
        return [sym for sym, _ in ranked]


# ════════════════════════════════════════════════════════════════
# MAIN CLASS
# ════════════════════════════════════════════════════════════════

class MetaCoordinator:
    """
    Unified System Brain — CTOS Governance Layer.
    
    REPLACES: meta_arbitrator.py (now .BAK)
    ABSORBS: GraphIntelligenceLayer vetoes, RegimeHorizonRouter, SymbolRanker,
             conflict resolution, and SystemInvariants enforcement.
    """
    def __init__(self):
        self.router = RegimeHorizonRouter()
        self.ranker = SymbolRanker()
        self.intent_queue: asyncio.Queue = asyncio.Queue()
        self.approved_queue: asyncio.Queue = asyncio.Queue()
        
        self.is_running = False
        self._task = None
        self.db = DatabaseHandler()
        
        # 🌐 GRAPH INTELLIGENCE LAYER (Institutional Core)
        try:
            from core.graph_intelligence import GraphIntelligenceLayer
            self.graph_layer = GraphIntelligenceLayer(symbols=Config.CRYPTO_FUTURES_PAIRS)
        except Exception as e:
            logger.warning(f"[META-COORD] GraphIntelligenceLayer init failed: {e}")
            self.graph_layer = None
        
        # Telemetry
        self._metrics = {
            'intents_received': 0,
            'intents_approved': 0,
            'intents_vetoed_invariant': 0,
            'intents_vetoed_graph': 0,
            'intents_vetoed_conflict': 0,
        }
        
        logger.info("🧠 [META-COORD] Unified MetaCoordinator initialized (CTOS Brain).")

    # ════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ════════════════════════════════════════════════════════════════
    
    def start(self):
        self.is_running = True
        self._task = asyncio.create_task(self._arbitration_loop())
        logger.info("🧠 [META-COORD] Unified Brain STARTED.")
        
    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()

    # ════════════════════════════════════════════════════════════════
    # PUBLIC API (Drop-in replacement for MetaArbitrator)
    # ════════════════════════════════════════════════════════════════
    
    async def submit_intent(self, event: SignalEvent):
        """Called by Engine to submit a Trade Intent."""
        self._metrics['intents_received'] += 1
        await self.intent_queue.put(event)
        
    async def get_approved_intent(self) -> SignalEvent:
        """Called by Engine to retrieve the next approved trade intent."""
        return await self.approved_queue.get()

    # ════════════════════════════════════════════════════════════════
    # INVARIANT ENFORCEMENT (NEW — from invariants.py)
    # ════════════════════════════════════════════════════════════════
    
    def _check_invariants(self, intent: SignalEvent) -> bool:
        """
        Applies SystemInvariants to the raw signal.
        Converts SignalEvent fields into a pseudo-TradeIntent for validation.
        """
        # Map SignalEvent to invariant-compatible structure
        from core.structs import TradeIntent
        
        direction = 'EXIT'
        if intent.signal_type == SignalType.LONG:
            direction = 'LONG'
        elif intent.signal_type == SignalType.SHORT:
            direction = 'SHORT'
        
        # EXIT signals ALWAYS pass invariants
        if direction == 'EXIT':
            return True
        
        pseudo_intent = TradeIntent(
            symbol=intent.symbol,
            direction=direction,
            confidence=getattr(intent, 'confidence', getattr(intent, 'strength', 0.5)),
            expected_mfe=0.0,
            expected_mae=0.0,
            horizon=getattr(intent, 'horizon', 'SCALPING'),
            regime_compatibility=1.0,
            liquidity_score=0.5,  # Default; will be enriched when SSOT is fully wired
            strategy_id=getattr(intent, 'strategy_id', 'unknown'),
            timestamp_ns=getattr(intent, 'timestamp_ns', 0),
        )
        
        passed, reason = invariants.check_all(pseudo_intent)
        if not passed:
            logger.warning(f"🛡️ [INVARIANT] {intent.symbol} {direction} BLOCKED: {reason}")
            self._metrics['intents_vetoed_invariant'] += 1
        return passed

    # ════════════════════════════════════════════════════════════════
    # GRAPH VETOES (Absorbed from MetaArbitrator._apply_graph_vetoes)
    # ════════════════════════════════════════════════════════════════
    
    def _apply_graph_vetoes(self, intent: SignalEvent) -> bool:
        """
        Applies Graph Theory and State Vector rules to veto bad intents.
        Returns True if approved, False if vetoed.
        """
        if intent.signal_type == SignalType.EXIT:
            return True  # Exits are always approved
        
        if not self.graph_layer:
            return True  # Fallback if graph not available
            
        direction = "LONG" if intent.signal_type == SignalType.LONG else "SHORT"
        state = self.graph_layer.state_matrix.get(intent.symbol)
        
        if not state:
            return True  # Fallback if state is missing
            
        # 1. GRAPH CONTAGION VETO
        if direction == "LONG":
            contagion_risk = self.graph_layer.get_contagion_risk(intent.symbol)
            if contagion_risk > 0.50:
                logger.info(f"🛡️ [VETO GRAFO] LONG bloqueado en {intent.symbol}: Contagio Bajista ({contagion_risk:.2f})")
                self._metrics['intents_vetoed_graph'] += 1
                return False
                
        # 2. MICROSTRUCTURE VECTOR VETO
        if direction == "LONG" and state.orderflow_imbalance < -0.60:
            logger.info(f"🛡️ [VETO VECTOR] LONG bloqueado en {intent.symbol}: OrderFlow Severamente Negativo ({state.orderflow_imbalance:.2f})")
            self._metrics['intents_vetoed_graph'] += 1
            return False
        if direction == "SHORT" and state.orderflow_imbalance > 0.60:
            logger.info(f"🛡️ [VETO VECTOR] SHORT bloqueado en {intent.symbol}: OrderFlow Severamente Positivo ({state.orderflow_imbalance:.2f})")
            self._metrics['intents_vetoed_graph'] += 1
            return False
            
        # 3. MACRO GRAVITY VETO
        ecosystem_gravity = self.graph_layer.get_ecosystem_gravity()
        if direction == "LONG" and ecosystem_gravity < -2.0 and state.eigenvector_centrality > 0.1:
            logger.info(f"🛡️ [VETO MACRO] LONG bloqueado en {intent.symbol}: Gravedad Ecosistémica Negativa ({ecosystem_gravity:.2f})")
            self._metrics['intents_vetoed_graph'] += 1
            return False
            
        return True

    # ════════════════════════════════════════════════════════════════
    # CONFLICT RESOLUTION (Absorbed from MetaArbitrator.resolve_intents)
    # ════════════════════════════════════════════════════════════════
    
    def resolve_intents(self, intents_to_process: List[SignalEvent]) -> tuple:
        """Resolves conflicts among a batch of intents using the unified ConsensusFilter."""
        from core.consensus_filter import get_consensus_filter
        consensus = get_consensus_filter()
        
        approved_intents = []
        rejected_intents = []
        
        # Filtro de Consenso Omnisciente
        passed_intents = []
        veto_reasons = {}
        
        for intent in intents_to_process:
            # Detectar precio actual
            price = getattr(intent, 'price', 0.0)
            if not price and hasattr(intent, 'metadata') and intent.metadata:
                price = intent.metadata.get('close', 0.0)
                
            passed, reason = consensus.check_signal(
                signal_event=intent,
                portfolio=getattr(self, 'portfolio', None),
                current_price=price,
                risk_manager=getattr(self, 'risk_manager', None),
                meta_coordinator=self
            )
            
            # CRITICAL FIX: Actually enforce Invariants and Graph Vetoes!
            if passed:
                if intent.signal_type == SignalType.EXIT:
                    # EXIT signals bypass invariants, graph vetoes, and opening validations
                    pass
                else:
                    if not self._check_invariants(intent):
                        passed = False
                        reason = "INVARIANT_VETO"
                    elif not self._apply_graph_vetoes(intent):
                        passed = False
                        reason = "GRAPH_VETO"
                    else:
                        from core.asset_intelligence import get_asset_intelligence
                        passed_ai, reason_ai = get_asset_intelligence().verify_opening(intent, getattr(self, 'portfolio', None))
                        if not passed_ai:
                            passed = False
                            reason = reason_ai
            
            if passed:
                passed_intents.append(intent)
            else:
                passed_intents.append(None) # Para alineación con el registro de pensamientos
                veto_reasons[id(intent)] = reason
                rejected_intents.append({"intent": intent, "reason": reason})

        # ════════════════════════════════════════════════════════════════
        # CTOS PHASE 3: OMNISCIENT EXIT TRACKING & CONFLICT RESOLUTION
        # ════════════════════════════════════════════════════════════════
        # Document all intents to thoughts DB before resolving final conflicts
        for i, intent in enumerate(intents_to_process):
            thought_id = f"THOUGHT_{uuid.uuid4().hex[:8]}"
            import dataclasses
            current_metadata = getattr(intent, 'metadata', None) or {}
            new_metadata = dict(current_metadata)
            new_metadata['thought_id'] = thought_id
            
            if dataclasses.is_dataclass(intent):
                intent = dataclasses.replace(intent, metadata=new_metadata)
                intents_to_process[i] = intent
            else:
                _meta = getattr(intent, 'metadata', None)
                if _meta is None:
                    _meta = {}
                    try:
                        object.__setattr__(intent, 'metadata', _meta)
                    except (AttributeError, TypeError):
                        pass
                _meta['thought_id'] = thought_id
            
            # Determine outcome so far
            status = "PENDING"
            reason = "Awaiting Conflict Resolution"
            
            if passed_intents[i] is None:
                status = "VETOED"
                reason = veto_reasons.get(id(intent), "UNKNOWN_VETO")
                
            direction = "EXIT" if intent.signal_type == SignalType.EXIT else ("LONG" if intent.signal_type == SignalType.LONG else "SHORT")
            
            self.db.log_thought(
                thought_id=thought_id,
                trade_id=getattr(intent, 'trade_id', None),
                symbol=intent.symbol,
                strategy_id=getattr(intent, 'strategy_id', 'UNKNOWN'),
                horizon=getattr(intent, 'horizon', 'SCALPING'),
                direction=f"{direction} ({status}: {reason})",
                market_state={}, # Will be enriched by engine
                metrics={"confidence": float(getattr(intent, 'confidence', getattr(intent, 'strength', 0)))}
            )
            
        # Limpiar la lista de pasados para el Step 3
        graph_passed = [x for x in passed_intents if x is not None]
                
        # Step 3: Intra-Horizon Conflict Resolution
        # Group by (symbol, horizon) to allow cross-horizon hedging
        grouped: Dict[tuple, List[SignalEvent]] = {}
        for intent in graph_passed:
            sym = intent.symbol
            horizon = getattr(intent, 'horizon', 'SCALPING')
            key = (sym, horizon)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(intent)
        
        # Get global state for ranking
        from core.global_state import global_state
        all_states = global_state.get_all_states()
        
        for (symbol, horizon), intents in grouped.items():
            longs = [i for i in intents if i.signal_type == SignalType.LONG]
            shorts = [i for i in intents if i.signal_type == SignalType.SHORT]
            exits = [i for i in intents if i.signal_type == SignalType.EXIT]
            
            # FORENSIC FIX: Deduplicate Exits!
            # If we received multiple exits for the same (symbol, horizon) in the same loop,
            # we only approve ONE to avoid Binance order spam / race condition.
            if exits:
                best_exit = max(exits, key=lambda i: getattr(i, 'confidence', getattr(i, 'strength', 0)))
                approved_intents.append(best_exit)
                # Log the exit decision explicitly for Phase 3 forensics
                self.db.log_exit_decision(
                    trade_id=getattr(best_exit, 'trade_id', f"UNKNOWN_{symbol}"),
                    symbol=symbol,
                    exit_reason="Approved by MetaCoordinator (Deduplicated)",
                    proposing_strategy=getattr(best_exit, 'strategy_id', 'UNKNOWN'),
                    oracle_verdict="CLOSE",
                    pnl_at_decision=0.0 # Engine needs to fill this later or we pull from SSOT
                )
                
                # Reject the duplicates
                for ext in exits:
                    if ext != best_exit:
                        rejected_intents.append({"intent": ext, "reason": "EXIT_DEDUPLICATION"})
                
            if longs and shorts:
                # Conflict within same horizon — pick highest confidence
                logger.warning(
                    f"⚔️ [META-COORD] Intra-horizon conflict on {symbol} ({horizon}): "
                    f"{len(longs)} LONG vs {len(shorts)} SHORT"
                )
                self._metrics['intents_vetoed_conflict'] += len(longs) + len(shorts) - 1
                
                all_directional = longs + shorts
                winner = max(all_directional, key=lambda i: getattr(i, 'confidence', getattr(i, 'strength', 0)))
                approved_intents.append(winner)
                
                for loser in all_directional:
                    if loser != winner:
                        rejected_intents.append({"intent": loser, "reason": "CONFLICT_RESOLUTION"})
            else:
                # No conflicts — approve all directional
                approved_intents.extend(longs)
                approved_intents.extend(shorts)
        
        self._metrics['intents_approved'] += len(approved_intents)
        return approved_intents, rejected_intents

    # ════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════════════
    
    async def _arbitration_loop(self):
        """Processes intents dynamically every 10ms."""
        while self.is_running:
            try:
                intents_to_process = []
                while not self.intent_queue.empty():
                    intents_to_process.append(self.intent_queue.get_nowait())
                    
                if not intents_to_process:
                    await asyncio.sleep(0.01)  # 10ms sleep
                    continue
                    
                approved, _ = self.resolve_intents(intents_to_process)
                
                for intent in approved:
                    await self.approved_queue.put(intent)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🚨 [META-COORD] Loop error: {e}")
                await asyncio.sleep(1)

    def get_metrics(self) -> Dict[str, int]:
        """Returns telemetry metrics for dashboard/monitoring."""
        return self._metrics.copy()


# ════════════════════════════════════════════════════════════════
# SINGLETON (Drop-in replacement for meta_arbitrator)
# ════════════════════════════════════════════════════════════════
meta_coordinator = MetaCoordinator()

# Backward-compatibility alias
meta_arbitrator = meta_coordinator
