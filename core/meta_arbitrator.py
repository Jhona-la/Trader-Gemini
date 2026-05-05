"""
Meta-Arbitrator & Supreme Co-ordinator
Replaces direct signal execution with an intentional, buffered, and globally-aware
routing and conflict resolution system.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import replace

from utils.logger import logger
from core.events import SignalEvent, EventType, SignalType
from core.symbol_state_matrix import symbol_state_matrix
from core.enums import OrderSide

class RegimeHorizonRouter:
    """
    Assigns dynamic weights to strategies based on the current market regime.
    """
    def __init__(self):
        self.regime = "UNKNOWN"

    def get_weights(self, global_regime: str) -> Dict[str, float]:
        """
        Returns multiplier weights for confidence based on the regime.
        """
        # Pseudo-ML logic based on user request
        if "TREND" in global_regime.upper():
            return {"SWING_TREND": 1.0, "BREAKOUT": 0.8, "SCALP_COUNTER": 0.2}
        elif "CHOP" in global_regime.upper() or "SIDEWAYS" in global_regime.upper():
            return {"SCALP_MEAN_REVERSION": 1.0, "RANGE_SCALP": 0.9, "SWING_BREAKOUT": 0.15}
        
        # Default fallback weights
        return {"SCALP": 1.0, "SWING": 1.0}

class SymbolRanker:
    """
    Ranks symbols by tradability.
    """
    def rank_symbols(self, states: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Returns a sorted list of symbols from most to least tradable.
        """
        def score(state):
            # Example heuristic: high liquidity + high trend/volatility = highly tradable
            return state.get("liquidity_score", 0.5) * 0.4 + abs(state.get("trend_score", 0)) * 0.6
            
        ranked = sorted(states.items(), key=lambda x: score(x[1]), reverse=True)
        return [sym for sym, state in ranked]

class MetaArbitrator:
    """
    The Supreme Meta-Coordinator brain.
    Intercepts Trade Intents (SignalEvents), resolves cross-horizon/cross-strategy conflicts,
    and forwards Approved Intents to the execution pipeline.
    """
    def __init__(self):
        self.router = RegimeHorizonRouter()
        self.ranker = SymbolRanker()
        self.intent_queue: asyncio.Queue = asyncio.Queue()
        self.approved_queue: asyncio.Queue = asyncio.Queue()
        
        self.is_running = False
        self._task = None
        
        # We can store recent signals to detect conflicts
        # Format: { symbol: { horizon: SignalEvent } }
        self._live_intent_buffer: Dict[str, List[SignalEvent]] = {}
        
    def start(self):
        self.is_running = True
        self._task = asyncio.create_task(self._arbitration_loop())
        logger.info("🧠 [META-ARBITRATOR] Supreme Meta-Coordinator STARTED.")
        
    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            
    async def submit_intent(self, event: SignalEvent):
        """Called by Engine to submit a Trade Intent instead of immediate execution."""
        await self.intent_queue.put(event)
        
    async def get_approved_intent(self) -> SignalEvent:
        """Called by Engine to retrieve the next approved trade intent."""
        return await self.approved_queue.get()
        
    def resolve_intents(self, intents_to_process: List[SignalEvent]) -> tuple[List[SignalEvent], List[dict]]:
        """Synchronously resolves conflicts among a list of intents."""
        approved_intents = []
        rejected_intents_with_reasons = []
        
        # Group by (symbol, horizon) to allow cross-horizon hedging [P0 FIX]
        grouped_intents: Dict[tuple, List[SignalEvent]] = {}
        for intent in intents_to_process:
            sym = intent.symbol
            horizon = getattr(intent, 'horizon', 'SCALPING')
            key = (sym, horizon)
            if key not in grouped_intents:
                grouped_intents[key] = []
            grouped_intents[key].append(intent)
            
        # Get global state
        all_states = symbol_state_matrix.get_all_states()
        ranked_symbols = self.ranker.rank_symbols(all_states)
        
        # Process grouped intents
        for (symbol, horizon), intents in grouped_intents.items():
            # Check for conflicts intra-horizon
            longs = [i for i in intents if i.signal_type == SignalType.LONG]
            shorts = [i for i in intents if i.signal_type == SignalType.SHORT]
            exits = [i for i in intents if i.signal_type == SignalType.EXIT]
            
            # Exits ALWAYS get priority and bypass conflict resolution
            for ext in exits:
                approved_intents.append(ext)
                
            if longs and shorts:
                # Conflict detected within the SAME horizon!
                logger.warning(f"⚔️ [META-ARBITRATOR] Intra-horizon conflict detected on {symbol} ({horizon}): {len(longs)} LONG vs {len(shorts)} SHORT intents.")
                
                # Arbitrate: Prioritize highest confidence within the same horizon
                winning_list = longs if max([getattr(l, 'confidence', l.strength) for l in longs] + [0]) >= max([getattr(s, 'confidence', s.strength) for s in shorts] + [0]) else shorts
                losing_list = shorts if winning_list == longs else longs
                    
                # Pass only the highest confidence winner
                if winning_list:
                    winner = max(winning_list, key=lambda i: getattr(i, 'confidence', i.strength))
                    approved_intents.append(winner)
                    logger.info(f"⚖️ [META-ARBITRATOR] Conflict Resolved. Winner: {winner.horizon} {winner.signal_type.name}")
                    
                    # Log losers for Alpha Leak accounting
                    for loser in winning_list:
                        if loser != winner:
                            rejected_intents_with_reasons.append({"intent": loser, "reason": "OUTBID_BY_HIGHER_CONFIDENCE"})
                    for loser in losing_list:
                        rejected_intents_with_reasons.append({"intent": loser, "reason": "REGIME_CONFLICT_VETO"})
            else:
                # No conflicts intra-horizon, approve all valid directional intents
                approved_intents.extend(longs)
                approved_intents.extend(shorts)
                    
        return approved_intents, rejected_intents_with_reasons

    async def _arbitration_loop(self):
        """
        Processes intents dynamically. To maintain nano-speeds, this runs frequently
        (e.g. every 10ms) to clear the buffer.
        """
        while self.is_running:
            try:
                # 1. Drain the current queue into the buffer
                intents_to_process = []
                while not self.intent_queue.empty():
                    intents_to_process.append(self.intent_queue.get_nowait())
                    
                if not intents_to_process:
                    await asyncio.sleep(0.01) # 10ms sleep
                    continue
                    
                # 2. Resolve synchronously
                approved_intents, _ = self.resolve_intents(intents_to_process)
                
                # 3. Put back in queue
                for intent in approved_intents:
                    await self.approved_queue.put(intent)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🚨 [META-ARBITRATOR] Loop error: {e}")
                await asyncio.sleep(1)

# Singleton
meta_arbitrator = MetaArbitrator()
