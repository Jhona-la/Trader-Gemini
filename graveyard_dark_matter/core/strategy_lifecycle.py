import logging
from enum import Enum
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class StrategyPhase(Enum):
    RESEARCH = "RESEARCH"
    PAPER_TRADING = "PAPER_TRADING"
    ACTIVE_MINIMUM = "ACTIVE_MINIMUM"
    MATURE = "MATURE"
    DECLINE = "DECLINE"
    RETIRED = "RETIRED"

class StrategyMetadata:
    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        self.phase = StrategyPhase.PAPER_TRADING
        self.cycles_in_phase = 0
        self.paper_trading_signals = 0
        self.paper_trading_pnl = 0.0
        
        # Live stats
        self.live_cycles = 0
        self.rolling_profit_factor = 1.0
        self.peak_profit_factor = 1.0
        self.historical_winrate = 0.0
        self.alpha_decay_warnings = 0
        
        # Scaling
        self.allocation_modifier = 0.0 # 0.0 for paper, 1.0 for mature

class StrategyLifecycleManager:
    """
    Bloque V del Prompt Supremo.
    Gestiona el ciclo de vida completo de cada estrategia (Investigación -> Retiro).
    """
    def __init__(self):
        self.strategies: Dict[str, StrategyMetadata] = {}
        
    def register_strategy(self, strategy_id: str, is_new: bool = False):
        if strategy_id not in self.strategies:
            meta = StrategyMetadata(strategy_id)
            if not is_new:
                # Si no es nueva (viene del código base y la confiamos), arranca madura
                meta.phase = StrategyPhase.MATURE
                meta.allocation_modifier = 1.0
            self.strategies[strategy_id] = meta
            logger.info(f"🧬 [LIFECYCLE] Estrategia registrada: {strategy_id} | Fase: {meta.phase.value}")
            
    def update_strategy_stats(self, strategy_id: str, pf: float, winrate: float):
        if strategy_id not in self.strategies: return
        meta = self.strategies[strategy_id]
        
        # Actualizar rolling metrics
        meta.rolling_profit_factor = (meta.rolling_profit_factor * 0.9) + (pf * 0.1)
        meta.peak_profit_factor = max(meta.peak_profit_factor, meta.rolling_profit_factor)
        meta.historical_winrate = (meta.historical_winrate * 0.9) + (winrate * 0.1)
        
    def evaluate_cycle_transitions(self):
        """Llamado al final de cada ciclo de 72h."""
        for sid, meta in self.strategies.items():
            meta.cycles_in_phase += 1
            if meta.phase in [StrategyPhase.MATURE, StrategyPhase.ACTIVE_MINIMUM]:
                meta.live_cycles += 1
                
            # Evaluar PAPER_TRADING -> ACTIVE_MINIMUM
            if meta.phase == StrategyPhase.PAPER_TRADING:
                if meta.cycles_in_phase >= 3 and meta.paper_trading_signals >= 50:
                    if meta.rolling_profit_factor > 1.8:
                        meta.phase = StrategyPhase.ACTIVE_MINIMUM
                        meta.allocation_modifier = 0.05  # 5% capital allocation
                        meta.cycles_in_phase = 0
                        logger.info(f"🚀 [LIFECYCLE] Estrategia {sid} PROMOCIONADA a ACTIVE_MINIMUM.")
                        
            # Evaluar ACTIVE_MINIMUM -> MATURE
            elif meta.phase == StrategyPhase.ACTIVE_MINIMUM:
                if meta.cycles_in_phase >= 5:
                    if meta.rolling_profit_factor > 1.5:
                        meta.phase = StrategyPhase.MATURE
                        meta.allocation_modifier = 1.0
                        meta.cycles_in_phase = 0
                        logger.info(f"💎 [LIFECYCLE] Estrategia {sid} PROMOCIONADA a MATURE (Full Allocation).")
                        
            # Evaluar Alpha Decay (MATURE -> DECLINE)
            elif meta.phase == StrategyPhase.MATURE:
                # Alerta si el PF cae por debajo de 1.7 de su pico
                if meta.rolling_profit_factor < 1.7 and meta.peak_profit_factor > 2.0:
                    meta.alpha_decay_warnings += 1
                    if meta.alpha_decay_warnings >= 3:
                        meta.phase = StrategyPhase.DECLINE
                        meta.allocation_modifier = 0.5 # Cortar tamaño a la mitad
                        meta.cycles_in_phase = 0
                        logger.warning(f"⚠️ [LIFECYCLE] ALPHA DECAY: Estrategia {sid} DEGRADADA a DECLINE (Size 50%).")
                else:
                    meta.alpha_decay_warnings = 0
                    
            # Evaluar DECLINE -> RETIRED
            elif meta.phase == StrategyPhase.DECLINE:
                if meta.rolling_profit_factor < 1.2 and meta.cycles_in_phase >= 5:
                    meta.phase = StrategyPhase.RETIRED
                    meta.allocation_modifier = 0.0
                    meta.cycles_in_phase = 0
                    logger.error(f"💀 [LIFECYCLE] Estrategia {sid} RETIRADA (Archivada por fallo de alpha).")
                    
    def get_allocation_modifier(self, strategy_id: str) -> float:
        """Devuelve el multiplicador de tamaño de posición permitido para esta estrategia."""
        if strategy_id not in self.strategies: return 1.0
        return self.strategies[strategy_id].allocation_modifier

    def can_execute_live(self, strategy_id: str) -> bool:
        """Devuelve True si la estrategia puede ejecutar órdenes reales."""
        mod = self.get_allocation_modifier(strategy_id)
        return mod > 0.0
