import logging
from typing import List, Tuple
from core.structs import TradeIntent
from core.global_state import global_state

logger = logging.getLogger(__name__)

class InvariantViolation(Exception):
    pass

class SystemInvariants:
    """
    Hard-Rules del sistema (Axiomas).
    Ningún TradeIntent puede convertirse en ExecutionPlan si viola estas reglas.
    """
    
    @staticmethod
    def check_all(intent: TradeIntent) -> Tuple[bool, str]:
        """Ejecuta todas las invariantes sobre una intención."""
        checks = [
            SystemInvariants._check_self_hedging,
            SystemInvariants._check_liquidity_threshold,
            SystemInvariants._check_max_exposure
        ]
        
        for check in checks:
            passed, reason = check(intent)
            if not passed:
                logger.warning(f"[INVARIANT VIOLATED] {intent.symbol} -> {reason}")
                return False, reason
                
        return True, "OK"

    @staticmethod
    def _check_self_hedging(intent: TradeIntent) -> Tuple[bool, str]:
        """Evita abrir un LONG si ya hay un SHORT activo en el mismo símbolo."""
        pos = global_state.get_open_position(intent.symbol)
        if pos and pos.quantity != 0:
            if pos.direction != intent.direction and intent.direction != 'EXIT':
                return False, f"Self-Hedging Blocked: Existing {pos.direction}, Intent {intent.direction}"
        return True, ""

    @staticmethod
    def _check_liquidity_threshold(intent: TradeIntent) -> Tuple[bool, str]:
        """Evita operar si la liquidez está por debajo de un umbral catastrófico."""
        if intent.liquidity_score < 0.1:
            return False, f"Liquidity too low ({intent.liquidity_score} < 0.1)"
        return True, ""

    @staticmethod
    def _check_max_exposure(intent: TradeIntent) -> Tuple[bool, str]:
        """Evita sobreexposición si la cuenta ya tiene el margen copado."""
        if global_state.risk.global_hazard_rate > 0.85:
            if intent.direction != 'EXIT':
                return False, "Global Hazard Rate too high (>0.85). Entry blocked."
        return True, ""

# Singleton para chequeos
invariants = SystemInvariants()
