import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Validación Walk-Forward con Purge y Embargo (Parte VI del Prompt Supremo).
    Unidad de tiempo: Ciclos de 3 días.
    IS: 70%
    Purge: 2 ciclos
    Embargo: 1 ciclo
    OOS: 30%
    """
    
    CYCLE_DAYS = 3
    PURGE_CYCLES = 2
    EMBARGO_CYCLES = 1
    
    def __init__(self, data_size_days: int):
        self.data_size_days = data_size_days
        
    def generate_splits(self) -> Dict[str, Any]:
        """
        Genera los índices de split basados en días/ciclos.
        """
        total_cycles = self.data_size_days // self.CYCLE_DAYS
        
        # Necesitamos un mínimo absoluto de ciclos para que esto funcione (ej. 10 ciclos = 30 días)
        if total_cycles < 10:
            logger.warning("Pocos datos para Walk-Forward CV. Se requiere un dataset más largo.")
            
        is_cycles = int(total_cycles * 0.70)
        oos_cycles = total_cycles - is_cycles - self.PURGE_CYCLES - self.EMBARGO_CYCLES
        
        return {
            "is_start": 0,
            "is_end": is_cycles * self.CYCLE_DAYS,
            "purge_start": is_cycles * self.CYCLE_DAYS,
            "purge_end": (is_cycles + self.PURGE_CYCLES) * self.CYCLE_DAYS,
            "embargo_start": (is_cycles + self.PURGE_CYCLES) * self.CYCLE_DAYS,
            "embargo_end": (is_cycles + self.PURGE_CYCLES + self.EMBARGO_CYCLES) * self.CYCLE_DAYS,
            "oos_start": (is_cycles + self.PURGE_CYCLES + self.EMBARGO_CYCLES) * self.CYCLE_DAYS,
            "oos_end": total_cycles * self.CYCLE_DAYS
        }

    def validate_degradation(self, is_score: float, oos_score: float) -> bool:
        """
        Verifica que OOS no degrade más de un 30% respecto a IS.
        """
        if is_score <= 0:
            return False # Falló IS, no es válido de todos modos
            
        # degradation formula = (IS - OOS) / IS
        degradation = (is_score - oos_score) / is_score
        
        if degradation > 0.30:
            logger.warning(f"❌ [WALK-FORWARD] Rechazado por degradación OOS ({degradation*100:.2f}% > 30%)")
            return False
            
        logger.info(f"✅ [WALK-FORWARD] Degradación aceptable: {degradation*100:.2f}%")
        return True
