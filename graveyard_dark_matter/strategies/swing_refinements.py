import logging

logger = logging.getLogger("SWING-Refinements")

class SwingRefinements:
    """
    AXIOMA: MÓDULO PREDICCIÓN - MEJORAS ESPECÍFICAS PARA SWING
    """
    
    @staticmethod
    def on_chain_integration_score(sopr: float, netflow: float, mvrv: float, direction: str) -> int:
        """
        MEJORA SWING-2: INTEGRACIÓN ON-CHAIN PROFUNDA
        Ajustes al THS de posiciones SWING.
        """
        score_adj = 0
        if direction == 'LONG':
            if sopr > 1.0:
                score_adj += 10
            if netflow < 0:
                score_adj += 8
            if mvrv < 1.5:
                # Mercado en zona de valor, fuerte sesgo largo
                score_adj += 20
        elif direction == 'SHORT':
            if mvrv < 1.5:
                # NO shortear en zona de valor
                score_adj -= 30
                
        return score_adj
        
    @staticmethod
    def funding_rate_adjustment(funding_percentile: float, direction: str) -> int:
        """
        MEJORA SWING-3: FUNDING RATE COMO PREDICTOR DE REVERSIÓN
        """
        if funding_percentile > 0.90:
            return -20 if direction == 'LONG' else 15
        elif funding_percentile < 0.10:
            return -20 if direction == 'SHORT' else 15
        return 0
        
    @staticmethod
    def wyckoff_detector_phase(bc: bool, ar: bool, st: bool, lps: bool, spring: bool) -> str:
        """
        MEJORA SWING-4: WYCKOFF DETECTOR AUTOMÁTICO
        Algoritmo para identificar automáticamente las fases Wyckoff.
        """
        if bc and ar and st and lps and spring:
            return 'ACUMULACION'
        return 'UNKNOWN'
