import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class MIDD:
    """
    Motor de Inteligencia Direccional Dual (MIDD).
    Calcula simultáneamente el Score Largo (SL) y el Score Corto (SC) para cada activo,
    y determina el Mapa de Estado Direccional (MED).
    """
    def __init__(self, data_provider=None, ml_strategy=None):
        self.data_provider = data_provider
        self.ml_strategy = ml_strategy

    def evaluate_asset(self, symbol: str) -> Dict[str, Any]:
        """
        Evalúa un activo de forma bidireccional.
        """
        sl = self._calculate_sl(symbol)
        sc = self._calculate_sc(symbol)
        isn = sl - sc
        med_state = self._determine_med_state(sl, sc, isn)

        return {
            "symbol": symbol,
            "SL": sl,
            "SC": sc,
            "ISN": isn,
            "MED_STATE": med_state,
        }

    def _calculate_sl(self, symbol: str) -> float:
        """
        Calcula el Score Largo (SL). Max 100 puntos.
        C1: Estructura alcista (25)
        C2: CVD alcista (20)
        C3: ML Signal Long (20)
        C4: Funding Long (15)
        C5: Short Liqs (10)
        C6: Multi-TF Confluence (10)
        """
        score = 0.0
        
        # Mock/Extracted logic (To be wired with actual data_provider metrics)
        # Assuming we fetch the latest features
        if self.data_provider:
            # Structure (mocked for now, needs real price structure logic)
            score += 15.0  
            
            # Funding
            funding = self.data_provider.get_funding_rate(symbol) if hasattr(self.data_provider, 'get_funding_rate') else 0.0
            if funding < -0.0002: # Extremo negativo: Excelente para Longs
                score += 15.0
            elif funding < 0:
                score += 10.0
            else:
                score += 5.0

            # ML Signal
            if self.ml_strategy:
                # We would call the ML model L here
                pass
                
        return min(score + 40.0, 100.0) # Base score for now until wired

    def _calculate_sc(self, symbol: str) -> float:
        """
        Calcula el Score Corto (SC). Max 100 puntos.
        """
        score = 0.0
        
        if self.data_provider:
            # Structure
            score += 15.0  
            
            # Funding
            funding = self.data_provider.get_funding_rate(symbol) if hasattr(self.data_provider, 'get_funding_rate') else 0.0
            if funding > 0.0005: # Extremo positivo: Excelente para Shorts
                score += 15.0
            elif funding > 0:
                score += 10.0
            else:
                score += 5.0
                
        return min(score + 40.0, 100.0) # Base score for now until wired

    def _determine_med_state(self, sl: float, sc: float, isn: float) -> str:
        """
        Determina el Estado MED basado en ISN y SL/SC.
        MED-1: Sesgo Largo Dominante (ISN > 30)
        MED-2: Sesgo Corto Dominante (ISN < -30)
        MED-3: Equilibrio Sesgo Alcista (ISN +10 a +30)
        MED-4: Equilibrio Sesgo Bajista (ISN -10 a -30)
        MED-5: Transición (detectado por flip rápido, por ahora simplificado)
        MED-6: Alta Incertidumbre (ISN cercano a 0 pero SL y SC muy altos)
        """
        if sl > 70 and sc > 70 and abs(isn) < 10:
            return "MED-6" # Alta energía bidireccional = Incertidumbre/Volatilidad cruzada
        
        if isn > 30:
            return "MED-1"
        elif 10 < isn <= 30:
            return "MED-3"
        elif -10 <= isn <= 10:
            # Podría ser MED-5 o simplemente neutral
            return "MED-NEUTRAL"
        elif -30 <= isn < -10:
            return "MED-4"
        elif isn < -30:
            return "MED-2"
            
        return "MED-NEUTRAL"
