"""
👁️ PROTOCOLO CRITERIO-AXIOMA: The Oracle & Root Cause Engine
=============================================================

QUÉ: Motor de diagnóstico forense de trades fallidos.
POR QUÉ: Saber que perdimos dinero no sirve para aprender; la red necesita aislar
      si el entorno cambió (Tesis), si no hubo liquidez (Profundidad) o si 
      hubo un bug en el engine (Cálculo).
PARA QUÉ: Dirigir el castigo del algoritmo genético exactamente al problema real.
CÓMO: Mide residuales de predicción y desviaciones macro.
DÓNDE: `sophia/axioma.py`
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum
import math
from utils.logger import logger
from utils.axioma_math import PrecisionAuditor

class FallaBase(Enum):
    CALCULO = "CALCULO"        # Bugs, Precision loss, Fatal Slippage
    PROFUNDIDAD = "PROFUNDIDAD"# Latency, Spikes, Spread Widenings
    TESIS_DECAY = "TESIS_DECAY"# Modelo falló (Z-Score aplastado por tendencia, Alpha Decay)
    NO_FALLA = "NO_FALLA"      # Trade Winner (Premio aplicable)

@dataclass
class AxiomDiagnosis:
    tipo_falla: FallaBase
    residual_pct: float             # Target price vs Realized price (Deviation %)
    accion_recomendada: str
    razon: str
    is_fatal: bool = False
    
class AxiomDiagnoser:
    """
    El Oráculo Forense: Identifica por qué perdimos.
    Solo se debe llamar al finalizar un trade desde NemesisEngine.
    """
    
    # Tolerancias
    MAX_SLIPPAGE_PCT = 0.005 # 0.5% (Fatal para Scalping)
    
    @classmethod
    def diagnose(cls, 
                 pnl: float, 
                 direction: str, 
                 trigger_price: float, 
                 fill_price: float,
                 sophia_report: Dict[str, Any],
                 duration_mins: float) -> AxiomDiagnosis:
        """
        Interroga el trade muerto y emite el veredicto para el algoritmo Genético/Engine.
        """
        # 1. Base Trivial (Ganamos)
        if pnl > 0:
            return AxiomDiagnosis(
                tipo_falla=FallaBase.NO_FALLA,
                residual_pct=0.0,
                accion_recomendada="PREMIAR_GENOTIPO",
                razon="Trade Winner. Tesis Cumplida."
            )
            
        # --- A partir de aquí: ROOT CAUSE ANALYSIS (Perdimos) ---
        
        # 2. Análisis de Residuales (Diferencia de Precio Ejecutado vs Buscado)
        # Slippage Analysis
        if trigger_price and fill_price and trigger_price > 0:
            slippage = abs(fill_price - trigger_price) / trigger_price
            
            # Sub-Root 1: Cálculo / Latency Extrema (Slippage Devastador)
            # FORENSIC-2: Dynamic Slippage Threshold scaling
            # Scale the maximum allowed slippage relative to the expected duration bounds.
            # Short durations (< 5 mins) need tighter slippage tolerance (e.g. ~0.15%)
            # Swing durations (> 60 mins) can tolerate wider natural slippage gaps (e.g. ~0.8%)
            dynamic_slippage_max = min(0.015, max(0.0015, 0.0015 + (duration_mins / 120.0) * 0.005))
            
            if slippage > dynamic_slippage_max:
                return AxiomDiagnosis(
                    tipo_falla=FallaBase.CALCULO,
                    residual_pct=slippage,
                    accion_recomendada="SAFE_MODE_VERIFY_EXECUTION",
                    razon=f"Slippage Fatal detectado ({slippage*100:.3f}% vs Límite dinámico {dynamic_slippage_max*100:.3f}%). Posible Bug de red o motor lógico.",
                    is_fatal=True
                )
                
            # Sub-Root 2: Profundidad (Mala Liquidez)
            # Para scalping, si la duración fue ultra-corta (< 1 minuto) y hubo stop-loss, 
            # asume Spike the liquidez/Flash crash (Profundidad).
            if duration_mins < 1.0:
                 return AxiomDiagnosis(
                    tipo_falla=FallaBase.PROFUNDIDAD,
                    residual_pct=slippage,
                    accion_recomendada="AUMENTAR_FILTRO_LIQUIDEZ",
                    razon="Stop loss hit en < 60s. Shock de microestructura detectado (Spike)."
                )

        # 3. Sub-Root 3: Fallo de Tesis (Alpha Decay / Regime Change)
        # El mercado absorbió la señal Z-Score direccional.
        # Recuperamos la entropía de la predicción fallida.
        entropy = sophia_report.get('decision_entropy', 0.5)
        
        # O calcular residual de price action (si el precio esperado de TP era X, y cerramos en SL Y)
        # Approximamos residual por distance
        
        razon = "El mercado invalidó el modelo cuantitativo para las condiciones macro actuales."
        if entropy > 0.4:
             razon += " (La red ya tenía Alta Entropía/Dudas al entrar)."

        return AxiomDiagnosis(
            tipo_falla=FallaBase.TESIS_DECAY,
            residual_pct=-1.0, # Target failed
            accion_recomendada="CASTIGAR_GENOTIPO",
            razon=razon
        )
