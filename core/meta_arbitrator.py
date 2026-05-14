"""
meta_arbitrator.py — REDIRECT MODULE (CTOS Migration)
═══════════════════════════════════════════════════════════════
QUÉ: Este módulo ahora redirige al MetaCoordinator unificado.
POR QUÉ: La lógica de MetaArbitrator fue absorbida por MetaCoordinator
  como parte de la migración CTOS. Mantener dos cerebros causaba
  inconsistencias en la gobernanza de señales.
PARA QUÉ: Backward-compatibility. Todos los imports existentes
  (engine.py L292, L346, L523, L763, L773) siguen funcionando.
CÓMO: Re-exporta el singleton meta_arbitrator desde meta_coordinator.py.
CUÁNDO: Siempre activo.
DÓNDE: core/meta_arbitrator.py (este archivo)
QUIÉN: Arquitecto Senior (Phase CTOS Brain Fusion)
═══════════════════════════════════════════════════════════════

ORIGINAL: Backed up to core/meta_arbitrator.BAK
"""

# Single redirect — all logic lives in meta_coordinator.py now
from core.meta_coordinator import meta_coordinator as meta_arbitrator

# Re-export the class for type-checking
from core.meta_coordinator import MetaCoordinator as MetaArbitrator

__all__ = ['meta_arbitrator', 'MetaArbitrator']
