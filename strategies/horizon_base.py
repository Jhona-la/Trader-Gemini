from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class StrategyHorizon(ABC):
    """
    Clase base para Estrategias con consciencia de Horizonte (Dual-Horizon).
    Fuerza la categorización en 'scalping' (Micro) o 'swing' (Macro).
    """
    
    def __init__(self, horizon_type: str = "scalping"):
        if horizon_type not in ["scalping", "swing"]:
            raise ValueError(f"Horizonte inválido: {horizon_type}. Debe ser 'scalping' o 'swing'.")
        self.horizon_type = horizon_type

    @abstractmethod
    def evaluate(self, symbol: str, data: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
        """
        Evalúa la data del mercado y emite una señal.
        
        Returns:
            Tuple: (signal, confidence, metadata)
                - signal: 1 (Buy), -1 (Sell), 0 (Hold)
                - confidence: 0.0 a 1.0
                - metadata: Diccionario con detalles de la orden, INCLUYENDO 'horizon'
        """
        pass
        
    def _create_metadata(self, base_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Añade el horizonte a los metadatos de la señal"""
        meta = base_metadata or {}
        meta["horizon"] = self.horizon_type
        return meta
