import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone

@dataclass
class SymbolStateVector:
    """
    Representación vectorial (State Vector) de un símbolo en el mercado.
    Mantiene todas las características matemáticas, estructurales y topológicas
    necesarias para la toma de decisiones algorítmicas en el Grafo.
    """
    symbol: str
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    
    # --- Microestructura (Microstructure) ---
    orderflow_imbalance: float = 0.0      # +1.0 (Compradores agresivos) a -1.0 (Vendedores agresivos)
    spread_cost_pct: float = 0.0          # Costo relativo del spread
    liquidity_depth: float = 0.0          # Volumen agregado en +- 0.5% del BBO
    
    # --- Momentum & Volatilidad (Kinetics) ---
    trend_score_m5: float = 0.0           # Fuerza direccional (-1 a 1)
    hurst_exponent: float = 0.5           # >0.5 Persistente (Tendencia), <0.5 Anti-persistente (Reversión)
    volatility_atr_pct: float = 0.0       # ATR relativo al precio (%)
    
    # --- Macro & Sentimiento (Context) ---
    funding_rate_bias: float = 0.0        # Desviación del funding base (predictivo de squeezes)
    regime_hmm: str = "UNKNOWN"           # Régimen detectado (TRENDING, RANGING, VOLATILE)
    
    # --- Propiedades de Grafo (Topology) ---
    eigenvector_centrality: float = 0.0   # Influencia del símbolo en el mercado global
    contagion_pressure: float = 0.0       # Riesgo de arrastre por caídas de símbolos correlacionados
    cluster_id: int = -1                  # Identificador del sub-grafo de correlación (ej. Memecoins=1, L1=2)
    
    def as_array(self) -> np.ndarray:
        """Vector numérico puro para procesamiento rápido en ML y Matrices."""
        return np.array([
            self.orderflow_imbalance,
            self.spread_cost_pct,
            self.liquidity_depth,
            self.trend_score_m5,
            self.hurst_exponent,
            self.volatility_atr_pct,
            self.funding_rate_bias,
            self.eigenvector_centrality,
            self.contagion_pressure
        ], dtype=np.float32)

    def update_from_dict(self, data: Dict[str, float]):
        """Actualiza el tensor de estado desde un diccionario de features."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.timestamp = datetime.now(timezone.utc).timestamp()
