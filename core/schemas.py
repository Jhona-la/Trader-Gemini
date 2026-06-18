import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass(frozen=True, slots=True)
class SophiaMetadata:
    win_probability: float
    confidence_score: float
    expected_drawdown: float

@dataclass(frozen=True, slots=True)
class SignalEvent:
    symbol: str
    signal_type: str
    horizon: str
    strength: float
    timestamp_ms: int
    sophia: Optional[SophiaMetadata] = None
    # Campo de escape termodinámico (si algo absolutamente necesita metadatos crudos)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TickEvent:
    symbol: str
    timestamp_ms: int
    bid_price: float
    ask_price: float
    bid_vol: float
    ask_vol: float
    trade_vol: float
    
    # Dimensiones de Córtex 10D (Features) pre-calculadas si no usa Rust Directo
    normalized_return: float = 0.0
    amihud_liquidity: float = 0.0
    order_imbalance: float = 0.0
    vpin: float = 0.0
    dark_alpha_vector: float = 0.0
    hurst_exponent: float = 0.0
    funding_elasticity: float = 0.0
    sophia_flow: float = 0.0
    portfolio_heat: float = 0.0
    ob_acceleration: float = 0.0
    
    def __post_init__(self):
        # Validación Fronteriza: El Punto de Colapso
        if self.bid_price <= 0 or self.ask_price <= 0:
            raise ValueError(f"Frontera Rígida: Precio nulo o negativo detectado en {self.symbol}")
            
    def to_cortex_tensor(self) -> np.ndarray:
        """
        PUENTE DE CONVERSIÓN 10D (O(1))
        Al estar el objeto congelado y validado, la extracción al tensor
        float32 es determinista y mapea la sinapsis exacta.
        """
        return np.array([
            self.normalized_return, # Dim 0
            self.amihud_liquidity,  # Dim 1
            self.order_imbalance,   # Dim 2
            self.vpin,              # Dim 3
            self.dark_alpha_vector, # Dim 4
            self.hurst_exponent,    # Dim 5
            self.funding_elasticity,# Dim 6
            self.sophia_flow,       # Dim 7
            self.portfolio_heat,    # Dim 8
            self.ob_acceleration    # Dim 9 (Nueva Física)
        ], dtype=np.float32)
