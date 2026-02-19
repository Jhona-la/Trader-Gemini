# import json (Removed Phase 3: Supreme Efficiency)
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

@dataclass
class Genotype:
    """
    Representa el ADN de una estrategia para un símbolo específico.
    Contiene todos los parámetros optimizables por el Algoritmo Genético.
    """
    symbol: str
    generation: int = 0
    fitness_score: float = 0.0
    parent_id: Optional[str] = None
    
    # Genes (Parámetros de Estrategia)
    genes: Dict[str, Any] = field(default_factory=lambda: {
        # Technical Indicators
        "bollinger_period": 20,
        "bollinger_std": 2.0,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "adx_threshold": 25,
        "strength_threshold": 0.6,
        
        # Risk Management (Institutional Optimization)
        "tp_pct": 0.015,
        "sl_pct": 0.02,
        "atr_sl_multiplier": 2.0,
        "trend_ema_period": 200,
        "trailing_activation_rsi": 65,
        
        # Weights (Neural/Hybrid)
        "weight_trend": 0.4,
        "weight_momentum": 0.4,
        "weight_volatility": 0.2,
        
        # Neural Bridge Weights (Phase 33)
        # Input Layer (25) -> Output Layer (4 Actions)
        # Flattened array: 25 * 4 = 100 weights
        # We start with None or random, but for JSON serializability we use list
        "brain_weights": [] 
    })

    def init_brain(self, input_size: int, output_size: int):
        """Initialize random neural weights if empty"""
        if not self.genes.get('brain_weights'):
            import numpy as np
            # Xavier Initialization-like range
            limit = np.sqrt(6 / (input_size + output_size))
            weights = np.random.uniform(-limit, limit, size=input_size * output_size)
            self.genes['brain_weights'] = weights.tolist()


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Genotype':
        return cls(**data)

    def save(self, filepath: str):
        """Persiste el genoma a disco (orjson High-Perf)"""
        from utils.fast_json import FastJson
        FastJson.dump_to_file(self.to_dict(), filepath)

    @classmethod
    def load(cls, filepath: str) -> 'Genotype':
        """Carga el genoma desde disco (orjson High-Perf)"""
        if not os.path.exists(filepath):
            return None
        
        from utils.fast_json import FastJson
        data = FastJson.load_from_file(filepath)
        if not data:
            return None
        return cls.from_dict(data)
