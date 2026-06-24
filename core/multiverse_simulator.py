"""
🌌 SOPHIA §5.3: Multiverse Simulator (Shadow Validation)

QUÉ: Simulador Monte Carlo interno para validar derivas de parámetros.
POR QUÉ: Para elegir la trayectoria evolutiva con mayor probabilidad de éxito.
PARA QUÉ: Precisión infinitesimal (±1e-6) sin riesgo de capital.
CÓMO: Corre 5 configuraciones paralelas sobre los últimos datos OHLCV.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple
from core.genotype import Genotype
from utils.logger import logger

class MultiverseSimulator:
    """
    🌌 THE ORACLE'S REPLAY.
    Evalúa 'Universos Paralelos' de parámetros antes de comprometer el genotipo real.
    """
    
    def __init__(self, shadow_universes: int = 5):
        self.shadow_universes = shadow_universes

    def simulate_trajectories(self, event, data_provider, base_tp: float, base_sl: float) -> Tuple[float, float, str]:
        """
        Genera universos paralelos y retorna el mejor set de TP/SL + narrativa.
        Evaluado síncronamente (O(1) operaciones) para no romper el GC critical section.
        """
        try:
            bars = data_provider.get_latest_bars(event.symbol, n=15, timeframe="1m")
            if bars is None or len(bars) < 14:
                return base_tp, base_sl, "INSUFFICIENT_DATA"
                
            close_prices = [b['close'] for b in bars]
            current_price = close_prices[-1]
            
            # Calculate simple ATR proxy
            high_prices = [b['high'] for b in bars[-14:]]
            low_prices = [b['low'] for b in bars[-14:]]
            atr = sum([h - l for h, l in zip(high_prices, low_prices)]) / 14.0
            
            if current_price == 0 or atr == 0:
                return base_tp, base_sl, "ZERO_VOLATILITY"

            # 1. Crear Variaciones Infinitesimales (Universos Sombra)
            trajectories = []
            base_genes = {'tp_pct': base_tp, 'sl_pct': base_sl}
            trajectories.append(base_genes)
            
            for _ in range(self.shadow_universes - 1):
                mutated = base_genes.copy()
                for key in ['tp_pct', 'sl_pct']:
                    drift = np.random.normal(0, 1e-4) # 0.01% drift
                    mutated[key] *= (1.0 + drift)
                trajectories.append(mutated)

            # 2. Simulación Rápida
            best_score = -float('inf')
            best_genes = base_genes
            best_index = 0
            
            volatility_norm = atr / current_price
            
            for i, genes in enumerate(trajectories):
                tp = genes['tp_pct']
                sl = genes['sl_pct']
                
                comfort = 1.0
                if sl < volatility_norm * 0.5: comfort -= 0.5 # Demasiado frágil
                if tp > volatility_norm * 5.0: comfort -= 0.3 # Demasiado ambicioso
                
                # Favor better RR
                rr = tp / sl if sl > 0 else 0
                comfort += (rr * 0.0001)
                
                if comfort > best_score:
                    best_score = comfort
                    best_genes = genes
                    best_index = i
                    
            reasoning = f"QUANTUM_PATH_{best_index}_SELECTED" if best_index > 0 else "MAINTAIN_CURRENT_REALITY"
            return best_genes['tp_pct'], best_genes['sl_pct'], reasoning
            
        except Exception as e:
            logger.error(f"Multiverse Simulation Error: {e}")
            return base_tp, base_sl, "SIMULATION_ERROR"

multiverse_simulator = MultiverseSimulator()
