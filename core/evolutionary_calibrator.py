import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

class EvolutionaryCalibrator:
    """
    AutoCalibrador Evolutivo (Fase 2).
    Ajusta dinámicamente los umbrales de entrada (hurdles) basándose en el
    Win Rate reciente de un horizonte específico.
    Si el Win Rate cae, el calibrador se vuelve más estricto (exige setups más perfectos).
    """
    
    def __init__(self, window_size: int = 5):
        # Mantiene historial de resultados por (symbol, horizon)
        self.history: Dict[str, List[float]] = {}
        self.window_size = window_size
        
        # Minimum hurdle multiplier
        self.min_multiplier = 0.5   # Relaxed (Alpha state)
        self.max_multiplier = 3.0   # Extremely strict (Injured state)
        
    def _get_key(self, symbol: str, horizon: str) -> str:
        return f"{symbol}_{horizon.upper()}"
        
    def register_trade_outcome(self, symbol: str, horizon: str, pnl: float):
        """Registra el resultado de un trade para ajustar el calibrador."""
        key = self._get_key(symbol, horizon)
        if key not in self.history:
            self.history[key] = []
            
        self.history[key].append(pnl)
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)
            
        # Log estado actual
        wr = self.get_win_rate(symbol, horizon)
        logger.debug(f"🧬 [EVOL_CALIBRATOR] {key} - Trade registrado: PnL={pnl:.4f} | Recent WR: {wr*100:.1f}%")
        
    def get_win_rate(self, symbol: str, horizon: str) -> float:
        """Calcula el Win Rate reciente."""
        key = self._get_key(symbol, horizon)
        if key not in self.history or len(self.history[key]) == 0:
            return 1.0 # Optimismo inicial por defecto
            
        wins = sum(1 for p in self.history[key] if p > 0)
        return wins / len(self.history[key])
        
    def get_hurdle_multiplier(self, symbol: str, horizon: str) -> float:
        """
        Calcula el multiplicador evolutivo para el hurdle base de entrada.
        Un Win Rate alto (< 1.0) reduce la fricción.
        Un Win Rate bajo aumenta la fricción severamente para detener la sangría.
        """
        key = self._get_key(symbol, horizon)
        
        # Si no hay suficiente historia, usamos un perfil neutro conservador.
        if key not in self.history or len(self.history[key]) < 3:
            return 1.0
            
        wr = self.get_win_rate(symbol, horizon)
        
        # Lógica de Adaptabilidad Evolutiva
        if horizon.upper() == "SCALPING":
            # SCALPING exige perfección absoluta (objetivo 100% WR)
            if wr == 1.0:
                multiplier = 0.8  # Premio: Relajar un poco (20%)
            elif wr >= 0.8:
                multiplier = 1.0  # Normal
            elif wr >= 0.5:
                multiplier = 1.5  # Restricción fuerte (50% más estricto)
            else:
                multiplier = 2.5  # Restricción extrema
        else:
            # SWING tolera menor WR debido a su ratio R:B asimétrico
            if wr >= 0.6:
                multiplier = 0.8
            elif wr >= 0.4:
                multiplier = 1.0
            elif wr >= 0.2:
                multiplier = 1.3
            else:
                multiplier = 2.0
                
        return np.clip(multiplier, self.min_multiplier, self.max_multiplier)

# Singleton global para que todos los motores consulten el mismo registro de evolución
evolutionary_calibrator = EvolutionaryCalibrator(window_size=5)
