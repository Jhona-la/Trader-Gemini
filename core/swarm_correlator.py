"""
🧬 SOPHIA §6.1: Swarm Correlator (Quantum Entanglement Fabric)

QUÉ: Calculador de correlación dinámica entre el líder del mercado y el enjambre.
POR QUÉ: Para que el bot entienda que las monedas no son universos aislados.
PARA QUÉ: Adaptación paramétrica proactiva basada en la interconexión del mercado.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from utils.logger import logger

class SwarmCorrelator:
    """
    🐝 SWARM ENTANGLEMENT ENGINE.
    Mide cómo cada moneda "baila" con el líder (BTC).
    """
    
    def __init__(self, leader_symbol: str = "BTC/USDT", window: int = 50):
        self.leader_symbol = leader_symbol
        self.window = window
        self.correlations = {} # symbol -> correlation_coefficient
        self.leader_returns = None
        
    def update_leader_data(self, btc_bars_1m):
        """Actualiza los retornos del líder para el cálculo de Pearson."""
        if len(btc_bars_1m) < self.window:
            return
        
        prices = btc_bars_1m['close'].astype(np.float64)
        returns = np.diff(prices) / prices[:-1]
        self.leader_returns = returns[-self.window:]

    def calculate_entanglement(self, symbol: str, symbol_bars_1m) -> float:
        """
        Calcula la correlación de Pearson entre el símbolo y el líder.
        Retorna el coeficiente (-1 a 1).
        """
        if self.leader_returns is None or len(symbol_bars_1m) < self.window:
            return 0.0
            
        try:
            prices = symbol_bars_1m['close'].astype(np.float64)
            returns = np.diff(prices) / prices[:-1]
            target_returns = returns[-self.window:]
            
            if len(target_returns) < self.window:
                return 0.0
                
            # Pearson Correlation
            corr = np.corrcoef(self.leader_returns, target_returns)[0, 1]
            if np.isnan(corr): corr = 0.0
            
            self.correlations[symbol] = corr
            return corr
            
        except Exception as e:
            logger.debug(f"🐝 [SWARM] Correlation error for {symbol}: {e}")
            return 0.0

    def get_swarm_pressure(self, symbol: str) -> float:
        """
        Determina la presión evolutiva del enjambre.
        1.0 = Total dependencia de BTC (Sigue al líder).
        0.0 = Autonomía absoluta (Busca su propio universo).
        """
        corr = self.correlations.get(symbol, 0.0)
        # Usamos el valor absoluto porque correlación inversa (-1) también es dependencia
        return abs(corr)

    def get_swarm_influence(self, symbol: str) -> float:
        """
        Retorna la fuerza de la influencia del líder sobre este símbolo.
        Útil para modular la agresividad de los SL/TP.
        """
        corr = self.correlations.get(symbol, 0.0)
        # Si la correlación es muy alta (>0.85), el enjambre está "entrelazado"
        return abs(corr)

swarm_correlator = SwarmCorrelator()
