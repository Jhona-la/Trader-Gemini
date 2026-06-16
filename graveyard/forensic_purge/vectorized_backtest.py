import logging
import numpy as np
from numba import njit, prange

logger = logging.getLogger("VectorizedBacktester")

@njit(parallel=True, fastmath=True, cache=True)
def simulate_positions_numba(
    signals_matrix: np.ndarray, # Shape: (N_candles, N_estrategias)
    prices: np.ndarray,         # Shape: (N_candles,)
    tps: np.ndarray,            # Shape: (N_estrategias,)
    sls: np.ndarray,            # Shape: (N_estrategias,)
    fees_taker: float = 0.0004,
    fees_maker: float = 0.0002,
    slippage: float = 0.0002    # Spread + latencia (2 bps conservador)
) -> np.ndarray:
    """
    AXIOMA: BACKTEST VECTORIZADO
    Procesa 10,000+ velas x 30 estrategias en C puro.
    Retorna la Equity Curve Shape: (N_estrategias, N_candles)
    """
    n_candles = signals_matrix.shape[0]
    n_estrategias = signals_matrix.shape[1]
    
    # Matriz de equity: arranca en 1.0 (100%)
    equity_curves = np.ones((n_estrategias, n_candles), dtype=np.float64)
    
    # Evaluamos en paralelo CADA ESTRATEGIA (aislada)
    for s in prange(n_estrategias):
        in_position = False
        entry_price = 0.0
        current_equity = 1.0
        tp = tps[s]
        sl = sls[s]
        
        for c in range(n_candles):
            price = prices[c]
            signal = signals_matrix[c, s]
            
            if not in_position:
                if signal > 0: # Buy signal
                    in_position = True
                    # Aplicar Slippage/Spread al precio de entrada (peor ejecución)
                    entry_price = price * (1.0 + slippage)
                    current_equity -= current_equity * fees_taker # Entry fee
            else:
                # Check target / stop
                price_change = (price - entry_price) / entry_price
                
                if price_change >= tp or price_change <= -sl:
                    in_position = False
                    # Close fee
                    current_equity += current_equity * price_change
                    current_equity -= current_equity * fees_maker # Limit close
                elif c == n_candles - 1: # Force close at end
                    in_position = False
                    current_equity += current_equity * price_change
                    current_equity -= current_equity * fees_taker
                    
            equity_curves[s, c] = current_equity
            
    return equity_curves

class VectorizedBacktester:
    def __init__(self):
        self.results = {}
        
    def run_sweep(self, signals, prices, tps, sls):
        logger.info(f"Running vectorized sweep: {signals.shape[0]} candles, {signals.shape[1]} configs...")
        curves = simulate_positions_numba(signals, prices, tps, sls)
        return curves
