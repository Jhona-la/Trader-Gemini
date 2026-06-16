import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from core.genotype import Genotype
from core.evolution import TradeResult

class SimDataProvider:
    """
    Proveedor de datos optimizado para simulación (Zero-Copy).
    Carga datos en arrays de NumPy y provee vistas rápidas.
    """
    def __init__(self, data: Dict[str, pd.DataFrame]):
        # Pre-process data into structured arrays for max speed
        self.arrays = {}
        for symbol, df in data.items():
            # Ensure safe types
            rec = df.to_records(index=True)
            # Rename index if needed (assumes 'timestamp' or index name)
            # We assume df has 'open', 'high', 'low', 'close', 'volume'
            self.arrays[symbol] = rec

    def get_window(self, symbol: str, end_idx: int, window_size: int) -> np.ndarray:
        """Retorna vista window_size hasta end_idx (exclusive)"""
        # Fast slicing on numpy array
        start = max(0, end_idx - window_size)
        return self.arrays[symbol][start:end_idx]

class SimulationEngine:
    """
    Motor de Simulacion 'Bare Metal' (Trinidad Omega - Phase 18).
    Ejecuta una estrategia sobre datos históricos lo más rápido posible.
    """
    def __init__(self, data_provider: SimDataProvider):
        self.data = data_provider
        
    def run(self, genotype: Genotype, symbol: str, start_idx: int = 0, end_idx: int = None, max_candles: int = None) -> List[TradeResult]:
        """
        Ejecuta el genotipo sobre los datos del símbolo.
        Retorna lista de TradeResults.
        Supports Start/End Index for Walk-Forward Analysis (Phase 3).
        Supports max_candles para Multi-Fidelidad F1-F4.
        """
        if symbol not in self.data.arrays:
            return []
            
        market_data = self.data.arrays[symbol]
        genes = genotype.genes
        
        # Unpack Genes
        tp_pct = genes.get('tp_pct', 0.015)
        sl_pct = genes.get('sl_pct', 0.02)
        
        # Check for Brain
        brain_weights = genes.get('brain_weights', [])
        use_brain = len(brain_weights) > 0
        weights_matrix = None
        
        if use_brain:
            from core.neural_bridge import neural_bridge
            # Reconstruct Matrix (Input 25 x Output 4)
            # We assume input dim is consistent with NeuralBridge
            input_dim = neural_bridge.input_dim # 25
            output_dim = 4
            if len(brain_weights) == input_dim * output_dim:
                weights_matrix = np.array(brain_weights).reshape(input_dim, output_dim)
            else:
                use_brain = False # Fallback if shape mismatch
        
        trades: List[TradeResult] = []
        position = None # None, 'LONG', 'SHORT'
        entry_price = 0.0
        entry_idx = 0
        
        closes = market_data['close']
        timestamps = market_data['timestamp'] # Assuming it exists
        
        # Define Simulation Range
        total_bars = len(market_data)
        actual_end = total_bars if end_idx is None else min(end_idx, total_bars)
        actual_start = max(50, start_idx) # Force warmup override
        
        if max_candles is not None:
            actual_end = min(actual_end, actual_start + max_candles)
        
        if actual_start >= actual_end:
            return []
        
        if not use_brain:
            # --- BARE METAL NUMBA EXECUTION ---
            from core.simulation_numba import technical_simulation_loop_njit
            
            window = int(genes.get('rsi_window', 14))
            fast_window = int(genes.get('macd_fast', 8))
            trend_conf = float(genes.get('trend_confirmation_threshold', 0.001))
            
            closes_arr = np.ascontiguousarray(closes, dtype=np.float64)
            
            trades_arr = technical_simulation_loop_njit(
                closes_arr,
                float(tp_pct),
                float(sl_pct),
                window,
                fast_window,
                trend_conf,
                actual_start,
                actual_end
            )
            
            for i in range(trades_arr.shape[0]):
                trades.append(TradeResult(trades_arr[i, 0], trades_arr[i, 1], bool(trades_arr[i, 2])))
                
            return trades

        # --- NEURAL EXECUTION LOOP (VECTORIZADO CON NUMBA) ---
        from core.simulation_numba import extract_features_njit, neural_feedforward_njit, execution_loop_njit
        
        closes_arr = np.ascontiguousarray(closes, dtype=np.float64)
        
        # 1. Extraer características para todas las velas (vectorizado 25D)
        from core.neural_bridge import neural_bridge
        window = neural_bridge.window
        features = extract_features_njit(closes_arr, window)
        
        # 2. Feed Forward Matricial (Vectorizado)
        # Weights matrix shape [25, 4] -> output [N, 4]
        probs = neural_feedforward_njit(features, weights_matrix)
        
        # 3. Hot-Loop de Ejecución en C/LLVM (Cero objetos Python)
        trades_arr = execution_loop_njit(
            probs,
            closes_arr,
            float(sl_pct),
            float(tp_pct),
            actual_start,
            actual_end
        )
        
        # Reconstruir TradeResults para la capa de análisis (Ocurre 1 vez al final)
        for i in range(trades_arr.shape[0]):
            trades.append(TradeResult(trades_arr[i, 0], trades_arr[i, 1], bool(trades_arr[i, 2])))
            
        return trades

    # --- HELPER: Fast RSI (Numpy) ---
    @staticmethod
    def calculate_rsi_numpy(closes, period=14):
        # Placeholder for vector logic
        pass
