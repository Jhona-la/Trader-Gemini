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
        
    def run(self, genotype: Genotype, symbol: str, start_idx: int = 0, end_idx: int = None) -> List[TradeResult]:
        """
        Ejecuta el genotipo sobre los datos del símbolo.
        Retorna lista de TradeResults.
        Supports Start/End Index for Walk-Forward Analysis (Phase 3).
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
        
        if actual_start >= actual_end:
            return []
        
        # Sim Loop
        for i in range(actual_start, actual_end):
            current_close = closes[i]
            
            # --- NEURAL EXECUTION ---
            if use_brain:
                from core.neural_bridge import neural_bridge
                # 1. Get State
                # Slice raw market data for window
                window_data = self.data.get_window(symbol, i+1, neural_bridge.window)
                
                # Mock Portfolio State
                port_state = {
                    'quantity': 1 if position else 0,
                    'pnl_pct': (current_close - entry_price)/entry_price if position == 'LONG' else (entry_price - current_close)/entry_price if position == 'SHORT' else 0.0,
                    'duration': i - entry_idx if position else 0
                }
                
                tensor = neural_bridge.get_state_tensor(window_data, port_state, genotype)
                
                # 2. Feed Forward (Simple Linear Layer for now)
                # Logits = Input @ Weights
                logits = np.dot(tensor, weights_matrix)
                
                # 3. Activation (Softmax)
                exp_logits = np.exp(logits - np.max(logits)) # Stable softmax
                probs = exp_logits / np.sum(exp_logits)
                
                # 4. Decode
                signal_type, conf = neural_bridge.decode_action(probs)
                
                # 5. Execute
                if position is None:
                    # ENTRY LOGIC
                    if signal_type and conf > 0.5: # Hard threshold for now
                        if isinstance(signal_type, str):
                            continue # Ignore string signals like "CLOSE" when flat
                            
                        if signal_type.name == 'LONG':
                            position = 'LONG'
                            entry_price = current_close
                            entry_idx = i
                        elif signal_type.name == 'SHORT':
                            position = 'SHORT'
                            entry_price = current_close
                            entry_idx = i
                
                else:
                    # EXIT LOGIC
                    # Check SL/TP first (Hard Risk Management)
                    pnl = (current_close - entry_price)/entry_price if position == 'LONG' else (entry_price - current_close)/entry_price
                    
                    if pnl <= -sl_pct or pnl >= tp_pct:
                        # Close Trade
                        trades.append(TradeResult(pnl, (i - entry_idx)*60, pnl > 0)) # Approx seconds
                        position = None
                        entry_idx = 0
                        continue # Trade done
                        
                    # Neural Exit
                    is_exit = False
                    if signal_type == "CLOSE":
                        is_exit = True
                    elif signal_type and position == 'LONG' and signal_type.name == 'SHORT':
                        is_exit = True
                    elif signal_type and position == 'SHORT' and signal_type.name == 'LONG':
                        is_exit = True

                    if is_exit:
                         trades.append(TradeResult(pnl, (i - entry_idx)*60, pnl > 0))
                         position = None
                         entry_idx = 0
            
            else:
                # Fallback / Hybrid Logic (Placeholder)
                pass
                
        return trades

    # --- HELPER: Fast RSI (Numpy) ---
    @staticmethod
    def calculate_rsi_numpy(closes, period=14):
        # Placeholder for vector logic
        pass
