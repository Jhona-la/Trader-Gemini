import numpy as np
from core.events import SignalEvent
from core.enums import SignalType

class NanoSignalBridge:
    """
    QUANTUM-NANO: O(1) Pre-computed Signal Lookup Table.
    Converts pre-calculated Numpy arrays of signals into runtime SignalEvents 
    with zero latency during the backtest loop.
    """
    def __init__(self, symbols, total_epochs):
        self.symbols = symbols
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        self.idx_to_symbol = {i: s for i, s in enumerate(symbols)}
        self.total_epochs = total_epochs
        
        # Dict of 2D arrays: strategy_id -> np.ndarray[symbol_idx, epoch_idx]
        self.signal_matrices = {}

    def register_signals(self, strategy_id: str, symbol: str, signal_array: np.ndarray):
        """
        Loads a pre-computed signal array (1D) into the lookup matrix.
        signal_array should contain 1 (LONG), -1 (SHORT), or 0 (NONE)
        """
        if strategy_id not in self.signal_matrices:
            self.signal_matrices[strategy_id] = np.zeros((len(self.symbols), self.total_epochs), dtype=np.int8)
            
        if symbol not in self.symbol_to_idx:
            return
            
        sym_idx = self.symbol_to_idx[symbol]
        length = min(len(signal_array), self.total_epochs)
        self.signal_matrices[strategy_id][sym_idx, :length] = signal_array[:length]

    def get_signals_for_epoch(self, epoch_idx: int, timestamp) -> list:
        """
        O(N_strats * N_active_symbols) extraction of SignalEvents for the current epoch.
        """
        events = []
        if epoch_idx >= self.total_epochs:
            return events
            
        for strat_id, matrix in self.signal_matrices.items():
            epoch_signals = matrix[:, epoch_idx]
            
            # Fast numpy filtering for non-zero signals
            active_indices = np.nonzero(epoch_signals)[0]
            
            for idx in active_indices:
                sig_val = epoch_signals[idx]
                sym = self.idx_to_symbol[idx]
                
                sig_type = SignalType.LONG if sig_val == 1 else SignalType.SHORT
                
                event = SignalEvent(
                    strategy_id=strat_id,
                    symbol=sym,
                    datetime=timestamp,
                    signal_type=sig_type,
                    strength=1.0, # Pre-computed signals assume max conviction
                    horizon="SCALPING",
                    priority=1
                )
                events.append(event)
                
        return events
