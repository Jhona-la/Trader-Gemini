
import numpy as np
import pandas as pd
from utils.logger import setup_logger
from utils.math_kernel import calculate_correlation_matrix_jit
from config import Config

logger = setup_logger("CorrelationManager")

class CorrelationManager:
    """
    🛡️ PHASE 13: DYNAMIC CORRELATION MANAGER
    Prevents portfolio concentration risk by monitoring asset correlations.
    """
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.correlation_matrix = None
        self.symbols = []
        self.MAX_CORRELATION = 0.85 # Threshold for blocking new trades
        self.last_update = 0
        
    def update_correlations(self):
        """
        Re-calculates the correlation matrix for all watched symbols.
        Expensive operation, should be throttled (e.g., every 5-15 mins).
        """
        try:
            # Gather history for all symbols
            # We need synchronized timestamps.
            # Strategy: Get DataFrame of closes for all symbols.
            
            # Since data_provider has ring buffers, we can extract them.
            # Ideally we want 1h close data for general correlation, 
            # or 5m data if we are scalping heavily. 
            # Let's use 5m data for short-term correlation relevance.
            
            df_dict = {}
            target_len = None
            
            # Collect 5m closes
            for symbol in self.data_provider.symbol_list:
                df = self.data_provider.get_latest_bars(symbol, n=100, timeframe='5m')
                if not df.empty:
                    # We assume roughly synced if real-time. 
                    # For exact sync, we'd need merge_asof, but that's slow.
                    # Fast approx: Take common length suffix.
                    if target_len is None: target_len = len(df)
                    target_len = min(target_len, len(df))
                    df_dict[symbol] = df['close'].values # Numpy array
            
            if not df_dict or target_len < 30:
                return # Not enough data
            
            # Build Matrix (N_samples, M_assets)
            self.symbols = list(df_dict.keys())
            m_assets = len(self.symbols)
            n_samples = target_len
            
            price_matrix = np.zeros((n_samples, m_assets), dtype=np.float32)
            
            for idx, sym in enumerate(self.symbols):
                # Take last N samples
                price_matrix[:, idx] = df_dict[sym][-target_len:]
                
            # JIT Calculation
            self.correlation_matrix = calculate_correlation_matrix_jit(price_matrix)
            self.last_update = 0 # Updates handled externally usually, or via timestamp check
            
            logger.info(f"🧩 Correlation Matrix Updated ({m_assets}x{m_assets})")
            
        except Exception as e:
            logger.error(f"Correlation Update Failed: {e}")

    def check_correlation_risk(self, new_symbol, current_positions):
        """
        Checks if adding 'new_symbol' violates correlation limits with EXISTING positions.
        Returns: (True/False, Reason)
        """
        if self.correlation_matrix is None or new_symbol not in self.symbols:
            return True, "NO_DATA" # Fail open (allow trade) or safe (deny)? Fail open + warning.
            
        new_idx = self.symbols.index(new_symbol)
        
        # Check against every currently held position
        for pos_symbol in current_positions:
            if pos_symbol == new_symbol: continue # Adding to same pos is fine here (size limit handles that)
            if pos_symbol not in self.symbols: continue
            
            pos_idx = self.symbols.index(pos_symbol)
            corr = self.correlation_matrix[new_idx, pos_idx]
            
            if corr > self.MAX_CORRELATION:
                return False, f"HIGH_CORRELATION: {new_symbol} vs {pos_symbol} ({corr:.3f})"
                
        return True, "SAFE"
