"""
🛡️ AEGIS-ULTRA: Maximum Adverse Excursion (MAE) Tracker
QUÉ: Tracks how far price moves AGAINST a trade before winning.
POR QUÉ: To tighten Stop Losses. If we always win with <0.5% drawdown, why risk 2%?
PARA QUÉ: Dynamic Stop Loss Optimization.
"""

import numpy as np
import pandas as pd
from utils.fast_json import FastJson as json
import os
from datetime import datetime

class MAETracker:
    def __init__(self, filepath="data/mae_stats.json"):
        self.filepath = filepath
        self.stats = self._load()
        
    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
        
    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"❌ MAE Save Error: {e}")

    def record_trade(self, symbol: str, direction: str, entry_price: float, 
                    exit_price: float, min_price: float, max_price: float):
        """
        Records the MAE/MFE for a closed trade.
        MAE = Worst price during trade vs Entry.
        MFE = Best price during trade vs Entry.
        """
        mae = 0.0
        mfe = 0.0
        
        if direction == 'LONG':
            # MAE: How low did it go below entry?
            mae_pct = (min_price - entry_price) / entry_price
            # MFE: How high did it go above entry?
            mfe_pct = (max_price - entry_price) / entry_price
        else: # SHORT
            # MAE: How high did it go above entry?
            mae_pct = (entry_price - max_price) / entry_price
            # MFE: How low did it go below entry?
            mfe_pct = (entry_price - min_price) / entry_price
            
        # MAE is typically negative (drawdown), but we store magnitude for "Excursion"
        # Standard def: MAE is positive distance from entry.
        # But for pct, usually negative means loss.
        # Let's track ABSOLUTE adverse excursion
        abs_mae = abs(min(0, mae_pct)) # Only count if it went against
        abs_mfe = max(0, mfe_pct)
        
        if symbol not in self.stats:
            self.stats[symbol] = {'count': 0, 'sum_mae': 0.0, 'sum_mfe': 0.0, 'max_mae': 0.0}
            
        s = self.stats[symbol]
        s['count'] += 1
        s['sum_mae'] += abs_mae
        s['sum_mfe'] += abs_mfe
        s['max_mae'] = max(s['max_mae'], abs_mae)
        
        self._save()
        
    def get_optimal_stop_loss(self, symbol: str, safety_factor: float = 1.5) -> float:
        """
        Returns suggested SL % based on historic MAE.
        Rule: SL should be slightly outside Average MAE.
        """
        if symbol not in self.stats or self.stats[symbol]['count'] < 5:
            return 0.02 # Default 2%
            
        s = self.stats[symbol]
        avg_mae = s['sum_mae'] / s['count']
        
        # Suggested SL = Avg MAE * 1.5
        suggested = avg_mae * safety_factor
        
        # Bounds
        return max(0.005, min(suggested, 0.05))

mae_tracker = MAETracker()
