"""
Efficacy Tracker - Manual Close Analysis & RL Feedback (Phase 99)
================================================================
QUÉ: Compara el precio de cierre manual contra la predicción original del bot.
POR QUÉ: Medir si el humano hizo mejor o peor que el modelo automático.
PARA QUÉ: Retroalimentar al sistema de RL con datos de eficacia real.
CÓMO: Calcula efficacy_ratio = (manual_close - entry) / (tp - entry).
CUÁNDO: Se invoca cuando UserDataStreamListener detecta un cierre manual.
DÓNDE: utils/efficacy_tracker.py
QUIÉN: Invocado por data/user_stream.py → on_manual_close.
"""

import os
import time
import csv
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from utils.logger import logger


class EfficacyTracker:
    """
    Tracks the efficacy of manual closes versus bot predictions.
    Provides RL feedback based on the comparison.
    """
    
    def __init__(self, log_path: str = "dashboard/data/efficacy_log.csv"):
        self.log_path = log_path
        self._ensure_csv_header()
        self.records = []  # In-memory buffer of recent efficacy events
    
    def _ensure_csv_header(self):
        """Create CSV file with header if it doesn't exist."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'symbol', 'side', 'entry_price', 
                        'manual_close_price', 'tp_price', 'sl_price',
                        'efficacy_ratio', 'premature_exit_cost', 
                        'pnl_pct', 'strategy_id'
                    ])
            except Exception as e:
                logger.warning(f"Could not create efficacy log: {e}")
    
    def record_manual_close(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        manual_close_price: float,
        tp_price: float,
        sl_price: float,
        strategy_id: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Records a manual close event and calculates efficacy metrics.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            side: 'LONG' or 'SHORT'
            entry_price: Original entry price
            manual_close_price: Price at which human closed
            tp_price: Bot's predicted Take Profit price
            sl_price: Bot's Stop Loss price
            strategy_id: Which strategy opened the position
        
        Returns:
            Dict with efficacy metrics
        """
        # Calculate Efficacy Ratio
        # For LONG: ratio = (close - entry) / (tp - entry)
        # For SHORT: ratio = (entry - close) / (entry - tp)
        # 1.0 = human closed exactly at TP
        # 0.0 = human closed at entry (breakeven)
        # >1.0 = human exceeded TP prediction
        # <0.0 = human closed at a loss
        
        if side == "LONG":
            tp_distance = tp_price - entry_price
            actual_distance = manual_close_price - entry_price
            premature_exit_cost = tp_price - manual_close_price
            pnl_pct = ((manual_close_price - entry_price) / entry_price) * 100
        else:  # SHORT
            tp_distance = entry_price - tp_price
            actual_distance = entry_price - manual_close_price
            premature_exit_cost = manual_close_price - tp_price
            pnl_pct = ((entry_price - manual_close_price) / entry_price) * 100
        
        # Avoid division by zero
        if abs(tp_distance) > 0:
            efficacy_ratio = actual_distance / tp_distance
        else:
            efficacy_ratio = 0.0
        
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'manual_close_price': manual_close_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'efficacy_ratio': round(efficacy_ratio, 4),
            'premature_exit_cost': round(premature_exit_cost, 4),
            'pnl_pct': round(pnl_pct, 4),
            'strategy_id': strategy_id
        }
        
        # Log the event
        self._log_to_csv(result)
        self.records.append(result)
        
        # Human-readable summary
        emoji = "✅" if efficacy_ratio >= 0.8 else ("⚠️" if efficacy_ratio >= 0.3 else "❌")
        logger.info(
            f"{emoji} EFFICACY: {symbol} {side} | "
            f"Ratio: {efficacy_ratio:.2f} | "
            f"PnL: {pnl_pct:+.2f}% | "
            f"Left on Table: ${abs(premature_exit_cost):.4f}"
        )
        
        return result
    
    def get_rl_outcome(self, efficacy_ratio: float) -> float:
        """
        Converts efficacy ratio to an RL outcome signal.
        
        Maps to [0.0, 1.0] range for update_recursive_weights:
        - efficacy >= 0.8: outcome = 1.0 (human agreed with bot direction)  
        - efficacy in [0.3, 0.8): outcome = efficacy (proportional)
        - efficacy < 0.3: outcome = 0.0 (model was wrong or premature)
        """
        if efficacy_ratio >= 0.8:
            return 1.0
        elif efficacy_ratio >= 0.0:
            return max(0.0, min(1.0, efficacy_ratio))
        else:
            return 0.0
    
    def _log_to_csv(self, record: Dict[str, Any]):
        """Append a record to the CSV log."""
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    record['timestamp'], record['symbol'], record['side'],
                    record['entry_price'], record['manual_close_price'],
                    record['tp_price'], record['sl_price'],
                    record['efficacy_ratio'], record['premature_exit_cost'],
                    record['pnl_pct'], record['strategy_id']
                ])
        except Exception as e:
            logger.warning(f"Failed to write efficacy log: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate efficacy statistics."""
        if not self.records:
            return {'total_manual_closes': 0}
        
        ratios = [r['efficacy_ratio'] for r in self.records]
        return {
            'total_manual_closes': len(self.records),
            'avg_efficacy': sum(ratios) / len(ratios),
            'best_efficacy': max(ratios),
            'worst_efficacy': min(ratios),
            'profitable_closes': sum(1 for r in self.records if r['pnl_pct'] > 0),
        }


# Module-level singleton
efficacy_tracker = EfficacyTracker()
