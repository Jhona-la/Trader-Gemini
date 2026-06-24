"""
LIQUIDATION HUNTER STRATEGY - Structural Edge Engine
=====================================================
Exploits thermodynamic imbalances in the derivatives market:
1. Extreme Funding Rates (Over-leveraged direction)
2. Flash-drops in Open Interest (Cascades / Liquidations)

This strategy replaces all ML and Technical engines.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional

from .strategy import Strategy
from core.events import SignalEvent
from core.enums import SignalType
from config import Config
from utils.logger import logger
from utils.common import validate_market_data, performance_timer
from utils.math_kernel import calculate_bollinger_jit, calculate_atr_jit

HAS_NANO = False
try:
    import core.nano_core as nano_core
    HAS_NANO = True
except ImportError:
    from core.exceptions import SystemIntegrityError
    raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

class LiquidationHunterStrategy(Strategy):
    """
    Hunts for liquidations and funding rate anomalies.
    Provides Structural Edge with large, organic Stop Losses 
    that bypass SIZING_FAILED safely.
    """
    
    def __init__(self, data_provider, events_queue, portfolio=None, priority: int = 1):
        super().__init__()
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.portfolio = portfolio
        self.priority = priority
        self.strategy_id = "STRUCTURAL_LIQ_HUNTER"
        
        # ── STRUCTURAL THRESHOLDS ──
        # Funding rate threshold: 0.1% per 8h is extremely high (usually it's 0.01%)
        self.FUNDING_EXTREME_THRESHOLD = 0.0005  # 0.05%
        
        # Open Interest flash-drop structural anomaly
        # Z-Score < -3.0 indicates a statistically significant, massive liquidation cascade
        self.OI_ZSCORE_THRESHOLD = -3.0 

        # We look back 15-30 mins to calculate OI drop
        self.PRIMARY_TF = '15m'
        
        # TP/SL Targets for Structural Edge (Asymmetric & Wide)
        self.TP_PCT = 0.12  # 12% Target Profit
        self.SL_PCT = 0.04  # 4% Stop Loss (Allows natural >$5 sizing for $13 capital)
        
        self.last_signal_time = {}
        self.signal_count = 0
        self.oi_history = {} # {symbol: [oi_values]}
        self.COOLDOWN_SECONDS = 3600  # 1 hour cooldown per symbol after a massive structural trade

    def _now(self):
        return datetime.now(tz=timezone.utc)
    
    @validate_market_data
    @performance_timer
    def calculate_signals(self, event):
        if not hasattr(event, 'symbol') or not event.symbol:
            target_symbols = self.data_provider.symbol_list
        else:
            target_symbols = [event.symbol]
            
        now = getattr(event, 'timestamp', self._now())
        
        for symbol in target_symbols:
            try:
                # 0. Cooldown
                if symbol in self.last_signal_time:
                    if (now - self.last_signal_time[symbol]).total_seconds() < self.COOLDOWN_SECONDS:
                        continue

                # 1. Fetch Structural Data (Zero-Copy)
                metrics = self.data_provider.get_derivatives_metrics(symbol)
                funding_rate = metrics['funding_rate']
                current_oi = metrics['open_interest']
                
                if current_oi == 0.0:
                    continue # No data available yet
                
                # Maintain OI history for calculating Z-Scores
                if symbol not in self.oi_history:
                    self.oi_history[symbol] = []
                    
                self.oi_history[symbol].append(current_oi)
                
                # Keep last 50 ticks for structural Z-Score calculation
                if len(self.oi_history[symbol]) > 50:
                    self.oi_history[symbol].pop(0)
                    
                if len(self.oi_history[symbol]) < 20:
                    continue # Need history for Z-Score
                    
                # Calculate Structural Z-Score of Open Interest
                if HAS_NANO:
                    try:
                        # Welford algorithm calculates streaming mean/std O(1)
                        z_scores = nano_core.welford_zscore_batch(self.oi_history[symbol], 20)
                        z_score_oi = z_scores[-1]
                    except Exception:
                        HAS_NANO = False
                        z_score_oi = 0.0
                
                if not HAS_NANO:
                    # Numpy fallback (O(N) computation over tiny array)
                    oi_slice = np.array(self.oi_history[symbol][-20:])
                    std_oi = np.std(oi_slice)
                    z_score_oi = (current_oi - np.mean(oi_slice)) / std_oi if std_oi > 0 else 0.0
                
                # Calculate simple drop for sizing multiplier later
                peak_oi = max(self.oi_history[symbol])
                oi_drop_pct = (peak_oi - current_oi) / peak_oi if peak_oi > 0 else 0.0
                
                # 2. Evaluate Structural Thermodynamics
                # Condition A: Extreme Funding Rate (Directional Bias)
                # If FR is highly positive, Longs are paying Shorts. The market is over-leveraged LONG.
                # A liquidation cascade will crash the price down (SHORT opportunity).
                
                # Condition B: Ignition (Z-Score drop in OI)
                # When OI drops abruptly by multiple sigmas, the cascade is happening NOW.
                
                signal_type = None
                strength = 0.0
                
                is_extreme_funding = abs(funding_rate) >= self.FUNDING_EXTREME_THRESHOLD
                is_liquidation_cascade = z_score_oi <= self.OI_ZSCORE_THRESHOLD

                
                if is_extreme_funding and is_liquidation_cascade:
                    if funding_rate > 0:
                        # Market was over-long, liquidations triggered -> go SHORT to ride the cascade 
                        # OR if the cascade finished, revert to mean (LONG). 
                        # Let's use structural momentum: ride the cascade.
                        signal_type = SignalType.SHORT
                        strength = min(1.0, oi_drop_pct / 0.05) # Max strength at 5% drop
                    else:
                        # Market was over-short, liquidations triggered -> go LONG
                        signal_type = SignalType.LONG
                        strength = min(1.0, oi_drop_pct / 0.05)
                        
                if not signal_type:
                    continue
                    
                # Fetch price for SignalEvent
                bars = self.data_provider.get_latest_bars(symbol, n=1, timeframe=self.PRIMARY_TF)
                if bars is None or len(bars) == 0:
                    continue
                current_price = bars['close'][-1]
                
                # We use fixed asymmetric TP/SL
                # R:R = 3:1 (12% TP, 4% SL)
                
                signal = SignalEvent(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    datetime=now,
                    signal_type=signal_type,
                    strength=strength,
                    atr=0.0, # Not relying on ATR for structural trades
                    tp_pct=self.TP_PCT,
                    sl_pct=self.SL_PCT,
                    current_price=float(current_price),
                    leverage=Config.BINANCE_LEVERAGE,
                    horizon="SWING", # Macro trades
                    priority=self.priority,
                    metadata={
                        'funding_rate': float(funding_rate),
                        'oi_drop_pct': float(oi_drop_pct),
                        'peak_oi': float(peak_oi),
                        'ml_confidence': float(strength) # Pass to Kelly compounding
                    }
                )
                
                self.events_queue.put(signal)
                self.last_signal_time[symbol] = now
                self.signal_count += 1
                logger.info(f"🌋 [LIQUIDATION HUNTER] #{self.signal_count}: {signal_type.name} {symbol} | FR: {funding_rate:.4%} | OI Z-Score: {z_score_oi:.2f} (Drop: {oi_drop_pct:.2%})")
                
            except Exception as e:
                logger.error(f"LiquidationHunter error for {symbol}: {e}")
