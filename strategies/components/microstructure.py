import numpy as np
from collections import deque
import time

class MicrostructureAnalyzer:
    """
    🌊 COMPONENT: Microstructure (The Dark Layer)
    QUÉ: Analiza la microestructura del mercado (LOB y Tape) en tiempo real.
    POR QUÉ: Detectar liquidez oculta (Icebergs) y flujo tóxico (VPIN) antes de que impacte el precio.
    PARA QUÉ: Evitar ser "atropellado" por institucionales y "front-run" liquidez oculta.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # --- Iceberg Detection ---
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.bid_qty = 0.0
        self.ask_qty = 0.0
        self.last_depth_update = 0
        
        # Track reload events: (timestamp, side, price)
        self.reload_events = deque(maxlen=20)
        self.iceberg_score = 0.0 # 0.0 to 1.0
        
        # --- VPIN (Toxic Flow) ---
        # Volume Buckets
        self.bucket_vol = 10000.0 # Default, should be dynamic based on ADX/Vol
        self.current_bucket_vol = 0.0
        self.buy_vol_bucket = 0.0
        self.sell_vol_bucket = 0.0
        
        self.vpin_history = deque(maxlen=50) # VPIN values for MA
        self.current_vpin = 0.5 # Neutral start
        
        # --- Intra-Candle Delta (Order Imbalance) ---
        self.tick_history = deque(maxlen=5000) # (timestamp, delta_vol)
        self.rolling_delta_60s = 0.0
        
    def on_depth(self, start_bid_p, start_bid_q, start_ask_p, start_ask_q):
        """
        Called when LOB updates.
        Detects "Reloads": Size increasing at BBO without price improvement.
        """
        now = time.time()
        
        # Check for reload at Bid
        if start_bid_p == self.best_bid:
            if start_bid_q > self.bid_qty:
                # RELOAD DETECTED
                delta = start_bid_q - self.bid_qty
                # Filter noise
                if delta > 100: # Min relevant size
                    self.reload_events.append((now, 'BID', start_bid_p))
                    
        # Check for reload at Ask
        if start_ask_p == self.best_ask:
            if start_ask_q > self.ask_qty:
                # RELOAD DETECTED
                delta = start_ask_q - self.ask_qty
                if delta > 100:
                    self.reload_events.append((now, 'ASK', start_ask_p))
                    
        # Update State
        self.best_bid = start_bid_p
        self.best_ask = start_ask_p
        self.bid_qty = start_bid_q
        self.ask_qty = start_ask_q
        self.last_depth_update = now
        
        # Calculate Iceberg Score
        self._calculate_iceberg_score()
        
    def on_trade(self, price, qty, is_buyer_maker):
        """
        Called on every trade.
        Updates VPIN buckets.
        """
        # Binance: is_buyer_maker=True -> SELL (Taker is Sell)
        # is_buyer_maker=False -> BUY (Taker is Buy)
        
        side = 'SELL' if is_buyer_maker else 'BUY'
        now = time.time()
        
        # VPIN Update
        self.current_bucket_vol += qty
        
        # Delta Tracking
        delta_val = qty if side == 'BUY' else -qty
        self.tick_history.append((now, delta_val))
        self.rolling_delta_60s += delta_val
        
        # Cleanup old ticks (>60s)
        while self.tick_history and (now - self.tick_history[0][0]) > 60.0:
            _, old_delta = self.tick_history.popleft()
            self.rolling_delta_60s -= old_delta
        
        if side == 'BUY':
            self.buy_vol_bucket += qty
        else:
            self.sell_vol_bucket += qty
            
        # Check Bucket Fill
        if self.current_bucket_vol >= self.bucket_vol:
            self._finalize_bucket()
            
    def _finalize_bucket(self):
        """Calculate VPIN for the closed bucket."""
        total = self.buy_vol_bucket + self.sell_vol_bucket
        if total > 0:
            order_imbalance = abs(self.buy_vol_bucket - self.sell_vol_bucket)
            vpin_packet = order_imbalance / total
            
            self.vpin_history.append(vpin_packet)
            
            # Simple MA of VPIN
            if len(self.vpin_history) > 0:
                self.current_vpin = sum(self.vpin_history) / len(self.vpin_history)
        
        # Reset Bucket
        self.current_bucket_vol = 0.0
        self.buy_vol_bucket = 0.0
        self.sell_vol_bucket = 0.0
        
    def _calculate_iceberg_score(self):
        """
        Decay old events and count frequency.
        """
        # Decay old events
        now = time.time()
        while self.reload_events and (now - self.reload_events[0][0]) > 10.0:
            self.reload_events.popleft()
            
        count = len(self.reload_events)
        # Normalize: >5 reloads in 10s = 1.0 Score
        self.iceberg_score = min(count / 5.0, 1.0)

    def get_metrics(self):
        # Update tick decay before returning
        now = time.time()
        while self.tick_history and (now - self.tick_history[0][0]) > 60.0:
            _, old_delta = self.tick_history.popleft()
            self.rolling_delta_60s -= old_delta
            
        return {
            'vpin': self.current_vpin,
            'iceberg_score': self.iceberg_score,
            'rolling_delta_60s': self.rolling_delta_60s,
            'is_toxic': self.current_vpin > 0.6 or self.iceberg_score > 0.8
        }
