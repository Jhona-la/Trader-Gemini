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
        
        # --- Mutación 16: Gravedad OBI ---
        self.current_obi = 0.0
        self.obi_history = deque(maxlen=20)
        self.obi_velocity = 0.0
        
        # --- Mutación 39: Bayesian Mirage Radar (Radar Anti-Spoofing Cuántico) ---
        self.is_spoofing = False
        self.spoofing_side = None
        self.spoofing_prob_buy = 0.0
        self.spoofing_prob_sell = 0.0
        
        # --- Mutación 14: Tick Volatility ---
        self.price_history = deque(maxlen=100)
        self.tick_volatility = 0.0001
        
        # --- Mutación 20: Liquidation Heatmap Cuántico ---
        from collections import defaultdict
        self.price_buckets = defaultdict(lambda: {'long_vol': 0.0, 'short_vol': 0.0})
        self.magnetic_pull_up = 0.0
        self.magnetic_pull_down = 0.0
        
        # --- Mutación 21: Filtro Gamma (Volume Acceleration) ---
        self.rolling_volume_60s = 0.0
        self.gamma_expansion_risk = False
        
        # --- Mutación 22: Z-Score Flash-Crash Anomaly ---
        self.tick_returns = deque(maxlen=200)
        self.flash_crash_anomaly = False
        self.flash_crash_direction = None
        self.last_price = 0.0
        
        # --- Mutación 23: Micro-Entropía (Shannon Proxy) ---
        self.tick_directions = deque(maxlen=200)
        self.high_micro_entropy = False
        self.current_entropy = 0.0
        
        # --- Mutación 33: Micro-Regime HMM ---
        self.hmm_state = 0 # 0: Normal, 1: Chop, 2: Toxic, 3: Flash Recovery

        # --- Mutación 34: Dark Pool & TWAP Decoder ---
        self.last_trade_time = 0.0
        self.twap_cluster_count = 0
        self.twap_cluster_side = None
        self.twap_cluster_vol = 0.0
        self.is_dark_pool_print = False
        self.dark_pool_side = None

        
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
        
        # Calculate OBI & OBI Velocity (Mutación 16)
        total_lob_qty = start_bid_q + start_ask_q
        if total_lob_qty > 0:
            self.current_obi = (start_bid_q - start_ask_q) / total_lob_qty
            self.obi_history.append(self.current_obi)
            if len(self.obi_history) >= 5:
                recent_5 = list(self.obi_history)[-5:]
                ma_5 = sum(recent_5) / 5.0
                self.obi_velocity = self.current_obi - ma_5
                
            # Mutación 39: Bayesian Mirage (Anti-Spoofing Bayesiano)
            if len(self.obi_history) > 1:
                prev_obi = self.obi_history[-2]
                
                # Base probability of spoofing (Prior: 10% of limit walls are fake)
                p_spoof = 0.10
                
                obi_drop_buy = max(0, prev_obi - self.current_obi)
                obi_drop_sell = max(0, self.current_obi - prev_obi)
                
                # Evidence of actual absorption (actual trades eating the wall)
                buy_absorption = max(0, self.rolling_delta_60s / (total_lob_qty + 1e-8))
                sell_absorption = max(0, -self.rolling_delta_60s / (total_lob_qty + 1e-8))
                
                # Bayes Rule para COMPRA Falsa
                if obi_drop_buy > 0.4 and buy_absorption < 0.05:
                    p_e_given_spoof = 0.95  # Alta prob de que desaparezca rápido sin trades si es spoof
                    p_e_given_real = 0.15   # Baja prob de que un institucional cancele su orden real sin fills
                    numerator = p_e_given_spoof * p_spoof
                    denominator = numerator + p_e_given_real * (1.0 - p_spoof)
                    self.spoofing_prob_buy = numerator / denominator
                else:
                    self.spoofing_prob_buy *= 0.9 # Decaimiento exponencial
                    
                # Bayes Rule para VENTA Falsa
                if obi_drop_sell > 0.4 and sell_absorption < 0.05:
                    p_e_given_spoof = 0.95
                    p_e_given_real = 0.15
                    numerator = p_e_given_spoof * p_spoof
                    denominator = numerator + p_e_given_real * (1.0 - p_spoof)
                    self.spoofing_prob_sell = numerator / denominator
                else:
                    self.spoofing_prob_sell *= 0.9
                    
                # Mantener compatibilidad con Mutación 17
                if self.spoofing_prob_buy > 0.6:
                    self.is_spoofing = True
                    self.spoofing_side = "BUY"
                elif self.spoofing_prob_sell > 0.6:
                    self.is_spoofing = True
                    self.spoofing_side = "SELL"
                else:
                    self.is_spoofing = False
                    self.spoofing_side = None
                    
    def on_trade(self, price, qty, is_buyer_maker):
        """
        Called on every trade.
        Updates VPIN buckets.
        """
        # Binance: is_buyer_maker=True -> SELL (Taker is Sell)
        # is_buyer_maker=False -> BUY (Taker is Buy)
        
        side = 'SELL' if is_buyer_maker else 'BUY'
        now = time.time()
        
        # --- Mutación 34: Dark Pool Decoder ---
        # Un humano no puede mandar múltiples órdenes en < 10ms. Si vemos esto es un algotrading TWAP
        if now - self.last_trade_time < 0.050: # 50ms window
            if side == self.twap_cluster_side:
                self.twap_cluster_count += 1
                self.twap_cluster_vol += qty
            else:
                self.twap_cluster_count = 1
                self.twap_cluster_side = side
                self.twap_cluster_vol = qty
        else:
            self.twap_cluster_count = 1
            self.twap_cluster_side = side
            self.twap_cluster_vol = qty
            
        self.last_trade_time = now
        
        # Si detectamos una ráfaga masiva unidireccional:
        if self.twap_cluster_count >= 15 and self.twap_cluster_vol > 50.0:
            self.is_dark_pool_print = True
            self.dark_pool_side = self.twap_cluster_side
        else:
            self.is_dark_pool_print = False
            self.dark_pool_side = None

        # VPIN Update
        self.current_bucket_vol += qty
        
        # Delta Tracking
        delta_val = qty if side == 'BUY' else -qty
        self.tick_history.append((now, delta_val))
        self.rolling_delta_60s += delta_val
        self.rolling_volume_60s += qty # Mutación 21
        
        # Tick Volatility (Mutación 14)
        self.price_history.append(price)
        if len(self.price_history) >= 20:
            import math
            mean_px = sum(self.price_history) / len(self.price_history)
            variance = sum((p - mean_px) ** 2 for p in self.price_history) / len(self.price_history)
            std_dev = math.sqrt(variance)
            self.tick_volatility = std_dev / price if price > 0 else 0.0001
        
        # --- Mutación 20: Liquidation Heatmap Cuántico Update ---
        if price > 1000: bucket_price = round(price, 0)
        elif price > 10: bucket_price = round(price, 2)
        else: bucket_price = round(price, 4)
            
        if side == 'BUY':
            self.price_buckets[bucket_price]['long_vol'] += qty
        else:
            self.price_buckets[bucket_price]['short_vol'] += qty
            
        # Cleanup memory: keep only buckets within 5% of current price
        keys_to_remove = [bp for bp in self.price_buckets.keys() if abs(bp - price) / price > 0.05]
        for k in keys_to_remove:
            del self.price_buckets[k]
            
        # Calculate Pull
        pull_down = 0.0
        pull_up = 0.0
        for bp, vols in self.price_buckets.items():
            if bp < price:
                # Price is above bucket. Longs are profitable, Shorts are underwater and could be liquidated UP.
                pull_up += vols['short_vol']
            else:
                # Price is below bucket. Shorts are profitable, Longs are underwater and could be liquidated DOWN.
                pull_down += vols['long_vol']
                
        self.magnetic_pull_up = pull_up
        self.magnetic_pull_down = pull_down
        
        # --- Mutación 22 & 23: Z-Score y Micro-Entropía ---
        if self.last_price > 0:
            tick_return = (price - self.last_price) / self.last_price
            if tick_return != 0:
                self.tick_returns.append(tick_return)
                self.tick_directions.append(1 if tick_return > 0 else -1)
                
            # Z-Score Calculation (Mutación 22)
            if len(self.tick_returns) >= 50:
                mean_ret = sum(self.tick_returns) / len(self.tick_returns)
                var_ret = sum((r - mean_ret)**2 for r in self.tick_returns) / len(self.tick_returns)
                std_ret = math.sqrt(var_ret)
                if std_ret > 0:
                    z_score = (tick_return - mean_ret) / std_ret
                    if z_score < -5.0:
                        self.flash_crash_anomaly = True
                        self.flash_crash_direction = "BUY"
                    elif z_score > 5.0:
                        self.flash_crash_anomaly = True
                        self.flash_crash_direction = "SELL"
                    else:
                        self.flash_crash_anomaly = False
                        self.flash_crash_direction = None
                        
            # Entropy Calculation (Mutación 23)
            if len(self.tick_directions) >= 100:
                ups = sum(1 for x in self.tick_directions if x == 1)
                p_up = ups / len(self.tick_directions)
                p_down = 1.0 - p_up
                if p_up > 0 and p_down > 0:
                    entropy = - (p_up * math.log2(p_up) + p_down * math.log2(p_down))
                else:
                    entropy = 0.0
                self.current_entropy = entropy
                self.high_micro_entropy = entropy > 0.98
                
            # HMM Calculation (Mutación 33)
            if len(self.tick_returns) >= 10:
                import numpy as np
                from utils.math_kernel import compute_micro_regime_hmm_jit
                self.hmm_state = compute_micro_regime_hmm_jit(np.array(self.tick_returns, dtype=np.float64))
                
        self.last_price = price
        
        # Cleanup old ticks (>60s)
        while self.tick_history and (now - self.tick_history[0][0]) > 60.0:
            _, old_delta = self.tick_history.popleft()
            self.rolling_delta_60s -= old_delta
            self.rolling_volume_60s -= abs(old_delta) # Mutación 21
        
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
            self.rolling_volume_60s -= abs(old_delta) # Mutación 21
            
        # --- Mutación 21: Gamma Expansion Risk Calculation ---
        # High volume but extremely low price volatility = Compression phase before a massive breakout
        if self.rolling_volume_60s > (self.bucket_vol * 2) and self.tick_volatility < 0.0005:
            self.gamma_expansion_risk = True
        else:
            self.gamma_expansion_risk = False
            
        return {
            'vpin': self.current_vpin,
            'iceberg_score': self.iceberg_score,
            'rolling_delta_60s': self.rolling_delta_60s,
            'is_toxic': self.current_vpin > 0.6 or self.iceberg_score > 0.8,
            'obi': self.current_obi,
            'obi_velocity': self.obi_velocity,
            'tick_volatility': self.tick_volatility,
            'is_spoofing': self.is_spoofing,
            'spoofing_side': self.spoofing_side,
            'spoofing_prob_buy': self.spoofing_prob_buy,   # Mutación 39
            'spoofing_prob_sell': self.spoofing_prob_sell, # Mutación 39
            'magnetic_pull_up': self.magnetic_pull_up,     # Mutación 20
            'magnetic_pull_down': self.magnetic_pull_down, # Mutación 20
            'gamma_expansion_risk': self.gamma_expansion_risk, # Mutación 21
            'flash_crash_anomaly': self.flash_crash_anomaly,   # Mutación 22
            'flash_crash_direction': self.flash_crash_direction, # Mutación 22
            'high_micro_entropy': self.high_micro_entropy, # Mutación 23
            'current_entropy': self.current_entropy, # Mutación 23
            'hmm_state': self.hmm_state, # Mutación 33
            'is_dark_pool_print': self.is_dark_pool_print, # Mutación 34
            'dark_pool_side': self.dark_pool_side # Mutación 34
        }
