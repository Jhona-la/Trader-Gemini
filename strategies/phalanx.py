import numpy as np
from numba import jit, float64, int64
from numba.experimental import jitclass

# JIT Spec for OnlineGARCH
garch_spec = [
    ('omega', float64),
    ('alpha', float64),
    ('beta', float64),
    ('variance', float64),
    ('last_return', float64)
]

@jitclass(garch_spec)
class OnlineGARCH:
    """
    ⚡ Online GARCH(1,1) Filter for HFT Volatility Clustering.
    Model: sigma^2_t = omega + alpha * ret^2_{t-1} + beta * sigma^2_{t-1}
    """
    def __init__(self, omega, alpha, beta, initial_variance):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.variance = initial_variance
        self.last_return = 0.0

    def update(self, current_return):
        """
        Updates variance estimate with new return.
        Stability check: Omega + Alpha + Beta < 1.0 (Strictly Stationary)
        """
        # GARCH Update
        new_variance = self.omega + self.alpha * (self.last_return ** 2) + self.beta * self.variance
        
        # Mean Reversion / Persistence Stability
        # Crypto volatility shouldn't explode. We use a floor and a cap.
        floor = 1e-6
        cap = 0.5 # 50% variance is huge even for crypto
        
        if new_variance > cap: new_variance = cap
        if new_variance < floor: new_variance = floor
        
        self.variance = new_variance
        self.last_return = current_return
        
        return np.sqrt(self.variance)
        
    def get_volatility(self):
        return np.sqrt(self.variance)

class OrderFlowAnalyzer:
    """
    🕵️‍♂️ Phalanx-Omega: Order Flow Imbalance & Absorption Logic.
    """
    def __init__(self):
        self.imbalance_threshold_long = 3.0 # 300%
        self.imbalance_threshold_short = 0.33 # 33%
        
    def analyze_imbalance(self, metrics: dict) -> dict:
        """
        Analyzes LOB metrics from binance_loader.
        metrics: { 'imbalance': float, 'delta': float, 'bid_vol_5': float, 'ask_vol_5': float }
        Returns: { 'signal': int, 'strength': float, 'reason': str, 'sniper': bool }
        """
        if not metrics:
            return {'signal': 0, 'strength': 0.0, 'reason': 'No Data', 'sniper': False}
            
        imb = metrics.get('imbalance', 1.0)
        delta = metrics.get('delta', 0.0)
        
        # SNIPER ENTRY: Imbalance > 3.0 (300%) AND Positive Delta (Market Aggression)
        is_sniper_long = (imb >= self.imbalance_threshold_long) and (delta > 0)
        is_sniper_short = (imb <= self.imbalance_threshold_short) and (delta < 0)

        # 1. Sniper Logic: > 300% Bid Vol = Buy Pressure
        if is_sniper_long:
            return {
                'signal': 1, 
                'strength': min(imb / 3.0, 3.0), 
                'reason': f"SNIPER_LONG_IMB_{imb:.2f}_DLT_{delta:.1f}",
                'sniper': True
            }
        elif imb >= self.imbalance_threshold_long:
            return {
                'signal': 1, 
                'strength': min(imb / 3.0, 2.0), # Lower strength if no delta confirmation
                'reason': f"LOB_IMBALANCE_LONG_{imb:.2f}",
                'sniper': False
            }
            
        # 2. Sniper Logic: > 300% Ask Vol = Sell Pressure
        if is_sniper_short:
             return {
                'signal': -1, 
                'strength': min((1.0/imb) / 3.0, 3.0),
                'reason': f"SNIPER_SHORT_IMB_{imb:.2f}_DLT_{delta:.1f}",
                'sniper': True
            }
        elif imb <= self.imbalance_threshold_short:
             return {
                'signal': -1, 
                'strength': min((1.0/imb) / 3.0, 2.0),
                'reason': f"LOB_IMBALANCE_SHORT_{imb:.2f}",
                'sniper': False
            }
            
        return {'signal': 0, 'strength': 0.0, 'reason': 'NEUTRAL', 'sniper': False}

    def is_absorption_detected(self, price_action, metrics: Optional[dict] = None) -> dict:
        """
        [PHASE 13] Absorption Detection (Stopping Volume + Delta Confirmation)
        Logic: High Relative Volume + Compressed Price Action + Delta Exhaustion
        Returns: { 'detected': bool, 'type': 'BULLISH'|'BEARISH'|'NONE', 'reason': str }
        """
        try:
            n = len(price_action)
            if n < 10:
                return {'detected': False, 'type': 'NONE', 'reason': 'INSUFFICIENT_DATA'}

            # 1. Volume Analysis (Relative Volume > 1.8x)
            last = price_action[-1]
            curr_vol = float(last['volume'])
            
            # Use NumPy vectorization
            avg_vol = np.mean(price_action[-11:-1]['volume'])
            
            if avg_vol > 0 and curr_vol < (avg_vol * 1.8):
                return {'detected': False, 'type': 'NONE', 'reason': ''}
            elif avg_vol == 0:
                 return {'detected': False, 'type': 'NONE', 'reason': 'ZERO_AVG_VOL'}

            # 2. Price Action Analysis (Spread/Range Compression)
            hi, lo, op, cl = float(last['high']), float(last['low']), float(last['open']), float(last['close'])
            rng = hi - lo
            if rng == 0: return {'detected': False, 'type': 'NONE', 'reason': 'ZERO_RANGE'}
            
            body = abs(cl - op)
            body_pct = body / rng
            
            # 3. Delta Confirmation (Institutional Signature)
            delta = metrics.get('delta', 0.0) if metrics else 0.0
            
            # 4. Detection Logic: High Effort (Vol) vs Low Result (Body)
            # If Delta is massive in one direction but price doesn't MOVE -> ABSORPTION
            if body_pct < 0.40: 
                # Context: Short-term Trend check
                start_price = float(price_action[-5]['close'])
                trend_delta = cl - start_price
                
                # BULLISH Absorption: Downtrend + Negative Delta + Price Stabilization
                if trend_delta < 0 and delta < 0: # Aggressive sellers met by passive buyers (muros)
                    return {
                        'detected': True, 
                        'type': 'BULLISH', 
                        'reason': f'ABSORPTION_OF_SELLERS_DLT_{delta:.1f}'
                    }
                # BEARISH Absorption: Uptrend + Positive Delta + Price Stabilization
                elif trend_delta > 0 and delta > 0: # Aggressive buyers met by passive sellers
                    return {
                        'detected': True, 
                        'type': 'BEARISH', 
                        'reason': f'ABSORPTION_OF_BUYERS_DLT_{delta:.1f}'
                    }
                    
            return {'detected': False, 'type': 'NONE', 'reason': 'VOL_NO_ABSORPTION'}
            
        except Exception as e:
            return {'detected': False, 'type': 'ERROR', 'reason': str(e)}
