import numpy as np
from typing import Optional
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

from core.events import SignalEvent, SignalType
from datetime import datetime, timezone
from strategies.strategy import Strategy
from utils.cooldown_manager import cooldown_manager

class PhalanxStrategy(Strategy):
    """
    Phalanx Multi-Signal Strategy
    Operates as an independent strategy using the OrderFlowAnalyzer to detect absorption.

    ═══════════════════════════════════════════════════════════════
    FORENSIC-DCA FIXES:
    #1: Añadido missing import 'Optional' (NameError en is_absorption_detected)
    #2: SL/TP ahora dinámicos basados en horizonte (SCALPING vs SWING)
    #3: Añadido cooldown de 60s por símbolo para prevenir spam
    #4: Fallback seguro si get_order_flow_metrics() no existe
    ═══════════════════════════════════════════════════════════════
    """
    def __init__(self, data_provider, events_queue, horizon="SCALPING", priority=1):
        super().__init__()
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.horizon = horizon
        self.priority = priority
        self.strategy_id = f"PHALANX_{horizon}"
        self.analyzer = OrderFlowAnalyzer()

    def generate_signals(self, event):
        from config import Config
        for symbol in Config.TRADING_PAIRS:
            # FORENSIC-DCA FIX #3: Cooldown por símbolo
            cooldown_key = f"PHALANX_{symbol}_{self.horizon}"
            if not cooldown_manager.check_custom_cooldown(cooldown_key, duration_seconds=60):
                continue

            data = self.data_provider.get_data(symbol)
            if data is None or len(data) < 20:
                continue
                
            # FORENSIC-DCA FIX #4: Fallback seguro para order flow metrics
            metrics = None
            if hasattr(self.data_provider, 'get_order_flow_metrics'):
                try:
                    metrics = self.data_provider.get_order_flow_metrics(symbol)
                except Exception:
                    metrics = None
            
            # Convert last 20 rows to list of dicts for analyzer
            price_action = data.iloc[-20:].to_dict('records')
            
            absorption = self.analyzer.is_absorption_detected(price_action, metrics)
            if absorption['detected']:
                signal_type = SignalType.LONG if absorption['type'] == 'BULLISH' else SignalType.SHORT
                current_price = price_action[-1]['close']
                
                # FORENSIC-DCA FIX #2: SL/TP dinámicos por horizonte
                # QUÉ: SL/TP ya no son 1% hardcoded — usan parámetros del horizonte.
                # POR QUÉ: 1% SL con 10x leverage = -10% loss, demasiado para $13.
                # PARA QUÉ: Consistencia con las SL/TP de las estrategias principales.
                if self.horizon == 'SCALPING':
                    tp_pct = Config.Strategies.SCALPING_PARAMS.get('tp_pct', 0.006)
                    sl_pct = Config.Strategies.SCALPING_PARAMS.get('sl_pct', 0.0075)
                else:
                    tp_pct = Config.Strategies.SWING_PARAMS.get('tp_pct', 0.045)
                    sl_pct = Config.Strategies.SWING_PARAMS.get('sl_pct', 0.025)
                
                # SOPHIA INTEGRATION
                sophia_report_dict = {}
                if hasattr(self, 'sophia') and self.sophia:
                    # Get market regime dynamically
                    market_regime = "UNKNOWN"
                    if hasattr(self, 'portfolio') and self.portfolio and hasattr(self.portfolio, 'market_regime') and self.portfolio.market_regime:
                        market_regime = self.portfolio.market_regime.get_current_regime()

                    sophia_report = self.sophia.analyze(
                        symbol=symbol,
                        direction=signal_type.name,
                        signal_strength=0.85,
                        setups={'reason': absorption['reason']},
                        confluence_score=1.0,
                        tp_pct=tp_pct,
                        sl_pct=sl_pct,
                        returns=None,
                        ttl_seconds=120.0 if self.horizon == 'SCALPING' else 900.0,
                        regime=market_regime
                    )
                    
                    # FORENSIC-V42: Dynamic threshold
                    veto_threshold = 0.65 if market_regime == "TRENDING" else 0.55
                    if sophia_report.win_probability < veto_threshold:
                        continue
                    sophia_report_dict = sophia_report.to_dict()

                signal = SignalEvent(
                    strategy_id="PHALANX",
                    symbol=symbol,
                    datetime=datetime.now(timezone.utc),
                    signal_type=signal_type,
                    strength=0.85,
                    atr=0.0,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    current_price=current_price,
                    horizon=self.horizon,
                    priority=self.priority,
                    metadata={'sophia': sophia_report_dict, 'reason': absorption['reason']}
                )
                self.events_queue.put(signal)
    
    def calculate_signals(self, event):
        self.generate_signals(event)
