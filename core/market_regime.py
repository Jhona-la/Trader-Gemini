import talib
import numpy as np

class MarketRegimeDetector:
    """
    Detects market regime to help strategies adapt their behavior.
    
    Regimes:
    - TRENDING_BULL: Strong uptrend (ADX>25, EMA50>EMA200, use ML)
    - TRENDING_BEAR: Strong downtrend (ADX>25, EMA50<EMA200, avoid trading)
    - RANGING: Sideways market (ADX<20, use mean reversion)
    - CHOPPY: Uncertain (ADX 20-25, reduce position size)
    """
    
    def __init__(self):
        self.last_regime = {}  # Per-symbol regime cache
    
    def detect_regime(self, symbol, bars_1m, bars_1h):
        """
        Detect current market regime for a symbol.
        
        Parameters:
        - symbol: Trading pair
        - bars_1m: 1-minute bars (for ADX)
        - bars_1h: 1-hour bars (for trend direction)
        
        Returns: String regime ('TRENDING_BULL', 'TRENDING_BEAR', 'RANGING', 'CHOPPY')
        """
        try:
            # Validate input
            if len(bars_1m) < 50 or len(bars_1h) < 200:
                # Not enough data, return last known or default
                return self.last_regime.get(symbol, 'RANGING')
            
            # 1. Calculate ADX from 1m bars (trend strength)
            closes_1m = np.array([b['close'] for b in bars_1m])
            highs_1m = np.array([b['high'] for b in bars_1m])
            lows_1m = np.array([b['low'] for b in bars_1m])
            
            adx = talib.ADX(highs_1m, lows_1m, closes_1m, timeperiod=14)[-1]
            
            # 2. Calculate trend direction from 1h bars (use closed candles only)
            closes_1h = np.array([b['close'] for b in bars_1h[:-1]])  # Exclude current open candle
            
            if len(closes_1h) >= 200:
                ema_50_1h = talib.EMA(closes_1h, timeperiod=50)[-1]
                ema_200_1h = talib.EMA(closes_1h, timeperiod=200)[-1]
                is_bullish = ema_50_1h > ema_200_1h
            else:
                # Fallback: use simple comparison
                is_bullish = closes_1h[-1] > closes_1h[-50]
            
            # 3. Calculate volatility (ATR %)
            atr = talib.ATR(highs_1m, lows_1m, closes_1m, timeperiod=14)[-1]
            current_price = closes_1m[-1]
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0.5
            
            # 4. Regime Classification
            if adx > 25:
                # Strong trend
                if is_bullish:
                    regime = 'TRENDING_BULL'
                else:
                    regime = 'TRENDING_BEAR'
            elif adx < 20:
                # Weak trend = ranging
                regime = 'RANGING'
            else:
                # ADX 20-25: uncertain/choppy
                regime = 'CHOPPY'
            
            # Cache result
            self.last_regime[symbol] = regime
            
            return regime
            
        except Exception as e:
            print(f"⚠️  Regime Detector Error for {symbol}: {e}")
            # Return last known or safe default
            return self.last_regime.get(symbol, 'RANGING')
    
    def get_regime_advice(self, regime):
        """
        Get trading advice for each regime.
        
        Returns: dict with recommended actions
        """
        advice = {
            'TRENDING_BULL': {
                'preferred_strategy': 'ML',
                'position_size_multiplier': 1.0,
                'description': 'Strong uptrend - ML aggressive'
            },
            'TRENDING_BEAR': {
                'preferred_strategy': 'NONE',
                'position_size_multiplier': 0.0,
                'description': 'Strong downtrend - CASH'
            },
            'RANGING': {
                'preferred_strategy': 'STATISTICAL',
                'position_size_multiplier': 1.0,
                'description': 'Sideways - Mean reversion'
            },
            'CHOPPY': {
                'preferred_strategy': 'TECHNICAL',
                'position_size_multiplier': 0.5,
                'description': 'Uncertain - Reduce size'
            }
        }
        
        return advice.get(regime, advice['RANGING'])
