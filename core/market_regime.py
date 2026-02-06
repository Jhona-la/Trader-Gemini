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
        self.global_regime = 'UNKNOWN'
        self.market_breadth = {'sentiment': 'UNKNOWN', 'bull_pct': 0.0, 'bear_pct': 0.0}
        self.regime_history = {}
        self.last_hurst = 0.5
    
    def detect_regime(self, symbol, bars_1m, bars_5m=None, bars_15m=None, bars_1h=None):
        """
        Detect current market regime for a symbol with MTF Confluence.
        """
        try:
            # Validate input
            if len(bars_1m) < 50:
                return self.last_regime.get(symbol, 'RANGING')
            
            # 1. MTF Trend Filter (5m & 15m)
            # PROFESSOR METHOD:
            # QUÉ: Filtro de confluencia Multi-Timeframe.
            # POR QUÉ: Evita entradas M1 en contra de la fuerza M5/M15.
            mtf_bias = 0 # 1=Bull, -1=Bear, 0=Neutral
            
            if bars_5m and len(bars_5m) >= 20:
                c5m = np.array([b['close'] for b in bars_5m], dtype=np.float64)
                ema20_5m = talib.EMA(c5m, timeperiod=20)[-1]
                mtf_bias += 1 if c5m[-1] > ema20_5m else -1
                
            if bars_15m and len(bars_15m) >= 20:
                c15m = np.array([b['close'] for b in bars_15m], dtype=np.float64)
                ema20_15m = talib.EMA(c15m, timeperiod=20)[-1]
                mtf_bias += 1 if c15m[-1] > ema20_15m else -1

            # 2. ADX & Metrics (1m)
            closes_1m = np.array([b['close'] for b in bars_1m], dtype=np.float64)
            highs_1m = np.array([b['high'] for b in bars_1m], dtype=np.float64)
            lows_1m = np.array([b['low'] for b in bars_1m], dtype=np.float64)
            
            adx = talib.ADX(highs_1m, lows_1m, closes_1m, timeperiod=14)[-1]
            
            # 3. Trend Direction (1h)
            is_bullish = True
            if bars_1h and len(bars_1h) >= 50:
                closes_1h = np.array([b['close'] for b in bars_1h], dtype=np.float64)
                ema50_1h = talib.EMA(closes_1h, timeperiod=50)[-1]
                is_bullish = closes_1h[-1] > ema50_1h
            
            # 4. Hurst (1m)
            from utils.statistics_pro import StatisticsPro
            hurst = StatisticsPro.calculate_hurst_exponent(closes_1m[-100:]) if len(closes_1m) >= 100 else 0.5
            self.last_hurst = hurst

            # 5. Logic
            raw_regime = 'CHOPPY'
            
            if adx > 25:
                if is_bullish and mtf_bias >= 0:
                    raw_regime = 'TRENDING_BULL'
                elif not is_bullish and mtf_bias <= 0:
                    raw_regime = 'TRENDING_BEAR'
            elif adx < 20:
                raw_regime = 'RANGING'
                if hurst < 0.4: raw_regime = 'MEAN_REVERTING'

            # Hysteresis
            if symbol not in self.regime_history: self.regime_history[symbol] = []
            self.regime_history[symbol].append(raw_regime)
            if len(self.regime_history[symbol]) > 3: self.regime_history[symbol].pop(0)
            
            if len(self.regime_history[symbol]) == 3 and all(x == raw_regime for x in self.regime_history[symbol]):
                final_regime = raw_regime
            else:
                final_regime = self.last_regime.get(symbol, raw_regime)
            
            self.last_regime[symbol] = final_regime
            return final_regime
            
        except Exception as e:
            logger.error(f"Regime Error {symbol}: {e}")
            return self.last_regime.get(symbol, 'RANGING')

    def calculate_market_context(self, active_symbols_data: Dict[str, Dict]):
        """
        SOVEREIGN MARKET CONTEXT (Swarm Intelligence).
        QUÉ: Calcula el sentimiento agregado de la canasta Elite.
        POR QUÉ: Evita dependencia de un solo símbolo y mide la amplitud real del mercado.
        
        active_symbols_data: {
            'BTC/USDT': {'1m': bars, '5m': bars, '1h': bars},
            ...
        }
        """
        regimes = []
        
        for symbol, data in active_symbols_data.items():
            r = self.detect_regime(
                symbol, 
                data.get('1m', []), 
                data.get('5m', []), 
                data.get('15m', []), 
                data.get('1h', [])
            )
            regimes.append(r)
            
        if not regimes:
            return self.market_breadth
            
        # Stats
        total = len(regimes)
        bulls = regimes.count('TRENDING_BULL')
        bears = regimes.count('TRENDING_BEAR')
        
        bull_pct = (bulls / total)
        bear_pct = (bears / total)
        
        # Determine Aggregate Sentiment
        # Phase 8.1 Rule: Consensus > 60%
        if bear_pct >= 0.60:
            sentiment = 'TRENDING_BEAR'
        elif bull_pct >= 0.60:
            sentiment = 'TRENDING_BULL'
        else:
            sentiment = 'MIXED'
            
        self.global_regime = sentiment # For backwards compatibility
        self.market_breadth = {
            'sentiment': sentiment,
            'bull_pct': bull_pct,
            'bear_pct': bear_pct,
            'regime_count': total
        }
        
        # LOGGING INSTITUCIONAL
        if sentiment == 'TRENDING_BEAR':
            logger.warning(f"🚨 [Sovereign Context] MARKET PANIC: {bear_pct:.0%} of assets are Bearish. Veto Active.")
        elif sentiment == 'TRENDING_BULL':
            logger.info(f"🐂 [Sovereign Context] MARKET FRENZY: {bull_pct:.0%} of assets are Bullish.")
            
        return self.market_breadth

    def detect_global_regime(self, btc_bars_1m, btc_bars_5m, btc_bars_1h):
        """
        DEPRECATED: Use calculate_market_context for breadth-based analysis.
        Kept for transition.
        """
        return self.detect_regime('BTC/USDT', btc_bars_1m, btc_bars_5m, None, btc_bars_1h)
    
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
                'preferred_strategy': 'NONE', # Or Short ML
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
            },
            'ZOMBIE': {
                'preferred_strategy': 'NONE',
                'position_size_multiplier': 0.0,
                'description': 'Flat/Frozen Market - Protection Active'
            }
        }
        
        return advice.get(regime, advice['RANGING'])
