import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .strategy import Strategy
from core.events import SignalEvent
from datetime import datetime
import talib

class MLStrategy(Strategy):
    """
    Ensemble Machine Learning Strategy combining Random Forest + XGBoost.
    Uses REGRESSION to predict return magnitude (not just direction).
    Features: RSI, MACD, Bollinger Bands, ATR, ADX, Volume, Sentiment.
    """
    def __init__(self, data_provider, events_queue, symbol='BTC/USDT', lookback=50, sentiment_loader=None):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.symbol = symbol
        self.lookback = lookback  # ENHANCED: Now uses 30 bars for prediction (was 20)
        self.sentiment_loader = sentiment_loader
        
        # Ensemble: Random Forest + XGBoost (REGRESSION for magnitude prediction)
        # OPTIMIZED: Increased capacity for better crypto pattern recognition
        self.rf_model = RandomForestRegressor(
            n_estimators=200,        # +100% trees for better ensemble
            max_depth=8,             # +60% depth (256 vs 32 combinations)
            min_samples_split=10,    # Prevent overfitting
            min_samples_leaf=4,      # Reduce noise
            max_features='sqrt',     # Feature diversity
            n_jobs=-1,               # Use all CPUs
            random_state=42
        )
        self.xgb_model = XGBRegressor(
            n_estimators=200,        # +100% trees
            max_depth=6,             # More depth for complex patterns
            learning_rate=0.05,      # Slower but more precise
            subsample=0.8,           # Row sampling (prevent overfit)
            colsample_bytree=0.8,    # Feature sampling
            min_child_weight=3,      # Regularization
            tree_method='hist',      # Faster training
            random_state=42
        )
        
        # ADAPTIVE WEIGHTS: Start at 50/50, then adjust based on performance
        self.rf_weight = 0.5  # Dynamic weight for Random Forest
        self.xgb_weight = 0.5  # Dynamic weight for XGBoost
        
        # RAM OPTIMIZATION: Reduced from 100 to 30 predictions
        self.prediction_history = []
        self.max_history = 30  # Track last 30 predictions (was 100)
        self.last_price = None
        self.weight_update_interval = 20
        self.predictions_since_update = 0
        
        self.is_trained = False
        self.min_bars_to_train = 150 # ENHANCED: Need more history for multi-timeframe
        self.last_processed_time = None
        self.retrain_interval = 60 # Retrain every 60 minutes (bars)
        self.bars_since_train = 0
        
        # AUTO-SUSPEND: Pause strategy if it can't get data
        self.consecutive_failures = 0
        self.max_failures_before_suspend = 10
        self.is_suspended = False
        self.suspend_check_interval = 60
        self.cycles_since_suspend = 0

    def _calculate_confluence(self, rsi_1m, rsi_5m, rsi_15m, rsi_1h):
        """
        Calculate multi-timeframe confluence score.
        Counts how many timeframes are bullish vs bearish.
        
        Returns:
            int: -4 (all bearish) to +4 (all bullish)
            
        Examples:
            All bullish (RSI>50): +4 (perfect confluence)
            3 bullish, 1 bearish: +2 (strong confluence)
            2 vs 2: 0 (conflict/neutral)
            1 bullish, 3 bearish: -2 (bearish confluence)
        """
        score = 0
        
        # Each timeframe votes based on RSI
        score += 1 if rsi_1m > 50 else -1
        score += 1 if rsi_5m > 50 else -1
        score += 1 if rsi_15m > 50 else -1
        score += 1 if rsi_1h > 50 else -1
        
        return score

    def _prepare_features(self, bars):
        df = pd.DataFrame(bars)
        
        # Ensure numeric types
        cols = ['close', 'open', 'high', 'low', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=5).std()
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Add RSI Feature
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # Add MACD
        macd, macdsignal, macdhist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['macd_hist'] = macdhist
        
        # Add Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = upper
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        
        # Add ATR (Volatility)
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # Add ADX (Trend Strength)
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # Add Volume SMA
        df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=20)
        df['volume_rel'] = df['volume'] / df['volume_sma']
        
        # CRYPTO-SPECIFIC FEATURES (for better crypto predictions)
        # On-Balance Volume (important for crypto volume analysis)
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        df['obv_ema'] = talib.EMA(df['obv'].values, timeperiod=20)
        
        # Money Flow Index (buying/selling pressure)
        df['mfi'] = talib.MFI(df['high'].values, df['low'].values, 
                              df['close'].values, df['volume'].values, timeperiod=14)
        
        # Stochastic RSI (better for volatile crypto markets)
        fastk, fastd = talib.STOCHRSI(df['close'].values, timeperiod=14)
        df['stoch_rsi'] = fastk
        
        # ATR as percentage (normalized volatility)
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # Volume surge detection
        df['volume_surge'] = df['volume'] / df['volume_sma']
        
        # MULTI-TIMEFRAME FEATURES: Enhanced with 5m, 15m, 30m, and 1h context
        # This gives the model a complete picture of market across all time horizons
        try:
            # 5-MINUTE TIMEFRAME (Short-term momentum)
            bars_5m = self.data_provider.get_latest_bars_5m(self.symbol, n=30)
            if len(bars_5m) >= 14:
                closes_5m = np.array([b['close'] for b in bars_5m])
                # 5m RSI (micro trend)
                rsi_5m = talib.RSI(closes_5m, timeperiod=14)[-1]
                df['rsi_5m'] = rsi_5m
                # 5m EMA crossover (fast trend changes)
                ema_fast_5m = talib.EMA(closes_5m, timeperiod=5)[-1]
                ema_slow_5m = talib.EMA(closes_5m, timeperiod=15)[-1]
                df['ema_cross_5m'] = 1 if ema_fast_5m > ema_slow_5m else -1
        except:
            df['rsi_5m'] = df['rsi']  # Fallback to 1m RSI
            df['ema_cross_5m'] = 0
        
        try:
            # 15-MINUTE TIMEFRAME (Medium-term trend)
            bars_15m = self.data_provider.get_latest_bars_15m(self.symbol, n=30)
            if len(bars_15m) >= 14:
                closes_15m = np.array([b['close'] for b in bars_15m])
                highs_15m = np.array([b['high'] for b in bars_15m])
                lows_15m = np.array([b['low'] for b in bars_15m])
                
                # 15m RSI (swing trend)
                rsi_15m = talib.RSI(closes_15m, timeperiod=14)[-1]
                df['rsi_15m'] = rsi_15m
                # 15m ADX (trend strength)
                adx_15m = talib.ADX(highs_15m, lows_15m, closes_15m, timeperiod=14)[-1]
                df['adx_15m'] = adx_15m
        except:
            df['rsi_15m'] = df['rsi']
            df['adx_15m'] = df['adx']
        
        # Existing 1h TIMEFRAME (Long-term context)
        try:
            bars_1h = self.data_provider.get_latest_bars_1h(self.symbol, n=50)
            if len(bars_1h) >= 20:
                # Calculate 1h features
                closes_1h = np.array([b['close'] for b in bars_1h])
                highs_1h = np.array([b['high'] for b in bars_1h])
                lows_1h = np.array([b['low'] for b in bars_1h])
                volumes_1h = np.array([b['volume'] for b in bars_1h])
                
                # 1h RSI (macro momentum)
                rsi_1h = talib.RSI(closes_1h, timeperiod=14)[-1]
                
                # 1h MACD
                macd_1h, signal_1h, hist_1h = talib.MACD(closes_1h, fastperiod=12, slowperiod=26, signalperiod=9)
                df['macd_1h'] = macd_1h[-1] if len(macd_1h) > 0 else 0
                
                # 1h ADX (trend strength)
                adx_1h = talib.ADX(highs_1h, lows_1h, closes_1h, timeperiod=14)[-1]
                
                # Broadcast 1h features to all 1m rows
                df['rsi_1h'] = rsi_1h
                df['macd_1h'] = df.get('macd_1h', 0)
                df['adx_1h'] = adx_1h
            else:
                # Fallback if not enough 1h data
                df['rsi_1h'] = 50.0  # Neutral
                df['macd_1h'] = 0
                df['adx_1h'] = 20
        except:
            # Fallback if 1h data unavailable
            df['rsi_1h'] = 50.0
            df['macd_1h'] = 0
            df['adx_1h'] = 20
        
        # WEIGHTED CONFLUENCE SCORE: Count timeframe agreement with importance weighting
        # Longer timeframes are more reliable (1h > 15m > 5m > 1m)
        # Returns: -1.0 (all bearish) to +1.0 (all bullish)
        # VECTORIZED for pandas Series compatibility
        confluence = 0.0
        confluence += df['rsi'].apply(lambda x: 0.15 if x < 50 else -0.15)  # 1m: 15% weight
        confluence += df.get('rsi_5m', pd.Series([50] * len(df))).apply(lambda x: 0.25 if x < 50 else -0.25)  # 5m: 25%
        confluence += df.get('rsi_15m', pd.Series([50] * len(df))).apply(lambda x: 0.30 if x < 50 else -0.30)  # 15m: 30%
        confluence += df.get('rsi_1h', pd.Series([50] * len(df))).apply(lambda x: 0.30 if x < 50 else -0.30)   # 1h: 30%
        df['confluence_score'] = confluence
        
        # Drop unused columns that might cause NaNs
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)
        
        # Drop NaNs created by rolling/shift
        df.dropna(inplace=True)
        
        return df

    def calculate_signals(self, event):
        """
        Main Strategy Logic:
        1. Updates Market Data.
        2. Checks if Retraining is needed (Online Learning).
        3. Generates Predictions (Inference).
        4. Emits Signals.
        """
        if event.type != 'MARKET':
            return

        # 1. DATA COLLECTION & SUFFICIENCY CHECK
        # We need enough bars for features + training
        bars = self.data_provider.get_latest_bars(self.symbol, n=2500)
        if len(bars) < self.min_bars_to_train:
            return
            
        # 2. ONLINE LEARNING (RETRAINING)
        # Check if we need to retrain the model
        self.bars_since_train += 1
        
        if not self.is_trained or self.bars_since_train >= self.retrain_interval:
            print(f"🧠 ML Strategy: Retraining models for {self.symbol}...")
            try:
                df = self._prepare_features(bars)
                
                # Create Targets (Shifted Returns)
                # We want to predict the NEXT 5-minute return
                # Target = (Close[t+5] - Close[t]) / Close[t]
                # FIXED: Was using df['close'].shift(-5).pct_change(periods=5).shift(-5) which double-shifted!
                # Correct formula:
                df['target'] = (df['close'].shift(-5) - df['close']) / df['close']
                
                # Drop NaNs created by shifting
                df.dropna(inplace=True)
                
                if len(df) > 100:
                    # Features (X) and Target (y)
                    feature_cols = [c for c in df.columns if c not in ['target', 'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                    X = df[feature_cols]
                    y = df['target'] * 100 # Scale to percentage
                    
                    # TRAIN MODELS (The Missing Link!)
                    self.rf_model.fit(X, y)
                    self.xgb_model.fit(X, y)
                    
                    self.is_trained = True
                    self.bars_since_train = 0
                    print(f"✅ ML Strategy: Training Complete. (Rows: {len(df)})")
                else:
                    print("⚠️  ML Strategy: Not enough data after processing to train.")
            except Exception as e:
                print(f"❌ ML Strategy Training Failed: {e}")
                return

        # 3. INFERENCE (PREDICTION)
        if not self.is_trained:
            return

        try:
            # Get fresh features for the CURRENT moment
            # We re-fetch to ensure we have the absolute latest bar
            current_bars = self.data_provider.get_latest_bars(self.symbol, n=100)
            df_current = self._prepare_features(current_bars)
            
            if df_current.empty:
                return

            last_row = df_current.iloc[[-1]] # Double brackets to keep DataFrame format
            feature_cols = [c for c in df_current.columns if c not in ['target', 'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Ensure columns match training
            # (In production, we should enforce schema, but here we assume consistency)
            X_pred = last_row[feature_cols]
            
            # Predict
            rf_pred = self.rf_model.predict(X_pred)[0]
            xgb_pred = self.xgb_model.predict(X_pred)[0]
            
            # Ensemble
            ensemble_return = (rf_pred * self.rf_weight) + (xgb_pred * self.xgb_weight)
            
            # Context
            current_price = current_bars[-1]['close']
            current_atr = df_current.iloc[-1]['atr']
            timestamp = current_bars[-1]['datetime']
            
            # Sentiment Adjustment
            sentiment_score = 0
            if self.sentiment_loader:
                sentiment_score = self.sentiment_loader.get_sentiment(self.symbol)
                if sentiment_score > 0.2:
                    ensemble_return += 0.1
                elif sentiment_score < -0.2:
                    ensemble_return -= 0.1

            # 4. SIGNAL GENERATION
            # Dynamic Threshold based on Volatility/Regime
            entry_threshold = 0.15 # Base threshold (0.15% move)
            
            # 1h Trend Filter
            trend_1h = 'NEUTRAL'
            # (Simplified 1h trend check)
            if 'rsi_1h' in last_row.columns:
                rsi_1h = last_row['rsi_1h'].values[0]
                if rsi_1h > 55: trend_1h = 'UP'
                elif rsi_1h < 45: trend_1h = 'DOWN'
            
            # Logic
            if ensemble_return > entry_threshold:
                if trend_1h != 'DOWN': # Don't buy in downtrend
                    strength = min(ensemble_return / 0.5, 1.0)
                    print(f"🚀 ML LONG Signal: Exp Return {ensemble_return:.2f}%")
                    self.events_queue.put(SignalEvent(1, self.symbol, timestamp, 'LONG', strength=strength, atr=current_atr))
            
            elif ensemble_return < -entry_threshold:
                if trend_1h != 'UP': # Don't short in uptrend
                    strength = min(abs(ensemble_return) / 0.5, 1.0)
                    print(f"🔻 ML SHORT Signal: Exp Return {ensemble_return:.2f}%")
                    self.events_queue.put(SignalEvent(1, self.symbol, timestamp, 'SHORT', strength=strength, atr=current_atr))
            
            # Update History
            self.prediction_history.append({'pred': ensemble_return, 'price': current_price})
            if len(self.prediction_history) > 30: self.prediction_history.pop(0)

        except Exception as e:
            print(f"❌ ML Strategy Inference Failed: {e}")
            import traceback
            traceback.print_exc()

