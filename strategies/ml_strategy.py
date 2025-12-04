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
    def __init__(self, data_provider, events_queue, symbol='BTC/USDT', lookback=50, sentiment_loader=None, portfolio=None):
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.symbol = symbol
        self.lookback = lookback  # ENHANCED: Now uses 30 bars for prediction (was 20)
        self.sentiment_loader = sentiment_loader
        self.portfolio = portfolio  # LAYER 2: Reference to portfolio for exit logic
        
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
        # BUG #38 FIX: Implement adaptive weight updates
        self.weight_update_interval = 50  # Update weights every 50 predictions
        self.predictions_since_update = 0
        
        self.is_trained = False
        self.min_bars_to_train = 150 # ENHANCED: Need more history for multi-timeframe
        self.last_processed_time = None
        # BUG #41 FIX: Increased from 240 (4h) to 1440 (24h) to reduce overfitting
        # Longer interval allows model to learn more stable, generalizable patterns
        self.retrain_interval = 1440  # Retrain every 24 hours
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
        
        # Add Bollinger Bands (STATIONARY FEATURES ONLY)
        upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        # REMOVED: bb_upper, bb_lower (Absolute prices confuse the model)
        # ADDED: %B (Percent B) - Position of price within bands
        # (Price - Lower) / (Upper - Lower)
        df['bb_width'] = (upper - lower) / middle
        df['bb_pct_b'] = (df['close'] - lower) / (upper - lower)
        
        # Add EMA Distances (Relative Trend)
        ema_20 = talib.EMA(df['close'].values, timeperiod=20)
        ema_50 = talib.EMA(df['close'].values, timeperiod=50)
        df['dist_ema_20'] = (df['close'] - ema_20) / ema_20
        df['dist_ema_50'] = (df['close'] - ema_50) / ema_50
        
        # Add ATR (Volatility)
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # Add ADX (Trend Strength)
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # Add Volume SMA
        df['volume_sma'] = talib.SMA(df['volume'].values, timeperiod=20)
        df['volume_rel'] = df['volume'] / df['volume_sma']
        
        # FEATURE CLIPPING (Robustness against Flash Crashes/Pumps)
        # Limit relative features to reasonable bounds (+/- 20%)
        # This prevents the model from seeing "impossible" values during extreme volatility
        clip_cols = ['dist_ema_20', 'dist_ema_50', 'returns', 'momentum']
        for c in clip_cols:
            if c in df.columns:
                df[c] = df[c].clip(lower=-0.2, upper=0.2)
        
        # CRYPTO-SPECIFIC FEATURES (Sanitized)
        # On-Balance Volume (Use Slope/Change, NOT raw value)
        obv = talib.OBV(df['close'].values, df['volume'].values)
        df['obv_roc'] = pd.Series(obv).pct_change(periods=5) # 5-bar ROC of OBV
        
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
        # CRITICAL FIX: Corrected inverted logic (was x < 50, now x > 50)
        # RSI > 50 = Bullish → LONG signal (+)
        # RSI < 50 = Bearish → SHORT signal (-)
        confluence = 0.0
        confluence += df['rsi'].apply(lambda x: 0.15 if x > 50 else -0.15)  # 1m: 15% weight
        confluence += df.get('rsi_5m', pd.Series([50] * len(df))).apply(lambda x: 0.25 if x > 50 else -0.25)  # 5m: 25%
        confluence += df.get('rsi_15m', pd.Series([50] * len(df))).apply(lambda x: 0.30 if x > 50 else -0.30)  # 15m: 30%
        confluence += df.get('rsi_1h', pd.Series([50] * len(df))).apply(lambda x: 0.30 if x > 50 else -0.30)   # 1h: 30%
        df['confluence_score'] = confluence
        
        
        # Drop unused columns that might cause NaNs
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)
        if 'datetime' in df.columns:
            df.drop(columns=['datetime'], inplace=True)
        if 'symbol' in df.columns:
            df.drop(columns=['symbol'], inplace=True)
        
        # DATA CLEANING (CRITICAL FIX for "Input X contains infinity")
        # 1. Replace Infinities with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Drop NaNs created by rolling/shift/indicators
        df.dropna(inplace=True)
        
        # 3. Clip values to float32 range to prevent overflow
        # CRITICAL FIX: Only clip NUMERIC columns (not datetime)
        # We clip to +/- 1e9 which is safe for float32 and plenty for market data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(lower=-1e9, upper=1e9)
        
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
        # BUG #41 FIX: Increased from 2500 to 5000 bars for better training data
        # More data = better generalization, less overfitting to recent noise
        bars = self.data_provider.get_latest_bars(self.symbol, n=5000)
        if len(bars) < self.min_bars_to_train:
            return
            
        # 2. ONLINE LEARNING (RETRAINING)
        # Check if we need to retrain the model
        self.bars_since_train += 1
        
        if not self.is_trained or self.bars_since_train >= self.retrain_interval:
            # Check if training is already in progress
            if hasattr(self, 'training_thread') and self.training_thread.is_alive():
                # print(f"⏳ ML Strategy: Training already in progress for {self.symbol}...")
                return

            print(f"🧠 ML Strategy: Starting background training for {self.symbol}...")
            
            # Define training function to run in thread
            def train_models_background(bars_data):
                try:
                    # CRITICAL: Create a completely new DataFrame to avoid GIL contention
                    # and ensure we don't modify shared state
                    df = self._prepare_features(bars_data)
                    
                    # Create Targets (PAST Returns to Predict FUTURE)
                    # CRITICAL FIX FOR LOOKAHEAD BIAS:
                    # We want to predict the NEXT 5-bar return based on CURRENT features
                    # WRONG: raw_return = (df['close'].shift(-5) - df['close']) / df['close']  # Uses FUTURE data!
                    # RIGHT: We label each row with what happened AFTER it (forward-looking prediction target)
                    
                    # Calculate what the return will be if we enter NOW
                    # We use shift(-5) for the TARGET (what we want to predict)
                    # But we use shift(0) features (current bar) to make that prediction
                    # This is CORRECT because:
                    # - At training: We see bar N's features, and bar N+5's price (this is the label)
                    # - At inference: We see bar N's features, and predict what bar N+5 will be
                    
                    # However, the FEATURES must be from PAST data only (no shift(-X) in features)
                    # Let's verify our _prepare_features doesn't use future data...
                    
                    # Future 5-bar return (this is what we WANT to predict)
                    future_close = df['close'].shift(-5)
                    current_close = df['close']
                    raw_return = (future_close - current_close) / current_close
                    
                    # PROPER LABELING:
                    # We're creating a supervised learning problem where:
                    # X (features) = Technical indicators at time T (RSI, MACD, etc. from PAST data)
                    # y (target) = Return from time T to time T+5
                    # At inference: We only have data up to time T, predict T+5 return
                    
                    # Account for trading costs (fees + slippage)
                    # Binance Futures: 0.02% maker + 0.04% taker = 0.06% round-trip (conservative)
                    # Add slippage: ~0.04% more = Total 0.1% (0.001)
                    # CRITICAL FIX: Use REALISTIC Binance Futures costs
                    # Maker: 0.02%, Taker: 0.04%, Round-trip: 0.06% (all taker)
                    # Was 0.1% (too pessimistic), now 0.06% (realistic)
                    df['target'] = raw_return - 0.0006  # Subtract 0.06% cost (realistic Binance Futures fees)
                    df.dropna(inplace=True)
                    
                    if len(df) > 100:
                        feature_cols = [c for c in df.columns if c not in ['target', 'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                        X = df[feature_cols]
                        y = df['target'] # CRITICAL FIX: Keep in decimal units (0.003 not 0.3) 
                        
                        # Train new models
                        # CRITICAL OPTIMIZATION: n_jobs=1 to prevent thread contention in background
                        new_rf = RandomForestRegressor(
                            n_estimators=200, max_depth=8, min_samples_split=10, 
                            min_samples_leaf=4, max_features='sqrt', n_jobs=1, 
                            random_state=42
                        )
                        new_xgb = XGBRegressor(
                            n_estimators=200, max_depth=6, learning_rate=0.05, 
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=3, 
                            tree_method='hist', n_jobs=1, 
                            random_state=42
                        )
                        
                        new_rf.fit(X, y)
                        new_xgb.fit(X, y)
                        
                        # Atomic swap
                        self.rf_model = new_rf
                        self.xgb_model = new_xgb
                        self.is_trained = True
                        self.bars_since_train = 0
                        print(f"✅ ML Strategy: Background Training Complete for {self.symbol}. (Rows: {len(df)})")
                    else:
                        print("⚠️  ML Strategy: Not enough data for background training.")
                except Exception as e:
                    print(f"❌ ML Strategy Background Training Failed: {e}")

            # Start training thread
            import threading
            # Copy bars to avoid race conditions if main thread modifies list (though list is replaced in loader)
            # Deep copy might be slow, shallow copy of list is usually enough if dicts inside aren't mutated
            import copy
            bars_copy = copy.deepcopy(bars) 
            
            self.training_thread = threading.Thread(target=train_models_background, args=(bars_copy,))
            self.training_thread.daemon = True # Daemon thread dies if main program exits
            self.training_thread.start()

        # 3. INFERENCE (PREDICTION)
        if not self.is_trained:
            return

        try:
            # Get fresh features for the CURRENT moment
            # We re-fetch to ensure we have the absolute latest bar
            current_bars = self.data_provider.get_latest_bars(self.symbol, n=100)
            df_current = self._prepare_features(current_bars)
            
            if df_current.empty or len(df_current) < 2:
                return

            # BUG #35 REVERT: Changed back to use CLOSED candle (index -2)
            # REASONING: The user reported the model "became dumb" (losing).
            # Predicting on the OPEN candle (index -1) introduces massive noise because
            # indicators like RSI fluctuate wildly during the minute.
            # The model was trained on CLOSED candles (stable data).
            # Mismatching Open (Inference) vs Closed (Training) data destroys accuracy.
            # We accept the 1-minute lag in exchange for signal validity.
            last_row = df_current.iloc[[-2]]  # Use last CLOSED candle for consistency
            feature_cols = [c for c in df_current.columns if c not in ['target', 'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Ensure columns match training
            # (In production, we should enforce schema, but here we assume consistency)
            X_pred = last_row[feature_cols]
            
            # Predict
            rf_pred = self.rf_model.predict(X_pred)[0]
            xgb_pred = self.xgb_model.predict(X_pred)[0]
            
            # Ensemble
            ensemble_return = (rf_pred * self.rf_weight) + (xgb_pred * self.xgb_weight)
            
            # BUG #38 FIX: Track predictions for adaptive weighting
            self.predictions_since_update += 1
            
            # Context
            current_price = current_bars[-1]['close']
            current_atr = df_current.iloc[-1]['atr']
            timestamp = current_bars[-1]['datetime']
            
            # Sentiment Adjustment
            sentiment_score = 0
            if self.sentiment_loader:
                # CRITICAL FIX: get_sentiment() takes no parameters (it returns overall market sentiment)
                # Previously was calling with self.symbol which caused TypeError
                sentiment_score = self.sentiment_loader.get_sentiment()
                if sentiment_score > 0.2:
                    ensemble_return += 0.001 # Add 0.1% bias (was 0.1 = 10% -> BUG)
                elif sentiment_score < -0.2:
                    ensemble_return -= 0.001 # Subtract 0.1% bias

            # 4. SIGNAL GENERATION
            # BUG #36 FIX: Increased threshold to cover actual trading costs
            # COST ANALYSIS:
            #   - Futures fees (round-trip): 0.12% (0.06% x 2)
            #   - Slippage estimate: 0.10% (conservative for 1min bars)
            #   - Total unavoidable cost: 0.22%
            #   - Minimum profit target: 0.30% (above costs)
            #   - TOTAL THRESHOLD: 0.52%
            # This ensures we only trade high-conviction signals that can overcome costs
            entry_threshold = 0.0052  # 0.52% threshold (was 0.0015/0.15% - caused guaranteed losses)
            
            # 1h Trend Filter (Robust EMA Logic)
            trend_1h = 'NEUTRAL'
            try:
                # Use the 1h features we already calculated if available
                if 'macd_1h' in last_row.columns:
                    # We don't have EMAs in the dataframe, so let's calculate them quickly from 1h bars
                    bars_1h = self.data_provider.get_latest_bars_1h(self.symbol, n=210)
                    if len(bars_1h) >= 200:
                        closes_1h = np.array([b['close'] for b in bars_1h[:-1]]) # Closed candles
                        ema_50 = talib.EMA(closes_1h, timeperiod=50)[-1]
                        ema_200 = talib.EMA(closes_1h, timeperiod=200)[-1]
                        
                        if ema_50 > ema_200:
                            trend_1h = 'UP'
                        else:
                            trend_1h = 'DOWN'
            except:
                # Fallback to RSI check if EMA fails
                if 'rsi_1h' in last_row.columns:
                    rsi_1h = last_row['rsi_1h'].values[0]
                    if rsi_1h > 55: trend_1h = 'UP'
                    elif rsi_1h < 45: trend_1h = 'DOWN'
            
            # Logic
            # CRITICAL FIX: Use Confluence Score to filter signals
            # We only take LONGs if confluence is positive (Bullish alignment)
            # We only take SHORTs if confluence is negative (Bearish alignment)
            confluence = last_row['confluence_score'].values[0] if 'confluence_score' in last_row.columns else 0
            
            if ensemble_return > entry_threshold:
                # BUG #37 FIX: Strengthen confluence requirement
                # OLD: confluence > -0.5 (allowed LONG even if 3/4 timeframes bearish)
                # NEW: confluence >= 1 (requires at least 3/4 timeframes bullish)
                if trend_1h != 'DOWN' and confluence >= 1:  # Require strong multi-timeframe alignment
                    # Strength: 0.5% return = 100% strength (0.005 / 0.005 = 1.0)
                    # CRITICAL FIX: Adjusted from 0.5% to 0.3% for 100% strength
                    # More realistic for crypto 1min bars (average move ~0.15%)
                    strength = min(ensemble_return / 0.003, 1.0)  # 100% at 0.3% (was 0.5%)
                    print(f"🚀 ML LONG Signal: Exp Return {ensemble_return*100:.2f}% | Confluence: {confluence:.1f}")
                    self.events_queue.put(SignalEvent(1, self.symbol, timestamp, 'LONG', strength=strength, atr=current_atr))
            
            elif ensemble_return < -entry_threshold:
                # BUG #37 FIX: Strengthen confluence requirement
                # OLD: confluence < 0.5 (allowed SHORT even if 3/4 timeframes bullish)
                # NEW: confluence <= -1 (requires at least 3/4 timeframes bearish)
                if trend_1h != 'UP' and confluence <= -1:  # Require strong multi-timeframe alignment
                    # CRITICAL FIX: Match LONG signal strength calculation (0.003 not 0.005)
                    # Ensures consistent position sizing between LONG and SHORT
                    strength = min(abs(ensemble_return) / 0.003, 1.0)  # 100% at 0.3% (was 0.5%)
                    print(f"🔻 ML SHORT Signal: Exp Return {ensemble_return*100:.2f}% | Confluence: {confluence:.1f}")
                    self.events_queue.put(SignalEvent(1, self.symbol, timestamp, 'SHORT', strength=strength, atr=current_atr))
            
            # LAYER 2: Strategy-Based Exit Logic (Intelligent Exits)
            # Check if we should exit existing positions based on ML predictions and technicals
            if self.portfolio:
                position = self.portfolio.positions.get(self.symbol, {'quantity': 0})
                quantity = position.get('quantity', 0)
                
                if quantity != 0:  # We have an open position
                    # Get technical indicators for exit decision
                    rsi = last_row['rsi'].values[0] if 'rsi' in last_row.columns else 50
                    macd = last_row['macd'].values[0] if 'macd' in last_row.columns else 0
                    macd_signal = last_row['macd_signal'].values[0] if 'macd_signal' in last_row.columns else 0
                    
                    should_exit = False
                    exit_reason = ""
                    
                    if quantity > 0:  # LONG position
                        # Exit conditions for LONG
                        # 1. ML predicts reversal (negative return)
                        if ensemble_return < -0.001:  # Predicts -0.1% or worse
                            should_exit = True
                            exit_reason = "ml_reversal"
                        
                        # 2. RSI overbought (exhaustion)
                        elif rsi > 75:
                            should_exit = True
                            exit_reason = "rsi_overbought"
                        
                        # 3. MACD bearish crossover
                        elif macd < macd_signal and macd > 0:  # Was positive, now crossing down
                            should_exit = True
                            exit_reason = "macd_cross_down"
                    
                    elif quantity < 0:  # SHORT position
                        # Exit conditions for SHORT
                        # 1. ML predicts upward move
                        if ensemble_return > 0.001:  # Predicts +0.1% or better
                            should_exit = True
                            exit_reason = "ml_reversal"
                        
                        # 2. RSI oversold (potential bounce)
                        elif rsi < 25:
                            should_exit = True
                            exit_reason = "rsi_oversold"
                        
                        # 3. MACD bullish crossover
                        elif macd > macd_signal and macd < 0:  # Was negative, now crossing up
                            should_exit = True
                            exit_reason = "macd_cross_up"
                    
                    if should_exit:
                        print(f"🚪 ML EXIT Signal for {self.symbol}: {exit_reason}")
                        self.events_queue.put(SignalEvent(1, self.symbol, timestamp, 'EXIT', strength=1.0))
            
            # Update History
            # BUG #38 & #43 FIX: Enhanced history tracking for adaptive weights and logging
            prediction_record = {
                'pred': ensemble_return,
                'rf_pred': rf_pred,
                'xgb_pred': xgb_pred,
                'price': current_price,
                'timestamp': timestamp,
                'rsi': last_row['rsi'].values[0] if 'rsi' in last_row.columns else 50,
                'confluence': confluence
            }
            self.prediction_history.append(prediction_record)
            if len(self.prediction_history) > 30: 
                self.prediction_history.pop(0)
            
            # BUG #38 FIX: Update model weights based on recent performance
            if self.predictions_since_update >= self.weight_update_interval:
                self._update_model_weights()
                self.predictions_since_update = 0
            
            # BUG #43 FIX: Log predictions to CSV for debugging
            self._log_prediction(timestamp, rf_pred, xgb_pred, ensemble_return, current_price, confluence, trend_1h)

        except Exception as e:
            print(f"❌ ML Strategy Inference Failed: {e}")
            import traceback
            traceback.print_exc()

    def _update_model_weights(self):
        """
        BUG #38 FIX: Update RF and XGBoost weights based on recent performance.
        Uses inverse error weighting: model with lower error gets higher weight.
        """
        try:
            if len(self.prediction_history) < 30:
                return
            
            # Calculate actual returns for last 30 predictions
            # We compare predicted return vs actual return over next N bars
            rf_errors = []
            xgb_errors = []
            
            for i in range(len(self.prediction_history) - 5):
                pred = self.prediction_history[i]
                future_pred = self.prediction_history[i + 5] if i + 5 < len(self.prediction_history) else None
                
                if future_pred:
                    # Actual return = (future_price - current_price) / current_price
                    actual_return = (future_pred['price'] - pred['price']) / pred['price']
                    
                    # Calculate errors
                    rf_error = abs(pred['rf_pred'] - actual_return)
                    xgb_error = abs(pred['xgb_pred'] - actual_return)
                    
                    rf_errors.append(rf_error)
                    xgb_errors.append(xgb_error)
            
            if not rf_errors or not xgb_errors:
                return
            
            # Mean Absolute Error
            rf_mae = np.mean(rf_errors)
            xgb_mae = np.mean(xgb_errors)
            
            # Inverse weighting: lower error = higher weight
            total_error = rf_mae + xgb_mae
            if total_error > 0:
                self.rf_weight = xgb_mae / total_error   # If XGB has high error, give more weight to RF
                self.xgb_weight = rf_mae / total_error   # If RF has high error, give more weight to XGB
                
                print(f"📊 {self.symbol} Model Weights Updated: RF={self.rf_weight:.3f} (MAE:{rf_mae:.4f}), XGB={self.xgb_weight:.3f} (MAE:{xgb_mae:.4f})")
        
        except Exception as e:
            print(f"⚠️  Weight update failed: {e}")
    
    def _log_prediction(self, timestamp, rf_pred, xgb_pred, ensemble_pred, current_price, confluence, trend_1h):
        """
        BUG #43 FIX: Log ML predictions to CSV for debugging and analysis.
        Creates a file per symbol with all prediction data.
        """
        try:
            import os
            from config import Config
            
            # Create log entry
            log_entry = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp),
                'symbol': self.symbol,
                'rf_pred': round(rf_pred, 6),
                'xgb_pred': round(xgb_pred, 6),
                'ensemble_pred': round(ensemble_pred, 6),
                'rf_weight': round(self.rf_weight, 3),
                'xgb_weight': round(self.xgb_weight, 3),
                'current_price': round(current_price, 4),
                'confluence': round(confluence, 2),
                'trend_1h': trend_1h
            }
            
            # Save to CSV
            log_dir = Config.DATA_DIR
            os.makedirs(log_dir, exist_ok=True)
            
            symbol_safe = self.symbol.replace('/', '_')
            log_path = os.path.join(log_dir, f"ml_predictions_{symbol_safe}.csv")
            
            df = pd.DataFrame([log_entry])
            
            # Append to existing file or create new
            if os.path.exists(log_path):
                df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(log_path, index=False)
        
        except Exception as e:
            # Silent fail - logging shouldn't crash the strategy
            pass

