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

    def train_model(self):
        try:
            # Fetch historical data (+ extra bars for proper validation)
            # ENHANCED: Increased training data from 1001 to 2500 bars (~41 hours)
            bars = self.data_provider.get_latest_bars(self.symbol, n=2500)
            if len(bars) < 100:
                return
            
            df = self._prepare_features(bars)
            
            if df.empty or len(df) < 50:
                return
            
            # REGRESSION TARGET: Predict next 5-minute return percentage
            df['target'] = (df['close'].shift(-5) / df['close'] - 1) * 100
            df.dropna(inplace=True)
            
            # FIX LOOK-AHEAD BIAS: Exclude last bar from training
            # MULTI-TIMEFRAME: Added 5m, 15m features + confluence_score
            features = ['returns', 'volatility', 'momentum', 'rsi', 'macd', 'macd_hist', 
                       'bb_width', 'atr', 'adx', 'volume_rel',
                       'obv_ema', 'mfi', 'stoch_rsi', 'atr_pct', 'volume_surge',
                       'rsi_5m', 'ema_cross_5m', 'rsi_15m', 'adx_15m',
                       'rsi_1h', 'macd_1h', 'adx_1h', 'confluence_score']
            X = df[features]
            y = df['target']
            
            if X.empty:
                return
            
            # CRITICAL FIX: Exclude last bar from training (no look-ahead)
            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            
            # Train BOTH models on historical data only
            self.rf_model.fit(X_train, y_train)
            self.xgb_model.fit(X_train, y_train)
            
            self.is_trained = True
            print(f"✅ Ensemble Trained on {len(X_train)} samples for {self.symbol} (Regression, No Look-Ahead).")
        except Exception as e:
            print(f"Error training model for {self.symbol}: {e}")
    
    def _update_adaptive_weights(self):
        """Calculate accuracy of each model and adjust weights accordingly."""
        if len(self.prediction_history) < 10:
            return
        
        # Count correct predictions for each model
        rf_correct = sum(1 for p in self.prediction_history if p['rf_pred'] == p['actual'])
        xgb_correct = sum(1 for p in self.prediction_history if p['xgb_pred'] == p['actual'])
        
        total = len(self.prediction_history)
        rf_accuracy = rf_correct / total
        xgb_accuracy = xgb_correct / total
        
        # Avoid division by zero
        total_accuracy = rf_accuracy + xgb_accuracy
        if total_accuracy == 0:
            return
        
        # Adjust weights proportionally to accuracy
        self.rf_weight = rf_accuracy / total_accuracy
        self.xgb_weight = xgb_accuracy / total_accuracy
        
        print(f"🔄 Adaptive Weights Updated ({self.symbol}): RF={self.rf_weight:.2f} ({rf_accuracy:.1%}) | XGB={self.xgb_weight:.2f} ({xgb_accuracy:.1%})")

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            # AUTO-SUSPEND CHECK: Skip if suspended
            if self.is_suspended:
                # SILENT MODE: Don't spam logs or try to wake up
                return
            
            # 1. Train Model if not yet trained
            if not self.is_trained:
                bars = self.data_provider.get_latest_bars(self.symbol, n=self.min_bars_to_train)
                
                # ENHANCED DATA UNAVAILABLE CHECK
                is_invalid = (
                    not bars or
                    len(bars) == 0 or
                    (len(bars) > 0 and not isinstance(bars[0].get('close'), (int, float)))
                )
                
                if is_invalid:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures_before_suspend:
                        print(f"⏸️  ML Strategy ({self.symbol}): PERMANENTLY SUSPENDED (no valid data)")
                        self.is_suspended = True
                    return
                
                self.consecutive_failures = 0
                
                current_len = len(bars)
                if current_len < self.min_bars_to_train:
                    # LOGGING: Print progress every 10 bars so user knows it's alive
                    if current_len == 1 or current_len % 10 == 0:
                        print(f"📉 ML Strategy ({self.symbol}): Collecting data... {current_len}/{self.min_bars_to_train} bars")
                    return
                
                self.train_model()
                self.bars_since_train = 0
                return

            # Check for retraining
            self.bars_since_train += 1
            if self.bars_since_train >= self.retrain_interval:
                self.train_model()
                self.bars_since_train = 0

            # 2. Predict
            # ENHANCED: Increased from 20 to 30 bars for better pattern recognition
            bars = self.data_provider.get_latest_bars(self.symbol, n=30)
            if len(bars) < 10:
                return

            # Deduplication: Check if we already processed this bar
            current_time = bars[-1]['datetime']
            if self.last_processed_time == current_time:
                return
            self.last_processed_time = current_time

            df = self._prepare_features(bars)
            if df.empty:
                return

            # Add Sentiment Feature
            sentiment_score = 0
            if self.sentiment_loader:
                sentiment_score = self.sentiment_loader.fetch_news() 
            
            # Use same features as training (including multi-timeframe + confluence)
            features = ['returns', 'volatility', 'momentum', 'rsi', 'macd', 'macd_hist', 
                       'bb_width', 'atr', 'adx', 'volume_rel',
                       'obv_ema', 'mfi', 'stoch_rsi', 'atr_pct', 'volume_surge',
                       'rsi_5m', 'ema_cross_5m', 'rsi_15m', 'adx_15m',
                       'rsi_1h', 'macd_1h', 'adx_1h', 'confluence_score']
            last_row = df.iloc[[-1]][features]
            
            # Current price and volatility for tracking
            current_price = bars[-1]['close']
            current_atr = last_row['atr'].values[0] if 'atr' in last_row else None
            current_atr_pct = last_row['atr_pct'].values[0] if 'atr_pct' in last_row else 0
            
            # CRITICAL FIX: Initialize regime BEFORE any early returns
            regime = 'LOW_VOL'
            entry_threshold = 0.40  # INCREASED: Was 0.15, now 0.40 to cover fees + profit
            
            # VOLATILITY FILTER (crypto optimization): Skip if market too quiet
            # Threshold adjusted for 1m candles: 0.05% (5 basis points)
            if current_atr_pct < 0.05:
                # print(f"  ⏭️  Skipping {self.symbol}: Low volatility ({current_atr_pct:.3f}%)")
                return  # Market too quiet, ML predictions unreliable
            
            # STRATEGY COORDINATION FIX #3: Market Regime Detection
            # Classify market to only trade in optimal conditions
            adx_value = last_row['adx'].values[0] if 'adx' in last_row else 0.0
            
            # DEBUG LOGGING (CRITICAL)
            print(f"🔍 {self.symbol} ANALYSIS: Price=${current_price:.2f} | ADX={adx_value:.2f} | ATR%={current_atr_pct:.3f}%")
            
            if adx_value == 0.0:
                 print(f"  ⚠️  WARNING: ADX is 0.0! Insufficient history or calculation error. Rows: {len(df)}")
            
            if adx_value > 25 and current_atr_pct > 1.5:
                regime = 'TRENDING_HIGH_VOL'
                entry_threshold = 0.4  # AGGRESSIVE: Lowered from 0.5 to capture volatility
            elif adx_value > 25 and current_atr_pct <= 1.5:
                regime = 'TRENDING_NORMAL'
                entry_threshold = 0.4  # Ideal for ML (was 0.2)
            elif adx_value <= 25 and current_atr_pct < 0.8: # FIXED: Covered 20-25 gap
                regime = 'RANGING'
                entry_threshold = 0.35  # AGGRESSIVE: Lowered from 0.45 to trade ranges

            elif adx_value < 20: # Fallback for high vol choppy
                regime = 'CHOPPY'
                # Skip choppy markets - low trend, medium-high volatility
                print(f"  ⏭️  Skipping {self.symbol}: Choppy market (ADX={adx_value:.1f})")
                return
            
            # PERFORMANCE TRACKING: Check previous prediction against actual result
            if self.last_price is not None:
                actual_direction = 1 if current_price > self.last_price else 0
                
                if len(self.prediction_history) > 0:
                    last_pred = self.prediction_history[-1]
                    if 'actual' not in last_pred:
                        last_pred['actual'] = actual_direction
                        
                        self.predictions_since_update += 1
                        if self.predictions_since_update >= self.weight_update_interval:
                            self._update_adaptive_weights()
                            self.predictions_since_update = 0
            
            # MULTI-TIMEFRAME ANALYSIS: 1h Trend
            # Fetch 1h bars to determine long-term trend
            # STABILITY FIX: Use CLOSED candles only (exclude last open candle)
            trend_1h = 'NEUTRAL'
            try:
                bars_1h = self.data_provider.get_latest_bars_1h(self.symbol, n=210)
                if len(bars_1h) >= 200:
                    # Exclude the last bar (current open candle) to prevent trend flip-flopping
                    closes_1h = np.array([b['close'] for b in bars_1h[:-1]])
                    
                    if len(closes_1h) >= 200:
                        ema_50_1h = talib.EMA(closes_1h, timeperiod=50)[-1]
                        ema_200_1h = talib.EMA(closes_1h, timeperiod=200)[-1]
                        
                        if ema_50_1h > ema_200_1h:
                            trend_1h = 'UP'
                        else:
                            trend_1h = 'DOWN'
            except:
                pass # Fallback to NEUTRAL if no 1h data
            
            # ENSEMBLE PREDICTION (REGRESSION - predicts return %)
            rf_prediction = self.rf_model.predict(last_row)[0]
            xgb_prediction = self.xgb_model.predict(last_row)[0]
            
            # Weighted Average of Predicted Returns
            ensemble_return = (self.rf_weight * rf_prediction) + (self.xgb_weight * xgb_prediction)
            
            # Direction based on sign of predicted return
            ensemble_direction = 'UP' if ensemble_return > 0 else 'DOWN'
            
            # PRINT ML CONCLUSIONS (User Request)
            # Only print if return is significant to avoid spam
            if abs(ensemble_return) > 0.05:
                # Extract 5m/15m features for display
                rsi_5m_val = last_row['rsi_5m'].values[0] if 'rsi_5m' in last_row else 0
                rsi_15m_val = last_row['rsi_15m'].values[0] if 'rsi_15m' in last_row else 0
                
                print(f"🤖 ML Strategy {self.symbol}: Pred={ensemble_return:.2f}% ({ensemble_direction}) | RF={rf_prediction:.2f} XGB={xgb_prediction:.2f} | Regime={regime} | 5m_RSI={rsi_5m_val:.1f} 15m_RSI={rsi_15m_val:.1f}")

            # Store prediction
            self.prediction_history.append({
                'rf_pred': rf_prediction,
                'xgb_pred': xgb_prediction,
                'ensemble_return': ensemble_return,
            })
            
            if len(self.prediction_history) > self.max_history:
                self.prediction_history.pop(0)
            
            self.last_price = current_price
            
            # Adjust based on Sentiment (modify predicted return)
            if sentiment_score > 0.2:
                ensemble_return += 0.1
            elif sentiment_score < -0.2:
                ensemble_return -= 0.1
            
            timestamp = bars[-1]['datetime']
            
            # Log ensemble details with predicted returns and regime
            print(f"ML Ensemble {self.symbol} [{regime}] (1h Trend: {trend_1h}): RF={rf_prediction:+.2f}%(w={self.rf_weight:.2f}) XGB={xgb_prediction:+.2f}%(w={self.xgb_weight:.2f}) → Expect {ensemble_return:+.2f}% return")
            
            # 3. Generate Signal Based on Expected Return (Dynamic Threshold)
            # entry_threshold set by regime detection above
            
            # MULTI-TIMEFRAME ADJUSTMENT: Boost/Penalize based on 1h trend
            # If 1h Trend is UP, boost LONG signals
            # If 1h Trend is DOWN, penalize LONG signals
            
            trend_boost = 0.0
            if trend_1h == 'UP':
                trend_boost = 0.2  # Boost expected return by 0.2% (Stronger trend alignment)
                entry_threshold -= 0.05 # Lower threshold slightly
            elif trend_1h == 'DOWN':
                trend_boost = -0.1 # Penalize expected return
                entry_threshold += 0.05 # Raise threshold
            
            adjusted_return = ensemble_return + trend_boost
            
            # Entry: Use dynamic threshold based on market conditions
            if adjusted_return > entry_threshold:
                strength = min(adjusted_return / 2.0, 1.0)
                # Boost strength if aligned with trend
                if trend_1h == 'UP':
                    strength = min(strength + 0.2, 1.0)
                
                self.events_queue.put(SignalEvent(3, self.symbol, timestamp, 'LONG', strength=strength, atr=current_atr))
                print(f"  → LONG signal (strength={strength:.2f}, threshold={entry_threshold:.2f}, 1h_trend={trend_1h})")
            
            # Exit threshold: Expect loss > -0.5% (Relaxed from -0.3% for crypto volatility)
            elif adjusted_return < -0.5:
                # HOLDING PERIOD CHECK: Don't exit too early unless drop is severe
                # If we just entered (less than 3 bars ago), give it room to breathe
                # unless the drop is catastrophic (<-1.0%)
                bars_held = 0
                # We don't track holding time here directly, but we can infer or skip for now.
                # Ideally we'd check portfolio, but strategy is stateless regarding position duration.
                # We'll rely on the relaxed threshold (-0.5%) to handle noise.
                
                strength = min(abs(adjusted_return) / 2.0, 1.0)
                self.events_queue.put(SignalEvent(3, self.symbol, timestamp, 'EXIT', strength=strength, atr=current_atr))
                print(f"  → EXIT signal (strength={strength:.2f})")

