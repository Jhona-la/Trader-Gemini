"""
🧪 MULTI-HORIZON BACKTEST v2.0 - Trader Gemini
Ejecuta backtests para 1D, 7D, 15D y 30D en los símbolos principales.
Reporta: PNL, Win Rate, Max Drawdown, Sharpe Ratio por horizonte y estrategia.

ESTRATEGIAS EVALUADAS:
- Technical (HybridScalpingStrategy): Momentum + Mean Reversion
- Sophia Intelligence: KMeans clustering adaptativo REAL
- ML Strategy (XGBoost Ensemble): Walk-forward XGBoost REAL

FIXES v2.0:
- BUG-001 FIX: ML_XGBoost ahora usa modelo real XGBoost con walk-forward training
- BUG-002 FIX: Sophia ahora usa KMeans clustering real sobre features multivariate
- BUG-003 FIX: Circuit breaker (kill switch) integrado en backtest engine
- BUG-004 FIX: Sharpe ratio requiere mínimo 3 días de datos para cálculo válido
- COMISIÓN: Documentada correctamente (entry + exit = round-trip 0.075%)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from queue import Queue
from binance.client import Client
from config import Config
import time
import json
import warnings
warnings.filterwarnings('ignore')

# ML Imports for real XGBoost and KMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── CONFIG ──────────────────────────────────────────────────────────────────
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT']
HORIZONS = [1, 7, 15, 30]  # Días
INITIAL_CAPITAL = Config.INITIAL_CAPITAL  # Usa el capital real configurado
LEVERAGE = Config.BINANCE_LEVERAGE
COMMISSION_PCT = Config.BINANCE_TAKER_FEE_BNB  # 0.0375% per side (entry + exit = 0.075% round-trip)
RISK_PER_TRADE = Config.MAX_RISK_PER_TRADE  # 5%

# ── BUG-003 FIX: Circuit Breaker Config ─────────────────────────────────────
KILL_SWITCH_DD_PCT = 0.02    # 2% max drawdown kills trading for session
KILL_SWITCH_COOLDOWN = 120   # 120 bars (2h at 1m) cooldown after kill switch

print(f"💰 Capital inicial configurado: ${INITIAL_CAPITAL:.2f}")
print(f"⚡ Leverage: {LEVERAGE}x | Fee: {COMMISSION_PCT*100:.4f}% per side")
print(f"🛡️ Kill Switch @ {KILL_SWITCH_DD_PCT*100:.1f}% DD | Cooldown: {KILL_SWITCH_COOLDOWN} bars")
print(f"🎯 Símbolos: {SYMBOLS}")
print(f"📅 Horizontes: {HORIZONS} días\n")

# ── DATA FETCH ───────────────────────────────────────────────────────────────
def fetch_data(symbol: str, days: int) -> pd.DataFrame:
    """Descarga velas 1m de Binance Spot (público, sin API key)"""
    print(f"  📡 Descargando {days}d para {symbol}...")
    client = Client()  # Público
    binance_symbol = symbol.replace('/', '')
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    all_klines = []
    current_start = start_time
    while current_start < end_time:
        batch_end = min(current_start + timedelta(hours=16), end_time)
        try:
            klines = client.get_historical_klines(
                binance_symbol,
                Client.KLINE_INTERVAL_1MINUTE,
                str(int(current_start.timestamp() * 1000)),
                str(int(batch_end.timestamp() * 1000)),
                limit=1000
            )
            if not klines:
                break
            all_klines.extend(klines)
        except Exception as e:
            print(f"    ⚠️ Error batch: {e}")
        current_start += timedelta(hours=16)
        time.sleep(0.15)

    if not all_klines:
        return None
    
    df = pd.DataFrame(all_klines, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_vol','trades','tbbase','tbquote','ignore'
    ])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df = df[['open','high','low','close','volume']]
    # Deduplicar índices
    df = df[~df.index.duplicated(keep='first')]
    print(f"    ✅ {len(df)} velas ({len(df)/1440:.1f}D)")
    return df


# ── INDICATORS ───────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame, warmup=200) -> pd.DataFrame:
    """Calcula indicadores técnicos vectorizados"""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    vols = df['volume'].values
    n = len(closes)
    
    # RSI 14
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(com=13, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(com=13, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # EMA 20, 50, 200
    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(closes).ewm(span=200, adjust=False).mean().values
    
    # Bollinger Bands 20, 2
    rolling_mean = pd.Series(closes).rolling(20).mean().values
    rolling_std = pd.Series(closes).rolling(20).std().values
    bb_upper = rolling_mean + 2 * rolling_std
    bb_lower = rolling_mean - 2 * rolling_std
    
    # ATR 14
    tr = np.maximum(highs - lows, np.maximum(
        np.abs(highs - np.roll(closes, 1)),
        np.abs(lows - np.roll(closes, 1))
    ))
    atr = pd.Series(tr).ewm(com=13, adjust=False).mean().values
    
    # MACD 12/26/9
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
    macd = ema12 - ema26
    macd_sig = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_hist = macd - macd_sig
    
    # Volume MA
    vol_ma = pd.Series(vols).rolling(20).mean().values
    vol_ratio = np.where(vol_ma > 0, vols / vol_ma, 1.0)
    
    # ADX proxy (trend strength)
    dx = np.abs(ema20 - ema50) / np.where(ema50 > 0, ema50, 1.0) * 100
    adx = pd.Series(dx).rolling(14).mean().values
    
    # Momentum (Rate of Change 10)
    roc = np.zeros(n)
    for idx in range(10, n):
        if closes[idx-10] > 0:
            roc[idx] = (closes[idx] - closes[idx-10]) / closes[idx-10] * 100
    
    # Amihud Illiquidity (microstructure feature)
    returns = np.abs(np.diff(closes, prepend=closes[0]) / np.where(closes > 0, closes, 1.0))
    amihud = np.where(vols > 0, returns / vols, 0.0)
    amihud_ma = pd.Series(amihud).rolling(20).mean().values
    
    # High-Low Spread proxy
    hl_spread = (highs - lows) / np.where(closes > 0, closes, 1.0)
    
    # === MTF (Multi-Timeframe) Features ===
    close_series = pd.Series(closes, index=df.index)
    df_15m = close_series.resample('15min').last().dropna()
    ema20_15m = df_15m.ewm(span=20, adjust=False).mean()
    ema50_15m = df_15m.ewm(span=50, adjust=False).mean()
    
    df_1h = close_series.resample('1h').last().dropna()
    ema20_1h = df_1h.ewm(span=20, adjust=False).mean()
    ema50_1h = df_1h.ewm(span=50, adjust=False).mean()
    
    df_temp = pd.DataFrame(index=df.index)
    df_temp['ema20_15m'] = ema20_15m
    df_temp['ema50_15m'] = ema50_15m
    df_temp['ema20_1h'] = ema20_1h
    df_temp['ema50_1h'] = ema50_1h
    df_temp = df_temp.ffill().bfill()
    
    htf_trend_15m = np.where(df_temp['ema20_15m'] > df_temp['ema50_15m'], 1, -1)
    htf_trend_1h = np.where(df_temp['ema20_1h'] > df_temp['ema50_1h'], 1, -1)
    
    df2 = df.copy()
    df2['rsi'] = rsi
    df2['ema20'] = ema20
    df2['ema50'] = ema50
    df2['ema200'] = ema200
    df2['bb_upper'] = bb_upper
    df2['bb_lower'] = bb_lower
    df2['atr'] = atr
    df2['macd'] = macd
    df2['macd_sig'] = macd_sig
    df2['macd_hist'] = macd_hist
    df2['vol_ratio'] = vol_ratio
    df2['atr_pct'] = atr / closes
    df2['adx'] = adx
    df2['roc'] = roc
    df2['amihud'] = amihud_ma
    df2['hl_spread'] = hl_spread
    df2['htf_trend_15m'] = htf_trend_15m
    df2['htf_trend_1h'] = htf_trend_1h
    return df2


# ── STRATEGY SIGNALS ─────────────────────────────────────────────────────────
def signal_technical(row, prev_row, params) -> tuple:
    """
    HybridScalpingStrategy: Momentum + Mean Reversion
    Retorna: (direction: 'long'/'short'/None, sl_pct, tp_pct)
    """
    close = row['close']
    rsi = row['rsi']
    atr_pct = row['atr_pct']
    bb_upper = row['bb_upper']
    bb_lower = row['bb_lower']
    in_uptrend = row['ema20'] > row['ema50'] and close > row['ema200']
    in_downtrend = row['ema20'] < row['ema50'] and close < row['ema200']
    macd_hist = row['macd_hist']
    prev_hist = prev_row['macd_hist'] if prev_row is not None else 0
    macd_accel = abs(macd_hist) > abs(prev_hist)
    vol_ratio = row['vol_ratio']
    
    # Dynamic RSI levels (percentile-based)
    rsi_buy = params.get('rsi_buy', 32)
    rsi_sell = params.get('rsi_sell', 68)
    
    # Dynamic SL/TP based on ATR
    sl_pct = min(max(atr_pct * 1.8, 0.008), 0.025)
    tp_pct = min(max(atr_pct * 3.5, 0.015), 0.06)
    
    # Mean Reversion setups
    at_lower = close <= bb_lower
    at_upper = close >= bb_upper
    oversold = rsi < rsi_buy
    overbought = rsi > rsi_sell
    high_vol = vol_ratio > 1.0
    
    htf_aligned_bull = row.get('htf_trend_15m', 0) == 1 and row.get('htf_trend_1h', 0) == 1
    htf_aligned_bear = row.get('htf_trend_15m', 0) == -1 and row.get('htf_trend_1h', 0) == -1
    
    # Filtro Institucional: no abrir cortos si la macro es muy alcista (y viceversa)
    if at_lower and oversold and high_vol and (in_uptrend or rsi < 35):
        if htf_aligned_bear: return None, sl_pct, tp_pct # Evitar long en macro bajista
        tp_adj = tp_pct * 1.5 if htf_aligned_bull else tp_pct
        return 'long', sl_pct, tp_adj
    if at_upper and overbought and high_vol and (in_downtrend or rsi > 65):
        if htf_aligned_bull: return None, sl_pct, tp_pct # Evitar short en macro alcista
        tp_adj = tp_pct * 1.5 if htf_aligned_bear else tp_pct
        return 'short', sl_pct, tp_adj
    
    # Momentum setups
    if row['macd'] > row['macd_sig'] and macd_hist > 0 and macd_accel and in_uptrend and vol_ratio > 1.2:
        if htf_aligned_bear: return None, sl_pct, tp_pct
        tp_adj = (tp_pct * 2.5) if htf_aligned_bull else (tp_pct * 1.5) # Asymmetric Reward
        return 'long', sl_pct, tp_adj
    if row['macd'] < row['macd_sig'] and macd_hist < 0 and macd_accel and in_downtrend and vol_ratio > 1.2:
        if htf_aligned_bull: return None, sl_pct, tp_pct
        tp_adj = (tp_pct * 2.5) if htf_aligned_bear else (tp_pct * 1.5) # Asymmetric Reward
        return 'short', sl_pct, tp_adj
    
    return None, sl_pct, tp_pct


# ═══════════════════════════════════════════════════════════════════════════
# BUG-002 FIX: SOPHIA CLUSTER ENGINE (Real KMeans Clustering)
# ═══════════════════════════════════════════════════════════════════════════

class SophiaClusterEngine:
    """
    🧠 Sophia Intelligence v2.0 - Real KMeans Clustering
    
    QUÉ: Motor de clustering que clasifica el régimen de mercado usando KMeans
         sobre features multivariable (RSI, ATR%, volume ratio, EMA ratios).
    POR QUÉ: La versión anterior usaba if/else estáticos. KMeans detecta 
         patrones no lineales y se adapta a datos cambiantes.
    PARA QUÉ: Generar señales coherentes por régimen, adaptando SL/TP/umbral
         dinámicamente según el cluster actual.
    CÓMO: Re-fit KMeans cada N bars sobre ventana rolling de features.
         Mapea clusters a regímenes por centroid analysis.
    CUÁNDO: Cada 120 bars (~2h) o al inicializar.
    DÓNDE: run_multi_horizon_backtest.py → SophiaClusterEngine
    QUIÉN: signal_sophia() usa esta clase.
    """
    
    def __init__(self, n_clusters=4, refit_interval=120, window_size=500):
        self.n_clusters = n_clusters
        self.refit_interval = refit_interval
        self.window_size = window_size
        self.kmeans = None
        self.scaler = StandardScaler()
        self.feature_buffer = []
        self.bars_since_fit = 0
        self.is_fitted = False
        self.current_regime = 'RANGING'
        self.cluster_regime_map = {}  # cluster_id -> regime_name
        
    def _extract_features(self, row):
        """Extract clustering features from a data row"""
        return [
            row['rsi'],
            row['atr_pct'] * 100,  # Normalize to percentage
            min(row['vol_ratio'], 5.0),
            row['ema20'] / row['ema50'] if row['ema50'] > 0 else 1.0,
            row['macd_hist'] / row['close'] * 1000 if row['close'] > 0 else 0,
            row.get('adx', 20.0),
            row.get('roc', 0.0),
        ]
    
    def update(self, row):
        """Add new data point and re-cluster if needed"""
        features = self._extract_features(row)
        self.feature_buffer.append(features)
        
        # Keep rolling window
        if len(self.feature_buffer) > self.window_size:
            self.feature_buffer = self.feature_buffer[-self.window_size:]
        
        self.bars_since_fit += 1
        
        # Re-fit periodically
        if len(self.feature_buffer) >= 100 and (
            not self.is_fitted or self.bars_since_fit >= self.refit_interval
        ):
            self._fit()
            
        # Predict current regime
        if self.is_fitted:
            X = np.array([features])
            try:
                X_scaled = self.scaler.transform(X)
                cluster = self.kmeans.predict(X_scaled)[0]
                self.current_regime = self.cluster_regime_map.get(cluster, 'RANGING')
            except Exception:
                pass
    
    def _fit(self):
        """Fit KMeans on buffered features"""
        try:
            X = np.array(self.feature_buffer)
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.kmeans = KMeans(
                n_clusters=self.n_clusters, 
                n_init=10, 
                random_state=42,
                max_iter=100
            )
            self.kmeans.fit(X_scaled)
            
            # Map clusters to regimes by analyzing centroids
            self._map_clusters_to_regimes()
            
            self.is_fitted = True
            self.bars_since_fit = 0
        except Exception:
            pass
    
    def _map_clusters_to_regimes(self):
        """
        Map cluster centroids to market regimes based on feature values.
        Features: [RSI, ATR%, vol_ratio, ema_ratio, macd_norm, adx, roc]
        """
        centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.cluster_regime_map = {}
        
        for i, centroid in enumerate(centroids):
            rsi_c = centroid[0]
            atr_c = centroid[1]
            vol_c = centroid[2]
            ema_ratio_c = centroid[3]
            adx_c = centroid[5] if len(centroid) > 5 else 20.0
            roc_c = centroid[6] if len(centroid) > 6 else 0.0
            
            # High ADX + strong EMA ratio difference = trending
            if adx_c > 25 and abs(ema_ratio_c - 1.0) > 0.005:
                if ema_ratio_c > 1.0 and roc_c > 0:
                    regime = 'TRENDING_BULL'
                elif ema_ratio_c < 1.0 and roc_c < 0:
                    regime = 'TRENDING_BEAR'
                else:
                    regime = 'TRENDING_BULL' if rsi_c > 50 else 'TRENDING_BEAR'
            # High ATR = choppy/volatile
            elif atr_c > 1.5:
                regime = 'CHOPPY'
            # Low ADX + moderate ATR = ranging
            else:
                regime = 'RANGING'
            
            self.cluster_regime_map[i] = regime
    
    def generate_signal(self, row, prev_row) -> tuple:
        """Generate trading signal based on current cluster regime"""
        close = row['close']
        rsi = row['rsi']
        atr_pct = row['atr_pct']
        vol_ratio = row['vol_ratio']
        regime = self.current_regime
        
        htf_aligned_bull = row.get('htf_trend_15m', 0) == 1 and row.get('htf_trend_1h', 0) == 1
        htf_aligned_bear = row.get('htf_trend_15m', 0) == -1 and row.get('htf_trend_1h', 0) == -1

        if regime == 'TRENDING_BULL':
            # Only longs in bull trend — buy dips
            sl_pct = min(max(atr_pct * 2.2, 0.01), 0.03)
            tp_pct = min(max(atr_pct * 5.0, 0.025), 0.10)
            if htf_aligned_bull: tp_pct *= 1.3  # Asymmetric Reward
            rsi_threshold = 45 if vol_ratio > 1.0 else 40
            if rsi < rsi_threshold and vol_ratio > 0.8 and row['ema20'] > row['ema50']:
                if not htf_aligned_bear: return 'long', sl_pct, tp_pct
                
        elif regime == 'TRENDING_BEAR':
            # Only shorts in bear trend — sell rallies
            sl_pct = min(max(atr_pct * 2.2, 0.01), 0.03)
            tp_pct = min(max(atr_pct * 5.0, 0.025), 0.10)
            if htf_aligned_bear: tp_pct *= 1.3
            rsi_threshold = 55 if vol_ratio > 1.0 else 60
            if rsi > rsi_threshold and vol_ratio > 0.8 and row['ema20'] < row['ema50']:
                if not htf_aligned_bull: return 'short', sl_pct, tp_pct
                
        elif regime == 'CHOPPY':
            # Very conservative in choppy — only extreme signals
            sl_pct = min(max(atr_pct * 1.2, 0.006), 0.015)
            tp_pct = min(max(atr_pct * 2.0, 0.010), 0.025)
            if rsi < 22 and close <= row['bb_lower'] and vol_ratio > 1.5:
                if not htf_aligned_bear: return 'long', sl_pct, tp_pct
            if rsi > 78 and close >= row['bb_upper'] and vol_ratio > 1.5:
                if not htf_aligned_bull: return 'short', sl_pct, tp_pct
                
        else:  # RANGING
            # Mean reversion in ranging
            sl_pct = min(max(atr_pct * 1.5, 0.008), 0.020)
            tp_pct = min(max(atr_pct * 2.5, 0.012), 0.040)
            if rsi < 28 and close <= row['bb_lower']:
                if not htf_aligned_bear: return 'long', sl_pct, tp_pct
            if rsi > 72 and close >= row['bb_upper']:
                if not htf_aligned_bull: return 'short', sl_pct, tp_pct
        
        return None, 0.015, 0.03


# ═══════════════════════════════════════════════════════════════════════════
# BUG-001 FIX: WALK-FORWARD XGBOOST (Real ML Model)
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardXGBoost:
    """
    🤖 Walk-Forward XGBoost v2.0 - Real ML Model
    
    QUÉ: Modelo XGBoost real que se entrena sobre datos históricos y predice 
         la dirección del precio con walk-forward validation.
    POR QUÉ: La versión anterior usaba scores lineales hardcodeados (BUG-001).
         Un modelo XGBoost real captura relaciones no lineales entre features.
    PARA QUÉ: Generar señales con probabilidades calibradas, permitiendo
         threshold adaptativo y confidence-based sizing.
    CÓMO: Train on [0:train_end], predict on [train_end:train_end+retrain_interval].
         Retrain every retrain_interval bars. Labels = forward return sign.
    CUÁNDO: Training at warmup, retraining every 1440 bars (~1 day).
    DÓNDE: run_multi_horizon_backtest.py → WalkForwardXGBoost
    QUIÉN: signal_xgboost_ml() usa esta clase.
    """
    
    FEATURE_COLS = [
        'rsi', 'atr_pct', 'vol_ratio', 'macd', 'macd_sig', 'macd_hist',
        'adx', 'roc', 'amihud', 'hl_spread'
    ]
    
    def __init__(self, retrain_interval=1440, min_train_size=500, 
                 lookahead=30, threshold=0.58):
        self.retrain_interval = retrain_interval
        self.min_train_size = min_train_size
        self.lookahead = lookahead
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.bars_since_train = 0
        self.train_accuracy = 0.0
        self.n_trainings = 0
        
    def _build_features(self, df_slice):
        """Build feature matrix from dataframe slice"""
        features = {}
        closes = df_slice['close'].values
        
        features['rsi'] = df_slice['rsi'].values
        features['atr_pct'] = df_slice['atr_pct'].values
        features['vol_ratio'] = np.clip(df_slice['vol_ratio'].values, 0, 5)
        features['macd'] = df_slice['macd'].values / np.where(closes > 0, closes, 1) * 1000
        features['macd_sig'] = df_slice['macd_sig'].values / np.where(closes > 0, closes, 1) * 1000
        features['macd_hist'] = df_slice['macd_hist'].values / np.where(closes > 0, closes, 1) * 1000
        features['adx'] = df_slice['adx'].values if 'adx' in df_slice else np.full(len(df_slice), 20.0)
        features['roc'] = df_slice['roc'].values if 'roc' in df_slice else np.zeros(len(df_slice))
        features['amihud'] = df_slice['amihud'].values if 'amihud' in df_slice else np.zeros(len(df_slice))
        features['hl_spread'] = df_slice['hl_spread'].values if 'hl_spread' in df_slice else np.zeros(len(df_slice))
        
        # BB position (normalized 0-1)
        bb_range = df_slice['bb_upper'].values - df_slice['bb_lower'].values
        features['bb_position'] = np.where(
            bb_range > 0, 
            (closes - df_slice['bb_lower'].values) / bb_range, 
            0.5
        )
        
        # EMA ratio
        features['ema_ratio'] = df_slice['ema20'].values / np.where(
            df_slice['ema50'].values > 0, df_slice['ema50'].values, 1.0
        )
        
        # Trend alignment
        features['trend_align'] = np.where(
            (df_slice['ema20'].values > df_slice['ema50'].values) & (closes > df_slice['ema200'].values),
            1.0,
            np.where(
                (df_slice['ema20'].values < df_slice['ema50'].values) & (closes < df_slice['ema200'].values),
                -1.0,
                0.0
            )
        )
        
        X = pd.DataFrame(features)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X
    
    def _create_labels(self, df_slice):
        """Create labels: 1 = price goes up, 0 = price goes down"""
        closes = df_slice['close'].values
        labels = np.zeros(len(closes))
        
        for i in range(len(closes) - self.lookahead):
            future_max = np.max(closes[i+1:i+1+self.lookahead])
            future_min = np.min(closes[i+1:i+1+self.lookahead])
            
            up_pct = (future_max - closes[i]) / closes[i]
            down_pct = (closes[i] - future_min) / closes[i]
            
            if up_pct > down_pct and up_pct > 0.003:  # 0.3% threshold
                # Phase 6.3: Drawdown Penalty (Reward Shaping) para Evitar Whipsaw en Lado Comprador
                drawdown_stress = down_pct / up_pct if up_pct > 0 else 0
                labels[i] = 1 if drawdown_stress < 0.75 else 0
            elif down_pct > up_pct and down_pct > 0.003:
                # Phase 6.3: Drawdown Penalty (Reward Shaping) Lado Vendedor
                drawdown_stress = up_pct / down_pct if down_pct > 0 else 0
                labels[i] = -1 if drawdown_stress < 0.75 else 0
            else:
                labels[i] = 0
        
        return labels
    
    def train(self, df_train):
        """Train XGBoost on training data"""
        if len(df_train) < self.min_train_size:
            return False
            
        try:
            X = self._build_features(df_train)
            y = self._create_labels(df_train)
            
            # Remove last lookahead rows (no valid labels)
            valid = len(X) - self.lookahead
            X = X.iloc[:valid]
            y = y[:valid]
            
            # Remove neutral labels for binary classifier
            mask = y != 0
            X = X[mask]
            y = y[mask]
            y = (y > 0).astype(int)  # 1 = bullish, 0 = bearish
            
            if len(X) < 50 or len(np.unique(y)) < 2:
                return False
            
            # Fit scaler
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train XGBoost with anti-overfit params
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
                random_state=42
            )
            
            # Time-series split for validation
            split_idx = int(len(X_scaled) * 0.8)
            X_t, X_v = X_scaled[:split_idx], X_scaled[split_idx:]
            y_t, y_v = y[:split_idx], y[split_idx:]
            
            # --- EVOLUTIONARY TIME DECAY (Logarithmic) ---
            # Recent data gets higher weight (1.0), oldest data gets lower weight
            n_samples = len(X_t)
            if n_samples > 0:
                time_decay_weights = np.linspace(0.1, 1.0, n_samples)
                sample_weights = np.exp(time_decay_weights * 2) / np.exp(2)
                self.model.fit(X_t, y_t, sample_weight=sample_weights)
            else:
                self.model.fit(X_t, y_t)
            
            # Validation accuracy
            if len(X_v) > 0:
                self.train_accuracy = self.model.score(X_v, y_v)
            else:
                self.train_accuracy = self.model.score(X_t, y_t)
            
            self.is_trained = True
            self.bars_since_train = 0
            self.n_trainings += 1
            
            return True
            
        except Exception as e:
            return False
    
    def predict(self, df_window) -> tuple:
        """Predict direction with confidence"""
        if not self.is_trained or self.model is None:
            return None, 0.015, 0.03
        
        self.bars_since_train += 1
        
        try:
            X = self._build_features(df_window)
            X_last = X.iloc[[-1]]
            X_scaled = self.scaler.transform(X_last)
            
            proba = self.model.predict_proba(X_scaled)[0]
            
            # proba[0] = P(bearish), proba[1] = P(bullish)
            p_bull = proba[1] if len(proba) > 1 else 0.5
            p_bear = proba[0] if len(proba) > 1 else 0.5
            
            row = df_window.iloc[-1]
            atr_pct = row['atr_pct']
            
            # Dynamic SL/TP
            sl_pct = min(max(atr_pct * 2.0, 0.01), 0.025)
            tp_pct = min(max(atr_pct * 4.0, 0.02), 0.08)
            
            # Adaptive threshold based on train accuracy
            effective_threshold = self.threshold
            if self.train_accuracy < 0.52:
                effective_threshold += 0.05  # More conservative if model is weak
                
            # --- Dynamic Sigmoid Thresholding by Volatility ---
            # Si el mercado está muy violento (ATR alto), subimos el listón con una sigmoide
            import math
            # Asumimos que la volatilidad normal/crítica cruza al 3.5% (0.035) de ATR
            vol_penalty = 0.15 / (1.0 + math.exp(-150.0 * (atr_pct - 0.035)))
            effective_threshold = min(0.85, effective_threshold + vol_penalty)
            
            if p_bull > effective_threshold:
                return 'long', sl_pct, tp_pct
            elif p_bear > effective_threshold:
                return 'short', sl_pct, tp_pct
            
            return None, sl_pct, tp_pct
            
        except Exception:
            return None, 0.015, 0.03
    
    def should_retrain(self):
        """Check if model should be retrained"""
        return self.bars_since_train >= self.retrain_interval


# ── REGIME DETECTOR (Enhanced) ────────────────────────────────────────────────
def detect_regime(df_window) -> str:
    """Classify market regime based on EMA slope, ADX proxy, and volatility"""
    if len(df_window) < 50:
        return 'RANGING'
    closes = df_window['close'].values[-50:]
    ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values
    
    slope20 = (ema20[-1] - ema20[-20]) / ema20[-20] if ema20[-20] > 0 else 0
    slope50 = (ema50[-1] - ema50[-20]) / ema50[-20] if ema50[-20] > 0 else 0
    
    volatility = np.std(np.diff(closes) / closes[:-1])
    
    if slope20 > 0.005 and slope50 > 0 and ema20[-1] > ema50[-1]:
        return 'TRENDING_BULL'
    elif slope20 < -0.005 and slope50 < 0 and ema20[-1] < ema50[-1]:
        return 'TRENDING_BEAR'
    elif volatility > 0.008:
        return 'CHOPPY'
    else:
        return 'RANGING'


# ── BACKTEST ENGINE (Enhanced with Circuit Breaker) ──────────────────────────
def run_strategy_backtest(df: pd.DataFrame, symbol: str, strategy_name: str, 
                          initial_capital: float, leverage: int) -> dict:
    """
    Motor de backtest unificado para las 3 estrategias.
    v2.0: Incluye circuit breaker, ML real, clustering real.
    """
    df = df.copy()
    df = compute_indicators(df)
    df = df.dropna()
    
    capital = initial_capital
    peak = initial_capital
    max_dd = 0.0
    position = None  # {'side', 'entry', 'sl', 'tp', 'size_usd', 'entry_idx'}
    trades = []
    equity_curve = [capital]
    
    # Warming RSI percentiles for dynamic levels
    rsi_window = []
    
    # BUG-002 FIX: Real Sophia Cluster Engine
    sophia_engine = SophiaClusterEngine(n_clusters=4, refit_interval=120) if strategy_name == 'Sophia' else None
    
    # BUG-001 FIX: Real Walk-Forward XGBoost
    xgb_engine = WalkForwardXGBoost(retrain_interval=1440, min_train_size=500) if strategy_name == 'ML_XGBoost' else None
    
    # ML state
    ml_state = {}
    
    # BUG-003 FIX: Circuit Breaker State
    kill_switch_active = False
    kill_switch_bar = 0
    kill_switch_triggers = 0
    
    warmup = 200
    cooldown_bars = {}  # {symbol: last_loss_bar}
    COOLDOWN = 30  # bars (= 30 min at 1m)
    
    rows = df.reset_index()
    total = len(rows)
    
    for i in range(warmup, total):
        row = rows.iloc[i]
        prev_row = rows.iloc[i-1] if i > 0 else None
        close = row['close']
        high = row['high']
        low = row['low']
        ts = row['datetime'] if 'datetime' in row else row.name
        
        # Update RSI window for dynamic levels
        rsi_window.append(row['rsi'])
        if len(rsi_window) > 200:
            rsi_window.pop(0)
        
        # Update Sophia clustering
        if sophia_engine is not None:
            sophia_engine.update(row)
        
        # ── BUG-003 FIX: CIRCUIT BREAKER CHECK ──────────────────────────
        if kill_switch_active:
            if (i - kill_switch_bar) >= KILL_SWITCH_COOLDOWN:
                kill_switch_active = False
                # Reset peak to current capital after recovery
                peak = capital
            else:
                equity_curve.append(capital)
                continue  # Skip trading during kill switch
        
        # Check if drawdown triggers kill switch
        current_dd = (peak - capital) / peak if peak > 0 else 0
        if current_dd >= KILL_SWITCH_DD_PCT and position is None:
            kill_switch_active = True
            kill_switch_bar = i
            kill_switch_triggers += 1
            equity_curve.append(capital)
            continue
        
        # ── CHECK EXITS (SL/TP with breakeven) ──────────────────────────────
        if position is not None:
            side = position['side']
            sl = position['sl']
            tp = position['tp']
            entry = position['entry']
            size_usd = position['size_usd']
            
            # Breakeven at 80% toward TP
            if side == 'long':
                tp_dist = tp - entry
                be_target = entry + tp_dist * 0.80
                if high >= be_target and sl < entry:
                    position['sl'] = entry * 1.0005  # mini-buffer
                    sl = position['sl']
                # Check exits
                if low <= sl:
                    pnl_pct = (sl - entry) / entry
                    pnl_usd = size_usd * pnl_pct
                    # Exit commission only (entry was already charged)
                    pnl_usd -= size_usd * COMMISSION_PCT
                    capital += pnl_usd
                    trades.append({'pnl_usd': pnl_usd, 'pnl_pct': pnl_pct*100, 
                                   'exit': 'SL', 'side': side, 'bars_held': i - position['entry_idx']})
                    if pnl_usd < 0:
                        cooldown_bars[symbol] = i
                    position = None
                elif high >= tp:
                    pnl_pct = (tp - entry) / entry
                    pnl_usd = size_usd * pnl_pct
                    pnl_usd -= size_usd * COMMISSION_PCT
                    capital += pnl_usd
                    trades.append({'pnl_usd': pnl_usd, 'pnl_pct': pnl_pct*100,
                                   'exit': 'TP', 'side': side, 'bars_held': i - position['entry_idx']})
                    position = None
            else:  # short
                tp_dist = entry - tp
                be_target = entry - tp_dist * 0.80
                if low <= be_target and sl > entry:
                    position['sl'] = entry * 0.9995
                    sl = position['sl']
                if high >= sl:
                    pnl_pct = (entry - sl) / entry
                    pnl_usd = size_usd * pnl_pct
                    pnl_usd -= size_usd * COMMISSION_PCT
                    capital += pnl_usd
                    trades.append({'pnl_usd': pnl_usd, 'pnl_pct': pnl_pct*100,
                                   'exit': 'SL', 'side': side, 'bars_held': i - position['entry_idx']})
                    if pnl_usd < 0:
                        cooldown_bars[symbol] = i
                    position = None
                elif low <= tp:
                    pnl_pct = (entry - tp) / entry
                    pnl_usd = size_usd * pnl_pct
                    pnl_usd -= size_usd * COMMISSION_PCT
                    capital += pnl_usd
                    trades.append({'pnl_usd': pnl_usd, 'pnl_pct': pnl_pct*100,
                                   'exit': 'TP', 'side': side, 'bars_held': i - position['entry_idx']})
                    position = None
        
        # Update equity/drawdown
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        equity_curve.append(capital)
        
        # ── ENTRY SIGNALS ────────────────────────────────────────────────────
        if position is None:
            # Cooldown check
            if symbol in cooldown_bars and (i - cooldown_bars[symbol]) < COOLDOWN:
                continue
            
            # Dynamic RSI percentile levels
            if len(rsi_window) >= 50:
                rsi_buy = max(20, min(np.percentile(rsi_window, 15), 40))
                rsi_sell = min(80, max(np.percentile(rsi_window, 85), 60))
                params = {'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell}
            else:
                params = {'rsi_buy': 30, 'rsi_sell': 70}
            
            # Select strategy
            if strategy_name == 'Technical':
                direction, sl_pct, tp_pct = signal_technical(row, prev_row, params)
                
            elif strategy_name == 'Sophia':
                # BUG-002 FIX: Use real clustering engine
                direction, sl_pct, tp_pct = sophia_engine.generate_signal(row, prev_row)
                
            elif strategy_name == 'ML_XGBoost':
                # BUG-001 FIX: Use real XGBoost model
                if xgb_engine is not None:
                    # Train/retrain if needed
                    if not xgb_engine.is_trained or xgb_engine.should_retrain():
                        train_start = max(0, i - xgb_engine.min_train_size - 100)
                        train_df = df.iloc[train_start:i]
                        if xgb_engine.train(train_df):
                            pass  # Training succeeded silently
                    
                    # Predict
                    pred_start = max(0, i - 50)
                    bars_window = df.iloc[pred_start:i+1]
                    direction, sl_pct, tp_pct = xgb_engine.predict(bars_window)
                else:
                    direction = None
                    sl_pct, tp_pct = 0.015, 0.03
            else:
                direction = None
            
            if direction is not None:
                # Kelly-based sizing (simplified fractional Kelly)
                if len(trades) >= 10:
                    wins = [t['pnl_usd'] for t in trades[-20:] if t['pnl_usd'] > 0]
                    losses = [abs(t['pnl_usd']) for t in trades[-20:] if t['pnl_usd'] <= 0]
                    p = len(wins) / 20 if len(trades) >= 20 else 0.52
                    avg_w = np.mean(wins) if wins else 0.01
                    avg_l = np.mean(losses) if losses else 0.01
                    b = avg_w / avg_l if avg_l > 0 else 1.0
                    kelly_raw = (p * b - (1 - p)) / b if b > 0 else 0
                    size_pct = max(0.05, min(kelly_raw * 0.25, 0.35))  # Quarter-Kelly, max 35%
                else:
                    size_pct = RISK_PER_TRADE
                
                # Apply leverage and position
                notional = capital * leverage * size_pct
                size_usd = min(notional, capital * leverage)  # Cap at full leverage
                
                # Entry commission
                commission = size_usd * COMMISSION_PCT
                capital -= commission
                
                if direction == 'long':
                    sl_price = close * (1 - sl_pct)
                    tp_price = close * (1 + tp_pct)
                    position = {
                        'side': 'long', 'entry': close, 'sl': sl_price, 
                        'tp': tp_price, 'size_usd': size_usd, 'entry_idx': i
                    }
                else:
                    sl_price = close * (1 + sl_pct)
                    tp_price = close * (1 - tp_pct)
                    position = {
                        'side': 'short', 'entry': close, 'sl': sl_price,
                        'tp': tp_price, 'size_usd': size_usd, 'entry_idx': i
                    }
    
    # Close any open position at last bar
    if position is not None:
        last_row = rows.iloc[-1]
        last_price = last_row['close']
        side = position['side']
        entry = position['entry']
        size_usd = position['size_usd']
        if side == 'long':
            pnl_pct = (last_price - entry) / entry
        else:
            pnl_pct = (entry - last_price) / entry
        pnl_usd = size_usd * pnl_pct - size_usd * COMMISSION_PCT
        capital += pnl_usd
        trades.append({'pnl_usd': pnl_usd, 'pnl_pct': pnl_pct*100, 
                       'exit': 'EOD', 'side': side, 'bars_held': total - position['entry_idx']})
    
    # ── METRICS ─────────────────────────────────────────────────────────────
    total_trades = len(trades)
    if total_trades == 0:
        return {
            'trades': 0, 'pnl_usd': 0, 'pnl_pct': 0,
            'win_rate': 0, 'max_drawdown': max_dd * 100,
            'sharpe': 0, 'sortino': 0, 'avg_trade_bars': 0,
            'profit_factor': 0, 'capital_final': capital,
            'kill_switch_triggers': kill_switch_triggers
        }
    
    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]
    total_pnl = sum(t['pnl_usd'] for t in trades)
    win_rate = len(wins) / total_trades
    
    # ── BUG-004 FIX: Annualized Sharpe (require minimum 3 daily data points) ──
    daily_returns = []
    chunk = 1440  # 1 day = 1440 minutes
    for j in range(0, len(equity_curve)-1, chunk):
        chunk_eq = equity_curve[j:j+chunk+1]
        if len(chunk_eq) > 1 and chunk_eq[0] > 0:
            daily_ret = (chunk_eq[-1] - chunk_eq[0]) / chunk_eq[0]
            daily_returns.append(daily_ret)
    
    # FIX: Require at least 3 daily returns for statistically meaningful Sharpe
    if len(daily_returns) >= 3:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)  # Use Bessel's correction
        rf_daily = 0.02 / 365  # Risk-free daily
        sharpe = (mean_ret - rf_daily) / std_ret * np.sqrt(365) if std_ret > 1e-10 else 0
        # Cap extreme values to avoid misleading results
        sharpe = np.clip(sharpe, -50, 50)
        
        # Sortino  
        negative_rets = [r for r in daily_returns if r < rf_daily]
        downside_std = np.std(negative_rets, ddof=1) if len(negative_rets) >= 2 else std_ret
        sortino = (mean_ret - rf_daily) / downside_std * np.sqrt(365) if downside_std > 1e-10 else 0
        sortino = np.clip(sortino, -50, 50)
    elif len(daily_returns) >= 1:
        # For 1-2 day horizons: report simple daily return, no annualization
        mean_ret = np.mean(daily_returns)
        sharpe = mean_ret / 0.001 if abs(mean_ret) > 0.0001 else 0  # Simple signal-to-noise
        sharpe = np.clip(sharpe, -10, 10)
        sortino = sharpe  # Same for insufficient data
    else:
        sharpe = 0
        sortino = 0
    
    # Profit Factor
    gross_win = sum(t['pnl_usd'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl_usd'] for t in losses)) if losses else 1
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0
    
    avg_bars = np.mean([t['bars_held'] for t in trades]) if trades else 0
    
    return {
        'trades': total_trades,
        'wins': len(wins),
        'losses': len(losses),
        'pnl_usd': round(total_pnl, 4),
        'pnl_pct': round((capital - initial_capital) / initial_capital * 100, 3),
        'win_rate': round(win_rate * 100, 2),
        'max_drawdown': round(max_dd * 100, 3),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'avg_trade_bars': round(avg_bars, 1),
        'capital_final': round(capital, 4),
        'kill_switch_triggers': kill_switch_triggers,
        'ml_trainings': xgb_engine.n_trainings if xgb_engine else 0,
        'ml_accuracy': round(xgb_engine.train_accuracy * 100, 1) if xgb_engine and xgb_engine.is_trained else 0,
        'equity_curve': equity_curve
    }


# ═══════════════════════════════════════════════════════════════════════════
# EVOLUTIONARY ADAPTIVE ORCHESTRATOR v3.0
# ═══════════════════════════════════════════════════════════════════════════

class AntiWhipsawOrchestrator:
    """
    🧬 Evolutionary Adaptive Orchestrator v3.0

    QUÉ: Meta-orquestador que se ADAPTA y EVOLUCIONA en tiempo real.
         Ningún parámetro es fijo — todos mutan en respuesta al mercado.

    POR QUÉ: Un sistema con hiperparámetros estáticos en crypto (que cambia
         de régimen en horas) inevitablemente fallará cuando el mercado
         encuentre la condición para la que sus parámetros son subóptimos.

    PARA QUÉ: Un bot que nunca falle: que en TRENDING concentre (baja T),
         que en CHOPPY diversifique (sube T), que en volatilidad alta
         responda rápido (sube alpha), que se auto-corrija si su Sharpe baja.

    CÓMO: 6 capas de adaptación evolutiva:
         [1] Volatility-Adaptive Alpha  → EMA reacciona más rápido si el mercado
                                          se mueve rápido (GARCH-inspired)
         [2] DD-Penalty Adaptativo      → λ sube en CHOPPY, baja en TRENDING
         [3] Regime-Aware Temperature   → Softmax + concentrado en trending,
                                          + diversificado en choppy
         [4] Adaptive Cooldown          → Rebalanceo más frecuente si Sharpe cae
         [5] Self-Calibrating Sharpe    → Mide su propio Sharpe rolling (7d)
                                          y ajusta parámetros para mejorarlo
         [6] Regime Detector puro-numpy → Detecta régimen desde retornos sin
                                          depender de talib (portable al backtest)

    CUÁNDO: Activo desde la barra post-warmup. Los parámetros evolucionan
            cada `adaptation_interval` barras (default: cada hora = 60 barras).
    DÓNDE:  run_multi_horizon_backtest.py → llamado desde main()
    QUIÉN:  main() lo instancia y llama a orchestrator.run(eq_tech, eq_soph, eq_xgb)
    """

    # ── Tabla de parámetros por régimen ──────────────────────────────────────
    # Cada régimen define la "personalidad" del orquestador para ese estado
    # QUÉ: lookup table que codifica el conocimiento de qué funciona en cada régimen
    # POR QUÉ: evita if/elif complejos y facilita extensión a nuevos regímenes
    REGIME_PARAMS = {
        #                alpha   lambda  temp    cooldown_mult
        'TRENDING_BULL': (0.08,  1.5,    0.05,   0.7),   # rápido, poco DD-penalty, concentrado, rebalanceo más rápido
        'TRENDING_BEAR': (0.10,  5.0,    0.12,   1.5),   # muy rápido, penaliza fuerte, diversifica, cooldown largo
        'RANGING':       (0.04,  2.5,    0.09,   1.0),   # lento, penalidad moderada, semi-concentrado
        'CHOPPY':        (0.03,  6.0,    0.18,   2.0),   # muy lento, penalidad máxima, muy diversificado, largo cooldown
        'MEAN_REVERTING':(0.05,  3.0,    0.10,   1.0),   # valores medios equilibrados
        'UNKNOWN':       (0.05,  3.0,    0.08,   1.0),   # defaults de seguridad
    }

    def __init__(
        self,
        # Valores BASE — el sistema los deriva como punto de partida, luego los adapta
        ema_alpha: float = 0.05,
        dd_penalty_lambda: float = 3.0,
        softmax_temperature: float = 0.08,
        rebalance_cooldown: int = 240,        # barras base (1m → 4h)
        min_warmup_bars: int = 2880,
        adaptation_interval: int = 60,        # cada 60 barras (~1h) adaptar parámetros
        sharpe_target: float = 1.5,           # Sharpe objetivo rolling para auto-calibración
        volatility_window: int = 60,          # ventana para medir vol instantánea (60 min)
        regime_window: int = 120,             # ventana para detectar régimen (2h de barras 1m)
    ):
        # Parámetros base (ancla de partida)
        self._base_alpha = ema_alpha
        self._base_lambda = dd_penalty_lambda
        self._base_temp = softmax_temperature
        self._base_cooldown = rebalance_cooldown
        self.warmup = min_warmup_bars
        self.adaptation_interval = adaptation_interval
        self.sharpe_target = sharpe_target
        self.vol_window = volatility_window
        self.regime_window = regime_window

        # Parámetros ACTUALES (se adaptan en runtime)
        self.alpha = ema_alpha
        self.dd_lambda = dd_penalty_lambda
        self.temperature = softmax_temperature
        self.cooldown = rebalance_cooldown

        # Estado interno
        self._ema_scores = None
        self._weights = None
        self._last_rebalance = 0
        self._last_adaptation = 0
        self._peak = {0: 0.0, 1: 0.0, 2: 0.0}
        self._current_regime = 'UNKNOWN'

        # Historial para auto-calibración
        self._meta_eq_window = []       # equity del portfolio últimas N barras
        self._rolling_sharpe = 0.0
        self._adaptation_log = []       # registro de cada adaptación (para análisis)

    # ── CAPA 6: Detección de Régimen (pure numpy, sin talib) ─────────────────
    def _detect_regime(self, portfolio_eq: list, idx: int) -> str:
        """
        QUÉ: Clasifica el régimen del portfolio combinado directamente desde
             su equity curve, sin necesitar datos OHLCV ni talib.
        POR QUÉ: El backtest no tiene acceso a bars OHLCV durante el META-PASS.
                 El régimen del portfolio es lo que realmente importa al orquestador.
        CÓMO:
          - Volatilidad reciente (std de retornos de 60 barras) → CHOPPY si alta
          - Pendiente EMA20 vs EMA50 de la equity curve → TRENDING si divergen
          - Ratio momentum 7d/2d → MEAN_REVERTING si fuerte reversión
        """
        start = max(0, idx - self.regime_window)
        if idx - start < 30:
            return 'UNKNOWN'

        eq_slice = np.array(portfolio_eq[start:idx + 1])
        if len(eq_slice) < 20 or eq_slice[0] <= 0:
            return 'UNKNOWN'

        rets = np.diff(eq_slice) / np.where(eq_slice[:-1] > 0, eq_slice[:-1], 1.0)
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

        # Volatilidad instantánea (últimas 60 barras vs ventana completa)
        vol_recent = np.std(rets[-min(60, len(rets)):]) if len(rets) >= 5 else 0.0
        vol_global = np.std(rets) if len(rets) >= 5 else 0.0

        # Proxy ADX: diferencia entre EMA20 y EMA50 de la equity
        if len(eq_slice) >= 50:
            ema20 = pd.Series(eq_slice).ewm(span=20, adjust=False).mean().values
            ema50 = pd.Series(eq_slice).ewm(span=50, adjust=False).mean().values
            ema_diff_pct = abs(ema20[-1] - ema50[-1]) / (ema50[-1] if ema50[-1] > 0 else 1.0)
            slope_20 = (ema20[-1] - ema20[-min(20, len(ema20))]) / (ema20[-min(20, len(ema20))] if ema20[-min(20, len(ema20))] > 0 else 1.0)
            trending_bull = ema20[-1] > ema50[-1] and slope_20 > 0.002
            trending_bear = ema20[-1] < ema50[-1] and slope_20 < -0.002
        else:
            ema_diff_pct = 0.0
            trending_bull = False
            trending_bear = False

        # Ratio momentum: si el mercado revirtió fuerte en las últimas 2d
        momentum_7d = float(np.sum(rets[-min(10080, len(rets)):]))  # 7d ~ 10080 min pero usamos lo disponible
        momentum_2d = float(np.sum(rets[-min(2880, len(rets)):]))

        # Clasificación
        if vol_recent > 2.5 * vol_global and vol_global > 1e-6:
            return 'CHOPPY'
        if trending_bull and ema_diff_pct > 0.003:
            return 'TRENDING_BULL'
        if trending_bear and ema_diff_pct > 0.003:
            return 'TRENDING_BEAR'
        if abs(momentum_2d) > 0 and abs(momentum_7d) > 0:
            # Si el momentum de corto plazo es de signo contrario al largo → mean-reversion
            if np.sign(momentum_2d) != np.sign(momentum_7d) and abs(momentum_2d) > 0.01:
                return 'MEAN_REVERTING'
        return 'RANGING'

    # ── CAPA 1: Volatility-Adaptive Alpha ────────────────────────────────────
    def _adaptive_alpha(self, rets_portfolio: np.ndarray) -> float:
        """
        QUÉ: Ajusta el alpha del EMA en función de la volatilidad ACTUAL.
        POR QUÉ: En alta volatilidad el mercado cambia rápido → el EMA debe
                 responder más rápido (alpha alto). En calma, responder lento
                 (alpha bajo) para no reaccionar a ruido.
        CÓMO: GARCH-inspired: alpha ∝ volatilidad_reciente / volatilidad_media.
              Bounded en [alpha_base/2, alpha_base*4].
        """
        if len(rets_portfolio) < self.vol_window:
            return self._base_alpha
        recent_vol = np.std(rets_portfolio[-self.vol_window:])
        global_vol = np.std(rets_portfolio) if len(rets_portfolio) > self.vol_window else recent_vol
        if global_vol < 1e-10:
            return self._base_alpha
        vol_ratio = recent_vol / global_vol
        # Alpha adaptativo: cuanto más volátil, más rápido responde
        adaptive = self._base_alpha * vol_ratio
        return float(np.clip(adaptive, self._base_alpha * 0.4, self._base_alpha * 4.0))

    # ── CAPA 5: Self-Calibrating Sharpe ──────────────────────────────────────
    def _rolling_sharpe_7d(self) -> float:
        """
        QUÉ: Calcula el Sharpe del propio portfolio en los últimos 7 días.
        POR QUÉ: Si el Sharpe propio cae, el orquestador sabe que sus parámetros
                 actuales no son óptimos para el estado del mercado actual.
        """
        if len(self._meta_eq_window) < 1440 * 2:
            return 0.0
        eq = np.array(self._meta_eq_window[-1440 * 7:])
        if len(eq) < 2 or eq[0] <= 0:
            return 0.0
        # Retornos diarios
        daily_rets = []
        for j in range(0, len(eq) - 1, 1440):
            chunk = eq[j:j + 1441]
            if len(chunk) > 1 and chunk[0] > 0:
                daily_rets.append((chunk[-1] - chunk[0]) / chunk[0])
        if len(daily_rets) < 2:
            return 0.0
        mu = np.mean(daily_rets)
        sigma = np.std(daily_rets, ddof=1)
        return float(np.clip(mu / sigma * np.sqrt(365), -10, 10)) if sigma > 1e-10 else 0.0

    # ── CAPAS 1-5: Adaptación Maestra (llamada cada adaptation_interval) ─────
    def _adapt_parameters(self, portfolio_eq: list, idx: int, rets_portfolio: np.ndarray):
        """
        QUÉ: El cerebro adaptativo del orquestador. Actualiza TODOS los parámetros
             en función del régimen, volatilidad y Sharpe propio.

        PROTOCOLO DE ADAPTACIÓN (por orden de prioridad):
          1. Detectar régimen actual → obtener parámetros de REGIME_PARAMS
          2. Aplicar multiplicador de volatilidad sobre alpha (Capa 1)
          3. Aplicar auto-corrección de Sharpe: si Sharpe < target → diversificar más
          4. Loguear el estado para análisis posterior
        """
        # — Paso 1: Régimen y lookup de parámetros base por régimen —
        regime = self._detect_regime(portfolio_eq, idx)
        self._current_regime = regime
        r_alpha, r_lambda, r_temp, r_cooldown_mult = self.REGIME_PARAMS.get(
            regime, self.REGIME_PARAMS['UNKNOWN']
        )

        # — Paso 2: Volatility-Adaptive Alpha (Capa 1) —
        vol_alpha = self._adaptive_alpha(rets_portfolio)
        # Blending: 50% régimen, 50% volatilidad dinámica
        self.alpha = float(np.clip(
            0.5 * r_alpha + 0.5 * vol_alpha,
            0.01, 0.25
        ))

        # — Paso 3: DD-lambda y temperatura por régimen (Capas 2+3) —
        self.dd_lambda = float(r_lambda)
        self.temperature = float(r_temp)

        # — Paso 4: Cooldown adaptativo (Capa 4) —
        self._rolling_sharpe = self._rolling_sharpe_7d()
        if self._rolling_sharpe < self.sharpe_target * 0.5:
            # Sharpe muy bajo → rebalancear más frecuente para buscar mejor estado
            cooldown_sharpe_adj = r_cooldown_mult * 0.6
        elif self._rolling_sharpe > self.sharpe_target * 1.5:
            # Sharpe excelente → no tocar, cooldown más largo
            cooldown_sharpe_adj = r_cooldown_mult * 1.4
        else:
            cooldown_sharpe_adj = r_cooldown_mult

        self.cooldown = max(30, int(self._base_cooldown * cooldown_sharpe_adj))

        # — Paso 5: Log de adaptación —
        self._adaptation_log.append({
            'bar': idx,
            'regime': regime,
            'alpha': round(self.alpha, 4),
            'lambda': round(self.dd_lambda, 2),
            'temperature': round(self.temperature, 3),
            'cooldown': self.cooldown,
            'rolling_sharpe': round(self._rolling_sharpe, 3),
        })
        # Mantener solo últimos 100 registros
        if len(self._adaptation_log) > 100:
            self._adaptation_log = self._adaptation_log[-100:]

    # ── Helpers (sin cambio respecto v2) ─────────────────────────────────────
    def _compute_dd(self, eq_curves: list, idx: int) -> np.ndarray:
        """Drawdown actual de cada estrategia en barra idx."""
        dds = np.zeros(len(eq_curves))
        for k, eq in enumerate(eq_curves):
            if idx < len(eq) and eq[idx] is not None:
                self._peak[k] = max(self._peak[k], eq[idx])
                if self._peak[k] > 0:
                    dds[k] = (self._peak[k] - eq[idx]) / self._peak[k]
        return dds

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """Softmax con temperatura adaptativa."""
        scaled = scores / max(self.temperature, 1e-8)
        scaled -= scaled.max()
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum()

    def _daily_return(self, eq: list, idx: int, window: int = 1440) -> float:
        """Retorno acumulado de la estrategia en la ventana dada."""
        start = max(0, idx - window)
        if eq[start] > 0:
            return (eq[idx] - eq[start]) / eq[start]
        return 0.0

    # ── RUN: Loop principal ──────────────────────────────────────────────────
    def run(
        self,
        eq_tech: list,
        eq_soph: list,
        eq_xgb: list,
        initial_capital: float,
    ) -> dict:
        """
        Ejecuta el Evolutionary Adaptive Orchestrator sobre las 3 equity curves.
        A diferencia de v2, los hiperparámetros evolucionan barra a barra.
        """
        n = min(len(eq_tech), len(eq_soph), len(eq_xgb))
        eq_curves = [eq_tech, eq_soph, eq_xgb]
        strategy_names_local = ['Technical', 'Sophia', 'ML_XGBoost']

        # Inicialización neutral
        self._ema_scores = np.array([0.33, 0.34, 0.33])
        self._weights = np.array([0.33, 0.34, 0.33])
        self._peak = {k: eq_curves[k][0] for k in range(3)}
        self._last_rebalance = 0
        self._last_adaptation = 0
        self._meta_eq_window = []

        meta_capital = initial_capital
        meta_peak = initial_capital
        meta_max_dd = 0.0
        meta_eq = [meta_capital]
        rebalance_count = 0
        adaptation_count = 0

        # Buffer de retornos del portfolio (para Volatility-Adaptive Alpha)
        portfolio_rets_buffer = []

        for i in range(1, n):
            # ── FASE WARMUP: Sophia por defecto ──────────────────────────────
            if i < self.warmup:
                ret = (eq_soph[i] - eq_soph[i-1]) / eq_soph[i-1] if eq_soph[i-1] > 0 else 0.0
                meta_capital *= (1 + ret)
                if meta_capital > meta_peak:
                    meta_peak = meta_capital
                dd = (meta_peak - meta_capital) / meta_peak if meta_peak > 0 else 0
                if dd > meta_max_dd:
                    meta_max_dd = dd
                meta_eq.append(meta_capital)
                portfolio_rets_buffer.append(ret)
                self._meta_eq_window.append(meta_capital)
                continue

            # ── CAPA 1-5: ADAPTACIÓN EVOLUTIVA (cada adaptation_interval) ───
            if (i - self._last_adaptation) >= self.adaptation_interval:
                rets_arr = np.array(portfolio_rets_buffer) if portfolio_rets_buffer else np.zeros(1)
                self._adapt_parameters(meta_eq, i, rets_arr)
                self._last_adaptation = i
                adaptation_count += 1

            # ── CAPA 1+2: EMA scores con alpha y lambda adaptativos ──────────
            raw_scores = np.array([
                self._daily_return(eq, i, window=1440 * 7)
                for eq in eq_curves
            ])

            # CAPA 2: Penalización DD con lambda ADAPTADO por régimen
            current_dds = self._compute_dd(eq_curves, i)
            dd_penalty = 1.0 - self.dd_lambda * current_dds
            dd_penalty = np.clip(dd_penalty, 0.05, 1.0)
            penalized_scores = raw_scores * dd_penalty

            # CAPA 1: EMA con alpha ADAPTADO por volatilidad
            self._ema_scores = (
                self.alpha * penalized_scores
                + (1 - self.alpha) * self._ema_scores
            )

            # ── CAPA 3+4: Rebalanceo Softmax con cooldown ADAPTADO ───────────
            bars_since_rebalance = i - self._last_rebalance
            if bars_since_rebalance >= self.cooldown:
                new_weights = self._softmax(self._ema_scores)

                # Anti-chasing: suavizar si el shift supera 20%
                max_weight_shift = np.max(np.abs(new_weights - self._weights))
                if max_weight_shift > 0.20 and bars_since_rebalance < self.cooldown * 3:
                    self._weights = 0.5 * self._weights + 0.5 * new_weights
                else:
                    self._weights = new_weights

                self._last_rebalance = i
                rebalance_count += 1

            # ── Calcular retorno del portfolio weightado ──────────────────────
            rets = np.array([
                (eq[i] - eq[i-1]) / eq[i-1] if eq[i-1] > 0 else 0.0
                for eq in eq_curves
            ])

            portfolio_ret = float(np.dot(self._weights, rets))
            meta_capital *= (1 + portfolio_ret)
            portfolio_rets_buffer.append(portfolio_ret)
            # Cota para evitar consumo excesivo de memoria en horizontes largos
            if len(portfolio_rets_buffer) > 50000:
                portfolio_rets_buffer = portfolio_rets_buffer[-20000:]

            if meta_capital > meta_peak:
                meta_peak = meta_capital
            dd = (meta_peak - meta_capital) / meta_peak if meta_peak > 0 else 0
            if dd > meta_max_dd:
                meta_max_dd = dd
            meta_eq.append(meta_capital)
            self._meta_eq_window.append(meta_capital)
            if len(self._meta_eq_window) > 1440 * 8:
                self._meta_eq_window = self._meta_eq_window[-1440 * 8:]

        # ── Métricas finales ──────────────────────────────────────────────────
        pnl_usd = meta_capital - initial_capital
        pnl_pct = pnl_usd / initial_capital * 100

        daily_returns = []
        chunk = 1440
        for j in range(0, len(meta_eq) - 1, chunk):
            chunk_eq = meta_eq[j:j + chunk + 1]
            if len(chunk_eq) > 1 and chunk_eq[0] > 0:
                daily_returns.append((chunk_eq[-1] - chunk_eq[0]) / chunk_eq[0])

        sharpe = 0.0
        sortino = 0.0
        if len(daily_returns) >= 3:
            std_ret = np.std(daily_returns, ddof=1)
            mean_ret = np.mean(daily_returns)
            rf_daily = 0.02 / 365
            if std_ret > 1e-10:
                sharpe = np.clip((mean_ret - rf_daily) / std_ret * np.sqrt(365), -50, 50)
            neg = [r for r in daily_returns if r < rf_daily]
            ds = np.std(neg, ddof=1) if len(neg) >= 2 else std_ret
            if ds > 1e-10:
                sortino = np.clip((mean_ret - rf_daily) / ds * np.sqrt(365), -50, 50)
        elif len(daily_returns) >= 1:
            m = np.mean(daily_returns)
            sharpe = np.clip(m / 0.001 if abs(m) > 0.0001 else 0, -10, 10)
            sortino = sharpe

        final_weights = {s: round(float(w), 3) for s, w in zip(strategy_names_local, self._weights)}

        return {
            'pnl_usd': round(pnl_usd, 4),
            'pnl_pct': round(pnl_pct, 3),
            'max_drawdown': round(meta_max_dd * 100, 3),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'rebalance_count': rebalance_count,
            'adaptation_count': adaptation_count,
            'final_regime': self._current_regime,
            'final_weights': final_weights,
            'adaptation_log': self._adaptation_log[-5:],   # últimas 5 adaptaciones
            'equity_curve': meta_eq,
        }
    """
    🛡️ Anti-Whipsaw Meta-Orchestrator v2.0
    
    QUÉ: Sistema de allocación dinámica entre estrategias que reemplaza la
         selección binaria winner-take-all (causa del efecto whipsaw).
    
    POR QUÉ: El orquestador v1 usaba max(ret_48h) para elegir estrategia,
         lo que en crypto (alta mean-reversion) garantiza entrar justo en el
         peak de cada curva de rendimiento → doubled drawdown 4.14%.
    
    PARA QUÉ: Reducir el max_drawdown del Orchestrator por debajo del promedio
         de las estrategias individuales, mejorando el Sharpe combinado.
    
    CÓMO: 4 capas de protección:
         1. EMA sobre scores (filtro pasa-bajos → elimina noise de 48h)
         2. Penalización de drawdown actual (castiga estrategia en caída)
         3. Softmax allocation (portfolio balanceado, no winner-take-all)
         4. Cooldown anti-chasing (rebalanceo máx cada 4h)
    
    CUÁNDO: En cada barra del backtest durante el META-ORCHESTRATOR PASS.
    DÓNDE: run_multi_horizon_backtest.py
    QUIÉN: Llamado desde main() por símbolo y horizonte.
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.05,          # ≈ EMA-20 barras diarias — responde a 7-20 días
        dd_penalty_lambda: float = 3.0,   # Penalización agresiva de drawdown actual
        softmax_temperature: float = 0.08, # Sharper → mayor concentración en líder claro
        rebalance_cooldown: int = 240,     # 4h a 1m bars — anti-chasing
        min_warmup_bars: int = 2880,       # 2 días antes de activar el orquestador
    ):
        self.alpha = ema_alpha
        self.dd_lambda = dd_penalty_lambda
        self.temperature = softmax_temperature
        self.cooldown = rebalance_cooldown
        self.warmup = min_warmup_bars
        
        # Estado interno del orquestador
        self._ema_scores = None           # [score_tech, score_soph, score_xgb]
        self._weights = None              # Pesos actuales del portfolio [w_t, w_s, w_x]
        self._last_rebalance = 0          # Barra del último rebalanceo
        self._peak = {0: 0.0, 1: 0.0, 2: 0.0}  # Peak equity de cada estrategia

    def _compute_dd(self, eq_curves: list, idx: int) -> np.ndarray:
        """Calcula el drawdown actual de cada estrategia en la barra idx."""
        dds = np.zeros(len(eq_curves))
        for k, eq in enumerate(eq_curves):
            if idx < len(eq) and eq[idx] is not None:
                self._peak[k] = max(self._peak[k], eq[idx])
                if self._peak[k] > 0:
                    dds[k] = (self._peak[k] - eq[idx]) / self._peak[k]
        return dds

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """Softmax con temperatura — más temperatura = más uniforme."""
        scaled = scores / max(self.temperature, 1e-8)
        scaled -= scaled.max()  # Estabilidad numérica
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum()

    def _daily_return(self, eq: list, idx: int, window: int = 1440) -> float:
        """Retorno diario promedio de la estrategia en la ventana dada."""
        start = max(0, idx - window)
        if eq[start] > 0:
            return (eq[idx] - eq[start]) / eq[start]
        return 0.0

    def run(
        self,
        eq_tech: list,
        eq_soph: list,
        eq_xgb: list,
        initial_capital: float,
    ) -> dict:
        """
        Ejecuta el orquestador Anti-Whipsaw sobre las 3 equity curves.
        Retorna dict con métricas del portfolio combinado.
        """
        n = min(len(eq_tech), len(eq_soph), len(eq_xgb))
        eq_curves = [eq_tech, eq_soph, eq_xgb]
        strategy_names_local = ['Technical', 'Sophia', 'ML_XGBoost']
        
        # CAPA 1: Inicializar EMA scores — arrancar neutral (igual peso)
        self._ema_scores = np.array([0.33, 0.34, 0.33])  # Neutral initial
        self._weights = np.array([0.33, 0.34, 0.33])
        self._peak = {k: eq_curves[k][0] for k in range(3)}
        self._last_rebalance = 0
        
        meta_capital = initial_capital
        meta_peak = initial_capital
        meta_max_dd = 0.0
        meta_eq = [meta_capital]
        rebalance_count = 0
        
        for i in range(1, n):
            # ── FASE DE CALENTAMIENTO: usa Sophia por defecto (mejor 1D) ──
            if i < self.warmup:
                ret = (eq_soph[i] - eq_soph[i-1]) / eq_soph[i-1] if eq_soph[i-1] > 0 else 0.0
                meta_capital *= (1 + ret)
                if meta_capital > meta_peak:
                    meta_peak = meta_capital
                dd = (meta_peak - meta_capital) / meta_peak if meta_peak > 0 else 0
                if dd > meta_max_dd:
                    meta_max_dd = dd
                meta_eq.append(meta_capital)
                continue
            
            # ── CAPA 1+2: Actualizar EMA scores con penalización de DD ──
            # Score bruto: retorno diario de cada estrategia (7-day rolling)
            raw_scores = np.array([
                self._daily_return(eq, i, window=1440 * 7)
                for eq in eq_curves
            ])
            
            # CAPA 2: Penalización por drawdown actual
            current_dds = self._compute_dd(eq_curves, i)
            dd_penalty = 1.0 - self.dd_lambda * current_dds
            dd_penalty = np.clip(dd_penalty, 0.05, 1.0)  # Mínimo 5% de peso siempre
            penalized_scores = raw_scores * dd_penalty
            
            # CAPA 1: EMA sobre scores penalizados (filtro pasa-bajos)
            self._ema_scores = (
                self.alpha * penalized_scores
                + (1 - self.alpha) * self._ema_scores
            )
            
            # ── CAPA 3+4: Rebalanceo con Softmax (con cooldown) ──
            bars_since_rebalance = i - self._last_rebalance
            if bars_since_rebalance >= self.cooldown:
                # Softmax sobre EMA scores — distribución continua, no winner-take-all
                new_weights = self._softmax(self._ema_scores)
                
                # CAPA 4: Detectar cambio brusco de peso (anti-chasing)
                max_weight_shift = np.max(np.abs(new_weights - self._weights))
                if max_weight_shift > 0.20 and bars_since_rebalance < self.cooldown * 3:
                    # Shift >20% dentro de cooldown triple → suavizar con 50% blend
                    self._weights = 0.5 * self._weights + 0.5 * new_weights
                else:
                    self._weights = new_weights
                
                self._last_rebalance = i
                rebalance_count += 1
            
            # ── Calcular retorno del portfolio weightado ──
            rets = np.array([
                (eq[i] - eq[i-1]) / eq[i-1] if eq[i-1] > 0 else 0.0
                for eq in eq_curves
            ])
            
            portfolio_ret = float(np.dot(self._weights, rets))
            meta_capital *= (1 + portfolio_ret)
            
            if meta_capital > meta_peak:
                meta_peak = meta_capital
            dd = (meta_peak - meta_capital) / meta_peak if meta_peak > 0 else 0
            if dd > meta_max_dd:
                meta_max_dd = dd
            meta_eq.append(meta_capital)
        
        # ── Métricas del portfolio ──
        pnl_usd = meta_capital - initial_capital
        pnl_pct = pnl_usd / initial_capital * 100
        
        daily_returns = []
        chunk = 1440
        for j in range(0, len(meta_eq) - 1, chunk):
            chunk_eq = meta_eq[j:j + chunk + 1]
            if len(chunk_eq) > 1 and chunk_eq[0] > 0:
                daily_returns.append((chunk_eq[-1] - chunk_eq[0]) / chunk_eq[0])
        
        sharpe = 0.0
        sortino = 0.0
        if len(daily_returns) >= 3:
            std_ret = np.std(daily_returns, ddof=1)
            mean_ret = np.mean(daily_returns)
            rf_daily = 0.02 / 365
            if std_ret > 1e-10:
                sharpe = np.clip((mean_ret - rf_daily) / std_ret * np.sqrt(365), -50, 50)
            neg = [r for r in daily_returns if r < rf_daily]
            ds = np.std(neg, ddof=1) if len(neg) >= 2 else std_ret
            if ds > 1e-10:
                sortino = np.clip((mean_ret - rf_daily) / ds * np.sqrt(365), -50, 50)
        elif len(daily_returns) >= 1:
            m = np.mean(daily_returns)
            sharpe = np.clip(m / 0.001 if abs(m) > 0.0001 else 0, -10, 10)
            sortino = sharpe
        
        # Pesos finales del orquestador
        final_weights = {s: round(float(w), 3) for s, w in zip(strategy_names_local, self._weights)}
        
        return {
            'pnl_usd': round(pnl_usd, 4),
            'pnl_pct': round(pnl_pct, 3),
            'max_drawdown': round(meta_max_dd * 100, 3),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'rebalance_count': rebalance_count,
            'final_weights': final_weights,
            'equity_curve': meta_eq,
        }


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("🚀 TRADER GEMINI - MULTI-HORIZON BACKTEST v2.0")
    print("   🛡️ Real ML | Real Clustering | Circuit Breaker | Fixed Sharpe")
    print("=" * 70)
    
    all_results = {}
    strategy_names = ['Technical', 'Sophia', 'ML_XGBoost']
    
    # Per horizon
    for days in HORIZONS:
        print(f"\n{'='*70}")
        print(f"📅 HORIZONTE: {days} DÍAS")
        print(f"{'='*70}")
        
        horizon_results = {}
        cached_data = {}
        
        for symbol in SYMBOLS:
            print(f"\n  🔹 {symbol}")
            
            # Fetch data once per symbol per horizon
            if symbol not in cached_data:
                df = fetch_data(symbol, days + 2)  # +2 for warmup overlap
                if df is None or len(df) < 500:
                    print(f"    ❌ Insuficientes datos")
                    continue
                cached_data[symbol] = df
            df = cached_data[symbol]
            
            symbol_results = {}
            for strategy in strategy_names:
                result = run_strategy_backtest(df, symbol, strategy, INITIAL_CAPITAL, LEVERAGE)
                symbol_results[strategy] = result
                status = "✅" if result['pnl_usd'] >= 0 else "❌"
                
                # Enhanced output with ML info
                extra = ""
                if strategy == 'ML_XGBoost' and result.get('ml_trainings', 0) > 0:
                    extra = f" ML_Acc: {result.get('ml_accuracy', 0):.0f}%"
                if result.get('kill_switch_triggers', 0) > 0:
                    extra += f" 🛡️KS:{result['kill_switch_triggers']}"
                    
                print(f"    [{strategy:12s}] {status} PNL: ${result['pnl_usd']:+.4f} "
                      f"WR: {result['win_rate']:.1f}% "
                      f"DD: {result['max_drawdown']:.2f}% "
                      f"Sharpe: {result['sharpe']:.2f} "
                      f"Trades: {result['trades']}{extra}")
            
            # --- META-ORCHESTRATOR PASS v2.0 (Anti-Whipsaw) ---
            eq_tech = symbol_results['Technical']['equity_curve']
            eq_soph = symbol_results['Sophia']['equity_curve']
            eq_xgb  = symbol_results['ML_XGBoost']['equity_curve']
            
            orchestrator = AntiWhipsawOrchestrator(
                ema_alpha=0.05,
                dd_penalty_lambda=3.0,
                softmax_temperature=0.08,
                rebalance_cooldown=240,
                min_warmup_bars=min(2880, len(eq_soph) // 4),
            )
            orch_result = orchestrator.run(eq_tech, eq_soph, eq_xgb, INITIAL_CAPITAL)
            
            pnl_usd    = orch_result['pnl_usd']
            pnl_pct    = orch_result['pnl_pct']
            meta_max_dd = orch_result['max_drawdown'] / 100.0   # keep float for compatibilidad
            sharpe     = orch_result['sharpe']
            fw = orch_result['final_weights']
            
            status = "✅" if pnl_usd >= 0 else "❌"
            print(
                f"    ⭐[{'Orch-AWv2':12s}] {status} "
                f"PNL: ${pnl_usd:+.4f} "
                f"WR: --.-% "
                f"DD: {orch_result['max_drawdown']:.2f}% "
                f"Sharpe: {sharpe:.2f} "
                f"Rebalances: {orch_result['rebalance_count']} "
                f"Weights: T={fw['Technical']:.2f} S={fw['Sophia']:.2f} ML={fw['ML_XGBoost']:.2f}"
            )
            
            symbol_results['Orchestrator'] = {
                'trades': sum(symbol_results[s]['trades'] for s in strategy_names) // 3,
                'win_rate': np.mean([symbol_results[s]['win_rate'] for s in strategy_names]),
                'max_drawdown': orch_result['max_drawdown'],
                'pnl_usd': pnl_usd,
                'pnl_pct': pnl_pct,
                'sharpe': sharpe,
                'sortino': orch_result['sortino'],
                'rebalance_count': orch_result['rebalance_count'],
                'final_weights': fw,
            }
            
            horizon_results[symbol] = symbol_results
        
        all_results[f"{days}D"] = horizon_results
        
        # Aggregate por horizonte
        print(f"\n  📊 RESUMEN {days}D:")
        for strategy in strategy_names:
            totals = [h[strategy] for h in horizon_results.values() if strategy in h]
            if not totals:
                continue
            avg_wr = np.mean([t['win_rate'] for t in totals])
            avg_dd = np.mean([t['max_drawdown'] for t in totals])
            avg_sharpe = np.mean([t['sharpe'] for t in totals])
            total_pnl = sum(t['pnl_usd'] for t in totals)
            total_trades = sum(t['trades'] for t in totals)
            status = "✅" if total_pnl >= 0 else "❌"
            print(f"  [{strategy:12s}] {status} Total PNL: ${total_pnl:+.3f} | "
                  f"WR: {avg_wr:.1f}% | DD: {avg_dd:.2f}% | Sharpe: {avg_sharpe:.2f} | "
                  f"Trades: {total_trades}")

    strategy_names_full = ['Technical', 'Sophia', 'ML_XGBoost', 'Orchestrator']
    
    # Final report
    print("\n" + "="*70)
    print("🏆 REPORTE FINAL CONSOLIDADO v3.0 (EVOLUTIVO)")
    print("="*70)
    print(f"{'Horizon':<8} {'Strategy':<14} {'PNL$':>8} {'PNL%':>8} {'WR%':>7} {'MaxDD%':>8} {'Sharpe':>8} {'KS':>4}")
    print("-"*70)
    
    for horizon, h_data in all_results.items():
        for strategy in strategy_names_full:
            totals = [h[strategy] for h in h_data.values() if strategy in h]
            if not totals:
                continue
            t_pnl = sum(t['pnl_usd'] for t in totals)
            t_pnl_pct = np.mean([t['pnl_pct'] for t in totals])
            avg_wr = np.mean([t['win_rate'] for t in totals])
            avg_dd = np.mean([t['max_drawdown'] for t in totals])
            avg_s = np.mean([t['sharpe'] for t in totals])
            total_ks = sum(t.get('kill_switch_triggers', 0) for t in totals)
            print(f"{horizon:<8} {strategy:<14} {t_pnl:>8.3f} {t_pnl_pct:>7.2f}% {avg_wr:>6.1f}% {avg_dd:>7.2f}% {avg_s:>8.2f} {total_ks:>4}")
    
    # Save JSON
    out_file = "multi_horizon_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📁 Resultados guardados en: {out_file}")
    
    return all_results


if __name__ == '__main__':
    results = main()
