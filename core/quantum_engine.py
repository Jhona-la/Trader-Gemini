import numpy as np
import polars as pl
import os
import glob
from numba import njit, float64, int64
from utils.math_kernel import (
    calculate_rsi_jit, calculate_bollinger_jit, calculate_ema_jit,
    calculate_macd_jit, calculate_atr_jit, calculate_adx_jit,
    calculate_quantum_features_batch_jit
)
from config import Config
from datetime import datetime, timezone
from utils.logger import logger
import joblib
import xgboost as xgb_lib

@njit(fastmath=True, cache=True)
def resolve_trades_jit(
    long_idx: np.ndarray,
    short_idx: np.ndarray,
    open_p: np.ndarray,
    high_p: np.ndarray,
    low_p: np.ndarray,
    close_p: np.ndarray,
    timestamps: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    leverage: float,
    round_trip_fee: float,
    max_duration: int,
    slippage_pct: float
):
    n = len(open_p)
    max_trades = len(long_idx) + len(short_idx)
    out_exit_ts = np.zeros(max_trades, dtype=np.int64)
    out_pct_ret = np.zeros(max_trades, dtype=np.float64)
    out_is_win = np.zeros(max_trades, dtype=np.int64) 
    out_is_long = np.zeros(max_trades, dtype=np.int64) 
    out_entry_ts = np.zeros(max_trades, dtype=np.int64)
    out_entry_price = np.zeros(max_trades, dtype=np.float64)
    count = 0
    
    for i in range(len(long_idx)):
        idx = long_idx[i]
        if idx >= n - 2: continue
        entry_price = open_p[idx+1] * (1.0 + slippage_pct)
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)
        
        first_tp = 99999999
        first_sl = 99999999
        
        limit = min(n, idx + 1 + max_duration)
        for j in range(idx + 1, limit):
            if low_p[j] <= sl_price and first_sl == 99999999:
                first_sl = j
            if high_p[j] >= tp_price and first_tp == 99999999:
                first_tp = j
                
            if first_sl != 99999999 or first_tp != 99999999:
                break
                
        if first_tp < first_sl:
            pct_return = (tp_pct - slippage_pct - round_trip_fee) * leverage
            out_exit_ts[count] = timestamps[first_tp]
            out_pct_ret[count] = pct_return
            out_is_win[count] = 1
            out_is_long[count] = 1
            out_entry_ts[count] = timestamps[idx+1]
            out_entry_price[count] = entry_price
            count += 1
        elif first_sl < first_tp:
            pct_return = (-sl_pct - slippage_pct - round_trip_fee) * leverage
            out_exit_ts[count] = timestamps[first_sl]
            out_pct_ret[count] = pct_return
            out_is_win[count] = 0
            out_is_long[count] = 1
            out_entry_ts[count] = timestamps[idx+1]
            out_entry_price[count] = entry_price
            count += 1
        else:
            exit_idx = limit - 1
            exit_price = close_p[exit_idx] * (1.0 - slippage_pct)
            pct_return = (((exit_price - entry_price) / entry_price) - round_trip_fee) * leverage
            out_exit_ts[count] = timestamps[exit_idx]
            out_pct_ret[count] = pct_return
            out_is_win[count] = 1 if pct_return > 0 else 0
            out_is_long[count] = 1
            out_entry_ts[count] = timestamps[idx+1]
            out_entry_price[count] = entry_price
            count += 1

    for i in range(len(short_idx)):
        idx = short_idx[i]
        if idx >= n - 2: continue
        entry_price = open_p[idx+1] * (1.0 - slippage_pct)
        tp_price = entry_price * (1.0 - tp_pct)
        sl_price = entry_price * (1.0 + sl_pct)
        
        first_tp = 99999999
        first_sl = 99999999
        
        limit = min(n, idx + 1 + max_duration)
        for j in range(idx + 1, limit):
            if high_p[j] >= sl_price and first_sl == 99999999:
                first_sl = j
            if low_p[j] <= tp_price and first_tp == 99999999:
                first_tp = j
                
            if first_sl != 99999999 or first_tp != 99999999:
                break
                
        if first_tp < first_sl:
            pct_return = (tp_pct - slippage_pct - round_trip_fee) * leverage
            out_exit_ts[count] = timestamps[first_tp]
            out_pct_ret[count] = pct_return
            out_is_win[count] = 1
            out_is_long[count] = 0
            out_entry_ts[count] = timestamps[idx+1]
            out_entry_price[count] = entry_price
            count += 1
        elif first_sl < first_tp:
            pct_return = (-sl_pct - slippage_pct - round_trip_fee) * leverage
            out_exit_ts[count] = timestamps[first_sl]
            out_pct_ret[count] = pct_return
            out_is_win[count] = 0
            out_is_long[count] = 0
            out_entry_ts[count] = timestamps[idx+1]
            out_entry_price[count] = entry_price
            count += 1
        else:
            exit_idx = limit - 1
            exit_price = close_p[exit_idx] * (1.0 + slippage_pct)
            pct_return = (((entry_price - exit_price) / entry_price) - round_trip_fee) * leverage
            out_exit_ts[count] = timestamps[exit_idx]
            out_pct_ret[count] = pct_return
            out_is_win[count] = 1 if pct_return > 0 else 0
            out_is_long[count] = 0
            out_entry_ts[count] = timestamps[idx+1]
            out_entry_price[count] = entry_price
            count += 1
            
    return out_exit_ts[:count], out_pct_ret[:count], out_is_win[:count], out_is_long[:count], out_entry_ts[:count], out_entry_price[:count]


try:
    from strategies.components.feature_engineering import FeatureEngineering
    _HAS_FE = True
except ImportError:
    _HAS_FE = False

class QuantumEngine:
    """
    Motor Cuántico Vectorizado (Fase 21).
    Procesa 15 días de data (millones de puntos) en nanosegundos / milisegundos netos
    usando operaciones vectoriales (Álgebra Matricial) en lugar del Event Loop de Python.
    """
    
    def __init__(self, data_dir="dashboard/data/futures", capital=13.0, horizon="SCALPING"):
        self.data_dir = data_dir
        self.capital = capital
        self.horizon = horizon.upper()
        self.symbols = Config.TRADING_PAIRS
        self.results = {}
        
        # Determine base parameters
        if self.horizon == 'SCALPING':
            self.h_params = getattr(Config.Horizons, 'Scalping', {})
            self.tp_pct = self.h_params['tp_pct']
            self.sl_pct = self.h_params['sl_pct']
        else:
            self.h_params = getattr(Config.Horizons, 'Swing', {})
            self.tp_pct = self.h_params['tp_pct']
            self.sl_pct = self.h_params['sl_pct']

        if _HAS_FE:
            self.fe = FeatureEngineering()
            
        # Symbolic mapping for Nano Engine
        self.symbol_to_id = {sym: idx for idx, sym in enumerate(self.symbols)}
            
    def _precalculate_omni_features(self, df_pd, symbol, horizon_type):
        """
        FASE 31: OMNI-COMPILATION ENGINE
        Calcula TODO el sistema de una vez en Polars/NumPy (200 Features de IA + 30 Estrategias).
        Devuelve el dataframe con las columnas de las señales estáticas pre-calculadas.
        """
        arrs = {}
        # 1. TECHNICAL PROXIES (Phalanx & StatArb)
        close = df_pd['close'].to_numpy()
        high = df_pd['high'].to_numpy()
        low = df_pd['low'].to_numpy()
        
        # Proxy Phalanx: Volatility Breakout (High - Low > ATR*2)
        arrs['phalanx_sig'] = np.zeros(len(df_pd))
        try:
            atr = calculate_atr_jit(high, low, close, 14)
            arrs['phalanx_sig'] = np.where((high - low) > (atr * 2), 1.0, 0.0)
        except Exception as e:
            logger.exception(f"Bare except ghost bug: {e}")
            pass
            
        # Proxy StatArb: Z-Score of Close (Mean Reversion)
        arrs['statarb_sig'] = np.zeros(len(df_pd))
        try:
            sma = calculate_ema_jit(close, 50)
            std = np.std(close) if np.std(close) > 0 else 1
            z_score = (close - sma) / std
            arrs['statarb_sig'] = np.where(z_score > 2, -1.0, np.where(z_score < -2, 1.0, 0.0))
        except Exception as e:
            logger.exception(f"Bare except ghost bug: {e}")
            pass

        # 2. MACHINE LEARNING 200-FEATURE COMPILATION
        arrs['ml_bull'] = np.zeros(len(df_pd))
        arrs['ml_bear'] = np.zeros(len(df_pd))
        
        if not getattr(self, 'fe', None):
            return arrs
            
        try:
            logger.info(f"🧠 [OMNI-COMPILATION] Generando IA (200 Features) y Proxies para {symbol} ({horizon_type})")
            features_df = self.fe.prepare_features(df_pd, symbol=symbol, horizon=horizon_type)
            
            h_str = "scalping" if horizon_type == "SCALPING" else "swing"
            clean_sym = symbol.replace("/", "")
            # FASE 32: Carga Nanosegundo (Metadata + UBJ)
            meta_path = f"models/{clean_sym}_{h_str}_meta.joblib"
            ubj_path = f"models/{clean_sym}_{h_str}_xgb.ubj"
            
            # Retro-compatibilidad si .models/ es usado
            if not os.path.exists(meta_path):
                meta_path = f".models/{clean_sym}_{h_str}_meta.joblib"
                ubj_path = f".models/{clean_sym}_{h_str}_xgb.ubj"
                
            if not os.path.exists(meta_path) or not os.path.exists(ubj_path):
                logger.warning(f"⚠️ [OMNI-ENGINE] Sin modelo C++ {ubj_path}. Ignorando IA.")
                return arrs
                
            model_data = joblib.load(meta_path)
            feature_cols = model_data['feature_cols']
            
            if not feature_cols:
                return arrs
                
            xgb = xgb_lib.XGBClassifier()
            xgb.load_model(ubj_path)
                
            missing = [c for c in feature_cols if c not in features_df.columns]
            for m in missing:
                features_df[m] = 0.0
                
            X = features_df[feature_cols].astype(np.float32).values
            booster = xgb.get_booster() if hasattr(xgb, 'get_booster') else xgb
            
            try:
                # FASE 67: XG-JIT NANO-SPEED COMPILATION
                from core.xgboost_compiler import compile_xgb_to_arrays, predict_xgb_jit_batch
                
                # Check if we already compiled it to memory for this horizon
                if not hasattr(self, '_xgb_arrays'):
                    self._xgb_arrays = {}
                    
                cache_key = f"{symbol}_{horizon_type}"
                if cache_key not in self._xgb_arrays:
                    feats, ths, lefts, rights, miss, vals, offs, base_score = compile_xgb_to_arrays(booster)
                    self._xgb_arrays[cache_key] = (feats, ths, lefts, rights, miss, vals, offs, base_score)
                else:
                    feats, ths, lefts, rights, miss, vals, offs, base_score = self._xgb_arrays[cache_key]

                # Paralelo y Vectorizado: Millones de barras en milisegundos
                preds = predict_xgb_jit_batch(X, feats, ths, lefts, rights, miss, vals, offs, base_score)
                
                arrs['ml_bull'] = preds
                arrs['ml_bear'] = 1.0 - preds
                
            except Exception as ml_e:
                logger.warning(f"JIT ML falló, usando predict_proba lento: {ml_e}")
                if hasattr(xgb, 'predict_proba'):
                    probs = xgb.predict_proba(X)
                    arrs['ml_bull'] = probs[:, 1]
                    arrs['ml_bear'] = probs[:, 0]
                    
        except Exception as e:
            logger.error(f"Omni-Compilation failed for {symbol}: {e}")
            
        return arrs
            
    def load_data(self, days=15):
        """
        Carga datos desde disk cache a memoria estructurada Polars en O(1) I/O.
        Soporta carga dual 1m y 1h para multi-horizonte.
        """
        self.data_1m = {}
        self.data_1h = {}
        target_ms = int(datetime.now().timestamp() * 1000) - (days * 24 * 60 * 60 * 1000)
        
        for symbol in self.symbols:
            clean_sym = symbol.replace("/", "")
            
            # Cargar 1m
            if self.horizon in ['SCALPING', 'BOTH']:
                path_1m = f"{self.data_dir}/{clean_sym}_1m.parquet"
                if not os.path.exists(path_1m): path_1m = f"{self.data_dir}/{clean_sym}_1m.csv"
                if os.path.exists(path_1m):
                    try:
                        df = pl.read_parquet(path_1m) if path_1m.endswith('.parquet') else pl.read_csv(path_1m)
                        if 'timestamp' in df.columns:
                            max_ts = df['timestamp'].max()
                            if max_ts is not None:
                                target_ms_sym = max_ts - (days * 24 * 60 * 60 * 1000)
                                df = df.filter(pl.col('timestamp') >= target_ms_sym)
                        
                        omni_arrs = self._precalculate_omni_features(df, symbol, "SCALPING")
                        for k, v in omni_arrs.items():
                            df = df.with_columns(pl.Series(k, v))
                        
                        self.data_1m[symbol] = {
                            'timestamp': df['timestamp'].to_numpy(),
                            'close': df['close'].to_numpy().astype(np.float64),
                            'open': df['open'].to_numpy().astype(np.float64),
                            'high': df['high'].to_numpy().astype(np.float64),
                            'low': df['low'].to_numpy().astype(np.float64),
                            'volume': df['volume'].to_numpy().astype(np.float64),
                            'ml_bull': df['ml_bull'].to_numpy(),
                            'ml_bear': df['ml_bear'].to_numpy(),
                            'phalanx_sig': df['phalanx_sig'].to_numpy(),
                            'statarb_sig': df['statarb_sig'].to_numpy()
                        }
                    except Exception as e:
                        logger.warning(f"QuantumEngine: Failed to load 1m {symbol}: {e}")
            
            # Cargar 1h
            if self.horizon in ['SWING', 'BOTH']:
                path_1h = f"{self.data_dir}/{clean_sym}_1h.parquet"
                if not os.path.exists(path_1h): path_1h = f"{self.data_dir}/{clean_sym}_1h.csv"
                if os.path.exists(path_1h):
                    try:
                        df = pl.read_parquet(path_1h) if path_1h.endswith('.parquet') else pl.read_csv(path_1h)
                        if 'timestamp' in df.columns:
                            max_ts = df['timestamp'].max()
                            if max_ts is not None:
                                target_ms_sym = max_ts - (days * 24 * 60 * 60 * 1000)
                                df = df.filter(pl.col('timestamp') >= target_ms_sym)
                        
                        omni_arrs = self._precalculate_omni_features(df, symbol, "SWING")
                        for k, v in omni_arrs.items():
                            df = df.with_columns(pl.Series(k, v))
                        
                        self.data_1h[symbol] = {
                            'timestamp': df['timestamp'].to_numpy(),
                            'close': df['close'].to_numpy().astype(np.float64),
                            'open': df['open'].to_numpy().astype(np.float64),
                            'high': df['high'].to_numpy().astype(np.float64),
                            'low': df['low'].to_numpy().astype(np.float64),
                            'volume': df['volume'].to_numpy().astype(np.float64),
                            'ml_bull': df['ml_bull'].to_numpy(),
                            'ml_bear': df['ml_bear'].to_numpy(),
                            'phalanx_sig': df['phalanx_sig'].to_numpy(),
                            'statarb_sig': df['statarb_sig'].to_numpy()
                        }
                    except Exception as e:
                        logger.warning(f"QuantumEngine: Failed to load 1h {symbol}: {e}")
                        
    def _simulate_horizon(self, data_dict, dna, prefix, gate_mult, global_trades, horizon_type):
        rsi_buy = dna[f'{prefix}rsi_buy']
        rsi_sell = dna[f'{prefix}rsi_sell']
        bb_std = dna[f'{prefix}bb_std']
        tp_pct = dna[f'{prefix}tp_pct']
        sl_pct = dna[f'{prefix}sl_pct']
        ema_f_period = dna[f'{prefix}ema_fast']
        ema_s_period = dna[f'{prefix}ema_slow']
        leverage = dna[f'{prefix}leverage']
        
        consensus_fee_mult = dna['consensus_fee_mult']
        
        round_trip_fee = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375) * 2
        fee_threshold = round_trip_fee * consensus_fee_mult
        
        # FASE 31: PESOS OMNI-COMPILADOS (Genetic DNA)
        w_ml = float(dna[f'{prefix}w_ml'])
        w_tech = float(dna[f'{prefix}w_technical'])
        w_phalanx = float(dna[f'{prefix}w_phalanx'])
        w_statarb = float(dna[f'{prefix}w_statarb'])
        ml_thresh_bull = float(dna[f'{prefix}ml_th_long'])
        ml_thresh_bear = float(dna[f'{prefix}ml_th_short'])
        omni_threshold = float(dna[f'{prefix}master_threshold'])
        slippage_pct = float(dna['slippage_pct']) # 2 BPS slippage default
        use_regime = float(dna['use_regime']) > 0.5 # Default to true
        
        for symbol, arrs in data_dict.items():
            timestamps = arrs['timestamp']
            close = arrs['close']
            high = arrs['high']
            low = arrs['low']
            n = len(close)
            if n < 50: continue
            
            rsi = calculate_rsi_jit(close, 14)
            bbu, bbm, bbl = calculate_bollinger_jit(close, 20, bb_std)
            atr = calculate_atr_jit(high, low, close, 14)
            atr_pct = atr / close
            valid_volatility = (atr_pct >= fee_threshold)
            
            # FASE 31: OMNI-SCORE VECTORIZATION
            ml_bull_val = arrs['ml_bull']
            ml_bear_val = arrs['ml_bear']
            ml_long_sig = (ml_bull_val >= ml_thresh_bull)
            ml_short_sig = (ml_bear_val >= ml_thresh_bear)
            
            tech_long = (rsi < rsi_buy) & (close <= bbl)
            tech_short = (rsi > rsi_sell) & (close >= bbu)
            
            phalanx = arrs['phalanx_sig']
            statarb = arrs['statarb_sig']
            
            score_long = (tech_long * w_tech) + (ml_long_sig * w_ml) + (phalanx * w_phalanx) + (np.isclose(statarb, 1.0) * w_statarb)
            score_short = (tech_short * w_tech) + (ml_short_sig * w_ml) + (np.isclose(statarb, -1.0) * w_statarb)
            
            long_cond = (score_long >= omni_threshold) & valid_volatility
            short_cond = (score_short >= omni_threshold) & valid_volatility
            
            # FASE 40: MARKET REGIME FILTERING (Hurst + ADX)
            if use_regime:
                adx = calculate_adx_jit(high, low, close, 14)
                hurst_arr, _, _ = calculate_quantum_features_batch_jit(close, np.zeros(n), np.zeros(n), 20)
                # Chop regime: Hurst < 0.45 OR (Hurst < 0.55 and ADX < 20)
                is_chop = (hurst_arr < 0.45) | ((hurst_arr < 0.55) & (adx < 20.0))
                # En chop, vetamos las señales direccionales (ML/Tech) a menos que sean mean reversion puras (statarb)
                long_cond = long_cond & (~is_chop | np.isclose(statarb, 1.0))
                short_cond = short_cond & (~is_chop | np.isclose(statarb, -1.0))
            
            long_idx = np.where(long_cond)[0]
            short_idx = np.where(short_cond)[0]
            
            # Simulador Cuántico Vectorizado (Numba)
            if len(long_idx) > 0 or len(short_idx) > 0:
                max_duration = 60
                
                # Garantizamos que leverage sea un float simple
                out_ts, out_ret, out_win, out_long, out_entry_ts, out_entry_prices = resolve_trades_jit(
                    long_idx, short_idx,
                    arrs['open'], high, low, close, timestamps,
                    float(tp_pct), float(sl_pct), float(leverage), float(round_trip_fee), int(max_duration), float(slippage_pct)
                )
                
                for k in range(len(out_ts)):
                    # Empaquetamos para el Nano Risk Engine
                    sym_id = self.symbol_to_id[symbol]
                    global_trades.append({
                        'entry_ts': out_entry_ts[k],
                        'exit_ts': out_ts[k],
                        'pct_ret': out_ret[k],
                        'is_win': out_win[k],
                        'is_long': out_long[k],
                        'entry_price': out_entry_prices[k],
                        'horizon': horizon_type,
                        'symbol_id': sym_id
                    })

    def run_vectorized_backtest(self, dna=None):
        """
        Ejecución O(1) Vectorizada Multi-Horizonte con Separación Concurrente.
        """
        global_trades = [] # (exit_timestamp, pct_return, is_win, horizon_type)
        dna = dna or {}
        
        # Procesar Scalping
        if self.horizon in ['SCALPING', 'BOTH']:
            self._simulate_horizon(self.data_1m, dna, prefix="scalp_", gate_mult=2.0, global_trades=global_trades, horizon_type="SCALP")
            
        # Procesar Swing
        if self.horizon in ['SWING', 'BOTH']:
            self._simulate_horizon(self.data_1h, dna, prefix="swing_", gate_mult=2.8, global_trades=global_trades, horizon_type="SWING")
        
        # FASE 51: NANO RISK ENGINE (Tratamiento Cuántico Fiel)
        if len(global_trades) == 0:
            return {'final_capital': self.capital, 'pnl': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0, 'trades': 0}
            
        import numpy as np
        from core.nano_risk_engine import simulate_nano_portfolio_events_jit
        
        n_trades = len(global_trades)
        entry_ts = np.zeros(n_trades, dtype=np.int64)
        exit_ts = np.zeros(n_trades, dtype=np.int64)
        pct_ret = np.zeros(n_trades, dtype=np.float64)
        is_win = np.zeros(n_trades, dtype=np.int32)
        is_long = np.zeros(n_trades, dtype=np.int32)
        entry_prices = np.zeros(n_trades, dtype=np.float64)
        horizons = np.zeros(n_trades, dtype=np.int32) # 0=SCALP, 1=SWING
        symbols = np.zeros(n_trades, dtype=np.int32)
        
        for idx, t in enumerate(global_trades):
            entry_ts[idx] = t['entry_ts']
            exit_ts[idx] = t['exit_ts']
            pct_ret[idx] = t['pct_ret']
            is_win[idx] = int(t['is_win'])
            is_long[idx] = int(t['is_long'])
            entry_prices[idx] = float(t['entry_price'])
            horizons[idx] = 0 if t['horizon'] == "SCALP" else 1
            symbols[idx] = t['symbol_id']

        # Extraer Kelly parameters del DNA si existen, si no, Default 0.30 (30% capital alocado por trade)
        leverage = float(dna['leverage'])
        # Para el Nano Engine, el Leverage que se aplica al margen debe ser conocido
        # Nota: si pasamos 10x, el notional_size = target_margin * leverage.
        
        # Call Numba JIT Function
        final_capital, global_pnl, max_drawdown, win_rate, total_trades = simulate_nano_portfolio_events_jit(
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            pct_ret=pct_ret,
            is_win=is_win,
            is_long=is_long,
            entry_prices=entry_prices,
            horizons=horizons,
            symbols=symbols,
            initial_capital=self.capital,
            min_notional=5.50,          # Binance minimum limits
            kelly_fraction=float(dna['kelly_fraction']),
            max_concurrent=int(dna['max_concurrent']),
            leverage=leverage,
            round_trip_fee=0.00075
        )
            
        return {
            'final_capital': final_capital,
            'pnl': global_pnl,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': total_trades
        }
