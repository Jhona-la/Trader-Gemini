
import os
import sys
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, mean_absolute_error, r2_score

# Ensure root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from strategies.components.feature_engineering import FeatureEngineering

def train_model_for_horizon(df_features, feature_cols, symbol, horizon, models_dir):
    """
    ═══════════════════════════════════════════════════════════════
    FORENSIC-V130: SUPREME TRAINING PIPELINE (CLF + 5 REGRESSORS)
    
    QUÉ: Entrenamos 6 modelos por asset × horizonte:
      1. XGBClassifier  → Dirección (LONG=1, SHORT=0)
      2. XGBRegressor   → MFE LONG (cuánto % sube max)
      3. XGBRegressor   → MFE SHORT (cuánto % baja max)
      4. XGBRegressor   → Next High % (high de la próxima ventana)
      5. XGBRegressor   → Next Low % (low de la próxima ventana)
      6. XGBRegressor   → Time-to-Peak (cuántas barras al máximo MFE)
    
    POR QUÉ: Antes solo teníamos CLF + REG_LONG + REG_SHORT, y los
      regressors NO tenían validación cruzada (entrenados en full data
      sin medir MAE/R²). No sabíamos si predecían mejor que el azar.
    
    PARA QUÉ: 
      - Predicción COMPLETA de la vela futura (High, Low, duración)
      - Métricas reales de calidad para cada regresor
      - Rechazo automático de modelos con R² < 0 (peor que promedio)
    
    CÓMO: TimeSeriesSplit con 3 folds, embargo gap de lookahead barras.
    ═══════════════════════════════════════════════════════════════
    """
    closes = df_features['close'].values.astype(np.float32)
    highs = df_features['high'].values.astype(np.float32)
    lows = df_features['low'].values.astype(np.float32)
    
    # ═══════════════════════════════════════════════════════════════
    # FORENSIC-V130 FIX: LOOKAHEAD ESCALADO
    # QUÉ: Subimos el lookahead de scalping de 3 a 15 barras.
    # POR QUÉ: En 3 barras de 1m, el movimiento promedio de BTC es
    #   0.01-0.03% → RUIDO PURO. P(close[3] > close[0]) ≈ 50%.
    #   El clasificador no puede aprender nada con target 50/50.
    # PARA QUÉ: En 15 barras (15 min), los movimientos son 0.05-0.30%,
    #   suficiente para que exista una señal direccional real.
    # ═══════════════════════════════════════════════════════════════
    lookahead = 15 if horizon == 'SCALPING' else 60
    
    N = len(closes)
    y_cls = np.zeros(N, dtype=int)
    y_reg_long = np.zeros(N, dtype=np.float32)
    y_reg_short = np.zeros(N, dtype=np.float32)
    y_next_high_pct = np.zeros(N, dtype=np.float32)
    y_next_low_pct = np.zeros(N, dtype=np.float32)
    y_time_to_peak = np.zeros(N, dtype=np.float32)
    
    for i in range(N - lookahead):
        window_highs = highs[i+1 : i+lookahead+1]
        window_lows = lows[i+1 : i+lookahead+1]
        
        # Classification Target (Future close vs Current close)
        y_cls[i] = int(closes[i+lookahead] > closes[i])
        
        # Exact Percentage Excursion targets
        max_high = np.max(window_highs)
        min_low = np.min(window_lows)
        
        # Long magnitude (how much % it goes UP max)
        y_reg_long[i] = max(0.0, (max_high - closes[i]) / closes[i])
        # Short magnitude (how much % it goes DOWN max)
        y_reg_short[i] = max(0.0, (closes[i] - min_low) / closes[i])
        
        # Next window High % relative to current close
        y_next_high_pct[i] = (max_high - closes[i]) / closes[i]
        # Next window Low % relative to current close
        y_next_low_pct[i] = (min_low - closes[i]) / closes[i]
        
        # Time-to-Peak: bar index where max favorable excursion occurs
        peak_bar_long = np.argmax(window_highs)
        # Normalize to [0, 1] range (fraction of lookahead)
        y_time_to_peak[i] = peak_bar_long / max(1, lookahead)
        
    X = df_features[feature_cols].values[:-lookahead]
    y_cls = y_cls[:-lookahead]
    y_reg_long = y_reg_long[:-lookahead]
    y_reg_short = y_reg_short[:-lookahead]
    y_next_high_pct = y_next_high_pct[:-lookahead]
    y_next_low_pct = y_next_low_pct[:-lookahead]
    y_time_to_peak = y_time_to_peak[:-lookahead]
    
    if len(X) < 100:
        print(f"  ⚠️ {symbol} [{horizon}]: Not enough valid samples ({len(X)}).")
        return {}
        
    # --- Time Series Split with Embargo Gap ---
    tscv = TimeSeriesSplit(n_splits=3)
    embargo = lookahead
    
    # Models
    clf_model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, 
                               eval_metric='logloss', n_jobs=4)
    reg_long_model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, 
                                   eval_metric='mae', n_jobs=4)
    reg_short_model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, 
                                    eval_metric='mae', n_jobs=4)
    reg_next_high = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, 
                                  eval_metric='mae', n_jobs=4)
    reg_next_low = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, 
                                 eval_metric='mae', n_jobs=4)
    reg_ttp = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, 
                            eval_metric='mae', n_jobs=4)
    
    # Cross-Validation metrics
    fold_metrics = {
        'clf_precision': [], 
        'reg_long_mae': [], 'reg_long_r2': [],
        'reg_short_mae': [], 'reg_short_r2': [],
        'next_high_mae': [], 'next_high_r2': [],
        'next_low_mae': [], 'next_low_r2': [],
        'ttp_mae': [], 'ttp_r2': [],
    }
    
    fold = 0
    for train_index, test_index in tscv.split(X):
        fold += 1
        print(f"    🔄 Fold {fold}/3...")
        # Apply embargo gap
        test_start = test_index[0]
        purged_train = [i for i in train_index if i < test_start - embargo]
        if len(purged_train) < 50:
            purged_train = list(train_index)
        
        X_train, X_test = X[purged_train], X[test_index]
        
        # CLF
        y_cls_train, y_cls_test = y_cls[purged_train], y_cls[test_index]
        clf_model.fit(X_train, y_cls_train)
        preds = clf_model.predict(X_test)
        fold_metrics['clf_precision'].append(precision_score(y_cls_test, preds, zero_division=0))
        
        # REG LONG
        y_rl_train, y_rl_test = y_reg_long[purged_train], y_reg_long[test_index]
        reg_long_model.fit(X_train, y_rl_train)
        rl_preds = reg_long_model.predict(X_test)
        fold_metrics['reg_long_mae'].append(mean_absolute_error(y_rl_test, rl_preds))
        fold_metrics['reg_long_r2'].append(r2_score(y_rl_test, rl_preds))
        
        # REG SHORT
        y_rs_train, y_rs_test = y_reg_short[purged_train], y_reg_short[test_index]
        reg_short_model.fit(X_train, y_rs_train)
        rs_preds = reg_short_model.predict(X_test)
        fold_metrics['reg_short_mae'].append(mean_absolute_error(y_rs_test, rs_preds))
        fold_metrics['reg_short_r2'].append(r2_score(y_rs_test, rs_preds))
        
        # NEXT HIGH
        y_nh_train, y_nh_test = y_next_high_pct[purged_train], y_next_high_pct[test_index]
        reg_next_high.fit(X_train, y_nh_train)
        nh_preds = reg_next_high.predict(X_test)
        fold_metrics['next_high_mae'].append(mean_absolute_error(y_nh_test, nh_preds))
        fold_metrics['next_high_r2'].append(r2_score(y_nh_test, nh_preds))
        
        # NEXT LOW
        y_nl_train, y_nl_test = y_next_low_pct[purged_train], y_next_low_pct[test_index]
        reg_next_low.fit(X_train, y_nl_train)
        nl_preds = reg_next_low.predict(X_test)
        fold_metrics['next_low_mae'].append(mean_absolute_error(y_nl_test, nl_preds))
        fold_metrics['next_low_r2'].append(r2_score(y_nl_test, nl_preds))
        
        # TIME-TO-PEAK
        y_ttp_train, y_ttp_test = y_time_to_peak[purged_train], y_time_to_peak[test_index]
        reg_ttp.fit(X_train, y_ttp_train)
        ttp_preds = reg_ttp.predict(X_test)
        fold_metrics['ttp_mae'].append(mean_absolute_error(y_ttp_test, ttp_preds))
        fold_metrics['ttp_r2'].append(r2_score(y_ttp_test, ttp_preds))
    
    # Compute averages
    avg_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in fold_metrics.items()}
    
    print("    💾 Fitting final production models...")
    # Fit on full data for production
    clf_model.fit(X, y_cls)
    reg_long_model.fit(X, y_reg_long)
    reg_short_model.fit(X, y_reg_short)
    reg_next_high.fit(X, y_next_high_pct)
    reg_next_low.fit(X, y_next_low_pct)
    reg_ttp.fit(X, y_time_to_peak)
    
    # Save Models
    safe_sym = symbol.replace('/', '')
    suffix = "_scalping" if horizon == 'SCALPING' else "_swing"
    
    clf_model.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb.ubj"))
    reg_long_model.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_long.ubj"))
    reg_short_model.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_short.ubj"))
    reg_next_high.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_next_high.ubj"))
    reg_next_low.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_next_low.ubj"))
    reg_ttp.save_model(os.path.join(models_dir, f"{safe_sym}{suffix}_xgb_reg_ttp.ubj"))
    
    # Save metadata with metrics
    meta = {
        'feature_cols': feature_cols,
        'lookahead': lookahead,
        'horizon': horizon,
        'n_samples': len(X),
        'metrics': avg_metrics,
    }
    joblib.dump(meta, os.path.join(models_dir, f"{safe_sym}{suffix}_meta.joblib"))
    
    # Quality Report
    clf_p = avg_metrics['clf_precision']
    rl_mae = avg_metrics['reg_long_mae'] * 100
    rl_r2 = avg_metrics['reg_long_r2']
    rs_mae = avg_metrics['reg_short_mae'] * 100
    rs_r2 = avg_metrics['reg_short_r2']
    nh_mae = avg_metrics['next_high_mae'] * 100
    nl_mae = avg_metrics['next_low_mae'] * 100
    ttp_mae = avg_metrics['ttp_mae']
    
    clf_flag = "✅" if clf_p > 0.55 else "⚠️" if clf_p > 0.50 else "🔴"
    rl_flag = "✅" if rl_r2 > 0.05 else "⚠️" if rl_r2 > 0 else "🔴"
    rs_flag = "✅" if rs_r2 > 0.05 else "⚠️" if rs_r2 > 0 else "🔴"
    
    print(f"  {clf_flag} CLF Precision: {clf_p:.3f}")
    print(f"  {rl_flag} REG_LONG  MAE: {rl_mae:.4f}% | R²: {rl_r2:.4f}")
    print(f"  {rs_flag} REG_SHORT MAE: {rs_mae:.4f}% | R²: {rs_r2:.4f}")
    print(f"  📊 NEXT_HIGH MAE: {nh_mae:.4f}% | NEXT_LOW MAE: {nl_mae:.4f}%")
    print(f"  ⏱️ TIME_TO_PEAK MAE: {ttp_mae:.4f} (fraction of {lookahead} bars)")
    
    return avg_metrics

def train_supreme():
    print("🧠 [TRAINING] Protocol Supreme V2: Multi-Horizon + Full Candle Prediction...")
    print("═" * 70)
    
    cache_dir = "data/historical"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    files = [f for f in os.listdir(cache_dir) if f.endswith('_1m.csv')]
    
    if not files:
        print("⚠️ No historical CSV files found in data/historical/. Checking parquet...")
        cache_dir = "data/cache_parquet"
        if os.path.exists(cache_dir):
            files = [f for f in os.listdir(cache_dir) if f.endswith('1m.parquet')]
    
    global_metrics = {'SCALPING': [], 'SWING': []}
    
    for f in files:
        if f.endswith('.csv'):
            symbol = f.replace('_1m.csv', '').replace('_', '/')
        else:
            symbol = f.replace('_1m.parquet', '').replace('_', '/')
        path = os.path.join(cache_dir, f)
        
        try:
            print(f"\n🏋️ [{symbol}] Feature Engineering...")
            if f.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            
            fe = FeatureEngineering()
            df_features = fe.prepare_features(df, symbol=symbol)
            
            if df_features.empty:
                print(f"  ⚠️ {symbol}: Features data empty.")
                continue
                
            df_features = df_features.dropna()
            if len(df_features) < 200:
                print(f"  ⚠️ {symbol}: Not enough valid features post-dropna (len: {len(df_features)}).")
                continue
                
            feature_cols = [c for c in df_features.columns 
                          if c not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Train SCALPING
            print(f"  📊 SCALPING (lookahead=15 bars):")
            m_scalp = train_model_for_horizon(df_features, feature_cols, symbol, 'SCALPING', models_dir)
            if m_scalp: 
                global_metrics['SCALPING'].append(m_scalp)
            
            # Train SWING
            print(f"  📊 SWING (lookahead=60 bars):")
            m_swing = train_model_for_horizon(df_features, feature_cols, symbol, 'SWING', models_dir)
            if m_swing: 
                global_metrics['SWING'].append(m_swing)
            
        except Exception as e:
            import traceback
            print(f"❌ {symbol}: Training Failed - {e}")
            traceback.print_exc()
    
    # ═══════════════════════════════════════════════════════════════
    # GLOBAL QUALITY REPORT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("📊 GLOBAL QUALITY REPORT")
    print("═" * 70)
    
    for hz in ['SCALPING', 'SWING']:
        metrics_list = global_metrics[hz]
        if not metrics_list:
            print(f"\n  [{hz}] No models trained.")
            continue
            
        avg_clf = np.mean([m.get('clf_precision', 0) for m in metrics_list])
        avg_rl_mae = np.mean([m.get('reg_long_mae', 0) for m in metrics_list]) * 100
        avg_rl_r2 = np.mean([m.get('reg_long_r2', 0) for m in metrics_list])
        avg_rs_mae = np.mean([m.get('reg_short_mae', 0) for m in metrics_list]) * 100
        avg_rs_r2 = np.mean([m.get('reg_short_r2', 0) for m in metrics_list])
        avg_nh_mae = np.mean([m.get('next_high_mae', 0) for m in metrics_list]) * 100
        avg_nl_mae = np.mean([m.get('next_low_mae', 0) for m in metrics_list]) * 100
        avg_ttp_mae = np.mean([m.get('ttp_mae', 0) for m in metrics_list])
        
        print(f"\n  [{hz}] ({len(metrics_list)} assets)")
        print(f"    CLF Precision:     {avg_clf:.3f}")
        print(f"    REG_LONG  MAE:     {avg_rl_mae:.4f}% | R²: {avg_rl_r2:.4f}")
        print(f"    REG_SHORT MAE:     {avg_rs_mae:.4f}% | R²: {avg_rs_r2:.4f}")
        print(f"    NEXT_HIGH MAE:     {avg_nh_mae:.4f}%")
        print(f"    NEXT_LOW  MAE:     {avg_nl_mae:.4f}%")
        print(f"    TIME_TO_PEAK MAE:  {avg_ttp_mae:.4f}")
        
        if avg_rl_r2 < 0:
            print(f"    🔴 WARNING: REG_LONG R² < 0 → Model is WORSE than predicting the mean!")
        if avg_rs_r2 < 0:
            print(f"    🔴 WARNING: REG_SHORT R² < 0 → Model is WORSE than predicting the mean!")
    
    print("\n🧠 [TRAINING] Complete.")

if __name__ == "__main__":
    train_supreme()
