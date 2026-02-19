
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import lightgbm as lgb
import gc
import logging
import traceback

# Setup basic logging for the worker process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ML_Worker")

from utils.shm_utils import load_shared_array

def train_model_process(symbol, X_shm_info, y_shm_info, params, prev_xgb_model=None, prev_gb_model=None):
    """
    Pure function that runs in a separate process to train the ML ensemble.
    
    Args:
        symbol (str): Symbol being trained
        X_shm_info (dict): {'name': str, 'shape': tuple, 'dtype': str, 'columns': list}
        y_shm_info (dict): {'name': str, 'shape': tuple, 'dtype': str}
        # ... rest same
    """
    try:
        # 1. Reconstruct X from Shared Memory (Zero-Copy Read)
        X_arr, X_shm = load_shared_array(X_shm_info['name'], X_shm_info['shape'], X_shm_info['dtype'])
        if X_arr is None:
            raise ValueError("Could not attach to SharedMemory for X")
            
        # Reconstruct DataFrame (Zero-Copy View if possible)
        # copy=False ensures we point to simple buffer
        X_df = pd.DataFrame(X_arr, columns=X_shm_info['columns'], copy=False)
        
        # 2. Reconstruct y from Shared Memory
        y_arr, y_shm = load_shared_array(y_shm_info['name'], y_shm_info['shape'], y_shm_info['dtype'])
        if y_arr is None:
             raise ValueError("Could not attach to SharedMemory for y")
        
        y_series = pd.Series(y_arr, copy=False)
        
        feature_cols = X_shm_info['columns']
        
        # Unpack params
        n_estimators = params.get('n_estimators', 100)
        max_depth_rf = params.get('max_depth_rf', 6)
        max_depth_xgb = params.get('max_depth_xgb', 5)
        max_depth_gb = params.get('max_depth_gb', 5)
        learning_rate = params.get('learning_rate', 0.1)
        subsample = params.get('subsample', 0.8)
        training_iteration = params.get('training_iteration', 0)
        base_rf_weight = params.get('rf_weight', 0.45)
        base_xgb_weight = params.get('xgb_weight', 0.35)
        base_gb_weight = params.get('gb_weight', 0.20)
        
        # Cross Validation Setup
        n_samples = len(X_df)
        if n_samples < 5:
            # Minimal mode
            indices = list(range(n_samples))
            splitter = [(indices, indices)]
        else:
            tscv = TimeSeriesSplit(n_splits=3)
            splitter = tscv.split(X_df)
            
        cv_scores = {'rf': [], 'xgb': [], 'gb': []}
        best_models = {'rf': None, 'xgb': None, 'gb': None}
        best_scaler = None
        
        # Loop folds
        for fold, (train_idx, test_idx) in enumerate(splitter):
            X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train).astype('float32')
            X_test_scaled = scaler.transform(X_test).astype('float32')
            
            # --- 1. Random Forest ---
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth_rf,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                n_jobs=1, # Single core per job
                random_state=42 + training_iteration,
                class_weight='balanced'
            )
            rf.fit(X_train_scaled, y_train)
            cv_scores['rf'].append(rf.score(X_test_scaled, y_test))
            
            # --- 2. XGBoost ---
            xgb = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth_xgb,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=0.8,
                n_jobs=1,
                tree_method='hist',
                random_state=42 + training_iteration,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )
            # Incremental support check
            xgb_model_arg = None
            if prev_xgb_model:
                # Basic check if features match could be done here or passed in
                xgb_model_arg = prev_xgb_model
                
            xgb.fit(X_train_scaled, y_train, xgb_model=xgb_model_arg)
            cv_scores['xgb'].append(xgb.score(X_test_scaled, y_test))
            
            # --- 3. Gradient Boosting ---
            gb = None
            if prev_gb_model:
                gb = prev_gb_model
                gb.n_estimators += 5
            else:
                gb = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth_gb,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    random_state=42 + training_iteration,
                    warm_start=True
                )
            gb.fit(X_train_scaled, y_train)
            cv_scores['gb'].append(gb.score(X_test_scaled, y_test))
            
            # Save best (last fold)
            is_last_fold = (fold == 2) or (n_samples < 5 and fold == 0)
            if is_last_fold:
                best_models['rf'] = rf
                best_models['xgb'] = xgb
                best_models['gb'] = gb
                best_scaler = scaler
                
            # Cleanup
            del X_train, X_test, X_train_scaled, X_test_scaled, rf, xgb, gb
            gc.collect()
            
        # Calc scores
        avg_rf = np.mean(cv_scores['rf'])
        avg_xgb = np.mean(cv_scores['xgb'])
        avg_gb = np.mean(cv_scores['gb'])
        
        ensemble_score = (avg_rf * base_rf_weight + 
                         avg_xgb * base_xgb_weight + 
                         avg_gb * base_gb_weight)
                         
        metrics = {
            'rf_score': avg_rf,
            'xgb_score': avg_xgb,
            'gb_score': avg_gb,
            'ensemble_score': ensemble_score
        }
        
        # Return everything needed to update the main process strategy
        
        # Cleanup Worker Side Attachments
        try:
            del X_df, y_series # Release Pandas refs to buffer
            X_shm.close()
            y_shm.close()
        except:
            pass
            
        return best_models, best_scaler, feature_cols, ensemble_score, metrics
        
    except Exception as e:
        logger.error(f"Worker Error for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return None, None, [], 0.0, {}
