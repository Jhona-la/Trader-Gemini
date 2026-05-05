import os
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any

logger = logging.getLogger("PETIM_Model")

class GeometryPredictor:
    """
    🧠 POST-ENTRY TRAJECTORY INTELLIGENCE MODULE (PETIM)
    Multi-task learning engine to predict MFE, MAE, Survival Time, and Continuation.
    """
    def __init__(self, symbol: str, timeframe: str = '5m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.models = {
            'direction': xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05),
            'mfe': xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05),
            'mae': xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05),
            'survival': xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, objective='reg:squaredlogerror')
        }
        self.features = []
        self.is_trained = False
        
    def train(self, df: pd.DataFrame, feature_cols: list):
        """Train the multi-task heads on the labeled PETIM dataset."""
        logger.info(f"Training PETIM Multi-Task Engine for {self.symbol} on {len(df)} samples...")
        self.features = feature_cols
        
        X = df[feature_cols].values
        
        # Labels
        y_dir = df['label_continuation'].values
        y_mfe = df['label_mfe'].values
        y_mae = df['label_mae'].values
        y_surv = df['label_survival_time'].values
        
        # Train Direction Head (Classifier)
        logger.info("Training Direction Head...")
        self.models['direction'].fit(X, y_dir)
        
        # Train MFE Head (Regressor)
        logger.info("Training MFE Head...")
        self.models['mfe'].fit(X, y_mfe)
        
        # Train MAE Head (Regressor)
        logger.info("Training MAE Head...")
        self.models['mae'].fit(X, y_mae)
        
        # Train Survival Head (Regressor - Log Error for time)
        logger.info("Training Survival Head...")
        self.models['survival'].fit(X, y_surv)
        
        self.is_trained = True
        logger.info("✅ PETIM Training Complete.")
        
    def predict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Inference for a single feature vector."""
        if not self.is_trained:
            return {"error": "Models not trained"}
            
        # Ensure 2D array
        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector
            
        p_cont = float(self.models['direction'].predict_proba(X)[0][1])
        exp_mfe = float(self.models['mfe'].predict(X)[0])
        exp_mae = float(self.models['mae'].predict(X)[0])
        exp_surv = float(self.models['survival'].predict(X)[0])
        
        # Hazard rate is a derived metric (e.g. 1 / exp_surv) or custom logic
        hazard = 1.0 / max(exp_surv, 1.0)
        
        return {
            "p_continuation": p_cont,
            "expected_mfe": exp_mfe,
            "expected_mae": exp_mae,
            "expected_duration": exp_surv,
            "hazard_rate": hazard
        }
        
    def save(self, directory: str):
        """Saves the 4 model heads to disk."""
        os.makedirs(directory, exist_ok=True)
        sym_safe = self.symbol.replace("/", "_")
        for name, model in self.models.items():
            path = os.path.join(directory, f"petim_{sym_safe}_{name}.json")
            model.save_model(path)
            
        with open(os.path.join(directory, f"petim_{sym_safe}_features.json"), "w") as f:
            json.dump(self.features, f)
            
    def load(self, directory: str) -> bool:
        """Loads models from disk."""
        sym_safe = self.symbol.replace("/", "_")
        try:
            for name, model in self.models.items():
                path = os.path.join(directory, f"petim_{sym_safe}_{name}.json")
                if os.path.exists(path):
                    model.load_model(path)
                else:
                    return False
                    
            feat_path = os.path.join(directory, f"petim_{sym_safe}_features.json")
            with open(feat_path, "r") as f:
                self.features = json.load(f)
                
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error loading PETIM models: {e}")
            return False
