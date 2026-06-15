import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
import joblib

# Ensure root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from scripts.vector_backtest.feature_engine import VectorFeatureEngine
from strategies.ml_strategy import UniversalEnsembleStrategy
from utils.logger import logger

class QuantumVectorTrainer:
    def __init__(self, days=30):
        self.days = days
        # Tomamos el TOP 5 para hacer la prueba Quantum
        self.symbols = Config.CORE_SYMBOLS[:5]
        self.horizons = ["SCALPING", "SWING"]
        
    def _generate_labels(self, df, symbol, horizon):
        profile = Config.AdaptiveProfileEngine.get(symbol, horizon)
        tp_pct = profile["track"]["tp_pct_cap"]
        sl_pct = profile["track"]["sl_pct_cap"]
        
        closes = df['close'].values.astype(np.float32)
        highs = df['high'].values.astype(np.float32)
        lows = df['low'].values.astype(np.float32)
        
        N = len(closes)
        y = np.ones(N, dtype=np.int8)  # 1 = Hold (por defecto)
        
        # Lookahead dinámico (15 barras = 15 min para SCALPING, 2880 barras = 48 horas para SWING)
        lookahead = 15 if horizon == "SCALPING" else 2880
        
        logger.info(f"    🏷️ Etiquetando con TP: {tp_pct*100:.2f}%, SL: {sl_pct*100:.2f}%, Ventana: {lookahead}")
        
        # O(N) Loop de Etiquetado "Future Exact Path"
        for i in range(N - lookahead):
            c_price = closes[i]
            target_tp_long = c_price * (1 + tp_pct)
            target_sl_long = c_price * (1 - sl_pct)
            target_tp_short = c_price * (1 - tp_pct)
            target_sl_short = c_price * (1 + sl_pct)
            
            for j in range(1, lookahead + 1):
                h = highs[i+j]
                l = lows[i+j]
                
                # Check Long condition
                if h >= target_tp_long and l > target_sl_long:
                    y[i] = 2  # 2 = Long
                    break
                elif l <= target_sl_long:
                    # Toca SL de Long. Veamos si tocaba TP de Short antes del SL de Short
                    if l <= target_tp_short and h < target_sl_short:
                        y[i] = 0  # 0 = Short
                        break
                    else:
                        break # Rompe y queda en Hold (1)
                
                # Check Short condition
                elif l <= target_tp_short and h < target_sl_short:
                    y[i] = 0  # 0 = Short
                    break
                elif h >= target_sl_short:
                    break
                    
        return y, lookahead

    def train_symbol_horizon(self, symbol, horizon, df_features, feature_cols):
        y, lookahead = self._generate_labels(df_features, symbol, horizon)
        
        X = df_features[feature_cols].fillna(0.0).values
        
        # Cortar la ventana final no resuelta
        X_train = X[:-lookahead]
        y_train = y[:-lookahead]
        
        classes, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(classes, counts))
        logger.info(f"    📊 Distribución de Clases para {horizon}: {class_dist} (0=Short, 1=Hold, 2=Long)")
        
        if len(classes) < 2:
            logger.warning(f"    ⚠️ No hay suficientes clases para entrenar {symbol} {horizon}. Saltando.")
            return
            
        logger.info(f"    🧠 Entrenando XGBoost Inplace (N={len(X_train)})...")
        # Optimizando para CPU rápida (hist)
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            tree_method='hist',
            n_jobs=-1,
            objective='multi:softprob',
            num_class=3
        )
        model.fit(X_train, y_train)
        
        # Guardar en el formato estricto que exige ml_batch_infer.py -> UniversalEnsembleStrategy
        suffix = f"_{horizon}"
        safe_sym = symbol.replace("/", "") + suffix
        xgb_dir = getattr(Config, "MODEL_DIR", ".models")
        os.makedirs(xgb_dir, exist_ok=True)
        
        ubj_path = os.path.join(xgb_dir, f"{safe_sym}_xgb.ubj")
        model.save_model(ubj_path)
        
        meta_path = os.path.join(xgb_dir, f"{safe_sym}_meta.joblib")
        joblib.dump({"feature_cols": feature_cols}, meta_path)
        
        # También guardar los regresores "Dummy" para evitar crashes en la estrategia de producción
        # (La Fase 5 usa XGBoost puro para clasificación rápida y sizing de Kelly)
        dummy_reg = xgb.XGBRegressor(n_estimators=1, max_depth=1, n_jobs=-1)
        dummy_X = np.zeros((10, X_train.shape[1]))
        dummy_y = np.zeros(10)
        dummy_reg.fit(dummy_X, dummy_y)
        
        dummy_reg.save_model(os.path.join(xgb_dir, f"{safe_sym}_xgb_reg_long.ubj"))
        dummy_reg.save_model(os.path.join(xgb_dir, f"{safe_sym}_xgb_reg_short.ubj"))
        
        logger.info(f"    ✅ Modelos guardados para {symbol} [{horizon}] en {ubj_path}")

    def run(self):
        logger.info(f"==================================================")
        logger.info(f"🧬 QUANTUM VECTOR TRAINER INICIADO (Fase 6)")
        logger.info(f"   Días de Historia : {self.days}")
        logger.info(f"   Símbolos         : {self.symbols}")
        logger.info(f"==================================================")
        
        data_dict = {}
        cache_dir = "data/cache_parquet"
        for symbol in self.symbols:
            safe_sym = symbol.replace("/", "")
            # Buscamos cualquier archivo que contenga el simbolo y ".parquet" excluyendo vision
            matched_files = [f for f in os.listdir(cache_dir) if safe_sym in f and f.endswith(".parquet") and "vision" not in f]
            if matched_files:
                # Tomar el mas reciente
                matched_files.sort(reverse=True)
                path = os.path.join(cache_dir, matched_files[0])
                logger.info(f"📥 Loading {symbol} from cache: {path}")
                df = pd.read_parquet(path)
                data_dict[symbol] = df
            else:
                logger.warning(f"⚠️ No cache found for {symbol}")
        
        for symbol in self.symbols:
            if symbol not in data_dict or len(data_dict[symbol]) < 1000:
                logger.warning(f"⚠️ Datos insuficientes para {symbol}")
                continue
                
            logger.info(f"\n🧬 PROCESANDO SÍMBOLO: {symbol}")
            
            # Generar Features (Una vez por símbolo)
            df_features_pl = VectorFeatureEngine.compute_all_features(data_dict[symbol], symbol)
            
            # Extract features from dataframe columns
            feature_cols = [c for c in df_features_pl.columns if c not in ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
            
            df_features = df_features_pl.to_pandas()
            
            for horizon in self.horizons:
                self.train_symbol_horizon(symbol, horizon, df_features, feature_cols)
                
        logger.info(f"==================================================")
        logger.info(f"🏆 ENTRENAMIENTO CUÁNTICO COMPLETADO")
        logger.info(f"==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Quantum Vector Trainer')
    parser.add_argument('--days', type=int, default=15, help='Days of historical data')
    args = parser.parse_args()
    
    trainer = QuantumVectorTrainer(days=args.days)
    trainer.run()
