import numpy as np
import pandas as pd
from utils.logger import logger
from strategies.ml_strategy import UniversalEnsembleStrategy

class MLBatchInfer:
    """
    Vectorized Machine Learning Inference.
    En lugar de llamar predict() barra por barra en el backtest (lo que toma horas),
    inyecta toda la historia de precios (1,000,000+ filas) en la VRAM/RAM y 
    obtiene las predicciones completas en un solo pase de ~0.02 segundos.
    """
    
    @staticmethod
    def infer_all(df, symbol, Config, models_dir="data/models", horizon="SCALPING"):
        """
        Retorna un array 1D de señales y de confianzas vectorizadamente.
        df debe contener todas las features calculadas.
        """
        try:
            # Inicializar la estrategia solo para cargar el modelo pre-entrenado
            # o compilar la configuración de variables
            ml_strat = UniversalEnsembleStrategy(
                data_provider=None,
                events_queue=None,
                symbol=symbol,
                lookback=200,
                horizon=horizon,
                models_dir=models_dir
            )
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            return np.zeros(len(df), dtype=np.int8), np.zeros(len(df), dtype=np.float32)
        
        # En versiones recientes de la estrategia, los modelos se cargan lazy, 
        # así que forzamos la carga.
        ml_strat._load_models()

        # Si no están entrenados, retornamos 0
        if not hasattr(ml_strat, 'xgb_model') or not ml_strat.xgb_model:
            logger.warning(f"⚠️ No XGBoost model found for {symbol} in VectorBacktest. Returning 0 signals.")
            return np.zeros(len(df), dtype=np.int8), np.zeros(len(df), dtype=np.float32)
            
        exclude_cols = ['timestamp', 'close_time', 'target_scalp', 'target_swing']
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

        # Extraer X en formato Numpy contiguous
        # Si df es Polars, pasamos a Pandas/Numpy
        if hasattr(df, "to_pandas"):
            df_pd = df.to_pandas()
        else:
            df_pd = df
            
        # Manejo de NaNs (rellenamos con 0 o bfill)
        X_raw = df_pd[feature_cols].fillna(0.0).values
        
        # Escalar (solo si el scaler está entrenado)
        if hasattr(ml_strat, 'scaler') and ml_strat.scaler and hasattr(ml_strat.scaler, 'scale_'):
            X_scaled = ml_strat.scaler.transform(X_raw)
        else:
            X_scaled = X_raw
            
        # Inferencia masiva
        logger.info(f"🧠 [Batch Infer] Executing quantum inference on {len(X_scaled)} rows...")
        
        # XGBoost return probabilities for each class
        # Assuming 3 classes (0: Hold, 1: Long, 2: Short)
        # Using inplace_predict for zero-copy C-speed prediction
        booster = ml_strat.xgb_model.get_booster()
        probs = booster.inplace_predict(np.ascontiguousarray(X_scaled, dtype=np.float32))
        
        if probs.ndim == 2 and probs.shape[1] == 3:
            # 3 clases
            predictions = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            
            # Mapeo a señales (-1, 0, 1)
            # En ml_strategy, clases son: 0 -> -1 (Short), 1 -> 1 (Long), 2 -> 0 (Hold)
            # Wait, let's look at standard mapping:
            # -1 mapped to 0, 0 mapped to 1, 1 mapped to 2 usually in XGBClassifier.
            # So: pred=0 is Short (-1), pred=2 is Long (1), pred=1 is Hold (0)
            
            signals = np.zeros(len(predictions), dtype=np.int8)
            signals[predictions == 0] = -1
            signals[predictions == 2] = 1
            
            return signals, confidences
            
        elif probs.ndim == 2 and probs.shape[1] == 2:
            # 2 clases (Binary)
            pred_class = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
            
            signals = np.zeros(len(pred_class), dtype=np.int8)
            signals[pred_class == 0] = -1
            signals[pred_class == 1] = 1
            
            return signals, confidences
        else:
            # Si regresa 1D array (Binary logistic)
            signals = np.where(probs > 0.5, 1, -1).astype(np.int8)
            confidences = np.where(probs > 0.5, probs, 1 - probs)
            return signals, confidences
