import polars as pl
from utils.logger import logger
from strategies.components.feature_engineering import FeatureEngineering

class VectorFeatureEngine:
    """
    Motor de características súper-masivo basado en Polars.
    Extrae indicadores técnicos, matemáticos y cuánticos sobre toda la serie 
    de tiempo en paralelo, sin iterar.
    """
    @staticmethod
    def compute_all_features(df_pd, symbol):
        """
        Toma OHLCV en Pandas y retorna Polars/Pandas con TODAS las columnas calculadas.
        Reutilizamos la lógica ya optimizada en FeatureEngineering pero forzamos 
        la salida en Polars para máxima velocidad.
        """
        logger.info(f"⚙️ [FeatureEngine] Starting vectorized calculation for {symbol} ({len(df_pd)} rows)...")
        
        # Instantiate base engine
        fe = FeatureEngineering()
        
        # Le indicamos que regrese el DataFrame en Polars (para máxima eficiencia)
        # Omitimos sentiment por ahora en vector mode extremo
        df_features = fe.prepare_features(
            bars=df_pd,
            market_regime="UNKNOWN", # Will be mapped if needed
            sentiment_loader=None,
            data_provider=None,
            symbol=symbol,
            feature_store=None,
            horizon="SCALPING",
            return_polars=True
        )
        
        logger.info(f"✅ [FeatureEngine] Computed {len(df_features.columns)} features for {symbol}.")
        
        return df_features
