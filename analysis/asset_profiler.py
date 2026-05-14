import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from utils.logger import logger
from data.database import DatabaseHandler

class AssetProfiler:
    """
    Asset Profiler
    Analyzes historical data to determine the 'size' and 'personality' of each asset:
    - ATR (Average True Range)
    - Volatility percentage
    - Relative Liquidity (Volume profile)
    Saves profiles to the database for omniscient use.
    """
    def __init__(self, db_path: str = "trader_gemini.db"):
        self.db = DatabaseHandler(db_path)
        
    def profile_asset(self, symbol: str, ohlcv_df: pd.DataFrame):
        """
        Profiles an asset given a dataframe of OHLCV data.
        ohlcv_df must contain: ['open', 'high', 'low', 'close', 'volume']
        """
        if ohlcv_df.empty or len(ohlcv_df) < 14:
            logger.warning(f"Not enough data to profile {symbol}")
            return
            
        # Calculate ATR(14)
        high = ohlcv_df['high']
        low = ohlcv_df['low']
        close = ohlcv_df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean().iloc[-1]
        
        # Calculate Volatility (%)
        returns = ohlcv_df['close'].pct_change()
        volatility_pct = returns.std() * np.sqrt(len(ohlcv_df)) # Annualized roughly if data is 1Y, else just period vol.
        
        # Average volume 24h
        # Assuming the df is 1-hour candles, last 24 rows = 24h
        if len(ohlcv_df) >= 24:
            avg_volume_24h = ohlcv_df['volume'].iloc[-24:].sum()
        else:
            avg_volume_24h = ohlcv_df['volume'].sum()
            
        # Simple Liquidity Score (Log of volume USD roughly, assuming price * volume)
        last_price = ohlcv_df['close'].iloc[-1]
        usd_volume = avg_volume_24h * last_price
        liquidity_score = np.log1p(usd_volume)
        
        self._save_profile(symbol, atr_14, avg_volume_24h, float(volatility_pct), float(liquidity_score))
        logger.info(f"📊 Profiled {symbol}: ATR={atr_14:.4f}, Vol={volatility_pct*100:.2f}%, Liq={liquidity_score:.2f}")
        
    def _save_profile(self, symbol: str, atr: float, vol: float, volatility_pct: float, liquidity_score: float):
        conn = self.db.get_connection()
        if not conn: return
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO asset_profiles 
                (symbol, atr_14, avg_volume_24h, volatility_pct, liquidity_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol, atr, vol, volatility_pct, liquidity_score, datetime.now(timezone.utc)
            ))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving asset profile for {symbol}: {e}")

if __name__ == "__main__":
    print("AssetProfiler initialized. Run via main engine or simulation to profile assets.")
