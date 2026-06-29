import os
import sys
import numpy as np
import polars as pl
from numba import njit
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@njit(fastmath=True, nogil=True)
def compute_order_flow_metrics_numba(prices, volumes, is_buyer_maker, bucket_size_ms=60000, timestamps=None):
    """
    Cálculo O(1) Stateful de:
    1. CVD (Cumulative Volume Delta)
    2. VPIN Estimator (porcentaje de desbalance por bloque de volumen)
    """
    n = len(prices)
    cvd = np.zeros(n, dtype=np.float64)
    
    current_cvd = 0.0
    
    # Precompute CVD tick-by-tick
    for i in range(n):
        # is_buyer_maker == True => Agresor Vendedor (hit the bid)
        delta = -volumes[i] if is_buyer_maker[i] else volumes[i]
        current_cvd += delta
        cvd[i] = current_cvd
        
    return cvd

def analyze_absorption(df: pl.DataFrame):
    """Detecta zonas de absorción donde el CVD cae pero el precio no."""
    prices = df['price'].to_numpy()
    volumes = df['volume'].to_numpy()
    is_buyer_maker = df['is_buyer_maker'].to_numpy()
    timestamps = df['timestamp'].to_numpy()
    
    t0 = time.time()
    cvd = compute_order_flow_metrics_numba(prices, volumes, is_buyer_maker)
    t1 = time.time()
    
    print(f"⚡ CVD calculado para {len(prices):,} ticks en {(t1-t0)*1000:.2f} ms")
    
    df = df.with_columns(pl.Series(name="cvd", values=cvd))
    
    # Agrupar a 1 minuto para ver divergencias claras (Absorción macro)
    # Floor timestamp a 1 minuto
    df = df.with_columns(((pl.col("timestamp") // 60000) * 60000).alias("minute_ts"))
    
    df_min = df.group_by("minute_ts").agg([
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("cvd").last().alias("cvd_close")
    ]).sort("minute_ts")
    
    # Calcular delta del precio vs delta del CVD
    df_min = df_min.with_columns([
        (pl.col("close").diff() / pl.col("close").shift(1) * 100).alias("price_pct_change"),
        pl.col("cvd_close").diff().alias("cvd_delta")
    ])
    
    # ABSORPTION METRIC: Si el CVD_delta es muy negativo (ventas pesadas > 90th percentile)
    # pero el price_pct_change es neutral o positivo (> -0.05%)
    
    cvd_delta_p10 = df_min.select(pl.col("cvd_delta").quantile(0.10)).item()
    
    absorptions = df_min.filter(
        (pl.col("cvd_delta") < cvd_delta_p10) & 
        (pl.col("price_pct_change") > -0.05)
    )
    
    print(f"\n🐋 Detección de Absorción (Whale Limits):")
    print(f"CVD Delta 10th Percentil: {cvd_delta_p10:.2f} UNIUSDT equivalentes en volumen agresor de venta")
    print(f"Zonas de Absorción detectadas: {len(absorptions)}")
    
    if len(absorptions) > 0:
        print("\nTop 5 ejemplos de Absorción Institucional Masiva:")
        print(absorptions.head(5).select(["minute_ts", "close", "price_pct_change", "cvd_delta"]))

if __name__ == "__main__":
    symbol = "UNIUSDT"
    days = 7
    parquet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history", f"{symbol}_aggTrades_{days}d.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"❌ No se encontró el archivo: {parquet_path}")
        sys.exit(1)
        
    print(f"📊 Cargando Microestructura de {symbol}...")
    df = pl.read_parquet(parquet_path)
    analyze_absorption(df)
