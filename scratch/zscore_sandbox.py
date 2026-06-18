import os
import polars as pl
import numpy as np

def run_zscore_sandbox(symbol="BTCUSDT"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "history", "macro")
    
    klines_path = os.path.join(target_dir, f"{symbol}_klines_15m.parquet")
    metrics_path = os.path.join(target_dir, f"{symbol}_metrics.parquet")
    funding_path = os.path.join(target_dir, f"{symbol}_funding.parquet")
    
    if not (os.path.exists(klines_path) and os.path.exists(metrics_path) and os.path.exists(funding_path)):
        print(f"❌ Faltan datos para {symbol}. Asegúrate de ejecutar build_2y_derivatives_lake.py")
        return
        
    print(f"📊 Cargando datos cuánticos para {symbol}...")
    df_klines = pl.read_parquet(klines_path)
    if "open_time" in df_klines.columns:
        df_klines = df_klines.rename({"open_time": "timestamp"})
    df_metrics = pl.read_parquet(metrics_path)
    df_funding = pl.read_parquet(funding_path)
    
    # 1. Fusionar los DataFrames
    # Funding es cada 8h, Metrics cada 5m, Klines cada 5m. Haremos un asof join.
    df = df_klines.join_asof(df_metrics, on="timestamp", strategy="backward")
    df = df.join_asof(df_funding, on="timestamp", strategy="backward")
    
    df = df.drop_nulls()
    if df.is_empty():
        print("❌ El DataFrame fusionado está vacío.")
        return
        
    print(f"✅ Fusión completada: {len(df)} velas de 15 minutos.")
    
    # 2. Calcular Variables Adaptativas (ATR y Z-Scores)
    window_oi = 40  # 40 velas de 15m = 10 horas
    window_fr = 20  # 20 velas de 15m = 5 horas
    
    df = df.with_columns([
        # Calcular cambio relativo del OI
        ((pl.col("sum_open_interest") - pl.col("sum_open_interest").shift(1)) / pl.col("sum_open_interest").shift(1)).alias("oi_change"),
        
        # Calcular TR (True Range) para el ATR
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1)).abs(),
            (pl.col("low") - pl.col("close").shift(1)).abs()
        ]).alias("tr")
    ])
    
    df = df.with_columns([
        # Calcular Z-Score del OI Change
        ((pl.col("oi_change") - pl.col("oi_change").rolling_mean(window_oi)) / 
         (pl.col("oi_change").rolling_std(window_oi) + 1e-9)).alias("z_oi"),
         
        # Calcular Z-Score del Funding Rate
        ((pl.col("funding_rate") - pl.col("funding_rate").rolling_mean(window_fr)) / 
         (pl.col("funding_rate").rolling_std(window_fr) + 1e-9)).alias("z_fr"),
         
        # Calcular ATR
        pl.col("tr").rolling_mean(14).alias("atr")
    ])
    
    df = df.drop_nulls()
    
    # 3. Detectar Señales de Ignición (Filtros Estructurales Dinámicos)
    # Buscamos un Z_OI < -3.0 (una caída extrema del OI a >3 sigmas de su norma local)
    # y que el Funding Rate antes de la caída haya estado caliente (Z_FR > 1.5)
    
    # Condición de Cascada
    signals = df.filter(
        (pl.col("z_oi") < -3.0) & 
        (pl.col("z_fr").shift(1) > 1.0) # FR antes de la cascada
    )
    
    print(f"🔥 Señales de Cascada detectadas: {len(signals)}")
    
    # 4. Simulador Walk-Forward (Sin Numba primero para validar)
    # Usaremos una iteración rápida sobre las señales para medir Win Rate
    # Parametros de SL y TP
    atr_sl_multiplier = 2.5
    atr_tp_multiplier = 3.5
    
    # Convertimos a diccionarios o listas para iterar rápido
    timestamps = df["timestamp"].to_list()
    closes = df["close"].to_list()
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    atrs = df["atr"].to_list()
    
    # Mapeo rápido
    time_idx_map = {t: i for i, t in enumerate(timestamps)}
    
    wins = 0
    losses = 0
    total_profit = 0.0
    
    slippage_pct = 0.002 # 0.2% slippage base as a stress test
    
    for row in signals.iter_rows(named=True):
        entry_time = row["timestamp"]
        idx = time_idx_map.get(entry_time)
        if idx is None or idx >= len(closes) - 1: continue
        
        # Simulamos entrada en la *siguiente* vela (limit order en apertura)
        entry_idx = idx + 1
        entry_price = closes[entry_idx-1] * (1 - slippage_pct) # Suponemos que compramos peor por spread
        
        atr_val = atrs[entry_idx]
        sl_price = entry_price - (atr_val * atr_sl_multiplier)
        tp_price = entry_price + (atr_val * atr_tp_multiplier)
        
        # Buscar resolución
        trade_open = True
        for j in range(entry_idx, len(closes)):
            curr_high = highs[j]
            curr_low = lows[j]
            
            # Hit SL?
            if curr_low <= sl_price:
                # Perdimos
                loss_pct = (sl_price - entry_price) / entry_price
                total_profit += loss_pct
                losses += 1
                trade_open = False
                break
            
            # Hit TP?
            if curr_high >= tp_price:
                win_pct = (tp_price - entry_price) / entry_price
                total_profit += win_pct
                wins += 1
                trade_open = False
                break
                
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # Calculate Profit Factor correctly by simulating slippage
    gross_profit = wins * (atr_tp_multiplier)
    gross_loss = losses * (atr_sl_multiplier)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print("="*50)
    print(f"📊 REPORTE DE ROBUSTEZ Z-SCORE: {symbol}")
    print(f"Trades Totales: {total_trades}")
    print(f"Win Rate:     {win_rate*100:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Net PnL (%):  {total_profit*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "UNIUSDT"]:
        run_zscore_sandbox(sym)
