import os
import sys
import polars as pl
import time

pl.Config.set_tbl_rows(10)

def merge_macro_state(symbol="UNIUSDT"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    macro_dir = os.path.join(base_dir, "data", "history", "macro")
    
    klines_path = os.path.join(macro_dir, f"{symbol}_klines_15m.parquet")
    metrics_path = os.path.join(macro_dir, f"{symbol}_metrics.parquet")
    funding_path = os.path.join(macro_dir, f"{symbol}_funding.parquet")
    
    if not os.path.exists(klines_path):
        print("❌ Faltan archivos de klines.")
        sys.exit(1)
        
    print(f"🧬 Iniciando Fusión MACRO (Zero-Copy) para {symbol}...")
    t0 = time.time()
    
    df_k = pl.scan_parquet(klines_path)
    # Renombrar open_time a timestamp para el join
    df_k = df_k.rename({"open_time": "timestamp"}).sort("timestamp")
    
    # Left join metrics
    if os.path.exists(metrics_path):
        df_m = pl.scan_parquet(metrics_path).drop_nulls("timestamp").sort("timestamp")
        df_k = df_k.join_asof(df_m, on="timestamp", strategy="backward")
    else:
        print("⚠️ No se encontró metrics.parquet")
        
    # Left join funding
    if os.path.exists(funding_path):
        df_f = pl.scan_parquet(funding_path).drop_nulls("timestamp").sort("timestamp")
        df_k = df_k.join_asof(df_f, on="timestamp", strategy="backward")
    else:
        print("⚠️ No se encontró funding.parquet")
        
    print("⏳ Materializando join en RAM...")
    final_df = df_k.collect()
    
    # Forward fill (para rellenar huecos si el primer registro de metrics es tardío)
    final_df = final_df.fill_null(strategy="forward").fill_null(0.0)
    
    t1 = time.time()
    print(f"✅ Fusión completada en {t1-t0:.2f}s")
    print(f"   Filas resultantes: {len(final_df):,}")
    
    out_path = os.path.join(macro_dir, f"{symbol}_merged_macro.parquet")
    final_df.write_parquet(out_path)
    print(f"💾 Guardado en: {out_path}")

if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "UNIUSDT"]
    for sym in symbols:
        try:
            merge_macro_state(sym)
        except SystemExit:
            continue
        except Exception as e:
            print(f"Error fusionando {sym}: {e}")
