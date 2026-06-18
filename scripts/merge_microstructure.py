import os
import sys
import polars as pl
import time

def merge_microstructure(symbol="UNIUSDT", days=7):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_dir = os.path.join(base_dir, "data", "history")
    
    agg_path = os.path.join(history_dir, f"{symbol}_aggTrades_{days}d.parquet")
    book_path = os.path.join(history_dir, f"{symbol}_bookTicker_{days}d.parquet")
    
    if not os.path.exists(agg_path) or not os.path.exists(book_path):
        print(f"❌ Faltan archivos base para {symbol}. Asegúrate de ejecutar los descargadores primero.")
        sys.exit(1)
        
    print(f"🧬 Iniciando Fusión As-Of (Zero-Copy) para {symbol}...")
    t0 = time.time()
    
    # Cargar LazyFrames para no reventar la RAM antes del join
    df_agg = pl.scan_parquet(agg_path)
    df_book = pl.scan_parquet(book_path)
    
    # Asegurar que ambos timestamps sean Int64 puros, ignorando cabeceras de texto
    df_agg = df_agg.with_columns(pl.col("timestamp").cast(pl.Int64, strict=False))
    df_book = df_book.with_columns(pl.col("timestamp").cast(pl.Int64, strict=False))
    
    # Filtrar posibles nulos y ordenar
    df_agg = df_agg.drop_nulls("timestamp").sort("timestamp")
    df_book = df_book.drop_nulls("timestamp").sort("timestamp")
    
    # As-Of Join Backward: Para cada aggTrade, buscar el bookTicker más reciente inmediatamente anterior o igual.
    # tolerance: opcional, si no queremos emparejar si el bookTicker es muy viejo. No usamos tolerancia aquí porque en crypto el book es continuo.
    df_merged = df_agg.join_asof(
        df_book,
        on="timestamp",
        strategy="backward"
    )
    
    # Materializar en memoria
    print("⏳ Materializando join en RAM...")
    final_df = df_merged.collect()
    
    t1 = time.time()
    print(f"✅ Fusión completada en {t1-t0:.2f}s")
    print(f"   Filas resultantes: {len(final_df):,}")
    
    # Imprimir un sample para verificar alineación
    print("\n🔍 Muestra de Microestructura Pura:")
    print(final_df.head(5).select(["timestamp", "price", "volume", "is_buyer_maker", "bid_qty", "ask_qty"]))
    
    out_path = os.path.join(history_dir, f"{symbol}_merged_microstructure_{days}d.parquet")
    final_df.write_parquet(out_path)
    print(f"💾 Guardado en: {out_path}")

if __name__ == "__main__":
    merge_microstructure("UNIUSDT", 7)
