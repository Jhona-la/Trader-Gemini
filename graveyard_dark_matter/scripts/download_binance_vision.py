"""
📥 CTOS VISION DOWNLOADER (Institutional Backtest Engine)
Descarga y procesa datos L2 y AggTrades históricos directamente de Binance Vision.
Convierte gigabytes de CSVs tick-by-tick en un Parquet de resolución 1-minuto.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
import time
import urllib.request
import zipfile
import datetime
import polars as pl
from utils.logger import logger

def get_vision_url(data_type: str, symbol: str, date_str: str) -> str:
    """Construye la URL de Binance Vision."""
    base = "https://data.binance.vision/data/futures/um/daily"
    # Format: /bookTicker/BTCUSDT/BTCUSDT-bookTicker-YYYY-MM-DD.zip
    # Format: /aggTrades/BTCUSDT/BTCUSDT-aggTrades-YYYY-MM-DD.zip
    return f"{base}/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.zip"

def download_and_extract_csv(url: str) -> io.BytesIO:
    """Descarga el ZIP en memoria y retorna el contenido del CSV."""
    logger.info(f"⬇️ Descargando: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            zip_bytes = response.read()
            
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_name = z.namelist()[0]
            return io.BytesIO(z.read(csv_name))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.warning(f"⚠️ Archivo no encontrado (404): {url}")
        else:
            logger.error(f"❌ Error HTTP {e.code} descargando {url}")
        return None
    except Exception as e:
        logger.error(f"❌ Error descargando {url}: {e}")
        return None

def process_daily_data(symbol: str, date: datetime.date) -> pl.DataFrame:
    """Descarga bookTicker y aggTrades para un día, resamplea a 1 minuto y cruza (join)."""
    date_str = date.strftime('%Y-%m-%d')
    sym_clean = symbol.replace("/", "").upper()
    
    # 1. Procesar BookTicker (L2)
    book_url = get_vision_url("bookTicker", sym_clean, date_str)
    book_csv = download_and_extract_csv(book_url)
    
    df_l2 = None
    if book_csv:
        # Schema: update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time
        try:
            df_book = pl.read_csv(book_csv)
            # Agrupar por minuto (transaction_time está en milisegundos)
            df_book = df_book.with_columns([
                ((pl.col("transaction_time") // 60000) * 60000).alias("timestamp_ms"),
                (pl.col("best_ask_price") - pl.col("best_bid_price")).alias("spread"),
                # Microprice: (Ask * BidQty + Bid * AskQty) / (BidQty + AskQty)
                ((pl.col("best_ask_price") * pl.col("best_bid_qty") + pl.col("best_bid_price") * pl.col("best_ask_qty")) / 
                 (pl.col("best_bid_qty") + pl.col("best_ask_qty") + 1e-9)).alias("microprice"),
                ((pl.col("best_ask_price") + pl.col("best_bid_price")) / 2.0).alias("mid_price")
            ])
            
            # Aproximación súper rápida de OFI (Order Flow Imbalance) en backtest:
            # delta_bid_qty - delta_ask_qty (simplificado por barra para no hacer cálculos iterativos costosos)
            df_book = df_book.with_columns([
                (pl.col("best_bid_qty").diff() - pl.col("best_ask_qty").diff()).fill_null(0).alias("ofi_tick"),
                ((pl.col("microprice") - pl.col("mid_price")) / (pl.col("mid_price") + 1e-9)).alias("micro_dist")
            ])
            
            df_l2 = df_book.group_by("timestamp_ms").agg([
                pl.col("spread").mean().alias("l2_spread"),
                pl.col("ofi_tick").sum().alias("l2_ofi"),
                pl.col("micro_dist").mean().alias("l2_microprice_dist")
            ])
        except Exception as e:
            logger.error(f"Error procesando BookTicker CSV para {date_str}: {e}")
            
    # 2. Procesar AggTrades (Whale Flow)
    trade_url = get_vision_url("aggTrades", sym_clean, date_str)
    trade_csv = download_and_extract_csv(trade_url)
    
    df_trades = None
    if trade_csv:
        # Schema: agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
        try:
            df_agg = pl.read_csv(trade_csv)
            # is_buyer_maker=True -> Venta a mercado. False -> Compra a mercado.
            df_agg = df_agg.with_columns([
                ((pl.col("transact_time") // 60000) * 60000).alias("timestamp_ms"),
                (pl.col("price") * pl.col("quantity")).alias("usd_value")
            ])
            
            # Whale flow: Solo trades > $100,000 USD
            df_agg = df_agg.with_columns([
                pl.when(pl.col("usd_value") > 100000)
                  .then(pl.when(pl.col("is_buyer_maker")).then(-pl.col("usd_value")).otherwise(pl.col("usd_value")))
                  .otherwise(0.0)
                  .alias("whale_flow_tick")
            ])
            
            df_trades = df_agg.group_by("timestamp_ms").agg([
                pl.col("whale_flow_tick").sum().alias("whale_flow")
            ])
        except Exception as e:
            logger.error(f"Error procesando AggTrades CSV para {date_str}: {e}")
            
    # 3. Combinar ambos
    if df_l2 is not None and df_trades is not None:
        df_merged = df_l2.join(df_trades, on="timestamp_ms", how="outer_coalesce").sort("timestamp_ms")
    elif df_l2 is not None:
        df_merged = df_l2.with_columns(pl.lit(0.0).alias("whale_flow")).sort("timestamp_ms")
    elif df_trades is not None:
        df_merged = df_trades.with_columns([pl.lit(0.0).alias("l2_spread"), pl.lit(0.0).alias("l2_ofi"), pl.lit(0.0).alias("l2_microprice_dist")]).sort("timestamp_ms")
    else:
        return None
        
    return df_merged

def generate_vision_cache(symbol: str, start_date: datetime.date, end_date: datetime.date):
    """Genera el cache local combinando múltiples días."""
    cache_dir = "data/cache_parquet"
    os.makedirs(cache_dir, exist_ok=True)
    sym_clean = symbol.replace("/", "").upper()
    cache_file = os.path.join(cache_dir, f"{sym_clean}_vision_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet")
    
    if os.path.exists(cache_file):
        logger.info(f"✅ Cache Vision ya existe: {cache_file}")
        return pl.read_parquet(cache_file)
        
    logger.info(f"🚀 Iniciando construcción de Cache Vision para {symbol} desde {start_date} hasta {end_date}")
    dfs = []
    
    curr_date = start_date
    while curr_date <= end_date:
        df_day = process_daily_data(symbol, curr_date)
        if df_day is not None and len(df_day) > 0:
            dfs.append(df_day)
        curr_date += datetime.timedelta(days=1)
        
    if not dfs:
        logger.error(f"❌ No se pudo recolectar data Vision para {symbol}")
        return None
        
    final_df = pl.concat(dfs).sort("timestamp_ms")
    
    # Fill nulls
    final_df = final_df.fill_null(0.0)
    
    # Save to parquet
    final_df.write_parquet(cache_file, compression="zstd")
    logger.info(f"💾 Guardado Cache Vision: {cache_file} ({len(final_df)} min-bars)")
    return final_df

if __name__ == "__main__":
    from config import Config
    import concurrent.futures
    
    # 3 Days of history for the backtest
    end = datetime.date.today() - datetime.timedelta(days=1) # Yesterday
    start = end - datetime.timedelta(days=2) # 3 days total
    
    pairs = getattr(Config.Data, 'CRYPTO_FUTURES_PAIRS', getattr(Config, 'CRYPTO_FUTURES_PAIRS', ["BTC/USDT", "ETH/USDT"]))
    
    logger.info(f"🚀 Iniciando descarga masiva de Binance Vision para {len(pairs)} pares. [{start} a {end}]")
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(generate_vision_cache, pair, start, end): pair for pair in pairs}
        for future in concurrent.futures.as_completed(futures):
            pair = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"❌ Error masivo en {pair}: {e}")
                
    logger.info(f"✅ Rehidratación masiva de datos Vision completada.")
