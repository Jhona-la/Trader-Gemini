import os
import sys
import asyncio
import aiohttp
import zipfile
import polars as pl
from datetime import datetime, timedelta
import io

pl.Config.set_tbl_rows(10)

async def download_and_extract(session, url, target_dir, zip_name, sem):
    csv_name = zip_name.replace('.zip', '.csv')
    csv_path = os.path.join(target_dir, csv_name)
    if os.path.exists(csv_path):
        return csv_path
        
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        content = await response.read()
                        with zipfile.ZipFile(io.BytesIO(content)) as z:
                            extracted_name = z.namelist()[0]
                            # Read into memory and write to avoid locking issues
                            data = z.read(extracted_name)
                            out_path = os.path.join(target_dir, extracted_name)
                            with open(out_path, 'wb') as f:
                                f.write(data)
                        return out_path
                    elif response.status == 404:
                        return None
            except Exception as e:
                await asyncio.sleep(1)
                continue
    return None

async def process_coin(symbol, start_date_str, days):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "history", "macro", symbol)
    os.makedirs(target_dir, exist_ok=True)
    
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    dates = [start_dt + timedelta(days=i) for i in range(days)]
    months = sorted(list(set([d.strftime("%Y-%m") for d in dates])))
    
    klines_url = "https://data.binance.vision/data/futures/um/daily/klines"
    metrics_url = "https://data.binance.vision/data/futures/um/daily/metrics"
    funding_url = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
    
    print(f"\n🚀 [{symbol}] Iniciando Ingesta MACRO ASYNC ({days} días)...")
    
    k_urls = [(f"{klines_url}/{symbol}/15m/{symbol}-15m-{d.strftime('%Y-%m-%d')}.zip", f"{symbol}-15m-{d.strftime('%Y-%m-%d')}.zip") for d in dates]
    m_urls = [(f"{metrics_url}/{symbol}/{symbol}-metrics-{d.strftime('%Y-%m-%d')}.zip", f"{symbol}-metrics-{d.strftime('%Y-%m-%d')}.zip") for d in dates]
    f_urls = [(f"{funding_url}/{symbol}/{symbol}-fundingRate-{m}.zip", f"{symbol}-fundingRate-{m}.zip") for m in months]
    
    all_tasks = k_urls + m_urls + f_urls
    
    downloaded_csvs = []
    print(f"⏳ [{symbol}] Descargando {len(all_tasks)} archivos (Concurrencia Máxima)...")
    
    # 50 concurrent connections to binance vision
    sem = asyncio.Semaphore(50)
    
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for url, zname in all_tasks:
            tasks.append(download_and_extract(session, url, target_dir, zname, sem))
            
        results = await asyncio.gather(*tasks)
        for res in results:
            if res:
                downloaded_csvs.append(res)
                
    k_csvs = [c for c in downloaded_csvs if "-15m-" in c]
    m_csvs = [c for c in downloaded_csvs if "-metrics-" in c]
    f_csvs = [c for c in downloaded_csvs if "-fundingRate-" in c]
    
    print(f"🔄 [{symbol}] Procesando DataFrames en memoria (Polars)...")
    
    k_dfs, m_dfs, f_dfs = [], [], []
    
    for csv in k_csvs:
        try:
            df = pl.read_csv(csv, has_header=False, infer_schema_length=0, new_columns=[
                "open_time", "open", "high", "low", "close", "volume", 
                "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ]).select(["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"]).filter(pl.col("open_time") != "open_time").with_columns([
                pl.col("open_time").cast(pl.Int64), pl.col("open").cast(pl.Float64), pl.col("high").cast(pl.Float64), pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64), pl.col("volume").cast(pl.Float64), pl.col("taker_buy_volume").cast(pl.Float64)
            ])
            k_dfs.append(df)
        except: pass
        finally:
            if os.path.exists(csv): os.remove(csv)
            
    df_klines = pl.concat(k_dfs).sort("open_time") if k_dfs else pl.DataFrame()
    
    for csv in m_csvs:
        try:
            df = pl.read_csv(csv, has_header=True)
            if 'create_time' in df.columns:
                df = df.rename({"create_time": "timestamp"})
                if df['timestamp'].dtype == pl.String:
                    try: df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.timestamp("ms"))
                    except: pass
                df = df.with_columns([pl.col("timestamp").cast(pl.Int64), pl.col("sum_open_interest").cast(pl.Float64)]).select(["timestamp", "sum_open_interest"])
                m_dfs.append(df)
        except: pass
        finally:
            if os.path.exists(csv): os.remove(csv)
            
    df_metrics = pl.concat(m_dfs).sort("timestamp") if m_dfs else pl.DataFrame()
    
    for csv in f_csvs:
        try:
            df = pl.read_csv(csv, has_header=True)
            if 'calc_time' in df.columns and 'last_funding_rate' in df.columns:
                df = df.rename({"calc_time": "timestamp", "last_funding_rate": "funding_rate"})
                df = df.with_columns([pl.col("timestamp").cast(pl.Int64), pl.col("funding_rate").cast(pl.Float64)]).select(["timestamp", "funding_rate"])
                f_dfs.append(df)
        except: pass
        finally:
            if os.path.exists(csv): os.remove(csv)
            
    df_funding = pl.concat(f_dfs).sort("timestamp") if f_dfs else pl.DataFrame()
    
    macro_out_dir = os.path.join(base_dir, "data", "history", "macro")
    if not df_klines.is_empty():
        df_klines.write_parquet(os.path.join(macro_out_dir, f"{symbol}_klines_15m.parquet"))
    if not df_metrics.is_empty():
        df_metrics.write_parquet(os.path.join(macro_out_dir, f"{symbol}_metrics.parquet"))
    if not df_funding.is_empty():
        df_funding.write_parquet(os.path.join(macro_out_dir, f"{symbol}_funding.parquet"))
        
    print(f"✅ [{symbol}] Completado: Klines={len(df_klines)}, Metrics={len(df_metrics)}, Funding={len(df_funding)}")

async def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "UNIUSDT"]
    start_date = "2022-01-01"
    
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime("2024-03-31", "%Y-%m-%d")
    days = (d2 - d1).days + 1
    
    print(f"🌍 Iniciando Descarga Masiva ASYNC (I/O Maximizado) para {len(symbols)} activos x {days} días")
    
    for sym in symbols:
        await process_coin(sym, start_date, days)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
