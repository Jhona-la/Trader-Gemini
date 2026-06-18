import os
import sys
import requests
import zipfile
import polars as pl
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time
import shutil

pl.Config.set_tbl_rows(10)

def download_and_extract(url, target_dir, zip_name):
    zip_path = os.path.join(target_dir, zip_name)
    csv_name = zip_name.replace('.zip', '.csv')
    csv_path = os.path.join(target_dir, csv_name)
    if os.path.exists(csv_path):
        return csv_path
        
    for attempt in range(3):
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                with zipfile.ZipFile(zip_path, 'r') as z:
                    extracted_name = z.namelist()[0]
                    z.extract(extracted_name, target_dir)
                os.remove(zip_path)
                return os.path.join(target_dir, extracted_name)
            elif r.status_code == 404:
                return None
        except Exception as e:
            time.sleep(1)
            continue
    return None

def process_coin(symbol, start_date_str, days):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "history", "macro", symbol)
    os.makedirs(target_dir, exist_ok=True)
    
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    dates = [start_dt + timedelta(days=i) for i in range(days)]
    months = sorted(list(set([d.strftime("%Y-%m") for d in dates])))
    
    klines_url = "https://data.binance.vision/data/futures/um/daily/klines"
    metrics_url = "https://data.binance.vision/data/futures/um/daily/metrics"
    funding_url = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
    
    print(f"\n🚀 [{symbol}] Iniciando Ingesta MACRO ({days} días)...")
    
    k_urls = [(f"{klines_url}/{symbol}/15m/{symbol}-15m-{d.strftime('%Y-%m-%d')}.zip", f"{symbol}-15m-{d.strftime('%Y-%m-%d')}.zip") for d in dates]
    m_urls = [(f"{metrics_url}/{symbol}/{symbol}-metrics-{d.strftime('%Y-%m-%d')}.zip", f"{symbol}-metrics-{d.strftime('%Y-%m-%d')}.zip") for d in dates]
    f_urls = [(f"{funding_url}/{symbol}/{symbol}-fundingRate-{m}.zip", f"{symbol}-fundingRate-{m}.zip") for m in months]
    
    all_tasks = k_urls + m_urls + f_urls
    
    downloaded_csvs = []
    print(f"⏳ [{symbol}] Descargando {len(all_tasks)} archivos desde Binance Vision...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for url, zname in all_tasks:
            futures.append(executor.submit(download_and_extract, url, target_dir, zname))
            
        for i, f in enumerate(futures):
            res = f.result()
            if res: downloaded_csvs.append(res)
            if (i+1) % 500 == 0:
                print(f"   [{symbol}] Progreso: {i+1}/{len(all_tasks)} archivos procesados.")
                
    k_csvs = [c for c in downloaded_csvs if "-15m-" in c]
    m_csvs = [c for c in downloaded_csvs if "-metrics-" in c]
    f_csvs = [c for c in downloaded_csvs if "-fundingRate-" in c]
    
    print(f"🔄 [{symbol}] Procesando Klines 15m ({len(k_csvs)})...")
    k_dfs = []
    for csv in k_csvs:
        try:
            df = pl.read_csv(csv, has_header=False, infer_schema_length=0, new_columns=[
                "open_time", "open", "high", "low", "close", "volume", 
                "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ])
            df = df.filter(pl.col("open_time") != "open_time")
            df = df.select(["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"])
            df = df.with_columns([
                pl.col("open_time").cast(pl.Int64),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("taker_buy_volume").cast(pl.Float64)
            ])
            k_dfs.append(df)
        except Exception as e: pass
        finally:
            if os.path.exists(csv): os.remove(csv)
            
    df_klines = pl.concat(k_dfs).sort("open_time") if k_dfs else pl.DataFrame()
    
    print(f"🔄 [{symbol}] Procesando Metrics ({len(m_csvs)})...")
    m_dfs = []
    for csv in m_csvs:
        try:
            df = pl.read_csv(csv, has_header=True)
            if 'create_time' in df.columns:
                df = df.rename({"create_time": "timestamp"})
                if df['timestamp'].dtype == pl.String:
                    try:
                        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.timestamp("ms"))
                    except: pass
                df = df.with_columns([
                    pl.col("timestamp").cast(pl.Int64),
                    pl.col("sum_open_interest").cast(pl.Float64),
                    pl.col("sum_toptrader_long_short_ratio").cast(pl.Float64)
                ])
                df = df.select(["timestamp", "sum_open_interest", "sum_toptrader_long_short_ratio"])
                m_dfs.append(df)
        except Exception as e: pass
        finally:
            if os.path.exists(csv): os.remove(csv)
            
    df_metrics = pl.concat(m_dfs).sort("timestamp") if m_dfs else pl.DataFrame()
    
    print(f"🔄 [{symbol}] Procesando Funding ({len(f_csvs)})...")
    f_dfs = []
    for csv in f_csvs:
        try:
            df = pl.read_csv(csv, has_header=True)
            if 'calc_time' in df.columns and 'last_funding_rate' in df.columns:
                df = df.rename({"calc_time": "timestamp", "last_funding_rate": "funding_rate"})
                df = df.with_columns([
                    pl.col("timestamp").cast(pl.Int64),
                    pl.col("funding_rate").cast(pl.Float64)
                ])
                df = df.select(["timestamp", "funding_rate"])
                f_dfs.append(df)
        except Exception as e: pass
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
        
    print(f"✅ [{symbol}] Ingesta completada.")
    print(f"   Klines: {len(df_klines)} filas")
    print(f"   Metrics: {len(df_metrics)} filas")
    print(f"   Funding: {len(df_funding)} filas")

if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "UNIUSDT"]
    start_date = "2022-01-01"
    
    # Calculate days to 2024-03-31
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime("2024-03-31", "%Y-%m-%d")
    days = (d2 - d1).days + 1
    
    print(f"🌍 Iniciando Descarga Masiva de {len(symbols)} activos x {days} días (2022-2024)")
    for sym in symbols:
        process_coin(sym, start_date, days)
