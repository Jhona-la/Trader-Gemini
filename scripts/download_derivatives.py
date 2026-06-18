import os
import sys
import requests
import zipfile
import polars as pl
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time

pl.Config.set_tbl_rows(10)

def download_daily_zip(base_url, symbol, date_str, target_dir, prefix):
    filename = f"{symbol}-{prefix}-{date_str}.zip"
    url = f"{base_url}/{symbol}/{filename}"
    zip_path = os.path.join(target_dir, filename)
    
    print(f"⬇️ Descargando {prefix}: {url}")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_filename = zip_ref.namelist()[0]
                zip_ref.extract(csv_filename, target_dir)
                
            csv_path = os.path.join(target_dir, csv_filename)
            os.remove(zip_path)
            return csv_path
        else:
            print(f"⚠️ Error {response.status_code} al descargar {url}")
    except Exception as e:
        print(f"❌ Excepción en {date_str} ({prefix}): {e}")
    return None

def download_monthly_zip(base_url, symbol, month_str, target_dir, prefix):
    filename = f"{symbol}-{prefix}-{month_str}.zip"
    url = f"{base_url}/{symbol}/{filename}"
    zip_path = os.path.join(target_dir, filename)
    
    print(f"⬇️ Descargando {prefix} Mensual: {url}")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_filename = zip_ref.namelist()[0]
                zip_ref.extract(csv_filename, target_dir)
                
            csv_path = os.path.join(target_dir, csv_filename)
            os.remove(zip_path)
            return csv_path
        else:
            print(f"⚠️ Error {response.status_code} al descargar {url}")
    except Exception as e:
        print(f"❌ Excepción en {month_str} ({prefix}): {e}")
    return None

def download_macro_data(symbol="UNIUSDT", start_date="2024-01-01", days=90):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "history", "macro")
    os.makedirs(target_dir, exist_ok=True)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    klines_url = "https://data.binance.vision/data/futures/um/daily/klines" # /UNIUSDT/15m/...
    metrics_url = "https://data.binance.vision/data/futures/um/daily/metrics"
    funding_url = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
    
    print(f"🚀 Iniciando Ingesta MACRO para {symbol} desde {start_date} ({days} días)...")
    
    dates = [start_dt + timedelta(days=i) for i in range(days)]
    months = list(set([d.strftime("%Y-%m") for d in dates]))
    
    # 1. Download Klines (15m)
    klines_csvs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            base_k = f"{klines_url}/{symbol}/15m"
            filename = f"{symbol}-15m-{date_str}.zip"
            url = f"{base_k}/{filename}"
            
            # Since download_daily_zip hardcodes /{symbol}/ in url, we must pass base_url differently
            # Or just rewrite how klines are fetched
            def dl_kline(d_str, url_str):
                z_path = os.path.join(target_dir, f"kline_{d_str}.zip")
                try:
                    r = requests.get(url_str, stream=True)
                    if r.status_code == 200:
                        with open(z_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                        with zipfile.ZipFile(z_path, 'r') as z:
                            csv_f = z.namelist()[0]
                            z.extract(csv_f, target_dir)
                        os.remove(z_path)
                        return os.path.join(target_dir, csv_f)
                except: pass
                return None
            futures.append(executor.submit(dl_kline, date_str, url))
            
        for f in futures:
            res = f.result()
            if res: klines_csvs.append(res)
            
    # 2. Download Metrics
    metrics_csvs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            futures.append(executor.submit(download_daily_zip, metrics_url, symbol, date_str, target_dir, "metrics"))
            
        for f in futures:
            res = f.result()
            if res: metrics_csvs.append(res)
            
    # 3. Download Funding (Monthly)
    funding_csvs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for m in months:
            futures.append(executor.submit(download_monthly_zip, funding_url, symbol, m, target_dir, "fundingRate"))
            
        for f in futures:
            res = f.result()
            if res: funding_csvs.append(res)
            
    print("🔄 Procesando Klines 15m...")
    k_dfs = []
    for csv in klines_csvs:
        try:
            # We use infer_schema_length=0 to read everything as string first
            df = pl.read_csv(csv, has_header=False, infer_schema_length=0, new_columns=[
                "open_time", "open", "high", "low", "close", "volume", 
                "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ])
            # Remove any header row if it exists
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
        except Exception as e:
            print(f"⚠️ Error procesando CSV {csv}: {e}")
        finally:
            if os.path.exists(csv): os.remove(csv)
        
    df_klines = pl.concat(k_dfs).sort("open_time") if k_dfs else pl.DataFrame()
    
    print("🔄 Procesando Metrics (Open Interest)...")
    m_dfs = []
    for csv in metrics_csvs:
        df = pl.read_csv(csv, has_header=True)
        # create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio
        if 'create_time' in df.columns:
            df = df.rename({"create_time": "timestamp"})
            
            # Algunas veces create_time es string con formato fecha, otras es timestamp.
            # Convertimos a Int64 asumiendo que es milisegundos o lo parseamos
            if df['timestamp'].dtype == pl.String:
                try:
                    df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").dt.timestamp("ms"))
                except:
                    pass
                    
            df = df.with_columns([
                pl.col("timestamp").cast(pl.Int64),
                pl.col("sum_open_interest").cast(pl.Float64),
                pl.col("sum_toptrader_long_short_ratio").cast(pl.Float64)
            ])
            df = df.select(["timestamp", "sum_open_interest", "sum_toptrader_long_short_ratio"])
            m_dfs.append(df)
        os.remove(csv)
        
    df_metrics = pl.concat(m_dfs).sort("timestamp") if m_dfs else pl.DataFrame()
    
    print("🔄 Procesando Funding Rates...")
    f_dfs = []
    for csv in funding_csvs:
        df = pl.read_csv(csv, has_header=True)
        if 'calc_time' in df.columns and 'last_funding_rate' in df.columns:
            df = df.rename({"calc_time": "timestamp", "last_funding_rate": "funding_rate"})
            df = df.with_columns([
                pl.col("timestamp").cast(pl.Int64),
                pl.col("funding_rate").cast(pl.Float64)
            ])
            df = df.select(["timestamp", "funding_rate"])
            f_dfs.append(df)
        os.remove(csv)
        
    df_funding = pl.concat(f_dfs).sort("timestamp") if f_dfs else pl.DataFrame()
    
    # Save raw
    if not df_klines.is_empty():
        df_klines.write_parquet(os.path.join(target_dir, f"{symbol}_klines_15m.parquet"))
    if not df_metrics.is_empty():
        df_metrics.write_parquet(os.path.join(target_dir, f"{symbol}_metrics.parquet"))
    if not df_funding.is_empty():
        df_funding.write_parquet(os.path.join(target_dir, f"{symbol}_funding.parquet"))
        
    print(f"✅ Ingesta Macro completada.")
    print(f"   Klines: {len(df_klines)} filas")
    print(f"   Metrics: {len(df_metrics)} filas")
    print(f"   Funding: {len(df_funding)} filas")

if __name__ == "__main__":
    download_macro_data("UNIUSDT", "2024-01-01", 90)
