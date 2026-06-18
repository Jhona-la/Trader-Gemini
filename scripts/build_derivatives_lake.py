import os
import glob
import requests
import zipfile
import polars as pl
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

pl.Config.set_tbl_rows(10)

def download_zip(url, target_dir):
    filename = url.split('/')[-1]
    zip_path = os.path.join(target_dir, filename)
    
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_filename = zip_ref.namelist()[0]
                zip_ref.extract(csv_filename, target_dir)
            
            csv_path = os.path.join(target_dir, csv_filename)
            os.remove(zip_path)
            return csv_path
    except Exception as e:
        print(f"❌ Excepción al descargar {url}: {e}")
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    return None

def download_symbol_derivatives(symbol, start_date="2024-01-01", days=90, max_workers=5):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "history", "macro")
    os.makedirs(target_dir, exist_ok=True)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start_dt + timedelta(days=i) for i in range(days)]
    months = sorted(list(set([d.strftime("%Y-%m") for d in dates])))
    
    metrics_url_base = "https://data.binance.vision/data/futures/um/daily/metrics"
    funding_url_base = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
    
    print(f"🚀 [DERIVATIVES] Descargando Metrics (OI) y Funding para {symbol}...")
    
    metrics_csvs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            url = f"{metrics_url_base}/{symbol}/{symbol}-metrics-{date_str}.zip"
            futures.append(executor.submit(download_zip, url, target_dir))
        for f in futures:
            res = f.result()
            if res: metrics_csvs.append(res)
            
    funding_csvs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for m in months:
            url = f"{funding_url_base}/{symbol}/{symbol}-fundingRate-{m}.zip"
            futures.append(executor.submit(download_zip, url, target_dir))
        for f in futures:
            res = f.result()
            if res: funding_csvs.append(res)
            
    # Procesar Metrics (Open Interest)
    m_dfs = []
    for csv in metrics_csvs:
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
                ]).select(["timestamp", "sum_open_interest", "sum_toptrader_long_short_ratio"])
                m_dfs.append(df)
        except Exception as e:
            print(f"Error procesando {csv}: {e}")
        finally:
            if os.path.exists(csv): os.remove(csv)
            
    df_metrics = pl.concat(m_dfs).sort("timestamp") if m_dfs else pl.DataFrame()
    
    # Procesar Funding Rates
    f_dfs = []
    for csv in funding_csvs:
        try:
            df = pl.read_csv(csv, has_header=True)
            if 'calc_time' in df.columns and 'last_funding_rate' in df.columns:
                df = df.rename({"calc_time": "timestamp", "last_funding_rate": "funding_rate"})
                df = df.with_columns([
                    pl.col("timestamp").cast(pl.Int64),
                    pl.col("funding_rate").cast(pl.Float64)
                ]).select(["timestamp", "funding_rate"])
                f_dfs.append(df)
        except Exception as e:
            print(f"Error procesando {csv}: {e}")
        finally:
            if os.path.exists(csv): os.remove(csv)
            
    df_funding = pl.concat(f_dfs).sort("timestamp") if f_dfs else pl.DataFrame()
    
    # Guardar
    if not df_metrics.is_empty():
        df_metrics.write_parquet(os.path.join(target_dir, f"{symbol}_metrics.parquet"))
    if not df_funding.is_empty():
        df_funding.write_parquet(os.path.join(target_dir, f"{symbol}_funding.parquet"))
        
    print(f"✅ [DERIVATIVES] {symbol} completado. OI: {len(df_metrics)} filas | Funding: {len(df_funding)} filas")

def build_data_lake(days=90):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, "data", "quantum_lake")
    
    # Identify active symbols by looking at cached .qbin files
    qbin_files = glob.glob(os.path.join(cache_dir, "*.qbin"))
    symbols = [os.path.basename(f).replace(".qbin", "") for f in qbin_files if not os.path.basename(f).startswith("DATAQUANTUM_LAKE")]
    
    # Remove slashes just in case they were cached as BNB/USDT (usually they are BNBUSDT)
    symbols = [sym.replace("/", "") for sym in symbols]
    
    if not symbols:
        print("⚠️ No symbols found in data/binance_cache. Defaulting to BNBUSDT.")
        symbols = ["BNBUSDT"]
        
    print(f"🌐 Iniciando Data Lake de Derivados para {len(symbols)} símbolos: {symbols}")
    
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Limitar concurrencia global para evitar bloqueos por rate-limit de Binance Vision
    for sym in symbols:
        download_symbol_derivatives(sym, start_date=start_date, days=days, max_workers=5)
        
    print("🚀 Data Lake Creado Exitosamente.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="Días de historial a minar")
    args = parser.parse_args()
    build_data_lake(days=args.days)
