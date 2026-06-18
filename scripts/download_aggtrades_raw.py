import os
import sys
import requests
import zipfile
import polars as pl
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# No import config

# Deshabilitar SettingWithCopyWarning de polars
pl.Config.set_tbl_rows(10)

def download_day(symbol, date_str, target_dir):
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{filename}"
    zip_path = os.path.join(target_dir, filename)
    
    print(f"⬇️ Descargando: {url}")
    
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
            
            # format: agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
            df = pl.read_csv(
                csv_path, 
                has_header=False,
                new_columns=["agg_trade_id", "price", "volume", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker"]
            )
            df = df.select(["timestamp", "price", "volume", "is_buyer_maker"])
            df = df.sort("timestamp")
            
            os.remove(zip_path)
            os.remove(csv_path)
            return df
        else:
            print(f"⚠️ Error {response.status_code} al descargar {date_str}.")
    except Exception as e:
        print(f"❌ Excepción en {date_str}: {e}")
    return None

def download_historical_aggtrades(symbol="UNIUSDT", start_date="2024-03-24", days=7):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "history")
    os.makedirs(target_dir, exist_ok=True)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    print(f"🚀 Iniciando descarga de {days} días de AggTrades RAW para {symbol} desde {start_date}...")
    
    dfs = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(days):
            current_dt = start_dt + timedelta(days=i)
            date_str = current_dt.strftime("%Y-%m-%d")
            futures.append(executor.submit(download_day, symbol, date_str, target_dir))
            
        for future in futures:
            res = future.result()
            if res is not None:
                dfs.append(res)
                
    if not dfs:
        print("❌ No se descargaron datos de AggTrades.")
        sys.exit(1)
        
    print("🔄 Concatenando DataFrames tick-by-tick...")
    final_df = pl.concat(dfs).sort("timestamp")
    
    print(f"✅ Total Ticks (Trades): {len(final_df):,}")
    
    out_parquet = os.path.join(target_dir, f"{symbol}_aggTrades_{days}d.parquet")
    final_df.write_parquet(out_parquet)
    print(f"💾 Guardado en: {out_parquet}")

if __name__ == "__main__":
    download_historical_aggtrades("UNIUSDT", "2024-03-24", 7)
