import os
import sys
import requests
import zipfile
import polars as pl
from datetime import datetime, timedelta

# Deshabilitar SettingWithCopyWarning de polars
pl.Config.set_tbl_rows(10)

def download_bookticker(symbol="UNIUSDT", start_date="2026-06-10", days=7):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "data", "history")
    os.makedirs(target_dir, exist_ok=True)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    print(f"🚀 Iniciando descarga de {days} días de BookTicker (Level 2) para {symbol}...")
    
    dfs = []
    
    for i in range(days):
        current_dt = start_dt + timedelta(days=i)
        date_str = current_dt.strftime("%Y-%m-%d")
        
        filename = f"{symbol}-bookTicker-{date_str}.zip"
        url = f"https://data.binance.vision/data/futures/um/daily/bookTicker/{symbol}/{filename}"
        
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
                
                # bookTicker CSVs don't always have headers in Binance Vision.
                # Format is usually: update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time
                df = pl.read_csv(
                    csv_path, 
                    has_header=True, 
                    ignore_errors=True
                )
                
                # Check columns to standardize
                cols = df.columns
                if 'transaction_time' in cols:
                    time_col = 'transaction_time'
                elif 'event_time' in cols:
                    time_col = 'event_time'
                elif 'timestamp' in cols:
                    time_col = 'timestamp'
                else:
                    # Fallback si no hay cabecera
                    df = pl.read_csv(csv_path, has_header=False)
                    # update_id, bid_price, bid_qty, ask_price, ask_qty, transaction_time, event_time
                    df.columns = ["update_id", "bid_price", "bid_qty", "ask_price", "ask_qty", "transaction_time", "event_time"]
                    time_col = "transaction_time"
                
                # Estandarizar columnas a los nombres esperados
                col_map = {}
                for c in df.columns:
                    if 'bid_price' in c or c == 'best_bid_price' or c == 'b': col_map[c] = 'bid_price'
                    if 'bid_qty' in c or c == 'best_bid_qty' or c == 'B': col_map[c] = 'bid_qty'
                    if 'ask_price' in c or c == 'best_ask_price' or c == 'a': col_map[c] = 'ask_price'
                    if 'ask_qty' in c or c == 'best_ask_qty' or c == 'A': col_map[c] = 'ask_qty'
                    if c == time_col: col_map[c] = 'timestamp'
                    
                df = df.rename(col_map, strict=False)
                
                # Select only relevant columns
                df = df.select(["timestamp", "bid_price", "bid_qty", "ask_price", "ask_qty"])
                
                # Ocasionalmente el bookTicker viene desordenado
                df = df.sort("timestamp")
                
                dfs.append(df)
                
                os.remove(zip_path)
                os.remove(csv_path)
            else:
                print(f"⚠️ Error {response.status_code} al descargar {date_str}. Puede que los datos no existan para este día.")
                
        except Exception as e:
            print(f"❌ Excepción en {date_str}: {e}")
            
    if not dfs:
        print("❌ No se descargaron datos de BookTicker.")
        sys.exit(1)
        
    print("🔄 Concatenando DataFrames BBO...")
    final_df = pl.concat(dfs).sort("timestamp")
    
    print(f"✅ Total BBO Updates: {len(final_df):,}")
    
    out_parquet = os.path.join(target_dir, f"{symbol}_bookTicker_{days}d.parquet")
    final_df.write_parquet(out_parquet)
    print(f"💾 Guardado en: {out_parquet}")

if __name__ == "__main__":
    download_bookticker("UNIUSDT", "2024-03-24", 7)
