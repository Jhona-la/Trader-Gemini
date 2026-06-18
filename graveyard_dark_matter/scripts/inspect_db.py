import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\dashboard\data\futures\trader_gemini.db"

def analyze_db():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    
    # 1. Show Tables
    print("--- 📋 Database Tables ---")
    try:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        print(tables)
    except Exception as e:
        print(f"Error listing tables: {e}")
        return

    # 2. Performance Stats (trades table)
    if 'trades' in tables['name'].values:
        print("\n--- 📊 Historical Performance Stats ---")
        try:
            trades = pd.read_sql_query("SELECT * FROM trades;", conn)
            if not trades.empty:
                total_trades = len(trades)
                winning_trades = len(trades[trades['pnl'] > 0]) if 'pnl' in trades.columns else 0
                total_pnl = trades['pnl'].sum() if 'pnl' in trades.columns else 0
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                print(f"Total Trades: {total_trades}")
                print(f"Win Rate: {win_rate:.2f}%")
                if 'pnl' in trades.columns:
                    print(f"Total PnL: {total_pnl:.6f}")
                    print(f"Avg PnL: {trades['pnl'].mean():.6f}")
                
                if 'symbol' in trades.columns:
                    print("\nPnL by Symbol:")
                    print(trades.groupby('symbol')['pnl'].sum())
            else:
                print("No trades found in the database.")
        except Exception as e:
            print(f"Error analyzing trades: {e}")

    # 3. Current State (Positions)
    if 'positions' in tables['name'].values:
        print("\n--- 💼 Current Positions ---")
        try:
            positions = pd.read_sql_query("SELECT * FROM positions;", conn)
            print(positions)
        except Exception as e:
            print(f"Error reading positions: {e}")
            
    # 4. Check Model files
    print("\n--- 🤖 Persisted Models (Filesystem) ---")
    models_dir = ".models"
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        if files:
            print(f"Models found in {models_dir}:")
            for f in files:
                mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(models_dir, f)))
                size = os.path.getsize(os.path.join(models_dir, f)) / 1024
                print(f" - {f} ({size:.2f} KB) - Last modified: {mtime}")
        else:
            print(f"No models found in {models_dir}")
    else:
        print(f"Directory {models_dir} not found")

    # 5. Persistence Efficiency
    print("\n--- ⚡ Persistence Efficiency ---")
    size_bytes = os.path.getsize(DB_PATH)
    print(f"Database size: {size_bytes / 1024:.2f} KB")
    
    total_records = 0
    for table in tables['name']:
        try:
            count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table};", conn).iloc[0]['count']
            print(f"Table '{table}': {count} records")
            total_records += count
        except:
             print(f"Table '{table}': Error counting")
             
    print(f"Total Database Records: {total_records}")
    if total_records > 0:
        print(f"Bytes per record: {size_bytes / total_records:.2f}")

    conn.close()

if __name__ == "__main__":
    analyze_db()
