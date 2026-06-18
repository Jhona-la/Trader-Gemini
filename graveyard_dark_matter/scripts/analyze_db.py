import sqlite3
import pandas as pd

conn = sqlite3.connect('data.db')
try:
    df = pd.read_sql_query("SELECT * FROM trades WHERE timestamp > datetime('now', '-1 day')", conn)
    if not df.empty:
        print(f"Total Trades: {len(df)}")
        print(f"Win Rate: {(df['pnl'] > 0).mean()*100:.2f}%")
        print(f"Total PnL: {df['pnl'].sum():.4f}")
        print(f"Average MFE: {df['mfe'].mean()*100:.4f}%" if 'mfe' in df.columns else "MFE not in DB")
    else:
        print("No trades found in the last day.")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
