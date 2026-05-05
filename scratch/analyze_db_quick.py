import sqlite3
import pandas as pd

conn = sqlite3.connect('data.db')
try:
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    if not df.empty:
        # Get only the trades from today
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df = df.sort_values('timestamp').tail(20) # get last 20 trades
        
        print(f"Total Trades Analyzed (Recent): {len(df)}")
        print(f"Win Rate: {(df['pnl'] > 0).mean()*100:.2f}%")
        print(f"Total PnL: {df['pnl'].sum():.4f}")
        if 'mfe' in df.columns:
            print(f"Average MFE: {df['mfe'].mean()*100:.4f}%")
        else:
            print("MFE not in DB")
            
        print("\nLast 5 trades:")
        print(df[['symbol', 'direction', 'pnl', 'pnl_pct', 'close_reason']].tail(5))
    else:
        print("No trades found in the DB.")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
