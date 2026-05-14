import sqlite3
import pandas as pd

for db in ['trader_gemini.db', 'data.db']:
    try:
        conn = sqlite3.connect(db)
        print(f"\n--- {db} ---")
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        print("Tables:", tables['name'].tolist())
        for table in tables['name']:
            print(f"Table '{table}' columns:")
            cols = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
            print("  " + ", ".join(cols['name']))
        conn.close()
    except Exception as e:
        print(e)
