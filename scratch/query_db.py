import sqlite3
import json

def get_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    conn.close()
    return tables

def find_trade(db_path, trade_id):
    tables = get_tables(db_path)
    results = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for table in tables:
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [c[1] for c in cursor.fetchall()]
            if 'trade_id' in columns:
                cursor.execute(f"SELECT * FROM {table} WHERE trade_id LIKE ?", (f"%{trade_id}%",))
                rows = cursor.fetchall()
                if rows:
                    results[table] = [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"Error querying {table}: {e}")
    conn.close()
    return results

if __name__ == "__main__":
    dbs = ['trader_gemini.db', 'data/backtest_governance.db', 'data/feature_store.db', 'data/optuna_studies.db']
    id_to_find = 'a996a033'
    for db in dbs:
        print(f"\nChecking {db}...")
        tables = get_tables(db)
        print(f"Tables: {tables}")
        found = find_trade(db, id_to_find)
        if found:
            print(f"Found trade {id_to_find} in {db}:")
            print(json.dumps(found, indent=2))
        else:
            print(f"Trade {id_to_find} not found in {db}")
