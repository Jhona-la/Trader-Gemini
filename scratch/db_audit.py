import sqlite3
import os

db_path = r'c:\Users\jhona\Documents\Proyectos\Trader Gemini\data\database.db'

if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Checking tables...")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"Tables: {tables}")

for table_tuple in tables:
    table = table_tuple[0]
    print(f"\nScanning table: {table}")
    try:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"Columns: {columns}")
        
        # Search for trade_id
        if 'trade_id' in columns:
            cursor.execute(f"SELECT * FROM {table} WHERE trade_id LIKE '%a996a033%';")
            results = cursor.fetchall()
            if results:
                print(f"FOUND in {table} by trade_id: {results}")
        
        # Search for entry price
        if 'entry_price' in columns:
            cursor.execute(f"SELECT * FROM {table} WHERE entry_price > 75788 AND entry_price < 75789;")
            results = cursor.fetchall()
            if results:
                print(f"FOUND in {table} by entry_price: {results}")
    except Exception as e:
        print(f"Error scanning {table}: {e}")

conn.close()
