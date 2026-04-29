import sqlite3
import json

def search_db():
    db_path = "data.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Try searching for trade_id if it exists
    # If not, search in the raw data or other columns
    try:
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in DB: {tables}")
        
        target_ids = [
            "a996a033-23ce-4bdc-bc5e-86e182bc83ee",
            "20f660cd-bc3a-4df9-aa76-4c07d6b4b7e9",
            "fc537edd-4bc3-44e2-8240-bece6031ff1e"
        ]
        
        for table in [t[0] for t in tables]:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [c[1] for c in cursor.fetchall()]
            
            if 'trade_id' in columns:
                for tid in target_ids:
                    cursor.execute(f"SELECT * FROM {table} WHERE trade_id = ?", (tid,))
                    result = cursor.fetchone()
                    if result:
                        print(f"Found {tid} in table {table}: {result}")
            else:
                # Search in all text columns
                for tid in target_ids:
                    for col in columns:
                        try:
                            cursor.execute(f"SELECT * FROM {table} WHERE {col} LIKE ?", (f"%{tid}%",))
                            result = cursor.fetchone()
                            if result:
                                print(f"Found partial {tid} in table {table}, column {col}: {result}")
                        except:
                            pass
                            
    except Exception as e:
        print(f"Error searching DB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    search_db()
