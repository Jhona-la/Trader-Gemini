import sqlite3
import json

def check_db():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE trade_id LIKE 'a996a033%'")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} trades in DB matching 'a996a033%'")
    for row in rows:
        print(row)
    conn.close()

if __name__ == "__main__":
    check_db()
