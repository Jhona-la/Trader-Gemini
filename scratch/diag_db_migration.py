import sqlite3
import os

def migrate_database(db_path):
    print(f"Migrating {db_path}...")
    if not os.path.exists(db_path):
        print("Database not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    columns_to_add = {
        'sl_pct': 'REAL',
        'tp_pct': 'REAL',
        'horizon': 'TEXT DEFAULT "SCALPING"',
        'strategy_id': 'TEXT DEFAULT "UNKNOWN"'
    }

    # Check existing columns
    cursor.execute("PRAGMA table_info(positions)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    for col_name, col_type in columns_to_add.items():
        if col_name not in existing_columns:
            try:
                print(f"Adding column {col_name}...")
                cursor.execute(f"ALTER TABLE positions ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError as e:
                print(f"Error adding {col_name}: {e}")

    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\data\trader_gemini.db"
    migrate_database(db_path)
