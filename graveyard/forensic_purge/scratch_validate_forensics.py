import sqlite3
import pandas as pd
import json

db_path = "dashboard/data/futures/trader_gemini.db"

def validate_forensic_db():
    print("🔍 CTOS Forensic DB Validation")
    print("="*40)
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. Thoughts Table
        cursor.execute("SELECT COUNT(*) FROM thoughts")
        thoughts_count = cursor.fetchone()[0]
        print(f"🧠 Total Thoughts Logged: {thoughts_count}")
        
        # 2. Exit Decisions
        cursor.execute("SELECT COUNT(*) FROM exit_decisions")
        exit_count = cursor.fetchone()[0]
        print(f"🚪 Total Exit Decisions: {exit_count}")
        
        # 3. Trades with thought_id
        cursor.execute("SELECT COUNT(*) FROM trades WHERE thought_id IS NOT NULL")
        trades_with_thought = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]
        print(f"📊 Trades linked to Thoughts: {trades_with_thought} / {total_trades}")
        
        print("\n--- SAMPLE EXIT DECISION ---")
        cursor.execute("SELECT * FROM exit_decisions ORDER BY timestamp DESC LIMIT 1")
        exit_row = cursor.fetchone()
        if exit_row:
            for k in exit_row.keys():
                print(f"  {k}: {exit_row[k]}")
                
        print("\n--- SAMPLE TRADE NOTIFICATION DATA ---")
        try:
            with open("dashboard/data/backtest_telemetry_spam.jsonl", "r") as f:
                lines = f.readlines()
                # Find the last trade close message
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception as je:
                        print(f"Failed to parse line: {repr(line)}: {je}")
                        continue
                    msg = data.get("message", "")
                    if "TRADE CERRADO" in msg:
                        print(msg)
                        break
        except Exception as e:
            print(f"Telemetry missing: {e}")

if __name__ == "__main__":
    validate_forensic_db()
