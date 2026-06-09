import sqlite3

def main():
    print("🔬 [DATABASE REMEDIATION AUDIT]")
    conn = sqlite3.connect("dashboard/data/futures/trader_gemini.db")
    cursor = conn.cursor()
    
    # 1. Fetch all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"  Available Tables: {tables}")
    
    # 2. Check strategy_report_card rows
    if "strategy_report_card" in tables:
        cursor.execute("SELECT * FROM strategy_report_card")
        rows = cursor.fetchall()
        print(f"  [PASS] strategy_report_card exists. Row count: {len(rows)}")
        for r in rows[:5]:
            print(f"    - Strategy: {r[0]} | Trades: {r[1]} | Wins: {r[2]} | Losses: {r[3]} | WR: {r[4]:.2f} | PnL: ${r[5]:.4f}")
    else:
        print("  [FAIL] strategy_report_card does not exist!")
        
    # 3. Check session_ledger rows
    if "session_ledger" in tables:
        cursor.execute("SELECT * FROM session_ledger")
        rows = cursor.fetchall()
        print(f"  [PASS] session_ledger exists. Row count: {len(rows)}")
        for r in rows[:5]:
            print(f"    - Session ID: {r[0]} | Start Equity: ${r[1]} | End Equity: ${r[2]} | Net PnL: ${r[8]}")
    else:
        print("  [FAIL] session_ledger does not exist!")
        
    conn.close()

if __name__ == "__main__":
    main()
