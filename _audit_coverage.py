"""Investigar la discrepancia de audit coverage"""
import sqlite3, os

DB = os.path.join("dashboard", "data", "futures", "trader_gemini.db")
conn = sqlite3.connect(DB)
c = conn.cursor()

# 1. Trades by side
print("=== TRADES BY SIDE ===")
rows = c.execute("SELECT side, COUNT(*) FROM trades GROUP BY side").fetchall()
for r in rows:
    print(f"  {r[0]}: {r[1]}")

# 2. Trades with pnl != 0 (actual closes)
close_count = c.execute("SELECT COUNT(*) FROM trades WHERE pnl != 0.0 AND pnl IS NOT NULL").fetchone()[0]
print(f"\n  Trades with non-zero PnL (real closes): {close_count}")

# 3. Trades with trade_id populated
tid_count = c.execute("SELECT COUNT(*) FROM trades WHERE trade_id IS NOT NULL AND trade_id != ''").fetchone()[0]
print(f"  Trades with trade_id: {tid_count}")

# 4. Unique trade_ids in prediction_audit
pa_tids = c.execute("SELECT COUNT(DISTINCT trade_id) FROM prediction_audit").fetchone()[0]
print(f"  Unique trade_ids in prediction_audit: {pa_tids}")

# 5. Check trade_ids in prediction_audit that match trades
matching = c.execute("""
    SELECT COUNT(DISTINCT pa.trade_id) 
    FROM prediction_audit pa 
    JOIN trades t ON pa.trade_id = t.trade_id
""").fetchone()[0]
print(f"  Matching trade_ids (audit↔trades): {matching}")

# 6. Distribution of trades over time
print("\n=== TRADES OVER TIME (by month) ===")
rows = c.execute("""
    SELECT substr(timestamp, 1, 7) as month, COUNT(*), 
           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
           SUM(pnl) as total_pnl
    FROM trades 
    GROUP BY month ORDER BY month
""").fetchall()
for r in rows:
    wr = r[2]/(r[2]+r[3])*100 if (r[2]+r[3]) > 0 else 0
    print(f"  {r[0]}: {r[1]} trades | W:{r[2]} L:{r[3]} WR:{wr:.0f}% | PnL: ${r[4]:.4f}")

# 7. Prediction audit over time
print("\n=== PREDICTION AUDIT OVER TIME ===")
rows = c.execute("""
    SELECT substr(entry_time, 1, 7) as month, COUNT(*),
           SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) as wins
    FROM prediction_audit
    GROUP BY month ORDER BY month
""").fetchall()
for r in rows:
    wr = r[2]/r[1]*100 if r[1] > 0 else 0
    print(f"  {r[0]}: {r[1]} audits | {r[2]} wins | WR: {wr:.0f}%")

# 8. Thought_id coverage
print("\n=== THOUGHT_ID COVERAGE ===")
trades_with_thought = c.execute("SELECT COUNT(*) FROM trades WHERE thought_id IS NOT NULL AND thought_id != ''").fetchone()[0]
audits_with_thought = c.execute("SELECT COUNT(*) FROM prediction_audit WHERE thought_id IS NOT NULL AND thought_id != ''").fetchone()[0]
print(f"  Trades with thought_id: {trades_with_thought}/{c.execute('SELECT COUNT(*) FROM trades').fetchone()[0]}")
print(f"  Audits with thought_id: {audits_with_thought}/{c.execute('SELECT COUNT(*) FROM prediction_audit').fetchone()[0]}")

conn.close()
