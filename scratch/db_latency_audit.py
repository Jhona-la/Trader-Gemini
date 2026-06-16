import time
import sqlite3
import os

db_path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\data\test_latency.db"
if os.path.exists(db_path): os.remove(db_path)

conn = sqlite3.connect(db_path)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
conn.execute("PRAGMA temp_store=MEMORY;")
conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")

latencies_ns = []
for i in range(1000):
    start = time.perf_counter_ns()
    conn.execute("INSERT INTO test (val) VALUES ('data')")
    conn.commit()
    end = time.perf_counter_ns()
    latencies_ns.append(end - start)

avg_latency = sum(latencies_ns) / len(latencies_ns)
print(f"Average Write + Commit Latency (WAL+NORMAL): {avg_latency:,.0f} ns")

conn.execute("PRAGMA synchronous=OFF;")
latencies_ns_off = []
for i in range(1000):
    start = time.perf_counter_ns()
    conn.execute("INSERT INTO test (val) VALUES ('data')")
    conn.commit()
    end = time.perf_counter_ns()
    latencies_ns_off.append(end - start)

avg_latency_off = sum(latencies_ns_off) / len(latencies_ns_off)
print(f"Average Write + Commit Latency (WAL+OFF): {avg_latency_off:,.0f} ns")

# Memory DB
conn_mem = sqlite3.connect(":memory:")
conn_mem.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
latencies_ns_mem = []
for i in range(1000):
    start = time.perf_counter_ns()
    conn_mem.execute("INSERT INTO test (val) VALUES ('data')")
    conn_mem.commit()
    end = time.perf_counter_ns()
    latencies_ns_mem.append(end - start)

avg_latency_mem = sum(latencies_ns_mem) / len(latencies_ns_mem)
print(f"Average Write + Commit Latency (:memory:): {avg_latency_mem:,.0f} ns")
