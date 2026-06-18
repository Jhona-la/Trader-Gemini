import os
import sys
import numpy as np

def audit_qbin_granularity(symbol="BTCUSDT"):
    qbin_path = f"data/quantum_lake/{symbol}.qbin"
    if not os.path.exists(qbin_path):
        print(f"File not found: {qbin_path}")
        return

    # From memory, QBIN structure is:
    # MAGIC (4 bytes), VERSION (4 bytes), N_ROWS (8), N_COLS (8)
    # Then N_COLS names (str length + str)
    # Then data (N_ROWS * N_COLS * float64)
    # Let's try to parse it
    try:
        with open(qbin_path, "rb") as f:
            magic = f.read(4)
            if magic != b'QBIN':
                print(f"Invalid magic: {magic}")
                return
            version = np.frombuffer(f.read(4), dtype=np.uint32)[0]
            n_rows = np.frombuffer(f.read(8), dtype=np.uint64)[0]
            n_cols = np.frombuffer(f.read(8), dtype=np.uint64)[0]
            
            print(f"QBIN {symbol}: Version {version}, Rows: {n_rows}, Cols: {n_cols}")
            
            cols = []
            for _ in range(n_cols):
                str_len = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                col_name = f.read(str_len).decode('utf-8')
                cols.append(col_name)
                
            print("Columns available:", cols)
            
            data = np.frombuffer(f.read(), dtype=np.float64).reshape((n_rows, n_cols))
            
            # Find timestamp, open_interest, and funding_rate
            ts_idx = cols.index("timestamp") if "timestamp" in cols else -1
            oi_idx = cols.index("open_interest_norm") if "open_interest_norm" in cols else -1
            if oi_idx == -1 and "sum_open_interest" in cols:
                oi_idx = cols.index("sum_open_interest")
                
            fr_idx = cols.index("funding_rate_norm") if "funding_rate_norm" in cols else -1
            if fr_idx == -1 and "funding_rate" in cols:
                fr_idx = cols.index("funding_rate")

            if ts_idx == -1:
                print("No timestamp column found.")
                return

            ts_data = data[:, ts_idx]
            
            # Check OI granularity
            if oi_idx != -1:
                oi_data = data[:, oi_idx]
                # Find where OI actually changes
                oi_diff = np.diff(oi_data)
                change_indices = np.where(oi_diff != 0)[0]
                if len(change_indices) == 0:
                    print("Open Interest NEVER changes (Constant).")
                else:
                    ts_changes = ts_data[change_indices + 1]
                    ts_diffs = np.diff(ts_changes)
                    median_diff_ms = np.median(ts_diffs)
                    min_diff_ms = np.min(ts_diffs)
                    print(f"Open Interest changes {len(change_indices)} times out of {n_rows} rows.")
                    print(f"Median time between OI changes: {median_diff_ms / 1000 / 60} minutes.")
                    print(f"Minimum time between OI changes: {min_diff_ms / 1000 / 60} minutes.")
                    
                    # Print a sample of changes
                    for i in range(min(5, len(change_indices))):
                        idx = change_indices[i]
                        print(f"  Change at row {idx}: {ts_data[idx]} -> {ts_data[idx+1]} ({ (ts_data[idx+1]-ts_data[idx])/60000} mins)")
            else:
                print("No Open Interest column found in QBIN.")
                
            # Check Funding Rate granularity
            if fr_idx != -1:
                fr_data = data[:, fr_idx]
                fr_diff = np.diff(fr_data)
                change_indices = np.where(fr_diff != 0)[0]
                if len(change_indices) == 0:
                    print("Funding Rate NEVER changes (Constant).")
                else:
                    ts_changes = ts_data[change_indices + 1]
                    ts_diffs = np.diff(ts_changes)
                    median_diff_ms = np.median(ts_diffs)
                    print(f"Funding Rate changes {len(change_indices)} times out of {n_rows} rows.")
                    print(f"Median time between Funding Rate changes: {median_diff_ms / 1000 / 60 / 60} hours.")
            else:
                print("No Funding Rate column found in QBIN.")
                
    except Exception as e:
        print(f"Error parsing QBIN: {e}")

if __name__ == "__main__":
    audit_qbin_granularity("SOLUSDT")
    audit_qbin_granularity("BTCUSDT")
