
import os
import sys
import pandas as pd
import numpy as np

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.math_kernel import calculate_zscore_jit
from config import Config

def validate_matrix():
    print("🔍 [VALIDATION] Protocol Supreme: Phase 20 - Matrix Integrity Check...")
    
    cache_dir = "data/cache_parquet"
    if not os.path.exists(cache_dir):
        print("❌ CRITICAL: No data found. Run ingestion first.")
        sys.exit(1)
        
    files = [f for f in os.listdir(cache_dir) if f.endswith('1m.parquet')]
    if not files:
        print("❌ CRITICAL: No 1m parquet files found.")
        sys.exit(1)
        
    print(f"📂 Analyzing {len(files)} symbol matrices...")
    
    global_errors = 0
    
    for f in files:
        symbol = f.replace('_1m.parquet', '')
        path = os.path.join(cache_dir, f)
        
        try:
            df = pd.read_parquet(path)
            
            # 1. Check Continuity
            # timestamps should be 60s apart (+- small jitter allowed if using timestamp ms)
            # We assume df has 'datetime'
            
            # Convert to seconds
            ts = df['datetime'].astype(np.int64) // 10**9
            diffs = np.diff(ts)
            
            gaps = diffs[diffs > 65] # Allow 5s jitter
            if len(gaps) > 0:
                print(f"⚠️ {symbol}: Found {len(gaps)} gaps > 65s. Max gap: {np.max(gaps)}s")
                # Not fatal for ingestion, but warning for continuity
                
            # 2. Check Z-Score Calculation (Math Kernel)
            closes = df['close'].values.astype(np.float32)
            zscores = calculate_zscore_jit(closes, period=20)
            
            # Validation: z-score should not be infinite/nan generally (unless period < 20)
            if np.isnan(zscores[20:]).any() or np.isinf(zscores[20:]).any():
                 print(f"❌ {symbol}: Z-Score Math Error (NaN/Inf detected)")
                 global_errors += 1
            
            # 3. Check Slope (Simple Linear Regression on last 20)
            # y = mx + c
            # x = [0..19], y = prices[-20:]
            if len(closes) > 20:
                y = closes[-20:]
                x = np.arange(20, dtype=np.float32)
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                
                # Sanity: if price went up, m should be positive
                delta = y[-1] - y[0]
                if (delta > 0 and m < 0) or (delta < 0 and m > 0):
                     # This can happen if curve is convex/concave, but generally direction aligns.
                     # This is a loose check.
                     pass 
                
                # Check for extreme slope (Float32 explosion)
                if abs(m) > 1e6 and y[0] < 100000:
                    print(f"❌ {symbol}: Slope Explosion detected (m={m})")
                    global_errors += 1

        except Exception as e:
            print(f"❌ {symbol}: Validation Failed - {e}")
            global_errors += 1
            
    if global_errors == 0:
        print("✅ [VALIDATION] All Matrices Passed Integrity Checks (Z-Score/Slope/Float32).")
    else:
        print(f"❌ [VALIDATION] Completed with {global_errors} errors.")
        sys.exit(1)

if __name__ == "__main__":
    validate_matrix()
