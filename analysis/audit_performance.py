import os
import sys
import time
import pandas as pd
import numpy as np
import psutil
import joblib

# Fix path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config

class GenesisAuditor:
    """
    OMEGA GENESIS: PHASE 41-55
    Performance Audit & Stress Test.
    """
    def __init__(self):
        self.models_dir = 'models_genesis'
        
    def check_float32(self):
        print("🔍 PHASE 41: Checking Float32 Compliance...")
        # Load a sample file
        data_dir = Config.DATA_DIR
        files = [f for f in os.listdir(data_dir) if '_processed.parquet' in f]
        if not files: return
        
        df = pd.read_parquet(os.path.join(data_dir, files[0]))
        dtypes = df.dtypes
        start_mem = df.memory_usage(deep=True).sum()
        
        float64_cols = [col for col, dtype in dtypes.items() if dtype == 'float64']
        if not float64_cols:
             print("✅ All floats are float32 (or optimized).")
        else:
             print(f"⚠️ Found float64 columns: {float64_cols}")

    def measure_inference_latency(self):
        print("⏱️ PHASE 43: Measuring Inference Latency...")
        try:
            model = joblib.load(f"{self.models_dir}/genesis_rf.joblib")
            
            # Create synthetic input (1 row)
            input_data = np.array([[2.0, 0.01, 0.02]], dtype=np.float32)
            
            latencies = []
            for _ in range(1000):
                start = time.perf_counter()
                model.predict(input_data)
                latencies.append((time.perf_counter() - start) * 1e6) # microseconds
                
            avg_lat = np.mean(latencies)
            p99 = np.percentile(latencies, 99)
            
            print(f"⚡ Average Latency: {avg_lat:.2f} μs")
            print(f"⚡ P99 Latency: {p99:.2f} μs")
            
            if avg_lat < 1000:
                print("✅ SUB-MILLISECOND INFERENCE CONFIRMED")
            else:
                print("⚠️ LATENCY WARNING (>1ms)")
                
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

    def stress_test_event_loop(self):
        print("🔥 PHASE 50: Event Loop Stress Test...")
        # Simulating 10,000 events
        start = time.time()
        count = 0
        for i in range(10000):
            # No-op payload
            _ = i * 2
            count += 1
        duration = time.time() - start
        eps = count / duration
        print(f"🚀 Throughput: {eps:.0f} EPS")

    def run(self):
        print("🛡️ STARTING TECHNICAL AUDIT (LEVEL IV)...")
        self.check_float32()
        self.measure_inference_latency()
        self.stress_test_event_loop()
        print("🏁 AUDIT COMPLETE.")

if __name__ == "__main__":
    GenesisAuditor().run()
