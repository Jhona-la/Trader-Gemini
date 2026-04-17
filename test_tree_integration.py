import sys
import queue
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification

sys.path.insert(0, '.')
from strategies.ml_strategy import ml_inference_worker_task
from ml.tree_compiler import compile_rf_to_numpy_batch, compile_gb_to_numpy_batch
from core.fused_strategy_kernel import predict_rf_jit, predict_gb_jit

# 1. Create Models
print("Training dummy models...")
X, y = make_classification(n_samples=50, n_features=10, random_state=42)
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(X, y)
gb = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42).fit(X, y)

rf_arrays = compile_rf_to_numpy_batch(rf)
gb_arrays = compile_gb_to_numpy_batch(gb)

# Warmup JIT
_ = predict_rf_jit(X[0].astype(np.float32), **rf_arrays)
_ = predict_gb_jit(X[0].astype(np.float32), **gb_arrays)

in_q = queue.Queue()
out_q = queue.Queue()

import threading
worker = threading.Thread(target=ml_inference_worker_task, args=(in_q, out_q), daemon=True)
worker.start()

# 2. Test passing Numba Arrays
print("Testing array worker fallback...")
packet = {
    'X': X[0].astype(np.float32),
    'rf': rf_arrays,
    'xgb': None,
    'gb': gb_arrays,
    'ts': time.time(),
    'weights': (0.5, 0.0, 0.5)
}

in_q.put(packet)
try:
    res = out_q.get(timeout=2)
    print("SUCCESS! Worker responded back:", res)
except queue.Empty:
    print("FAILED! Worker hanging.")
