from test_arrays import arrays
import sys
sys.path.insert(0, '.')
from core.fused_strategy_kernel import predict_rf_jit
import numpy as np

# A valid X for the tree
X = np.random.randn(10).astype(np.float32)

print("Starting JIT Warmup...")
try:
    res = predict_rf_jit(
        X,
        arrays['children_left'],
        arrays['children_right'],
        arrays['feature'],
        arrays['threshold'],
        arrays['value'],
        arrays['tree_offsets']
    )
    print("JIT Success! Result:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
