import os

filepath = 'core/fused_strategy_kernel.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Generate new content
new_content = '''import numpy as np
import ctypes
import math
import os

_rust_lib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'rust_engine', 'target', 'release', 'quantum_engine.dll')
_rust_lib = None
if os.path.exists(_rust_lib_path):
    try:
        _rust_lib = ctypes.CDLL(_rust_lib_path)
        
        # ffi_predict_rf
        _rust_lib.ffi_predict_rf.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t
        ]
        _rust_lib.ffi_predict_rf.restype = ctypes.c_double
        
        # ffi_predict_gb
        _rust_lib.ffi_predict_gb.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
            ctypes.c_double, ctypes.c_double
        ]
        _rust_lib.ffi_predict_gb.restype = ctypes.c_double
        
        # ffi_fused_compute_step
        _rust_lib.ffi_fused_compute_step.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)
        ]
        
    except Exception as e:
        print(f"[FFI] Error setting up ML bounds: {e}")

# =====================================================================
# MACHINE LEARNING INFERENCE KERNELS (Nano-Latency)
# =====================================================================

def fused_compute_step(
    closes: np.ndarray,
    volumes: np.ndarray,
    portfolio_state: np.ndarray,
    gene_params: np.ndarray,
    brain_weights: np.ndarray,
    l2_state: np.ndarray,
    window: int = 5
) -> np.ndarray:
    """[NANO-SPEED] Rust FFI Reemplazo de Fused Compute Step."""
    out = np.zeros(4, dtype=np.float64)
    if not _rust_lib:
        return out
        
    c = np.asarray(closes, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    ps = np.asarray(portfolio_state, dtype=np.float64)
    gp = np.asarray(gene_params, dtype=np.float64)
    bw = np.asarray(brain_weights, dtype=np.float64)
    l2 = np.asarray(l2_state, dtype=np.float64)
    
    _rust_lib.ffi_fused_compute_step(
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(c),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        l2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        window,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    return out.astype(np.float32)

def predict_rf_jit(
    X: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
    tree_offsets: np.ndarray
) -> float:
    """[NANO-SPEED] Rust FFI Reemplazo de RF."""
    if not _rust_lib:
        return 0.5
        
    x = np.asarray(X, dtype=np.float64)
    cl = np.asarray(children_left, dtype=np.int64)
    cr = np.asarray(children_right, dtype=np.int64)
    feat = np.asarray(feature, dtype=np.int64)
    thresh = np.asarray(threshold, dtype=np.float64)
    val = np.asarray(value, dtype=np.float64)
    to = np.asarray(tree_offsets, dtype=np.int64)
    
    return _rust_lib.ffi_predict_rf(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(x),
        cl.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        cr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        feat.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        thresh.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(cl),
        to.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(to)
    )

def predict_gb_jit(
    X: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
    tree_offsets: np.ndarray,
    init_score: float,
    learning_rate: float
) -> float:
    """[NANO-SPEED] Rust FFI Reemplazo de GB."""
    if not _rust_lib:
        return 0.5
        
    x = np.asarray(X, dtype=np.float64)
    cl = np.asarray(children_left, dtype=np.int64)
    cr = np.asarray(children_right, dtype=np.int64)
    feat = np.asarray(feature, dtype=np.int64)
    thresh = np.asarray(threshold, dtype=np.float64)
    val = np.asarray(value, dtype=np.float64)
    to = np.asarray(tree_offsets, dtype=np.int64)
    
    return _rust_lib.ffi_predict_gb(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(x),
        cl.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        cr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        feat.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        thresh.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(cl),
        to.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(to),
        float(init_score), float(learning_rate)
    )
'''

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(new_content)
