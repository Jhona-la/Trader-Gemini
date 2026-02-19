
import numpy as np

def aligned_empty(shape, dtype=np.float64, align=64):
    """
    🔬 PHASE 30: L2 CACHE ALIGNMENT
    Allocates a numpy array with memory aligned to 'align' bytes (default 64).
    Crucial for SIMD (AVX2/AVX512) and minimizing cache line splits.
    """
    dtype = np.dtype(dtype)
    n_bytes = dtype.itemsize * np.prod(shape)
    
    # Allocate extra bytes to ensure we can shift to alignment
    # + align to provide room for offset
    # + itemsize to be safe
    # We use uint8 for byte-level manipulation
    buffer = np.empty(n_bytes + align, dtype=np.uint8)
    
    # Find the aligned offset
    start_index = buffer.ctypes.data % align
    offset = 0
    if start_index != 0:
        offset = align - start_index
        
    # Create a view into the aligned region
    aligned_view = buffer[offset : offset + n_bytes].view(dtype)
    aligned_view = aligned_view.reshape(shape)
    
    return aligned_view

def aligned_zeros(shape, dtype=np.float64, align=64):
    arr = aligned_empty(shape, dtype, align)
    arr.fill(0)
    return arr
