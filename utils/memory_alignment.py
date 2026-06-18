import numpy as np

def aligned_zeros(shape, dtype=np.float32, alignment=64):
    """
    Allocates an array aligned to the specified byte boundary.
    Required for CPU cacheline alignment (avoiding false sharing)
    and future AVX-512 SIMD operations.
    """
    size = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(size + alignment, dtype=np.uint8)
    offset = (alignment - (buffer.ctypes.data % alignment)) % alignment
    aligned_buffer = buffer[offset:offset+size].view(dtype).reshape(shape)
    aligned_buffer.fill(0)
    return aligned_buffer
