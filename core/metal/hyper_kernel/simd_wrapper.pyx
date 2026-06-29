# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from libc.stddef cimport size_t
import numpy as np
cimport numpy as cnp

cdef extern from "simd_math.cpp":
    void compute_obi_acceleration_simd(
        const float* obi_t0,
        const float* obi_t1,
        const float* obi_t2,
        float* accel_out,
        size_t len
    ) nogil

def compute_obi_accel(const float[::1] obi_t0, const float[::1] obi_t1, const float[::1] obi_t2, float[::1] accel_out):
    cdef size_t length = obi_t0.shape[0]
    with nogil:
        compute_obi_acceleration_simd(
            &obi_t0[0],
            &obi_t1[0],
            &obi_t2[0],
            &accel_out[0],
            length
        )
