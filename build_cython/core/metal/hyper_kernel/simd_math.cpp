#include <immintrin.h>
#include <cstdint>
#include <cmath>

extern "C" {

    // Calcula la segunda derivada (aceleración) del Order Book Imbalance (OBI)
    // Utiliza intrinsics AVX2 (__m256) para procesar 8 floats a la vez.
    // obi_t0: OBI actual
    // obi_t1: OBI hace 1 lag
    // obi_t2: OBI hace 2 lags
    // accel_out: Output (Segunda Derivada) = (obi_t0 - obi_t1) - (obi_t1 - obi_t2) = obi_t0 - 2*obi_t1 + obi_t2
    void compute_obi_acceleration_simd(
        const float* obi_t0,
        const float* obi_t1,
        const float* obi_t2,
        float* accel_out,
        size_t len
    ) {
        size_t i = 0;
        
        // Procesamiento en batch de 8 floats usando AVX2
        for (; i + 7 < len; i += 8) {
            __m256 v_t0 = _mm256_loadu_ps(&obi_t0[i]);
            __m256 v_t1 = _mm256_loadu_ps(&obi_t1[i]);
            __m256 v_t2 = _mm256_loadu_ps(&obi_t2[i]);
            
            // v_t1_x2 = v_t1 * 2.0
            __m256 v_two = _mm256_set1_ps(2.0f);
            __m256 v_t1_x2 = _mm256_mul_ps(v_t1, v_two);
            
            // v_accel = v_t0 - v_t1_x2 + v_t2
            __m256 v_diff1 = _mm256_sub_ps(v_t0, v_t1_x2);
            __m256 v_accel = _mm256_add_ps(v_diff1, v_t2);
            
            _mm256_storeu_ps(&accel_out[i], v_accel);
        }
        
        // Tail processing para los elementos restantes
        for (; i < len; ++i) {
            accel_out[i] = obi_t0[i] - 2.0f * obi_t1[i] + obi_t2[i];
        }
    }

}
