use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};

#[pyclass]
pub struct QuantumTensors;

#[pymethods]
impl QuantumTensors {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    /// Fast Hurst Exponent calculation using zero-copy slice and math.
    /// Replaces the memory-allocating Python numpy loop in `calculate_hurst_exponent`.
    pub fn simd_hurst_exponent<'py>(
        &self,
        prices: PyReadonlyArray1<'py, f64>,
        max_lags: usize,
    ) -> f64 {
        let prices_s = prices.as_slice().unwrap_or(&[]);
        let n = prices_s.len();
        
        if n < max_lags * 2 || max_lags < 2 {
            return 0.5; // Random walk assumption
        }
        
        let num_lags = max_lags - 1;
        let mut sum_log_lags = 0.0;
        let mut sum_log_tau = 0.0;
        let mut sum_log_lags_sq = 0.0;
        let mut sum_log_lags_tau = 0.0;
        let mut valid_count = 0;
        
        for lag in 2..=max_lags {
            let count = n - lag;
            let mut diff_sum = 0.0;
            let mut diff_sq_sum = 0.0;
            
            for i in 0..count {
                let diff = prices_s[i + lag] - prices_s[i];
                diff_sum += diff;
                diff_sq_sum += diff * diff;
            }
            
            let count_f = count as f64;
            let mean = diff_sum / count_f;
            let variance = (diff_sq_sum / count_f) - (mean * mean);
            
            if variance > 0.0 {
                let std_dev = variance.sqrt();
                let log_lag = (lag as f64).ln();
                let log_tau = std_dev.ln();
                
                sum_log_lags += log_lag;
                sum_log_tau += log_tau;
                sum_log_lags_sq += log_lag * log_lag;
                sum_log_lags_tau += log_lag * log_tau;
                valid_count += 1;
            }
        }
        
        if valid_count < 3 {
            return 0.5;
        }
        
        let count_f = valid_count as f64;
        let mean_x = sum_log_lags / count_f;
        let mean_y = sum_log_tau / count_f;
        
        let cov_xy = (sum_log_lags_tau / count_f) - (mean_x * mean_y);
        let var_x = (sum_log_lags_sq / count_f) - (mean_x * mean_x);
        
        if var_x > 1e-10 {
            cov_xy / var_x
        } else {
            0.5
        }
    }

    /// O(1) Zero-Copy implementation of _reconstruct_neural_state.
    /// Builds a 25-dimensional float32 state tensor from raw market arrays.
    pub fn build_state_tensor<'py>(
        &self,
        py: Python<'py>,
        closes: PyReadonlyArray1<'py, f64>,
        volumes: PyReadonlyArray1<'py, f64>,
        ps: PyReadonlyArray1<'py, f64>,
        gene_params: PyReadonlyArray1<'py, f64>,
        l2_state: PyReadonlyArray1<'py, f64>,
        window: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let closes_s = closes.as_slice()?;
        let volumes_s = volumes.as_slice()?;
        let ps_s = ps.as_slice()?;
        let genes_s = gene_params.as_slice()?;
        let l2_s = l2_state.as_slice()?;

        let n = closes_s.len();
        let mut state = vec![0.0f32; 25];

        if n < 30 || volumes_s.len() != n {
            return Ok(state.into_pyarray_bound(py));
        }

        // 1. Returns (5)
        for i in 0..window {
            let idx = n - window + i;
            if idx > 0 {
                let prev = closes_s[idx - 1];
                if prev != 0.0 {
                    state[i] = ((closes_s[idx] - prev) / prev) as f32;
                }
            }
        }

        // 2. Volumes (5)
        let vol_start = if n >= 20 { n - 20 } else { 0 };
        let mut vol_sum = 0.0;
        for j in vol_start..n {
            vol_sum += volumes_s[j];
        }
        let mean_vol = if vol_sum > 0.0 { vol_sum / 20.0 } else { 1.0 };

        for i in 0..window {
            let idx = n - window + i;
            state[5 + i] = (volumes_s[idx] / mean_vol) as f32;
        }

        // 3. Momentum Proxy (5)
        for i in 0..window {
            let idx = n - window + i;
            let mut mom = 0.0;
            if idx >= 2 {
                let prev2 = closes_s[idx - 2];
                if prev2 != 0.0 {
                    mom = (closes_s[idx] / prev2) - 1.0;
                }
            }
            state[10 + i] = mom as f32;
        }

        // Placeholder (state[15..20] is 0.0, but we inject L2 Data below)
        
        // Inject L2 Data
        if l2_s.len() >= 2 {
            state[18] = l2_s[0] as f32;
            state[19] = l2_s[1] as f32;
        }

        // 4. Portfolio & Genes
        if ps_s.len() >= 3 {
            state[20] = ps_s[0] as f32;
            state[21] = ps_s[1] as f32;
            state[22] = ps_s[2] as f32;
        }
        if genes_s.len() >= 2 {
            state[23] = genes_s[0] as f32;
            state[24] = genes_s[1] as f32;
        }

        Ok(state.into_pyarray_bound(py))
    }
}
