use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, IntoPyArray};

#[pyclass]
pub struct QuantumMath;

#[pymethods]
impl QuantumMath {
    #[new]
    pub fn new() -> Self {
        QuantumMath {}
    }

    /// Fast Exponential Moving Average (EMA) O(N) using Zero-Copy rust-numpy
    #[staticmethod]
    pub fn ema<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let data_slice = match data.as_slice() {
            Ok(s) => s,
            Err(_) => return Err(pyo3::exceptions::PyValueError::new_err("Numpy array is not contiguous")),
        };
        
        let mut result = vec![f64::NAN; data_slice.len()];
        if data_slice.len() < period || period == 0 {
            return Ok(result.into_pyarray_bound(py));
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut sma = 0.0;
        for i in 0..period {
            sma += data_slice[i];
        }
        sma /= period as f64;

        let mut ema_prev = sma;
        result[period - 1] = ema_prev;

        for i in period..data_slice.len() {
            ema_prev = (data_slice[i] - ema_prev) * multiplier + ema_prev;
            result[i] = ema_prev;
        }

        Ok(result.into_pyarray_bound(py))
    }

    /// Fast Relative Strength Index (RSI) O(N) using Zero-Copy rust-numpy
    #[staticmethod]
    pub fn rsi<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, f64>, period: usize) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let data_slice = match data.as_slice() {
            Ok(s) => s,
            Err(_) => return Err(pyo3::exceptions::PyValueError::new_err("Numpy array is not contiguous")),
        };
        
        let mut result = vec![f64::NAN; data_slice.len()];
        if data_slice.len() <= period || period == 0 {
            return Ok(result.into_pyarray_bound(py));
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..=period {
            let change = data_slice[i] - data_slice[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        let mut avg_gain = gains / period as f64;
        let mut avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            result[period] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[period] = 100.0 - (100.0 / (1.0 + rs));
        }

        for i in (period + 1)..data_slice.len() {
            let change = data_slice[i] - data_slice[i - 1];
            let mut gain = 0.0;
            let mut loss = 0.0;

            if change > 0.0 {
                gain = change;
            } else {
                loss = -change;
            }

            avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

            if avg_loss == 0.0 {
                result[i] = 100.0;
            } else {
                let rs = avg_gain / avg_loss;
                result[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        Ok(result.into_pyarray_bound(py))
    }
}
