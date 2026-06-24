import os

kernels_code = """
// =========================================================
// VECTORIZED TECHNICAL INDICATORS
// =========================================================

#[inline(always)]
pub fn compute_ema_vectorized(data: &[f64], period: usize, out: &mut [f64]) {
    let n = data.len();
    if n == 0 || period == 0 || out.len() != n {
        return;
    }
    let k = 2.0 / (period as f64 + 1.0);
    out[0] = data[0];
    for i in 1..n {
        out[i] = data[i] * k + out[i - 1] * (1.0 - k);
    }
}

#[inline(always)]
pub fn compute_rsi_vectorized(data: &[f64], period: usize, out: &mut [f64]) {
    let n = data.len();
    if n < period || period == 0 || out.len() != n {
        for i in 0..n { out[i] = 50.0; } // Default safe value
        return;
    }
    
    let mut gain = 0.0;
    let mut loss = 0.0;
    
    // Seed first window
    for i in 1..period {
        let diff = data[i] - data[i - 1];
        if diff > 0.0 {
            gain += diff;
        } else {
            loss -= diff;
        }
    }
    
    gain /= period as f64;
    loss /= period as f64;
    
    // Fill until period with 50.0 to prevent artifacting
    for i in 0..period {
        out[i] = 50.0;
    }
    
    if loss == 0.0 {
        out[period - 1] = 100.0;
    } else {
        let rs = gain / loss;
        out[period - 1] = 100.0 - (100.0 / (1.0 + rs));
    }
    
    // Smoothed Wilders moving average
    for i in period..n {
        let diff = data[i] - data[i - 1];
        if diff > 0.0 {
            gain = (gain * (period as f64 - 1.0) + diff) / period as f64;
            loss = (loss * (period as f64 - 1.0)) / period as f64;
        } else {
            gain = (gain * (period as f64 - 1.0)) / period as f64;
            loss = (loss * (period as f64 - 1.0) - diff) / period as f64;
        }
        if loss == 0.0 {
            out[i] = 100.0;
        } else {
            let rs = gain / loss;
            out[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
}

#[inline(always)]
pub fn compute_bollinger_bands(data: &[f64], period: usize, std_dev_mult: f64, out_up: &mut [f64], out_mid: &mut [f64], out_low: &mut [f64]) {
    let n = data.len();
    if n < period || period == 0 {
        for i in 0..n {
            out_mid[i] = data[i];
            out_up[i] = data[i];
            out_low[i] = data[i];
        }
        return;
    }
    
    for i in 0..period-1 {
        out_mid[i] = data[i];
        out_up[i] = data[i];
        out_low[i] = data[i];
    }
    
    let window = period as f64;
    for i in (period - 1)..n {
        let mut sum = 0.0;
        for j in 0..period {
            sum += data[i - j];
        }
        let mean = sum / window;
        
        let mut variance = 0.0;
        for j in 0..period {
            let diff = data[i - j] - mean;
            variance += diff * diff;
        }
        let std_dev = (variance / window).sqrt();
        
        out_mid[i] = mean;
        out_up[i] = mean + std_dev_mult * std_dev;
        out_low[i] = mean - std_dev_mult * std_dev;
    }
}

#[inline(always)]
pub fn compute_macd(data: &[f64], fast_period: usize, slow_period: usize, signal_period: usize, out_macd: &mut [f64], out_signal: &mut [f64], out_hist: &mut [f64]) {
    let n = data.len();
    if n == 0 { return; }
    
    let mut fast_ema = vec![0.0; n];
    let mut slow_ema = vec![0.0; n];
    
    compute_ema_vectorized(data, fast_period, &mut fast_ema);
    compute_ema_vectorized(data, slow_period, &mut slow_ema);
    
    for i in 0..n {
        out_macd[i] = fast_ema[i] - slow_ema[i];
    }
    
    compute_ema_vectorized(&out_macd, signal_period, out_signal);
    
    for i in 0..n {
        out_hist[i] = out_macd[i] - out_signal[i];
    }
}
"""

lib_code = """
// =========================================================
// INDICATORS EXPORTS (FFI)
// =========================================================

#[no_mangle]
pub extern "C" fn ffi_compute_ema(
    data_ptr: *const f64, data_len: usize,
    period: usize,
    out_ptr: *mut f64
) {
    if data_ptr.is_null() || out_ptr.is_null() || data_len == 0 { return; }
    let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, data_len) };
    math_kernels::compute_ema_vectorized(data_slice, period, out_slice);
}

#[no_mangle]
pub extern "C" fn ffi_compute_rsi(
    data_ptr: *const f64, data_len: usize,
    period: usize,
    out_ptr: *mut f64
) {
    if data_ptr.is_null() || out_ptr.is_null() || data_len == 0 { return; }
    let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, data_len) };
    math_kernels::compute_rsi_vectorized(data_slice, period, out_slice);
}

#[no_mangle]
pub extern "C" fn ffi_compute_bbands(
    data_ptr: *const f64, data_len: usize,
    period: usize, std_dev: f64,
    out_up: *mut f64, out_mid: *mut f64, out_low: *mut f64
) {
    if data_ptr.is_null() || out_up.is_null() || out_mid.is_null() || out_low.is_null() || data_len == 0 { return; }
    let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
    let up_slice = unsafe { std::slice::from_raw_parts_mut(out_up, data_len) };
    let mid_slice = unsafe { std::slice::from_raw_parts_mut(out_mid, data_len) };
    let low_slice = unsafe { std::slice::from_raw_parts_mut(out_low, data_len) };
    math_kernels::compute_bollinger_bands(data_slice, period, std_dev, up_slice, mid_slice, low_slice);
}

#[no_mangle]
pub extern "C" fn ffi_compute_macd(
    data_ptr: *const f64, data_len: usize,
    fast_period: usize, slow_period: usize, signal_period: usize,
    out_macd: *mut f64, out_signal: *mut f64, out_hist: *mut f64
) {
    if data_ptr.is_null() || out_macd.is_null() || out_signal.is_null() || out_hist.is_null() || data_len == 0 { return; }
    let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
    let macd_slice = unsafe { std::slice::from_raw_parts_mut(out_macd, data_len) };
    let signal_slice = unsafe { std::slice::from_raw_parts_mut(out_signal, data_len) };
    let hist_slice = unsafe { std::slice::from_raw_parts_mut(out_hist, data_len) };
    math_kernels::compute_macd(data_slice, fast_period, slow_period, signal_period, macd_slice, signal_slice, hist_slice);
}
"""

with open(r'core\rust_engine\src\math_kernels.rs', 'a', encoding='utf-8') as f:
    f.write(kernels_code)

with open(r'core\rust_engine\src\lib.rs', 'a', encoding='utf-8') as f:
    f.write(lib_code)
