pub mod execution;
pub mod parsers;
pub mod portfolio;
pub mod risk;
pub mod math_kernels;
pub mod quantum_arena;
pub mod stateful_engine;
pub mod dark_alpha_router;
pub mod trailing;
pub mod networking;
pub mod executor;
pub mod orderbook;
pub mod ffi_networking;
pub mod unified_engine;

pub mod config;
pub mod ml_inference;

pub use quantum_arena::{QuantumRingBuffer, FEATURE_SIZE, QuantumStateArena};
pub use stateful_engine::StatefulEngine;
pub use trailing::{evaluate_quantum_trailing, TrailingResult};
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn ffi_update_portfolio(usdt_balance: f64) {
    portfolio::Portfolio::update_balance(usdt_balance);
}

#[no_mangle]
pub extern "C" fn ffi_set_position(horizon: i32, side: i32, entry_price: f64, qty: f64) {
    portfolio::Portfolio::set_position(horizon, side, entry_price, qty);
}

#[no_mangle]
pub extern "C" fn ffi_clear_position(horizon: i32) {
    portfolio::Portfolio::clear_position(horizon);
}

#[no_mangle]
pub extern "C" fn ffi_can_open_position(horizon: i32, requested_qty: f64, current_price: f64) -> bool {
    risk::RiskManager::can_open_position(horizon, requested_qty, current_price)
}

#[no_mangle]
pub extern "C" fn ffi_check_drawdown(current_price: f64) -> bool {
    risk::RiskManager::check_drawdown(current_price)
}

#[no_mangle]
pub extern "C" fn quantum_ring_new() -> *mut QuantumRingBuffer {
    let b = Box::new(QuantumRingBuffer::new());
    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn quantum_ring_free(ptr: *mut QuantumRingBuffer) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn engine_new() -> *mut StatefulEngine {
    let e = Box::new(StatefulEngine::new());
    Box::into_raw(e)
}

#[no_mangle]
pub extern "C" fn engine_free(ptr: *mut StatefulEngine) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn engine_process_and_inject(
    engine: *mut StatefulEngine,
    ring: *mut QuantumRingBuffer,
    reader_idx: usize,
    price: f64,
    volume: f64
) -> bool {
    if engine.is_null() || ring.is_null() {
        return false;
    }
    
    let result = std::panic::catch_unwind(|| {
        unsafe {
            (*engine).process_tick(price, volume);
            
            let mut f32_payload = [0.0f32; FEATURE_SIZE];
            (*engine).export_f32(&mut f32_payload);
            
            (*ring).write_tick(reader_idx, &f32_payload)
        }
    });

    match result {
        Ok(success) => success,
        Err(_) => {
            // Panic occurred. Caught safely to prevent C abort.
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn get_engine_drop_counter() -> usize {
    stateful_engine::DROP_COUNTER.load(std::sync::atomic::Ordering::SeqCst)
}

#[no_mangle]
pub extern "C" fn quantum_ring_read_tick(
    ring: *const QuantumRingBuffer,
    read_idx: usize,
    out_payload: *mut f32
) -> bool {
    if ring.is_null() || out_payload.is_null() {
        return false;
    }
    
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(out_payload, FEATURE_SIZE);
        let mut temp_payload = [0.0f32; FEATURE_SIZE];
        let success = (*ring).read_tick(read_idx, &mut temp_payload);
        if success {
            out_slice.copy_from_slice(&temp_payload);
        }
        success
    }
}



// =========================================================
// MATH KERNELS EXPORTS (FFI)
// =========================================================

#[no_mangle]
pub extern "C" fn ffi_compute_kelly_fraction(
    p: f64, b: f64, apply_mult: bool, kelly_mult: f64, stress_score: f64, max_exposure: f64
) -> f64 {
    math_kernels::compute_kelly_fraction(p, b, apply_mult, kelly_mult, stress_score, max_exposure)
}

#[no_mangle]
pub extern "C" fn ffi_extract_kelly_stats(
    pnl_ptr: *const f64, pnl_len: usize, 
    is_win_ptr: *const bool, is_win_len: usize,
    out_p: *mut f64, out_b: *mut f64
) {
    if pnl_ptr.is_null() || is_win_ptr.is_null() || out_p.is_null() || out_b.is_null() || pnl_len != is_win_len {
        return;
    }
    let pnl_slice = unsafe { std::slice::from_raw_parts(pnl_ptr, pnl_len) };
    let is_win_slice = unsafe { std::slice::from_raw_parts(is_win_ptr, is_win_len) };
    
    let (p, b) = math_kernels::extract_kelly_stats(pnl_slice, is_win_slice);
    unsafe {
        *out_p = p;
        *out_b = b;
    }
}

#[no_mangle]
pub extern "C" fn ffi_compute_cvar(
    loss_ptr: *const f64, loss_len: usize, confidence_level: f64
) -> f64 {
    if loss_ptr.is_null() || loss_len == 0 {
        return 0.0;
    }
    let loss_slice = unsafe { std::slice::from_raw_parts(loss_ptr, loss_len) };
    math_kernels::compute_cvar(loss_slice, confidence_level)
}

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

// =====================================================================
// FFI C-ABI EXPORTS: MACHINE LEARNING INFERENCE
// =====================================================================

#[no_mangle]
pub unsafe extern "C" fn ffi_predict_rf(
    x_ptr: *const f64,
    x_len: usize,
    cl_ptr: *const i64,
    cr_ptr: *const i64,
    feat_ptr: *const i64,
    thresh_ptr: *const f64,
    val_ptr: *const f64,
    nodes_len: usize,
    to_ptr: *const i64,
    to_len: usize,
) -> f64 {
    let x = std::slice::from_raw_parts(x_ptr, x_len);
    let cl = std::slice::from_raw_parts(cl_ptr, nodes_len);
    let cr = std::slice::from_raw_parts(cr_ptr, nodes_len);
    let feat = std::slice::from_raw_parts(feat_ptr, nodes_len);
    let thresh = std::slice::from_raw_parts(thresh_ptr, nodes_len);
    let val = std::slice::from_raw_parts(val_ptr, nodes_len);
    let to = std::slice::from_raw_parts(to_ptr, to_len);

    math_kernels::predict_rf(x, cl, cr, feat, thresh, val, to)
}

#[no_mangle]
pub unsafe extern "C" fn ffi_predict_gb(
    x_ptr: *const f64,
    x_len: usize,
    cl_ptr: *const i64,
    cr_ptr: *const i64,
    feat_ptr: *const i64,
    thresh_ptr: *const f64,
    val_ptr: *const f64,
    nodes_len: usize,
    to_ptr: *const i64,
    to_len: usize,
    init_score: f64,
    learning_rate: f64,
) -> f64 {
    let x = std::slice::from_raw_parts(x_ptr, x_len);
    let cl = std::slice::from_raw_parts(cl_ptr, nodes_len);
    let cr = std::slice::from_raw_parts(cr_ptr, nodes_len);
    let feat = std::slice::from_raw_parts(feat_ptr, nodes_len);
    let thresh = std::slice::from_raw_parts(thresh_ptr, nodes_len);
    let val = std::slice::from_raw_parts(val_ptr, nodes_len);
    let to = std::slice::from_raw_parts(to_ptr, to_len);

    math_kernels::predict_gb(x, cl, cr, feat, thresh, val, to, init_score, learning_rate)
}

