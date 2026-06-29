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

use quantum_arena::TradeDecision;

#[no_mangle]
pub extern "C" fn execute_oracle_kernel(arena: *const QuantumStateArena) -> TradeDecision {
    if arena.is_null() {
        return TradeDecision {
            action: 0,
            position_size: 0.0,
            stop_loss: 0.0,
            take_profit: 0.0,
            confidence: 0.0,
            error_code: 1,
            mempool_panic: 0.0,
            net_liq_pressure: 0.0,
            liquidation_cascade: 0.0,
        };
    }

    let a = unsafe { &*arena };

    // Very basic placeholder Oracle decision
    // Will be augmented with XGBoost trees in Phase III
    let conf = 0.85 - (a.mempool_panic_score * 0.1);

    let action = if a.net_liq_pressure > 0.0 {
        1 // Long
    } else if a.net_liq_pressure < 0.0 {
        -1 // Short
    } else {
        0 // Hold
    };

    TradeDecision {
        action,
        position_size: 1.0,
        stop_loss: 0.01,
        take_profit: 0.02,
        confidence: conf,
        error_code: 0,
        mempool_panic: a.mempool_panic_score,
        net_liq_pressure: a.net_liq_pressure,
        liquidation_cascade: 0.0, // To be populated properly with the DarkAlphaRouter
    }
}

use serde::Deserialize;

#[derive(Deserialize)]
struct BinanceWSKline {
    t: i64,      // Kline start time
    c: String,   // Close price
    v: String,   // Base asset volume
    x: bool,     // Is this kline closed?
}

#[derive(Deserialize)]
struct BinanceWSData {
    e: String,             // Event type
    E: i64,                // Event time
    k: Option<BinanceWSKline>, // Kline struct
}

#[derive(Deserialize)]
struct BinanceWSMessage {
    stream: Option<String>,
    data: Option<BinanceWSData>,
}

#[no_mangle]
pub extern "C" fn ingest_raw_ws_frame(
    arena_ptr: *mut QuantumStateArena,
    raw_bytes_ptr: *const u8,
    length: usize,
    index: usize,
) -> i32 {
    if arena_ptr.is_null() || raw_bytes_ptr.is_null() {
        return -1; // Null pointer error
    }

    let raw_slice = unsafe { std::slice::from_raw_parts(raw_bytes_ptr, length) };

    // Deserialize directly from the byte slice (Zero-Copy where possible)
    let msg: Result<serde_json::Value, _> = serde_json::from_slice(raw_slice);

    match msg {
        Ok(parsed_msg) => {
            if let Some(data) = parsed_msg.get("data") {
                if let Some(e) = data.get("e") {
                    if e == "kline" {
                        if let Some(kline) = data.get("k") {
                            unsafe {
                                let arena = &mut *arena_ptr;
                                if index < arena.tensor_len {
                                    let prices = std::slice::from_raw_parts_mut(arena.prices as *mut f32, arena.tensor_len);
                                    let volumes = std::slice::from_raw_parts_mut(arena.volumes as *mut f32, arena.tensor_len);
                                    
                                    if let (Some(c), Some(v)) = (kline.get("c"), kline.get("v")) {
                                        if let (Ok(p), Ok(vol)) = (c.as_str().unwrap_or("").parse::<f32>(), v.as_str().unwrap_or("").parse::<f32>()) {
                                            prices[index] = p;
                                            volumes[index] = vol;
                                            
                                            let t_val = kline.get("t").and_then(|t| t.as_i64()).unwrap_or(0);
                                            arena.timestamp_ns = t_val * 1_000_000;
                                            
                                            let is_closed = kline.get("x").and_then(|x| x.as_bool()).unwrap_or(false);
                                            return if is_closed { 1 } else { 0 };
                                        }
                                        return -8; // Parse f32 failed
                                    }
                                    return -7; // c or v missing
                                }
                                return -6; // index out of bounds
                            }
                        }
                        return -5; // k missing
                    }
                    return -4; // e != kline
                }
                return -3; // e missing
            }
            -10 // successfully parsed JSON, but no 'data' key (e.g. not multiplexed)
        }
        Err(e) => {
            eprintln!("Rust JSON parse error: {:?}", e);
            -2
        }
    }
}


#[no_mangle]
pub extern "C" fn run_sandbox_trial(
    highs: *const f64,
    lows: *const f64,
    closes: *const f64,
    signals: *const i8,
    len: usize,
    sl_pct: f64,
    kinematic_umbral: f64,
    out_pnl: *mut f64,
    out_duration: *mut i32,
    out_stats: *mut f64, 
) -> usize {
    if highs.is_null() || lows.is_null() || closes.is_null() || signals.is_null() || out_pnl.is_null() || out_duration.is_null() || out_stats.is_null() {
        return 0;
    }

    let h = unsafe { std::slice::from_raw_parts(highs, len) };
    let l = unsafe { std::slice::from_raw_parts(lows, len) };
    let c = unsafe { std::slice::from_raw_parts(closes, len) };
    let sigs = unsafe { std::slice::from_raw_parts(signals, len) };
    
    let pnl_out = unsafe { std::slice::from_raw_parts_mut(out_pnl, len) };
    let dur_out = unsafe { std::slice::from_raw_parts_mut(out_duration, len) };
    let stats_out = unsafe { std::slice::from_raw_parts_mut(out_stats, 3) };
    
    let mut trade_count = 0;
    let mut in_position = false;
    let mut entry_price = 0.0;
    let mut pos_side = 0i8; 
    let mut bars_held = 0;
    
    let mut wins = 0;
    let mut v_t = 0.0;
    
    for i in 0..len {
        let current_close = c[i];
        let current_high = h[i];
        let current_low = l[i];
        
        if in_position {
            bars_held += 1;
            
            let pnl_pct = if pos_side == 1 {
                (current_close - entry_price) / entry_price
            } else {
                (entry_price - current_close) / entry_price
            };
            
            let prev_v = v_t;
            v_t = pnl_pct; 
            let a_t = v_t - prev_v;
            
            let mut exit_price = 0.0;
            let mut exit_reason = 0; 
            
            if pos_side == 1 {
                let sl_price = entry_price * (1.0 - sl_pct);
                if current_low <= sl_price {
                    exit_price = sl_price;
                    exit_reason = 2;
                } else if pnl_pct > 0.005 && a_t < kinematic_umbral {
                    exit_price = current_close;
                    exit_reason = 1;
                }
            } else {
                let sl_price = entry_price * (1.0 + sl_pct);
                if current_high >= sl_price {
                    exit_price = sl_price;
                    exit_reason = 2;
                } else if pnl_pct > 0.005 && a_t < kinematic_umbral {
                    exit_price = current_close;
                    exit_reason = 1;
                }
            }
            
            if exit_reason > 0 {
                let raw_pnl = if pos_side == 1 {
                    (exit_price - entry_price) / entry_price
                } else {
                    (entry_price - exit_price) / entry_price
                };
                
                let net_pnl = raw_pnl - 0.0012; // Taker fees
                pnl_out[trade_count] = net_pnl;
                dur_out[trade_count] = bars_held;
                
                if net_pnl > 0.0 {
                    wins += 1;
                }
                
                trade_count += 1;
                in_position = false;
                bars_held = 0;
                v_t = 0.0;
                
                // PRUNER NATIVO (Phase II)
                if trade_count == 100 {
                    let wr = (wins as f64) / 100.0;
                    if wr < 0.70 {
                        stats_out[0] = wr;
                        stats_out[1] = trade_count as f64;
                        stats_out[2] = 1.0; 
                        return trade_count;
                    }
                }
                continue;
            }
        }
        
        if !in_position {
            let sig = sigs[i];
            if sig != 0 {
                in_position = true;
                pos_side = sig;
                entry_price = current_close;
                bars_held = 0;
                v_t = 0.0;
            }
        }
    }
    
    let wr = if trade_count > 0 { (wins as f64) / (trade_count as f64) } else { 0.0 };
    stats_out[0] = wr;
    stats_out[1] = trade_count as f64;
    stats_out[2] = 0.0; // SUCCESS
    
    trade_count
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

#[no_mangle]
pub unsafe extern "C" fn ffi_fused_compute_step(
    closes_ptr: *const f64,
    closes_len: usize,
    volumes_ptr: *const f64,
    portfolio_ptr: *const f64, // len 3
    gene_ptr: *const f64,      // len 2
    brain_ptr: *const f64,     // len 100
    l2_ptr: *const f64,        // len 2
    window: usize,
    out_ptr: *mut f64          // len 4
) {
    let closes = std::slice::from_raw_parts(closes_ptr, closes_len);
    let volumes = std::slice::from_raw_parts(volumes_ptr, closes_len);
    let port = &*(portfolio_ptr as *const [f64; 3]);
    let gene = &*(gene_ptr as *const [f64; 2]);
    let brain = &*(brain_ptr as *const [f64; 100]);
    let l2 = &*(l2_ptr as *const [f64; 2]);
    let out = &mut *(out_ptr as *mut [f64; 4]);

    math_kernels::fused_compute_step(closes, volumes, port, gene, brain, l2, window, out);
}

