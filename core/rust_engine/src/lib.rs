mod quantum_arena;
mod stateful_engine;

pub use quantum_arena::{QuantumRingBuffer, FEATURE_SIZE};
pub use stateful_engine::StatefulEngine;

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
