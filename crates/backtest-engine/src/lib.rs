pub mod vectorized;

#[repr(C)]
#[derive(Clone)]
pub struct UnifiedConfig {
    pub sl_pct: f64,
    pub tp_pct: f64,
    pub ml_threshold_l: f64,
    pub ml_threshold_s: f64,
    pub tech_threshold_l: f64,
    pub tech_threshold_s: f64,
    pub starting_capital: f64,
    pub scalp_leverage: f64,
    pub swing_leverage: f64,
    pub scalp_sl_ratio: f64,
    pub scalp_tp_ratio: f64,
    pub dyn_atr_min: f64,
    pub dyn_obi: f64,
    pub dyn_ema: f64,
    pub dyn_ofi: f64,
}

pub fn run_backtest_native(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[f64],
    cfg: &UnifiedConfig,
    out_pnl: &mut [f64],
    out_stats: &mut [f64],
    _symbol: &str,
) -> usize {
    let len = closes.len();
    
    // Axioma VII: Paridad Absoluta. Usamos la misma Arena y Core que producción.
    use std::sync::Arc;
    use god_engine_core::GodEngineCore;
    
    use quantum_arena::GlobalArena;
    use std::sync::atomic::Ordering;

    let arena = Arc::new(GlobalArena::new(cfg.starting_capital));
    // Apply configs from Evolution to GlobalArena
    arena.config.scalp_tp_base.store(cfg.tp_pct * cfg.scalp_tp_ratio, Ordering::Relaxed);
    arena.config.scalp_sl_base.store(cfg.sl_pct * cfg.scalp_sl_ratio, Ordering::Relaxed);
    arena.config.global_leverage.store(cfg.scalp_leverage, Ordering::Relaxed);
    arena.config.scalp_obi_threshold.store(cfg.ml_threshold_l, Ordering::Relaxed); // Simplified mapping
    
    arena.config.dynamic_atr_min.store(cfg.dyn_atr_min, Ordering::Relaxed);
    arena.config.dynamic_obi_threshold.store(cfg.dyn_obi, Ordering::Relaxed);
    arena.config.dynamic_ema_trend.store(cfg.dyn_ema, Ordering::Relaxed);
    arena.config.dynamic_ofi_threshold.store(cfg.dyn_ofi, Ordering::Relaxed);
    
    // Set static kelly for backtest baseline
    arena.config.scalp_kelly_fraction.store(0.2, Ordering::Relaxed);
    
    let mut core = GodEngineCore::new(arena.clone());
    
    let mut wins = 0;
    let mut trades = 0;
    
    let mut peak_capital = cfg.starting_capital;
    let mut max_dd = 0.0;
    
    for i in 0..len {
        let current_close = closes[i];
        let current_vol = volumes[i];
        
        let prev_close = if i > 0 { closes[i - 1] } else { current_close };
        let delta = current_close - prev_close;
        
        // Synthesize microstructure (Order Flow Imbalance estimation)
        // If price goes up, buyers were aggressive -> higher bid qty.
        let mut bid_ratio = 0.5;
        if delta > 0.0 {
            bid_ratio = 1.0; // 100% buys
        } else if delta < 0.0 {
            bid_ratio = 0.0; // 100% sells
        }
        
        let bid_qty = current_vol * bid_ratio;
        let ask_qty = current_vol * (1.0 - bid_ratio);
        
        // Interpolación HFT (FASE C): Dividimos la vela en 10 micro-ticks para estimular OFI/OBI
        let num_ticks = 10;
        let price_step = delta / num_ticks as f64;
        let vol_step = current_vol / num_ticks as f64;
        let bid_qty_step = bid_qty / num_ticks as f64;
        let ask_qty_step = ask_qty / num_ticks as f64;
        
        let mut sim_price = prev_close;
        let mut closed_sc = None;
        let mut closed_sw = None;
        
        let mut synthetic_omni = [0.0; 54];
        
        for t in 0..num_ticks {
            sim_price += price_step;
            let is_kline = t == num_ticks - 1; // Solo el último tick cierra la vela
            
            // Simular spread real (2 bps)
            let sim_bid = sim_price * 0.9999;
            let sim_ask = sim_price * 1.0001;
            
            // Llenar synthetic_omni con lo que haya calculado el feature_engine localmente
            let local_features = core.get_features(0);
            for (idx, &val) in local_features.iter().enumerate() {
                if idx < 54 { synthetic_omni[idx] = val as f64; }
            }
            
            let (_, _, sc, sw) = core.process_event(
                0,
                true, is_kline, true,
                sim_price,
                vol_step,
                sim_bid, sim_ask,
                bid_qty_step, ask_qty_step,
                0.5, 0.0,
                (i as u64 * 1000) + (t as u64 * 100),
                false, &synthetic_omni);
                
            if sc.is_some() { closed_sc = sc; }
            if sw.is_some() { closed_sw = sw; }
        }
        
        // Registrar resultados
        if let Some((_is_long, pnl, _qty)) = closed_sc {
            let margin = 5.0; // Minimal placeholder margin
            let pct = pnl / margin;
            out_pnl[trades] = pct;
            if pct > 0.0 { wins += 1; }
            trades += 1;
        }
        
        if let Some((_is_long, pnl, _qty)) = closed_sw {
            let margin = 5.0;
            let pct = pnl / margin;
            out_pnl[trades] = pct;
            if pct > 0.0 { wins += 1; }
            trades += 1;
        }
        
        let current_cap = core.arena.unified_capital.load(Ordering::Relaxed);
        if current_cap > peak_capital { peak_capital = current_cap; }
        let dd = if peak_capital > 0.0 { (peak_capital - current_cap) / peak_capital } else { 0.0 };
        if dd > max_dd { max_dd = dd; }
        
        if current_cap <= 0.0 {
            break;
        }
    }
    
    let final_cap = core.arena.unified_capital.load(Ordering::Relaxed);
    let win_rate = if trades > 0 { wins as f64 / trades as f64 } else { 0.0 };
    
    // Calcular Sharpe, Avg Win, Avg Loss
    let mut total_win_pct = 0.0;
    let mut total_loss_pct = 0.0;
    let mut count_wins = 0.0;
    let mut count_losses = 0.0;
    let mut sum_pnl = 0.0;
    
    for i in 0..trades {
        let p = out_pnl[i];
        sum_pnl += p;
        if p > 0.0 {
            total_win_pct += p;
            count_wins += 1.0;
        } else {
            total_loss_pct += p;
            count_losses += 1.0;
        }
    }
    
    let avg_win = if count_wins > 0.0 { total_win_pct / count_wins } else { 0.0 };
    let avg_loss = if count_losses > 0.0 { total_loss_pct / count_losses } else { 0.0 };
    let mean_pnl = if trades > 0 { sum_pnl / trades as f64 } else { 0.0 };
    
    let mut variance = 0.0;
    if trades > 1 {
        for i in 0..trades {
            variance += (out_pnl[i] - mean_pnl).powi(2);
        }
        variance /= (trades - 1) as f64;
    }
    let std_dev = variance.sqrt();
    let sharpe = if std_dev > 0.0 { mean_pnl / std_dev } else { 0.0 };

    out_stats[0] = win_rate;
    out_stats[1] = trades as f64;
    out_stats[2] = final_cap;
    out_stats[3] = max_dd;
    out_stats[4] = sharpe;
    out_stats[5] = avg_win;
    out_stats[6] = avg_loss;

    trades
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_run_unified_backtest(
    closes_ptr: *const f64,
    highs_ptr: *const f64,
    lows_ptr: *const f64,
    volumes_ptr: *const f64,
    len: usize,
    config: *const UnifiedConfig,
    out_pnl_ptr: *mut f64,
    out_stats_ptr: *mut f64,
    symbol_ptr: *const std::os::raw::c_char,
) -> usize {
    if closes_ptr.is_null() || highs_ptr.is_null() || lows_ptr.is_null() || config.is_null() || out_pnl_ptr.is_null() || out_stats_ptr.is_null() || symbol_ptr.is_null() {
        return 0;
    }

    let closes = unsafe { std::slice::from_raw_parts(closes_ptr, len) };
    let highs = unsafe { std::slice::from_raw_parts(highs_ptr, len) };
    let lows = unsafe { std::slice::from_raw_parts(lows_ptr, len) };
    let volumes = unsafe { std::slice::from_raw_parts(volumes_ptr, len) };
    let cfg = unsafe { &*config };
    let out_pnl = unsafe { std::slice::from_raw_parts_mut(out_pnl_ptr, len) };
    let out_stats = unsafe { std::slice::from_raw_parts_mut(out_stats_ptr, 4) };
    
    let sym_c = unsafe { std::ffi::CStr::from_ptr(symbol_ptr) };
    let sym_str = sym_c.to_str().unwrap_or("BTCUSDT");

    run_backtest_native(closes, highs, lows, volumes, cfg, out_pnl, out_stats, sym_str)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_run_unified_backtest_mmap(
    filepath_ptr: *const std::os::raw::c_char,
    len: usize,
    config: *const UnifiedConfig,
    out_pnl_ptr: *mut f64,
    out_stats_ptr: *mut f64,
    symbol_ptr: *const std::os::raw::c_char,
) -> usize {
    if filepath_ptr.is_null() || config.is_null() || out_pnl_ptr.is_null() || out_stats_ptr.is_null() || symbol_ptr.is_null() {
        return 0;
    }
    
    let filepath_c = unsafe { std::ffi::CStr::from_ptr(filepath_ptr) };
    let filepath = match filepath_c.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let file = match std::fs::File::open(filepath) {
        Ok(f) => f,
        Err(_) => return 0,
    };

    let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(_) => return 0,
    };

    let expected_bytes = len * 4 * 8;
    if mmap.len() < expected_bytes {
        return 0;
    }

    let ptr = mmap.as_ptr() as *const f64;
    let closes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let highs = unsafe { std::slice::from_raw_parts(ptr.add(len), len) };
    let lows = unsafe { std::slice::from_raw_parts(ptr.add(len * 2), len) };
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(len * 3), len) };

    let cfg = unsafe { &*config };
    let out_pnl = unsafe { std::slice::from_raw_parts_mut(out_pnl_ptr, len) };
    let out_stats = unsafe { std::slice::from_raw_parts_mut(out_stats_ptr, 4) };

    let sym_c = unsafe { std::ffi::CStr::from_ptr(symbol_ptr) };
    let sym_str = sym_c.to_str().unwrap_or("BTCUSDT");

    run_backtest_native(closes, highs, lows, volumes, cfg, out_pnl, out_stats, sym_str)
}

#[unsafe(no_mangle)]
pub extern "C" fn ffi_run_polars_backtest_mmap(
    filepath_ptr: *const std::os::raw::c_char,
    len: usize,
    config: *const UnifiedConfig,
    out_stats_ptr: *mut f64,
) -> usize {
    if filepath_ptr.is_null() || config.is_null() || out_stats_ptr.is_null() {
        return 0;
    }
    
    let filepath_c = unsafe { std::ffi::CStr::from_ptr(filepath_ptr) };
    let filepath = match filepath_c.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let file = match std::fs::File::open(filepath) {
        Ok(f) => f,
        Err(_) => return 0,
    };

    let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(_) => return 0,
    };

    let expected_bytes = len * 4 * 8;
    if mmap.len() < expected_bytes {
        return 0;
    }

    let ptr = mmap.as_ptr() as *const f64;
    let closes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let highs = unsafe { std::slice::from_raw_parts(ptr.add(len), len) };
    let lows = unsafe { std::slice::from_raw_parts(ptr.add(len * 2), len) };
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(len * 3), len) };

    let cfg = unsafe { &*config };
    let out_stats = unsafe { std::slice::from_raw_parts_mut(out_stats_ptr, 4) };

    if let Ok((final_cap, max_dd, trades, wins)) = vectorized::run_vectorized_hybrid(closes, highs, lows, volumes, cfg) {
        let win_rate = if trades > 0 { wins as f64 / trades as f64 } else { 0.0 };
        out_stats[0] = win_rate;
        out_stats[1] = trades as f64;
        out_stats[2] = final_cap;
        out_stats[3] = max_dd;
        return trades as usize;
    }
    0
}




