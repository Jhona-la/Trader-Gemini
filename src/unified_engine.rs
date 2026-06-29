use crate::stateful_engine::{StatefulEngine, MarketRegime};
use crate::ml_inference::NanoForest;
use crate::risk::RiskManager;

#[repr(C)]
#[derive(Clone)]
pub struct UnifiedConfig {
    pub sl_pct: f64,
    pub tp_pct: f64,
    pub ml_threshold_l: f64,
    pub ml_threshold_s: f64,
    pub tech_threshold_l: f64,
    pub tech_threshold_s: f64,
}

pub fn run_backtest_native(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[f64],
    cfg: &UnifiedConfig,
    out_pnl: &mut [f64],
    out_stats: &mut [f64],
) -> usize {
    let len = closes.len();
    
    // We instantiate TWO engines to emulate the production environment which separates Ticks (Scalping) and 1H Klines (Swing).
    // In our backtest CSV, we only have 1M or 1H data. We will simulate both on the same data stream for now, 
    // but correctly segment the Logic and Regimes.
    
    let mut scalp_engine = StatefulEngine::new();
    let mut swing_engine = StatefulEngine::new();
    
    let mut risk_manager = RiskManager::new(0.05, 1.0); 
    
    let mut capital = 13.0; // Phase 13 rule: 13 USD starting capital
    let mut peak_capital = 13.0;
    let mut max_dd = 0.0;
    
    let mut scalp_in_pos = false;
    let mut scalp_side = 0_i32; 
    let mut scalp_entry = 0.0;
    let mut scalp_qty = 0.0;
    
    let mut swing_in_pos = false;
    let mut swing_side = 0_i32; 
    let mut swing_entry = 0.0;
    let mut swing_qty = 0.0;
    
    let mut wins = 0;
    let mut trades = 0;

    let fee_rate = 0.0004; // Binance taker fee per side (0.04%)

    for i in 0..len {
        let current_close = closes[i];
        
        scalp_engine.process_tick(current_close, volumes[i]);
        swing_engine.process_tick(current_close, volumes[i]);
        
        let scalp_regime = scalp_engine.get_market_regime();
        let swing_regime = swing_engine.get_market_regime();

        let fast = swing_engine.ema_fast;
        let slow = swing_engine.ema_slow;

        let mut ml_prob = 0.0;
        if i > 10 {
            let features = scalp_engine.get_features();
            ml_prob = crate::ml_inference::NanoForest::predict_global(&features);
        }

        // --- 1. EVALUATE EXITS ---
        if scalp_in_pos {
            let pnl_pct = if scalp_side == 1 { (current_close - scalp_entry) / scalp_entry } else { (scalp_entry - current_close) / scalp_entry };
            let mut exit = false;
            let mut exit_price = current_close;
            let scalp_tp = cfg.tp_pct * 0.33; // Must match production: scalp uses tighter TP
            let scalp_sl = cfg.sl_pct * 0.33; // Must match production: scalp uses tighter SL

            if pnl_pct >= scalp_tp { exit = true; }
            else if pnl_pct <= -scalp_sl { exit = true; }
            // Symmetrical Intra-Bar Evaluation for Stop Loss & Take Profit
            else if scalp_side == 1 && highs[i] >= scalp_entry * (1.0 + scalp_tp) { exit = true; exit_price = scalp_entry * (1.0 + scalp_tp); }
            else if scalp_side == -1 && lows[i] <= scalp_entry * (1.0 - scalp_tp) { exit = true; exit_price = scalp_entry * (1.0 - scalp_tp); }
            else if scalp_side == 1 && lows[i] <= scalp_entry * (1.0 - scalp_sl) { exit = true; exit_price = scalp_entry * (1.0 - scalp_sl); }
            else if scalp_side == -1 && highs[i] >= scalp_entry * (1.0 + scalp_sl) { exit = true; exit_price = scalp_entry * (1.0 + scalp_sl); }
            
            // Adaptive Regime Exit
            if scalp_regime == MarketRegime::Swing && pnl_pct > 0.0 { exit = true; }

            if exit {
                let pnl_amount = scalp_qty * (exit_price - scalp_entry) * (scalp_side as f64);
                let fee_amount = scalp_qty * (scalp_entry + exit_price) * fee_rate; // Round-trip: entry + exit
                let net_pnl_amount = pnl_amount - fee_amount;
                
                capital += net_pnl_amount;
                let margin_used = (scalp_qty * scalp_entry) / 50.0;
                let net_pnl_pct = net_pnl_amount / margin_used;
                
                risk_manager.report_trade_result_local("BTCUSDT", net_pnl_pct > 0.0, net_pnl_pct);
                
                out_pnl[trades] = net_pnl_pct;
                if net_pnl_pct > 0.0 { wins += 1; }
                trades += 1;
                
                if capital > peak_capital { peak_capital = capital; }
                let dd = (peak_capital - capital) / peak_capital;
                if dd > max_dd { max_dd = dd; }
                
                scalp_in_pos = false;
            }
        }

        if swing_in_pos {
            let pnl_pct = if swing_side == 1 { (current_close - swing_entry) / swing_entry } else { (swing_entry - current_close) / swing_entry };
            let mut exit = false;
            let mut exit_price = current_close;
            let swing_tp = cfg.tp_pct;
            let swing_sl = cfg.sl_pct;

            if pnl_pct >= swing_tp { exit = true; }
            else if pnl_pct <= -swing_sl { exit = true; }
            // Symmetrical Intra-Bar Evaluation for Stop Loss & Take Profit
            else if swing_side == 1 && highs[i] >= swing_entry * (1.0 + swing_tp) { exit = true; exit_price = swing_entry * (1.0 + swing_tp); }
            else if swing_side == -1 && lows[i] <= swing_entry * (1.0 - swing_tp) { exit = true; exit_price = swing_entry * (1.0 - swing_tp); }
            else if swing_side == 1 && lows[i] <= swing_entry * (1.0 - swing_sl) { exit = true; exit_price = swing_entry * (1.0 - swing_sl); }
            else if swing_side == -1 && highs[i] >= swing_entry * (1.0 + swing_sl) { exit = true; exit_price = swing_entry * (1.0 + swing_sl); }
            
            // Adaptive Regime Exit
            if swing_regime == MarketRegime::Scalping && pnl_pct > 0.0 { exit = true; }

            if exit {
                let pnl_amount = swing_qty * (exit_price - swing_entry) * (swing_side as f64);
                let fee_amount = swing_qty * (swing_entry + exit_price) * fee_rate; // Round-trip: entry + exit
                let net_pnl_amount = pnl_amount - fee_amount;
                
                capital += net_pnl_amount;
                let margin_used = (swing_qty * swing_entry) / 15.0; // Swing leverage 15.0
                let net_pnl_pct = net_pnl_amount / margin_used;
                
                risk_manager.report_trade_result_local("BTCUSDT_SWING", net_pnl_pct > 0.0, net_pnl_pct);
                
                out_pnl[trades] = net_pnl_pct;
                if net_pnl_pct > 0.0 { wins += 1; }
                trades += 1;
                
                if capital > peak_capital { peak_capital = capital; }
                let dd = (peak_capital - capital) / peak_capital;
                if dd > max_dd { max_dd = dd; }
                
                swing_in_pos = false;
            }
        }

        // Check Bankruptcy
        if capital <= 0.0 {
            break;
        }

        // --- 2. EVALUATE ENTRIES ---
        if !scalp_in_pos && (scalp_regime == MarketRegime::Neutral || scalp_regime == MarketRegime::Scalping) {
            if ml_prob > cfg.ml_threshold_l as f32 {
                if let Some(qty) = risk_manager.calculate_micro_position_size_local("BTCUSDT", current_close, 50.0, capital) {
                    scalp_in_pos = true;
                    scalp_side = 1;
                    scalp_entry = current_close;
                    scalp_qty = qty;
                }
            } else if ml_prob < (1.0 - cfg.ml_threshold_s as f32) {
                if let Some(qty) = risk_manager.calculate_micro_position_size_local("BTCUSDT", current_close, 50.0, capital) {
                    scalp_in_pos = true;
                    scalp_side = -1;
                    scalp_entry = current_close;
                    scalp_qty = qty;
                }
            }
        }

        // Production only enters swing on 1H kline close (is_closed gate).
        // On 1m data, this means only every 60th bar to match production behavior.
        let is_hourly_close = (i % 60 == 59) && (i > 10);
        if !swing_in_pos && swing_regime == MarketRegime::Swing && is_hourly_close {
            if fast > slow * (1.0 + cfg.tech_threshold_l) {
                if let Some(qty) = risk_manager.calculate_micro_position_size_local("BTCUSDT_SWING", current_close, 15.0, capital) {
                    swing_in_pos = true;
                    swing_side = 1;
                    swing_entry = current_close;
                    swing_qty = qty;
                }
            } else if fast < slow * (1.0 - cfg.tech_threshold_s) {
                if let Some(qty) = risk_manager.calculate_micro_position_size_local("BTCUSDT_SWING", current_close, 15.0, capital) {
                    swing_in_pos = true;
                    swing_side = -1;
                    swing_entry = current_close;
                    swing_qty = qty;
                }
            }
        }
    }

    out_stats[0] = if trades > 0 { (wins as f64) / (trades as f64) } else { 0.0 };
    out_stats[1] = trades as f64;
    out_stats[2] = capital;
    out_stats[3] = max_dd;

    trades
}

#[no_mangle]
pub extern "C" fn ffi_run_unified_backtest(
    closes_ptr: *const f64,
    highs_ptr: *const f64,
    lows_ptr: *const f64,
    volumes_ptr: *const f64,
    len: usize,
    config: *const UnifiedConfig,
    out_pnl_ptr: *mut f64,
    out_stats_ptr: *mut f64,
) -> usize {
    if closes_ptr.is_null() || highs_ptr.is_null() || lows_ptr.is_null() || config.is_null() || out_pnl_ptr.is_null() || out_stats_ptr.is_null() {
        return 0;
    }

    let closes = unsafe { std::slice::from_raw_parts(closes_ptr, len) };
    let highs = unsafe { std::slice::from_raw_parts(highs_ptr, len) };
    let lows = unsafe { std::slice::from_raw_parts(lows_ptr, len) };
    let volumes = unsafe { std::slice::from_raw_parts(volumes_ptr, len) };
    let cfg = unsafe { &*config };
    let out_pnl = unsafe { std::slice::from_raw_parts_mut(out_pnl_ptr, len) };
    let out_stats = unsafe { std::slice::from_raw_parts_mut(out_stats_ptr, 4) };

    run_backtest_native(closes, highs, lows, volumes, cfg, out_pnl, out_stats)
}

#[no_mangle]
pub extern "C" fn ffi_run_unified_backtest_mmap(
    filepath_ptr: *const std::os::raw::c_char,
    len: usize,
    config: *const UnifiedConfig,
    out_pnl_ptr: *mut f64,
    out_stats_ptr: *mut f64,
) -> usize {
    if filepath_ptr.is_null() || config.is_null() || out_pnl_ptr.is_null() || out_stats_ptr.is_null() {
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

    run_backtest_native(closes, highs, lows, volumes, cfg, out_pnl, out_stats)
}
