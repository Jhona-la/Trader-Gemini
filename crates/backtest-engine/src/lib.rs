pub mod vectorized;

use quantum_arena::genome::SuperGenotype;

/// F3.3: contrato de tamaño del buffer out_stats. El código anterior escribía
/// 8 floats en buffers de 4 creados por los wrappers FFI → corrupción de heap
/// y pánico garantizado en polars_evolver (vec![0.0] para pnl). Todo caller
/// DEBE usar STATS_LEN.
pub const STATS_LEN: usize = 8;

pub fn run_backtest_native(
    closes: &[f64],
    _highs: &[f64],
    _lows: &[f64],
    volumes: &[f64],
    cfg: &SuperGenotype,
    out_pnl: &mut [f64],
    out_stats: &mut [f64],
    _symbol: &str,
    initial_capital: f64,
) -> usize {
    let len = closes.len();

    // F3.4: capital 0/NaN ⇒ arena rota y métricas basura. Rechazo explícito:
    // el caller debe extraer el balance real (API en demo/prod).
    if !initial_capital.is_finite() || initial_capital <= 0.0 {
        return 0;
    }
    if out_stats.len() < STATS_LEN {
        return 0; // contrato violado: sin corrupción silenciosa
    }

    // 🚨 AXIOMA FASE 33: ALERTA DE DIVERGENCIA FORENSE 🚨
    // Este motor sintetiza micro-ticks a partir de Klines 1m: la microestructura
    // NO refleja la del WS real. Motor rápido para GA; la VERDAD de validación es
    // audit_forensic_backtest (aggTrades + fees reales de la cuenta).

    // Axioma VII: Paridad Absoluta de Modelos. Usamos la misma Arena y Core que producción.
    use god_engine_core::GodEngineCore;
    use std::sync::Arc;

    use quantum_arena::GlobalArena;
    use std::sync::atomic::Ordering;

    let arena = Arc::new(GlobalArena::new(initial_capital)); // Dynamic capital for absolute GA evaluation
    // Apply configs from Evolution to GlobalArena
    cfg.apply_to_arena(&arena);

    let mut core = GodEngineCore::new(arena.clone());

    let mut gross_wins = 0;
    let mut net_wins = 0;
    let mut trades = 0;
    let mut gross_pnl_sum = 0.0;
    let mut net_pnl_sum = 0.0;
    let mut fees_est_sum = 0.0;
    // Fee estimado por trade: qty × precio_barra × fee medio.
    // ESTIMACIÓN documentada (el PnL emitido ya es net): para métricas pre-fee
    // exactas usar audit_forensic (fees reales de la cuenta vía API).
    // Floor realista Binance futures (0.02%/0.04%) hasta que F3.4 cablee la
    // commission API — el genoma JAMÁS puede evolucionar los costos a cero.
    let avg_fee_est = (0.0002 + 0.0004) * 0.5;

    let mut peak_capital = initial_capital;
    let mut max_dd = 0.0;

    // F3.2 — FIX LOOK-AHEAD: el OFI sintético se deriva ahora del delta de la
    // vela ANTERIOR (información disponible al abrir la vela actual). Antes:
    // delta de la vela COMPLETA — el motor "veía" el resultado del bar antes
    // de simular sus micro-ticks (win rate inflado por construcción).
    let mut prev_delta = 0.0;

    for i in 0..len {
        let current_close = closes[i];
        let current_vol = volumes[i];

        let prev_close = if i > 0 { closes[i - 1] } else { current_close };
        let delta = current_close - prev_close;

        // OFI NO anticipado: señal de la vela PREVIA (disponible en t=0 de esta).
        let mut bid_ratio = 0.5;
        if prev_delta > 0.0 {
            bid_ratio = 0.8;
        } else if prev_delta < 0.0 {
            bid_ratio = 0.2;
        }

        let bid_qty = current_vol * bid_ratio;
        let ask_qty = current_vol * (1.0 - bid_ratio);

        // Interpolación (modelado): la trayectoria lineal al cierre conocido
        // permanece como suposición de suavizado — divergencia documentada arriba.
        let num_ticks = 10;
        let price_step = delta / num_ticks as f64;
        let vol_step = current_vol / num_ticks as f64;
        let bid_qty_step = bid_qty / num_ticks as f64;
        let ask_qty_step = ask_qty / num_ticks as f64;

        let mut sim_price = prev_close;
        let mut closed_sc = None;
        let mut closed_sw = None;

        let mut noise_seed = (i as u64).wrapping_mul(1103515245).wrapping_add(12345);

        for t in 0..num_ticks {
            noise_seed = noise_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise_fract = (noise_seed % 1000) as f64 / 1000.0 - 0.5; // -0.5 to +0.5
            let vol_noise = (noise_seed % 100) as f64 / 100.0;

            let volatility = prev_close * cfg.base_slippage_floor; // Genome-evolved noise amplitude
            sim_price += price_step + (noise_fract * volatility);

            let is_kline = t == num_ticks - 1; // Solo el último tick cierra la vela

            // Simulate spread using genome-derived maker_spread_pct
            let half_spread = cfg.maker_spread_pct.max(0.00005); // Min 0.5 bps safety
            let sim_bid = sim_price * (1.0 - half_spread);
            let sim_ask = sim_price * (1.0 + half_spread);

            let sim_bid_qty = bid_qty_step * (0.5 + vol_noise);
            let sim_ask_qty = ask_qty_step * (1.5 - vol_noise);

            let (_, _, sc, sw) = core.process_event(
                0,
                true,
                is_kline,
                true,
                sim_price,
                vol_step,
                sim_bid,
                sim_ask,
                sim_bid_qty,
                sim_ask_qty,
                0.5,
                0.0,
                (i as u64 * 1000) + (t as u64 * 100),
                false,
                &[0.0; 54],
            );

            if sc.is_some() {
                closed_sc = sc;
            }
            if sw.is_some() {
                closed_sw = sw;
            }
        }

        if let Some((_is_long, net_pnl, qty)) = closed_sc {
            if trades < out_pnl.len() {
                out_pnl[trades] = net_pnl;
            }
            let fees_est = qty * current_close * avg_fee_est;
            fees_est_sum += fees_est;
            gross_pnl_sum += net_pnl + fees_est;
            net_pnl_sum += net_pnl;
            if net_pnl + fees_est > 0.0 {
                gross_wins += 1;
            }
            if net_pnl > 0.0 {
                net_wins += 1;
            }

            trades += 1;
        }

        if let Some((_is_long, net_pnl, qty)) = closed_sw {
            if trades < out_pnl.len() {
                out_pnl[trades] = net_pnl;
            }
            let fees_est = qty * current_close * avg_fee_est;
            fees_est_sum += fees_est;
            gross_pnl_sum += net_pnl + fees_est;
            net_pnl_sum += net_pnl;
            if net_pnl + fees_est > 0.0 {
                gross_wins += 1;
            }
            if net_pnl > 0.0 {
                net_wins += 1;
            }

            trades += 1;
        }
        let _ = fees_est_sum; // expuesto vía stats[7]/stats[5] abajo

        // F3.2: el delta de ESTA vela queda disponible para la SIGUIENTE.
        prev_delta = delta;

        let current_cap = core.arena.unified_capital.load(Ordering::Relaxed);
        if current_cap > peak_capital {
            peak_capital = current_cap;
        }
        let dd = if peak_capital > 0.0 {
            (peak_capital - current_cap) / peak_capital
        } else {
            0.0
        };
        if dd > max_dd {
            max_dd = dd;
        }

        if current_cap <= 0.0 {
            break;
        }
    }

    let mut final_cap = core.arena.unified_capital.load(Ordering::Relaxed);

    // Add unrealized PnL from open positions to prevent missing large drawdowns at end
    let last_price = closes[len - 1];
    let scalp_pos = &core.arena.coins[0].positions.scalp_position;
    if scalp_pos.is_open() {
        let entry = scalp_pos.entry_price.load(Ordering::Relaxed);
        let qty = scalp_pos.quantity.load(Ordering::Relaxed);
        let is_long = scalp_pos.is_long.load(Ordering::Relaxed);
        let unrealized = (last_price - entry) * qty * if is_long { 1.0 } else { -1.0 };
        final_cap += unrealized;
    }
    let swing_pos = &core.arena.coins[0].positions.swing_position;
    if swing_pos.is_open() {
        let entry = swing_pos.entry_price.load(Ordering::Relaxed);
        let qty = swing_pos.quantity.load(Ordering::Relaxed);
        let is_long = swing_pos.is_long.load(Ordering::Relaxed);
        let unrealized = (last_price - entry) * qty * if is_long { 1.0 } else { -1.0 };
        final_cap += unrealized;
    }

    let gross_win_rate = if trades > 0 {
        gross_wins as f64 / trades as f64
    } else {
        0.0
    };
    let net_win_rate = if trades > 0 {
        net_wins as f64 / trades as f64
    } else {
        0.0
    };

    let mean_pnl = if trades > 0 {
        net_pnl_sum / trades as f64
    } else {
        0.0
    };
    let mut variance = 0.0;
    if trades > 1 {
        // Solo los PnL realmente escritos en el buffer participan.
        let stored = trades.min(out_pnl.len());
        for pnl in out_pnl[..stored].iter() {
            variance += (pnl - mean_pnl).powi(2);
        }
        variance /= (stored - 1) as f64;
    }
    let std_dev = variance.sqrt();
    let sharpe = if std_dev > 0.0 {
        mean_pnl / std_dev
    } else {
        0.0
    };

    out_stats[0] = net_win_rate;
    out_stats[1] = trades as f64;
    out_stats[2] = final_cap;
    out_stats[3] = max_dd;
    out_stats[4] = sharpe;
    out_stats[5] = gross_pnl_sum;
    out_stats[6] = net_pnl_sum;
    out_stats[7] = gross_win_rate;

    trades
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_run_unified_backtest(
    closes_ptr: *const f64,
    highs_ptr: *const f64,
    lows_ptr: *const f64,
    volumes_ptr: *const f64,
    len: usize,
    config: *const quantum_arena::genome::SuperGenotype,
    out_pnl_ptr: *mut f64,
    out_stats_ptr: *mut f64,
    symbol_ptr: *const std::os::raw::c_char,
) -> usize {
    if closes_ptr.is_null()
        || highs_ptr.is_null()
        || lows_ptr.is_null()
        || config.is_null()
        || out_pnl_ptr.is_null()
        || out_stats_ptr.is_null()
        || symbol_ptr.is_null()
    {
        return 0;
    }

    let closes = unsafe { std::slice::from_raw_parts(closes_ptr, len) };
    let highs = unsafe { std::slice::from_raw_parts(highs_ptr, len) };
    let lows = unsafe { std::slice::from_raw_parts(lows_ptr, len) };
    let volumes = unsafe { std::slice::from_raw_parts(volumes_ptr, len) };
    let cfg = unsafe { &*config };
    let out_pnl = unsafe { std::slice::from_raw_parts_mut(out_pnl_ptr, len) };
    let out_stats = unsafe { std::slice::from_raw_parts_mut(out_stats_ptr, STATS_LEN) };

    let sym_c = unsafe { std::ffi::CStr::from_ptr(symbol_ptr) };
    let sym_str = sym_c.to_str().unwrap_or("BTCUSDT");

    let initial_capital = std::env::var("INITIAL_CAPITAL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    run_backtest_native(
        closes,
        highs,
        lows,
        volumes,
        cfg,
        out_pnl,
        out_stats,
        sym_str,
        initial_capital,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_run_unified_backtest_mmap(
    filepath_ptr: *const std::os::raw::c_char,
    len: usize,
    config: *const quantum_arena::genome::SuperGenotype,
    out_pnl_ptr: *mut f64,
    out_stats_ptr: *mut f64,
    symbol_ptr: *const std::os::raw::c_char,
) -> usize {
    if filepath_ptr.is_null()
        || config.is_null()
        || out_pnl_ptr.is_null()
        || out_stats_ptr.is_null()
        || symbol_ptr.is_null()
    {
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
    let out_stats = unsafe { std::slice::from_raw_parts_mut(out_stats_ptr, STATS_LEN) };

    let sym_c = unsafe { std::ffi::CStr::from_ptr(symbol_ptr) };
    let sym_str = sym_c.to_str().unwrap_or("BTCUSDT");

    let initial_capital = std::env::var("INITIAL_CAPITAL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    run_backtest_native(
        closes,
        highs,
        lows,
        volumes,
        cfg,
        out_pnl,
        out_stats,
        sym_str,
        initial_capital,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ffi_run_polars_backtest_mmap(
    filepath_ptr: *const std::os::raw::c_char,
    len: usize,
    config: *const quantum_arena::genome::SuperGenotype,
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
    let out_stats = unsafe { std::slice::from_raw_parts_mut(out_stats_ptr, STATS_LEN) };

    if let Ok((final_cap, max_dd, trades, wins)) =
        vectorized::run_vectorized_hybrid(closes, highs, lows, volumes, cfg)
    {
        let win_rate = if trades > 0 {
            wins as f64 / trades as f64
        } else {
            0.0
        };
        out_stats[0] = win_rate;
        out_stats[1] = trades as f64;
        out_stats[2] = final_cap;
        out_stats[3] = max_dd;
        return trades as usize;
    }
    0
}
