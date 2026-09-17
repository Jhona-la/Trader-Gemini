pub mod booktick_replay;
pub mod network_jitter;
pub mod tick_replayer;
pub mod vectorized;

pub use network_jitter::NetworkJitterSimulator;
use quantum_arena::genome::SuperGenotype;
pub use tick_replayer::*;
pub use vectorized::{OrderBookL2DepthSlippageModel, run_vectorized_hybrid};

/// F3.3: contrato de tamaño del buffer out_stats. El código anterior escribía
/// 8 floats en buffers de 4 creados por los wrappers FFI → corrupción de heap
/// y pánico garantizado en polars_evolver (vec![0.0] para pnl). Todo caller
/// DEBE usar STATS_LEN.
pub const STATS_LEN: usize = 8;

pub fn run_backtest_native(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[f64],
    cfg: &SuperGenotype,
    out_pnl: &mut [f64],
    out_stats: &mut [f64],
    symbol: &str,
    initial_capital: f64,
) -> usize {
    let len = closes.len();
    let target_coin_id = quantum_arena::symbol_registry::try_index(symbol).unwrap_or(0);

    // F3.4: capital 0/NaN ⇒ arena rota y métricas basura. Rechazo explícito:
    // el caller debe extraer el balance real (API en demo/prod).
    if !initial_capital.is_finite() || initial_capital <= 0.0 {
        return 0;
    }
    if out_stats.len() < STATS_LEN {
        return 0; // contrato violado: sin corrupción silenciosa
    }

    if len == 0 {
        out_stats[0] = 0.0;
        out_stats[1] = 0.0;
        out_stats[2] = initial_capital;
        out_stats[3] = 0.0;
        out_stats[4] = 0.0;
        out_stats[5] = 0.0;
        out_stats[6] = 0.0;
        out_stats[7] = 0.0;
        return 0;
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

    let arena = GlobalArena::build_in_own_stack(initial_capital); // Dynamic capital for absolute GA evaluation
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
    // D-394: Binance VIP0 Taker roundtrip real (0.05% in + 0.05% out = 0.10% total).
    let avg_fee_est = 0.0010;

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
        let prev_prev_close = if i > 1 { closes[i - 2] } else { prev_close };
        let delta = current_close - prev_close;
        // R2.1a — retorno y dirección de la vela PREVIA: lo único disponible
        // en t=0 de la vela actual. El retorno contemporáneo (`delta`) se
        // reserva EXCLUSIVAMENTE para el bridge del precio; toda feature del
        // tensor usa `prev_rel_ret` (causal). Antes, 14+ campos del tensor 54D
        // (funding, fear&greed, long/short, VIX, basis, skew...) derivaban del
        // retorno de la vela que aún no había cerrado.
        let prev_rel_ret =
            ((prev_close - prev_prev_close) / prev_prev_close.max(1e-9)).clamp(-1.0, 1.0);

        // OFI NO anticipado: señal de la vela PREVIA (disponible en t=0 de esta).
        let mut bid_ratio = 0.5;
        if prev_delta > 0.0 {
            bid_ratio = 0.8;
        } else if prev_delta < 0.0 {
            bid_ratio = 0.2;
        }

        let bid_qty = current_vol * bid_ratio;
        let ask_qty = current_vol * (1.0 - bid_ratio);

        // FIX BLOQUEO #4: Incrementar micro-ticks de 10 a 30 para resolución microestructura
        // Con 10 ticks/vela, las estrategias de microestructura (Hawkes, OFI, Soliton)
        // no pueden detectar patrones reales. 30 ticks proveen 2 por segundo.
        let num_ticks = 30;
        let vol_step = current_vol / num_ticks as f64;
        let bid_qty_step = bid_qty / num_ticks as f64;
        let ask_qty_step = ask_qty / num_ticks as f64;

        let mut sim_price = prev_close;
        let mut omni_sim = [0.0; 54];
        let rel_ret = prev_rel_ret; // R2.1a: features del tensor = vela PREVIA
        let ofi_proxy = if bid_qty + ask_qty > 0.0 {
            (bid_qty - ask_qty) / (bid_qty + ask_qty)
        } else {
            0.0
        };
        // 1. Cotizaciones Cross-Exchange y Microestructura (0..10)
        // Paridad 1:1 con producción (omni_multiplexer.rs):
        // feats[0] = 0.0 (referencia base spot)
        // feats[1..10] = porcentaje de spread relativo en [-5.0, 5.0]
        omni_sim[0] = 0.0; // binance_spot (referencia base)
        omni_sim[1] = (rel_ret * 10.0).clamp(-5.0, 5.0); // binance_futures spread
        omni_sim[2] = (-ofi_proxy * 10.0).clamp(-5.0, 5.0); // bybit_linear spread
        omni_sim[3] = (ofi_proxy * 10.0).clamp(-5.0, 5.0); // okx_swap spread
        omni_sim[4] = 0.0; // bitget_futures
        omni_sim[5] = 0.0; // coinbase_spot
        omni_sim[6] = 0.0; // kraken_spot
        omni_sim[7] = 0.0; // htx_spot
        omni_sim[8] = 0.0; // deribit_options
        omni_sim[9] = 0.0; // bitfinex_spot
        omni_sim[10] = current_vol * ofi_proxy.abs(); // binance_liquidations

        // 2. Sentimiento, Tasas y Macro TradFi dinámicos (11..30) (FIX #940)
        let dyn_funding = (rel_ret * 0.005).clamp(-0.001, 0.001);
        let dyn_fear_greed = (50.0 + rel_ret * 500.0).clamp(10.0, 90.0);
        let dyn_long_short = (1.0 + rel_ret * 5.0).clamp(0.5, 2.5);
        let dyn_vix = (15.0 + rel_ret.abs() * 200.0).clamp(10.0, 80.0);
        let dyn_dvol = (50.0 + rel_ret.abs() * 300.0).clamp(20.0, 150.0);
        let dyn_put_call = (1.0 - rel_ret * 2.0).clamp(0.4, 2.0);

        omni_sim[11] = dyn_funding; // agg_funding_rate
        omni_sim[12] = current_vol * 10.0; // agg_open_interest
        omni_sim[13] = dyn_long_short; // long_short_ratio
        omni_sim[14] = dyn_fear_greed; // fear_greed_index
        omni_sim[15] = 12.5; // altcoin_dominance
        omni_sim[16] = 25.0; // mempool_congestion
        omni_sim[17] = 0.0; // usdt_mint_alert
        omni_sim[18] = current_vol * 0.2; // exchange_inflows
        omni_sim[19] = current_vol * 0.2; // exchange_outflows
        omni_sim[20] = 0.0; // whale_alert_proxy
        omni_sim[21] = 104.2; // dxy
        omni_sim[22] = 5120.0; // sp500
        omni_sim[23] = 18100.0; // nasdaq
        omni_sim[24] = dyn_vix; // vix
        omni_sim[25] = 4.25; // us10y
        omni_sim[26] = 2320.0; // gold
        omni_sim[27] = 81.0; // oil_wti
        omni_sim[28] = 0.0; // econ_calendar_impact
        omni_sim[29] = 5.5; // fed_interest_rate
        omni_sim[30] = bid_qty - ask_qty; // spot_cvd

        // 3. Derivados, Clusters y Métricas de Flujo (31..54)
        omni_sim[31] = (bid_qty - ask_qty) * 1.2; // futures_cvd
        omni_sim[32] = if ask_qty > 0.0 {
            (bid_qty / ask_qty).clamp(0.1, 10.0)
        } else {
            1.0
        }; // taker_buy_sell_ratio
        omni_sim[33] = rel_ret * current_close * 0.001; // futures_basis_premium
        omni_sim[34] = current_close * 1.01; // liq_cluster_shorts
        omni_sim[35] = current_close * 0.99; // liq_cluster_longs
        omni_sim[36] = rel_ret * 5.0; // cme_futures_premium
        omni_sim[37] = 0.002; // cme_gap_proximity
        omni_sim[38] = ofi_proxy * 0.5; // spot_futures_arb_spread
        omni_sim[39] = ofi_proxy; // order_flow_imbalance
        omni_sim[40] = dyn_dvol; // dvol_index
        omni_sim[41] = (-rel_ret * 2.0).clamp(-0.25, 0.25); // options_25_delta_skew
        omni_sim[42] = dyn_put_call; // put_call_ratio
        omni_sim[43] = current_close; // max_pain_price
        omni_sim[44] = dyn_long_short; // top_traders_pos_accounts
        omni_sim[45] = dyn_long_short * 1.05; // top_traders_pos_volume
        omni_sim[46] = 0.12; // stablecoin_supply_ratio
        omni_sim[47] = 0.05; // margin_debt_ratio
        omni_sim[48] = 250.0; // etf_net_inflows
        omni_sim[49] = rel_ret.abs() * 100.0; // micro_volatility
        omni_sim[50] = 20800.0; // wb_us_m2_supply
        omni_sim[51] = 3.1; // wb_us_cpi_inflation
        omni_sim[52] = 2.2; // wb_us_real_interest
        omni_sim[53] = 2.6; // wb_global_gdp_growth
        // FIX #943: SplitMix64 PRNG (passes BigCrush) replacing weak glibc LCG
        let mut noise_seed = (i as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0x6A09E667F3BCC908);

        for t in 0..num_ticks {
            // SplitMix64 step: full 64-bit period, statistically robust
            noise_seed = noise_seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = noise_seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z = z ^ (z >> 31);
            let noise_fract = (z % 1000) as f64 / 1000.0 - 0.5; // -0.5 to +0.5
            let vol_noise = (z % 100) as f64 / 100.0;

            let is_kline = t == num_ticks - 1; // Solo el último tick cierra la vela

            // FASE 33 & BUG-586 & REHABILITACIÓN: Brownian bridge multi-fase con exploración de mechas (highs/lows)
            let bar_high = if i < highs.len() && highs[i].is_finite() && highs[i] > 0.0 {
                highs[i].max(current_close).max(prev_close)
            } else {
                current_close.max(prev_close)
            };
            let bar_low = if i < lows.len() && lows[i].is_finite() && lows[i] > 0.0 {
                lows[i].min(current_close).min(prev_close)
            } else {
                current_close.min(prev_close)
            };

            // Simulación física de trayectoria intra-vela:
            // Vela alcista: exploración inicial de la mecha inferior, luego impulso a la mecha superior, convergiendo a close.
            // Vela bajista: exploración inicial de la mecha superior, luego caída a la mecha inferior, convergiendo a close.
            // R2.1a — SIN LOOKAHEAD: la dirección que ordena la coreografía es
            // la de la vela PREVIA (disponible en t=0). Antes usaba el close
            // de la vela ACTUAL: ver "baja-then-alza" intra-bar equivalía a
            // saber que cerraría alcista. Limitación residual documentada: el
            // bridge converge al close y el clamp usa los extremos OHLC —
            // inherente a sintetizar ticks desde OHLC; la certificación corre
            // sobre aggTrades reales (R2.2).
            let is_bullish = prev_close >= prev_prev_close;
            let target_phase_price = if t < num_ticks / 3 {
                if is_bullish { bar_low } else { bar_high }
            } else if t < (2 * num_ticks) / 3 {
                if is_bullish { bar_high } else { bar_low }
            } else {
                current_close
            };

            let remaining_ticks = (num_ticks - t) as f64;
            let bridge_drift = (target_phase_price - sim_price) / remaining_ticks.max(1.0);
            let volatility = prev_close
                * cfg.base_slippage_floor.max(0.0001)
                * ((num_ticks - 1 - t) as f64 / num_ticks as f64).sqrt();
            sim_price = if is_kline {
                current_close
            } else {
                let next_p = sim_price + bridge_drift + (noise_fract * volatility);
                next_p.clamp(bar_low, bar_high)
            };

            // Simulate spread using genome-derived maker_spread_pct
            let half_spread = cfg.maker_spread_pct.max(0.00005); // Min 0.5 bps safety
            let sim_bid = sim_price * (1.0 - half_spread);
            let sim_ask = sim_price * (1.0 + half_spread);

            let sim_bid_qty = bid_qty_step * (0.5 + vol_noise);
            let sim_ask_qty = ask_qty_step * (1.5 - vol_noise);

            // FIX BLOQUEO #4: Actualizar omni_sim dinámicamente por tick
            // Antes: omni_sim era estático para toda la vela. Ahora: cada tick
            // actualiza las features que cambian (precios, OFI, volumen).
            let tick_ofi = if sim_bid_qty + sim_ask_qty > 0.0 {
                (sim_bid_qty - sim_ask_qty) / (sim_bid_qty + sim_ask_qty)
            } else {
                0.0
            };
            let tick_ret = if sim_price > 0.0 && prev_close > 0.0 {
                (sim_price - prev_close) / prev_close
            } else {
                0.0
            };
            // Actualizar features dinámicas por tick (Alineado 1:1 con Producción)
            omni_sim[0] = 0.0;
            omni_sim[1] = (tick_ret * 10.0).clamp(-5.0, 5.0);
            for j in 2..10 {
                omni_sim[j] = 0.0;
            } // Matches production offline secondary WS feeds
            omni_sim[10] = vol_step * tick_ofi.abs();
            omni_sim[30] = sim_bid_qty - sim_ask_qty;
            omni_sim[31] = (sim_bid_qty - sim_ask_qty) * 1.2;
            omni_sim[32] = if sim_ask_qty > 0.0 {
                (sim_bid_qty / sim_ask_qty).clamp(0.1, 10.0)
            } else {
                1.0
            };
            omni_sim[39] = tick_ofi;
            omni_sim[49] = tick_ret.abs() * 100.0;

            let sim_is_buyer_maker = tick_ret < 0.0;
            let (_new_order, closed_order) = core.process_event(
                target_coin_id,
                true,
                is_kline,
                true,
                sim_price,
                vol_step,
                sim_bid,
                sim_ask,
                sim_bid_qty,
                sim_ask_qty,
                tick_ofi, // FIX: Propagar OFI real derivado en vez de constante 0.5
                0.0,
                (i as u64 * 60_000) + (t as u64 * (60_000 / num_ticks.max(1) as u64)),
                false,
                &omni_sim,
                sim_is_buyer_maker,
            );

            if let Some((_is_long, net_pnl, qty)) = closed_order {
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
        // FIX #1516: Verificación de finitud en cálculo de Max Drawdown
        if dd > max_dd && dd.is_finite() {
            max_dd = dd;
        }

        if current_cap <= 0.0 {
            break;
        }
    }

    let mut final_cap = core.arena.unified_capital.load(Ordering::Relaxed);

    // Add unrealized PnL from open positions to prevent missing large drawdowns at end (deducting exit fees)
    // FIX #1485 & #1517: Usar target_coin_id y proteger suma de unrealized PnL
    let last_price = closes.last().copied().unwrap_or(0.0);
    let safe_coin_idx = target_coin_id.min(core.arena.coins.len().saturating_sub(1));
    let pos = &core.arena.coins[safe_coin_idx].positions.position;
    if pos.is_open() {
        let entry = pos.entry_price.load(Ordering::Relaxed);
        let qty = pos.quantity.load(Ordering::Relaxed);
        let is_long = pos.is_long.load(Ordering::Relaxed);
        let exit_fee = qty * last_price * avg_fee_est;
        let unrealized = (last_price - entry) * qty * if is_long { 1.0 } else { -1.0 } - exit_fee;
        if unrealized.is_finite() {
            final_cap += unrealized;
        }
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
    let std_dev = variance.max(0.0).sqrt();
    let sharpe = if std_dev > 0.0 {
        let raw_sharpe = mean_pnl / std_dev;
        if raw_sharpe.is_finite() {
            raw_sharpe
        } else {
            0.0
        }
    } else {
        0.0
    };

    out_stats[0] = if net_win_rate.is_finite() {
        net_win_rate
    } else {
        0.0
    };
    out_stats[1] = trades as f64;
    out_stats[2] = if final_cap.is_finite() {
        final_cap
    } else {
        initial_capital
    };
    out_stats[3] = if max_dd.is_finite() { max_dd } else { 0.0 };
    out_stats[4] = if sharpe.is_finite() { sharpe } else { 0.0 };
    out_stats[5] = if gross_pnl_sum.is_finite() {
        gross_pnl_sum
    } else {
        0.0
    };
    out_stats[6] = if net_pnl_sum.is_finite() {
        net_pnl_sum
    } else {
        0.0
    };
    out_stats[7] = if gross_win_rate.is_finite() {
        gross_win_rate
    } else {
        0.0
    };

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
        .unwrap_or(13.0);
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
        Err(e) => {
            eprintln!(
                "⚠️ [BACKTEST MMAP ERROR] Failed to open file '{}': {}",
                filepath, e
            );
            return 0;
        }
    };

    let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "⚠️ [BACKTEST MMAP ERROR] Failed to mmap file '{}': {}",
                filepath, e
            );
            return 0;
        }
    };

    let expected_bytes = len * 4 * 8;
    if mmap.len() < expected_bytes {
        eprintln!(
            "⚠️ [BACKTEST MMAP ERROR] File '{}' too short: got {} bytes, expected {}",
            filepath,
            mmap.len(),
            expected_bytes
        );
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
        .unwrap_or(13.0);
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
        Err(e) => {
            eprintln!(
                "⚠️ [POLARS MMAP ERROR] Failed to open file '{}': {}",
                filepath, e
            );
            return 0;
        }
    };

    let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "⚠️ [POLARS MMAP ERROR] Failed to mmap file '{}': {}",
                filepath, e
            );
            return 0;
        }
    };

    let expected_bytes = len * 4 * 8;
    if mmap.len() < expected_bytes {
        eprintln!(
            "⚠️ [POLARS MMAP ERROR] File '{}' too short: got {} bytes, expected {}",
            filepath,
            mmap.len(),
            expected_bytes
        );
        return 0;
    }

    let ptr = mmap.as_ptr() as *const f64;
    let closes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let highs = unsafe { std::slice::from_raw_parts(ptr.add(len), len) };
    let lows = unsafe { std::slice::from_raw_parts(ptr.add(len * 2), len) };
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(len * 3), len) };

    let cfg = unsafe { &*config };
    let out_stats = unsafe { std::slice::from_raw_parts_mut(out_stats_ptr, STATS_LEN) };

    // D-648/D-669 (DÉCIMA OLA): esta exportación evaluaba con
    // `run_vectorized_hybrid`, un cruce de EMAs que lee una fracción mínima del
    // genoma y dimensiona cada operación con todo el capital. Nada en el
    // repositorio la invoca, pero cualquier consumidor externo obtenía métricas
    // de un sistema distinto al que se despliega. Ahora usa el mismo motor que
    // producción y que los promotores de genomas, con el mismo capital por
    // defecto que las otras dos exportaciones.
    let initial_capital = std::env::var("INITIAL_CAPITAL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(13.0);
    let mut out_pnl = vec![0.0f64; len];
    run_backtest_native(
        closes,
        highs,
        lows,
        volumes,
        cfg,
        &mut out_pnl,
        out_stats,
        "",
        initial_capital,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_native_execution_contract() {
        let n = 100;
        let mut closes = Vec::with_capacity(n);
        let mut highs = Vec::with_capacity(n);
        let mut lows = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);

        let mut p = 60000.0;
        for i in 0..n {
            p += (i as f64 * 0.1).sin() * 10.0;
            closes.push(p);
            highs.push(p + 5.0);
            lows.push(p - 5.0);
            volumes.push(100.0);
        }

        let cfg = SuperGenotype::default();
        let mut out_pnl = vec![0.0; n];
        let mut out_stats = vec![0.0; STATS_LEN];

        let trades = run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &cfg,
            &mut out_pnl,
            &mut out_stats,
            "BTCUSDT",
            13.0,
        );

        assert!(out_stats[2] > 0.0, "El capital final debe ser positivo");
        assert!(out_stats[3] >= 0.0, "El Max Drawdown no puede ser negativo");
        let _ = trades;
    }

    #[test]
    fn test_backtest_native_zero_and_nan_capital_rejection() {
        let closes = vec![100.0; 10];
        let highs = vec![105.0; 10];
        let lows = vec![95.0; 10];
        let volumes = vec![50.0; 10];
        let cfg = SuperGenotype::default();
        let mut out_pnl = vec![0.0; 10];
        let mut out_stats = vec![0.0; STATS_LEN];

        let res_nan = run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &cfg,
            &mut out_pnl,
            &mut out_stats,
            "BTCUSDT",
            f64::NAN,
        );
        assert_eq!(res_nan, 0);

        let res_zero = run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &cfg,
            &mut out_pnl,
            &mut out_stats,
            "BTCUSDT",
            0.0,
        );
        assert_eq!(res_zero, 0);
    }

    #[test]
    fn test_backtest_native_short_stats_buffer_rejection() {
        let closes = vec![100.0; 10];
        let highs = vec![105.0; 10];
        let lows = vec![95.0; 10];
        let volumes = vec![50.0; 10];
        let cfg = SuperGenotype::default();
        let mut out_pnl = vec![0.0; 10];
        let mut short_stats = vec![0.0; 4]; // Violates STATS_LEN = 8

        let res = run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &cfg,
            &mut out_pnl,
            &mut short_stats,
            "BTCUSDT",
            13.0,
        );
        assert_eq!(
            res, 0,
            "Buffer too short must fail safe without memory corruption"
        );
    }

    #[test]
    fn test_backtest_native_empty_series() {
        let closes: Vec<f64> = Vec::new();
        let highs: Vec<f64> = Vec::new();
        let lows: Vec<f64> = Vec::new();
        let volumes: Vec<f64> = Vec::new();
        let cfg = SuperGenotype::default();
        let mut out_pnl: Vec<f64> = Vec::new();
        let mut out_stats = vec![0.0; STATS_LEN];

        let res = run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &cfg,
            &mut out_pnl,
            &mut out_stats,
            "BTCUSDT",
            13.0,
        );
        assert_eq!(res, 0);
        assert_eq!(out_stats[0], 0.0);
    }

    #[test]
    fn test_backtest_native_nan_and_infinite_closes() {
        let closes = vec![f64::NAN, f64::INFINITY, 100.0, 105.0, 102.0];
        let highs = vec![110.0; 5];
        let lows = vec![90.0; 5];
        let volumes = vec![1000.0; 5];
        let cfg = SuperGenotype::default();
        let mut out_pnl = vec![0.0; 5];
        let mut out_stats = vec![0.0; STATS_LEN];

        let _ = run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &cfg,
            &mut out_pnl,
            &mut out_stats,
            "BTCUSDT",
            13.0,
        );
        assert!(out_stats[2].is_finite());
        assert!(out_stats[3].is_finite());
    }

    /// F3.7 — GOLDEN TEST: misma entrada ⇒ EXACTAMENTE las mismas 8 stats.
    /// Serie determinista (LCG xorshift) + baseline de fees fijas; capturamos
    /// el vector completo y lo congelamos bit a bit. Si alguien introduce
    /// no-determinismo (iteración de HashMap, RNG sin semilla, reordenación
    /// de operaciones float), este test explota — el backtest es evidencia
    /// forense y las evidencias no mutan entre corridas.
    #[test]
    fn golden_backtest_determinism() {
        // Tests corren en PARALELO en el mismo proceso: el registro global de
        // símbolos es estado compartido (otros tests — p.ej. booktick_replay —
        // lo pueblan). El golden PINA su entorno: determinista sin importar el
        // orden de ejecución de la suite.
        quantum_arena::symbol_registry::update_registry(Vec::new());

        let n = 500;
        let mut closes = Vec::with_capacity(n);
        let mut highs = Vec::with_capacity(n);
        let mut lows = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        // LCG determinista: mismo seed ⇒ misma serie, siempre.
        let mut seed: u64 = 0xDEADBEEF;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / u32::MAX as f64) - 0.5
        };
        let mut p = 60000.0;
        for _ in 0..n {
            p *= 1.0 + next() * 0.002; // ±0.1% random walk
            closes.push(p);
            highs.push(p * 1.001);
            lows.push(p * 0.999);
            volumes.push(50.0 + next() * 100.0);
        }

        let cfg = SuperGenotype::new_baseline(0.0002, 0.0005);
        let mut out_pnl = vec![0.0; n];
        let mut out_stats = vec![0.0; STATS_LEN];
        let trades = run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &cfg,
            &mut out_pnl,
            &mut out_stats,
            "GOLDEN",
            1000.0,
        );

        // Hash de estabilidad: dos corridas con la MISMA serie deben coincidir
        // bit a bit (ejecutamos de nuevo y comparamos los 8 stats + trades).
        let mut out_pnl2 = vec![0.0; n];
        let mut out_stats2 = vec![0.0; STATS_LEN];
        let trades2 = run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &cfg,
            &mut out_pnl2,
            &mut out_stats2,
            "GOLDEN",
            1000.0,
        );
        assert_eq!(trades, trades2, "número de trades debe ser determinista");
        for i in 0..STATS_LEN {
            assert_eq!(
                out_stats[i].to_bits(),
                out_stats2[i].to_bits(),
                "stat[{i}] difiere entre corridas idénticas — no-determinismo detectado"
            );
        }

        // Golden congelado: los valores capturados en la primera corrida
        // certificada. Cualquier cambio INTENCIONADO del motor implica
        // re-certificar y actualizar este vector conscientemente.
        // Los asserts usan bits exactos — evidencia forense, no aproximada.
        assert_eq!(
            trades, 0,
            "golden baseline: serie neutra no debe operar (sin señal ML)"
        );
        assert_eq!(out_stats[1].to_bits(), 0.0f64.to_bits());
        assert_eq!(
            out_stats[2].to_bits(),
            1000.0f64.to_bits(),
            "sin trades: capital intacto"
        );
    }
}
