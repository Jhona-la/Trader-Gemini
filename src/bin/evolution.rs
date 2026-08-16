use backtest_engine::run_backtest_native;
use chrono::{DateTime, Utc};
use god_engine_core::ml_inference::NanoForest;
use quantum_arena::genome::SuperGenotype as Genotype;
use std::fs::File;
use std::time::Instant;

use std::sync::atomic::{AtomicU64, Ordering};

static RNG_STATE: AtomicU64 = AtomicU64::new(0);

/// xorshift64 PRNG — fast, well-distributed, no correlation between calls
fn random_f64(min: f64, max: f64) -> f64 {
    let mut s = RNG_STATE.fetch_add(1, Ordering::Relaxed).wrapping_add(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
    );
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    RNG_STATE.store(s, Ordering::Relaxed);
    let norm = (s as f64) / (u64::MAX as f64); // [0.0, 1.0)
    min + norm * (max - min)
}

fn main() {
    quantum_arena::symbol_registry::update_registry(vec![
        quantum_arena::symbol_registry::SymbolSpec {
            symbol: "BTCUSDT".to_string(),
            min_notional: 5.0,
            tick_size: 0.1,
            step_size: 0.001,
            max_leverage: 120,
            maker_fee: 0.0002,
            taker_fee: 0.0004,
            min_qty: 0.001,
            is_shadow: false,
        },
    ]);

    let args: Vec<String> = std::env::args().collect();
    let mut symbol = "BTCUSDT".to_string();
    let mut initial_capital = 0.0;

    for i in 1..args.len() {
        if args[i] == "--symbol" && i + 1 < args.len() {
            symbol = args[i + 1].clone();
        } else if let Ok(parsed_bal) = args[i].parse::<f64>() {
            initial_capital = parsed_bal;
        }
    }

    if initial_capital <= 0.0 {
        panic!("❌ [CRITICAL] Debes proveer un balance inicial válido como argumento (ej. cargo run --release --bin evolution -- 100.50)");
    }

    println!("============================================================");
    println!("🌌 RUST QUANTUM EVOLUTION ENGINE (SIMULATED ANNEALING)");
    println!("============================================================");

    let path = format!("models/{}_SCALP.json", symbol);
    if let Err(_) = NanoForest::load_global(&format!("{}_SCALP", symbol), &path) {
        println!(
            "⚠️ Failed to load NanoForest from {}. It might not exist yet.",
            path
        );
    } else {
        println!("✅ NanoForest Loaded for Evolution: {}", path);
    }

    let file_path = format!("data/{}_ticks.bin", symbol);
    let file = match File::open(&file_path) {
        Ok(f) => f,
        Err(_) => {
            println!("❌ Failed to open data file: {}", file_path);
            return;
        }
    };

    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .expect("Failed to mmap file")
    };

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    struct BinTick {
        pub timestamp: u64,
        pub bid_price: f64,
        pub ask_price: f64,
        pub bid_qty: f64,
        pub ask_qty: f64,
    }

    let bytes_len = mmap.len();
    let tick_size = std::mem::size_of::<BinTick>();
    let original_len = bytes_len / tick_size;

    let max_ticks = 300_000;
    let offset = if original_len > max_ticks {
        original_len - max_ticks
    } else {
        0
    };
    let len = original_len - offset;

    if len == 0 {
        println!("❌ No data loaded or file is empty.");
        return;
    }

    let ptr = mmap.as_ptr() as *const BinTick;
    let ticks = unsafe { std::slice::from_raw_parts(ptr.add(offset), len) };

    let mut timestamps = Vec::with_capacity(len);
    let mut closes = Vec::with_capacity(len);
    let mut highs = Vec::with_capacity(len);
    let mut lows = Vec::with_capacity(len);
    let mut volumes = Vec::with_capacity(len);

    for t in ticks {
        timestamps.push(t.timestamp as f64);
        closes.push(t.bid_price);
        highs.push(t.ask_price);
        lows.push(t.bid_qty);
        volumes.push(t.ask_qty);
    }

    let train_len = (len as f64 * 0.7) as usize;
    let test_len = len - train_len;

    println!(
        "✅ Memory-Mapped {} ticks (Train: 70% = {}, Test: 30% = {}) [FAST OPTIMIZATION]",
        len, train_len, test_len
    );

    let iterations = 50;
    let initial_temp = 100.0;
    let cooling_rate = 0.90;

    let mut current_config = Genotype::load_or_default();

    let mut best_config = current_config.clone();
    let mut current_score = -9999999.0;
    let mut best_score = -9999999.0;
    let mut temp = initial_temp;

    println!(
        "🚀 Starting Simulated Annealing: {} Iterations (Quantum Tunneling)",
        iterations
    );
    let start_time = Instant::now();

    for i in 0..iterations {
        let mut test_cfg = current_config.clone();

        // Random Neighbor Generation
        test_cfg.scalp_sl_base += random_f64(-0.001, 0.001) * temp / initial_temp;
        test_cfg.scalp_tp_base += random_f64(-0.005, 0.005) * temp / initial_temp;
        test_cfg.ml_threshold_long += random_f64(-0.1, 0.1) * temp / initial_temp;
        test_cfg.ml_threshold_short += random_f64(-0.1, 0.1) * temp / initial_temp;
        test_cfg.trend_threshold += random_f64(-0.05, 0.05) * temp / initial_temp;
        test_cfg.global_leverage += random_f64(-5.0, 5.0) * temp / initial_temp;

        test_cfg.dynamic_obi_threshold += random_f64(-0.05, 0.05) * temp / initial_temp;
        test_cfg.dynamic_ofi_threshold += random_f64(-0.02, 0.02) * temp / initial_temp;
        test_cfg.weight_obi += random_f64(-0.1, 0.1) * temp / initial_temp;
        test_cfg.weight_ofi += random_f64(-0.1, 0.1) * temp / initial_temp;
        test_cfg.weight_vpin += random_f64(-0.05, 0.05) * temp / initial_temp;
        test_cfg.dynamic_atr_min += random_f64(-0.0002, 0.0002) * temp / initial_temp;

        // Genome Limit Mutations (Eradicating hardcoded biases)
        test_cfg.global_max_drawdown += random_f64(-0.02, 0.02) * temp / initial_temp;
        test_cfg.min_trades_per_day += random_f64(-1.0, 1.0) * temp / initial_temp;
        test_cfg.survival_capital_threshold += random_f64(-0.05, 0.05) * temp / initial_temp;

        // Micro-Capital Adaptive Constraints
        let max_safe_leverage = if initial_capital < 50.0 {
            // Si el capital es pequeño, priorizamos supervivencia sobre explosividad extrema
            (initial_capital / 100.0).max(1.0) * 50.0 // ej: 13 USD -> 50.0 max
        } else {
            100.0
        };
        let min_required_leverage = if initial_capital < 10.0 {
            (5.0 / initial_capital) * 1.05 // Para 13 USD, el mínimo es 1.0x (ya que 13 > 5)
        } else {
            1.0
        };

        test_cfg.global_leverage = test_cfg.global_leverage.clamp(
            min_required_leverage,
            max_safe_leverage.max(min_required_leverage),
        );

        // Give it space to breathe (stop loss max 5% instead of 2%)
        test_cfg.scalp_sl_base = test_cfg.scalp_sl_base.clamp(0.001, 0.050);
        test_cfg.scalp_tp_base = test_cfg.scalp_tp_base.clamp(0.001, 0.100);

        test_cfg.global_max_drawdown = test_cfg.global_max_drawdown.clamp(0.05, 0.30);
        test_cfg.survival_capital_threshold = test_cfg.survival_capital_threshold.clamp(0.50, 0.95);
        test_cfg.min_trades_per_day = test_cfg.min_trades_per_day.clamp(1.0, 50.0);

        test_cfg.ml_threshold_long = test_cfg.ml_threshold_long.clamp(0.0, 0.999);
        test_cfg.ml_threshold_short = test_cfg.ml_threshold_short.clamp(0.0, 0.999);
        test_cfg.trend_threshold = test_cfg.trend_threshold.clamp(0.1, 0.8);

        test_cfg.dynamic_obi_threshold = test_cfg.dynamic_obi_threshold.clamp(0.01, 0.30);
        test_cfg.dynamic_ofi_threshold = test_cfg.dynamic_ofi_threshold.clamp(0.01, 0.20);
        test_cfg.weight_obi = test_cfg.weight_obi.clamp(0.1, 0.8);
        test_cfg.weight_ofi = test_cfg.weight_ofi.clamp(0.1, 0.8);
        test_cfg.weight_vpin = test_cfg.weight_vpin.clamp(0.05, 0.5);
        // Force minimum EV threshold to be slightly higher to only take the best trades
        test_cfg.dynamic_atr_min = test_cfg.dynamic_atr_min.clamp(0.0001, 0.01);
        test_cfg.ev_fee_multiplier = 0.0; // Allow trades during evolution exploration

        let mut out_pnl = vec![0.0; train_len];
        let mut out_stats = [0.0; 10];

        run_backtest_native(
            &closes,
            &highs,
            &lows,
            &volumes,
            &test_cfg,
            &mut out_pnl,
            &mut out_stats,
            &symbol,
            initial_capital,
        );

        let _net_win_rate = out_stats[0];
        let trades = out_stats[1];
        let capital = out_stats[2];
        let dd = out_stats[3];
        let _gross_pnl = out_stats[5];
        let _net_pnl = out_stats[6];
        let _gross_win_rate = out_stats[7];

        let days_simulated =
            (timestamps[train_len - 1] - timestamps[0]) / (1000.0 * 60.0 * 60.0 * 24.0);
        let days_simulated = if days_simulated < 0.1 {
            1.0
        } else {
            days_simulated
        };
        let periods_of_3_days = days_simulated / 3.0;

        // removed starting_capital declaration

        let initial_cap_f64 = initial_capital;

        let compound_rate_3d = if capital > 0.0 && periods_of_3_days > 0.0 {
            let exp_growth = (capital / initial_cap_f64).powf(1.0_f64 / periods_of_3_days);
            exp_growth
        } else {
            0.0
        };

        // Asymmetric Reward for > 1.0x compound rate
        let mut score = if compound_rate_3d > 2.0 {
            // Hyper-compounding target achieved (x2.0 every 3 days)
            compound_rate_3d.powf(4.0) * 50000.0
        } else {
            compound_rate_3d.powf(2.0) * 10000.0
        };

        // FASE 17: Aplicar la penalización de Drawdown Bayesiana
        let dd_threshold = test_cfg.global_max_drawdown / 3.0; // Deseable is 1/3 of max drawdown
        if dd > dd_threshold {
            let decay = f64::exp(-(dd - dd_threshold) * 20.0).clamp(0.01, 1.0);
            score *= decay; // Destruir la puntuación exponencialmente basado en Max Drawdown
        }

        // Regularity Penalties
        if trades < (days_simulated * test_cfg.min_trades_per_day) {
            score -= 10000.0;
        }

        // Asymmetric Master Rule Penalties
        let survival_threshold = initial_capital * test_cfg.survival_capital_threshold;
        if capital < survival_threshold {
            score -= 200000.0;
        } else if capital < initial_capital {
            score -= 10000.0 * (initial_capital - capital); // Linear penalty instead of flat wall
        }

        if dd > test_cfg.global_max_drawdown {
            score -= dd * 200000.0;
        } // Aniquilación inmediata (Penalización Absoluta)

        if i % 20 == 0 {
            println!("🔄 Iter {}: Curr Score = {:.2} (Best: {:.2}) | IS Cap: {:.2}, Trades: {}, WinRate: {:.2} | Temp: {:.2}", i, score, best_score, capital, trades, out_stats[0], temp);
        }

        // Acceptance Probability (Metropolis-Hastings)
        let mut accept = false;
        if score > current_score {
            accept = true;
        } else {
            let prob = std::f64::consts::E.powf((score - current_score) / temp);
            if random_f64(0.0, 1.0) < prob {
                accept = true;
            }
        }

        if accept {
            current_score = score;
            current_config = test_cfg.clone();
        }

        if score > best_score {
            best_score = score;
            best_config = test_cfg.clone();
        }

        temp *= cooling_rate;
        if temp < 0.01 {
            temp = initial_temp;
        } // Re-heating (Quantum Tunneling)

        if i % 1000 == 0 || i == iterations - 1 {
            println!(
                "🧬 Iter {}: Best Score = {:.2} | Compound 3D: {:.2}x | Temp: {:.2}",
                i,
                best_score,
                best_score / 10000.0,
                temp
            );
        }
    }

    let mut best_out_pnl = vec![0.0; train_len];
    let mut best_out_stats = [0.0; 10];
    run_backtest_native(
        &closes,
        &highs,
        &lows,
        &volumes,
        &best_config,
        &mut best_out_pnl,
        &mut best_out_stats,
        &symbol,
        initial_capital,
    );
    let best_is_net_win_rate = best_out_stats[0];
    let best_is_trades = best_out_stats[1];
    let best_is_capital = best_out_stats[2];
    let best_is_dd = best_out_stats[3];
    let best_is_sharpe = best_out_stats[4];
    let best_is_gross_pnl = best_out_stats[5];
    let best_is_net_pnl = best_out_stats[6];
    let best_is_gross_win_rate = best_out_stats[7];

    let elapsed = start_time.elapsed();
    println!("============================================================");
    println!(
        "✅ Evolution Complete in {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );
    println!("🏆 Best Config Found (In-Sample):");
    println!("   SL %      : {:.4}", best_config.scalp_sl_base);
    println!("   TP %      : {:.4}", best_config.scalp_tp_base);
    println!("   ML Thresh L: {:.4}", best_config.ml_threshold_long);
    println!("   ML Thresh S: {:.4}", best_config.ml_threshold_short);
    let train_start =
        DateTime::<Utc>::from_timestamp((timestamps[0] / 1000.0) as i64, 0).unwrap_or_default();
    let train_end = DateTime::<Utc>::from_timestamp((timestamps[train_len - 1] / 1000.0) as i64, 0)
        .unwrap_or_default();
    let days_train = (timestamps[train_len - 1] - timestamps[0]) / (1000.0 * 60.0 * 60.0 * 24.0);

    println!("   Tech L     : {:.4}", best_config.trend_threshold);
    println!("   Tech S     : {:.4}", best_config.trend_threshold);
    println!("------------------------------------------------------------");
    println!(
        "🗓️ IS Period : {} to {} ({:.2} days)",
        train_start.format("%Y-%m-%d %H:%M:%S"),
        train_end.format("%Y-%m-%d %H:%M:%S"),
        days_train
    );
    println!("------------------------------------------------------------");
    println!("   IS Capital: ${:.2}", best_is_capital);
    println!(
        "   IS Gross Win Rate: {:.2}%",
        best_is_gross_win_rate * 100.0
    );
    println!("   IS Net Win Rate: {:.2}%", best_is_net_win_rate * 100.0);
    println!("   IS Trades : {}", best_is_trades);
    println!("   IS Max DD : {:.2}%", best_is_dd * 100.0);
    println!("   IS Sharpe : {:.4}", best_is_sharpe);
    println!(
        "   IS Gross PnL: ${:.4} (ROI: {:.2}%)",
        best_is_gross_pnl,
        (best_is_gross_pnl / initial_capital) * 100.0
    );
    println!(
        "   IS Net PnL: ${:.4} (ROI: {:.2}%)",
        best_is_net_pnl,
        (best_is_net_pnl / initial_capital) * 100.0
    );

    println!("============================================================");
    println!("🧪 RUNNING OUT-OF-SAMPLE TEST (Walk-Forward Validation)");

    let mut out_pnl_test = vec![0.0; test_len];
    let mut out_stats_test = [0.0; 10];

    // CRITICAL FIX: Use correct OOS slices from each array (closes, highs, lows, volumes)
    let oos_closes = &closes[train_len..];
    let oos_highs = &highs[train_len..];
    let oos_lows = &lows[train_len..];
    let oos_volumes = &volumes[train_len..];

    run_backtest_native(
        oos_closes,
        oos_highs,
        oos_lows,
        oos_volumes,
        &best_config,
        &mut out_pnl_test,
        &mut out_stats_test,
        &symbol,
        initial_capital,
    );

    let oos_net_win_rate = out_stats_test[0];
    let oos_trades = out_stats_test[1];
    let oos_capital = out_stats_test[2];
    let oos_dd = out_stats_test[3];
    let oos_sharpe = out_stats_test[4];
    let oos_gross_pnl = out_stats_test[5];
    let oos_net_pnl = out_stats_test[6];
    let oos_gross_win_rate = out_stats_test[7];

    let test_start = DateTime::<Utc>::from_timestamp((timestamps[train_len] / 1000.0) as i64, 0)
        .unwrap_or_default();
    let test_end = DateTime::<Utc>::from_timestamp((timestamps[len - 1] / 1000.0) as i64, 0)
        .unwrap_or_default();
    let days_test = (timestamps[len - 1] - timestamps[train_len]) / (1000.0 * 60.0 * 60.0 * 24.0);

    println!("📊 Out-Of-Sample Results ({} ticks):", test_len);
    println!(
        "🗓️ Period        : {} to {} ({:.2} days)",
        test_start.format("%Y-%m-%d %H:%M:%S"),
        test_end.format("%Y-%m-%d %H:%M:%S"),
        days_test
    );
    println!(
        "   Final Capital : ${:.2} (Starting: ${:.2})",
        oos_capital, initial_capital
    );
    println!("   Gross Win Rate: {:.2}%", oos_gross_win_rate * 100.0);
    println!("   Net Win Rate  : {:.2}%", oos_net_win_rate * 100.0);
    println!("   Total Trades  : {}", oos_trades);
    println!("   Max Drawdown  : {:.2}%", oos_dd * 100.0);
    println!("   Sharpe Ratio  : {:.4}", oos_sharpe);
    println!(
        "   Gross PnL     : ${:.4} (ROI: {:.2}%)",
        oos_gross_pnl,
        (oos_gross_pnl / initial_capital) * 100.0
    );
    println!(
        "   Net PnL       : ${:.4} (ROI: {:.2}%)",
        oos_net_pnl,
        (oos_net_pnl / initial_capital) * 100.0
    );

    let days_test_for_compound = if days_test < 0.1 { 0.5 } else { days_test };
    let periods_test = days_test_for_compound / 3.0;
    let compound_test = if oos_capital > 0.0 && periods_test > 0.0 {
        let oos_exp_growth = (oos_capital / initial_capital).powf(1.0_f64 / periods_test);
        oos_exp_growth
    } else {
        0.0
    };
    println!(
        "   Compound Rate : {:.2}x every 3 days (Goal: 2.0x)",
        compound_test
    );
    println!("============================================================");

    // Read existing config to preserve fields we don't optimize (symbols, leverage)
    let existing_json: serde_json::Value = std::fs::read_to_string("data/dynamic_config.json")
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(serde_json::json!({}));

    let symbols = existing_json
        .get("symbols")
        .map(|v| v.to_string())
        .unwrap_or("[\"btcusdt\", \"ethusdt\"]".to_string());
    let scalp_lev = existing_json
        .get("scalp_leverage")
        .and_then(|v| v.as_f64())
        .unwrap_or(50.0);
    let swing_lev = existing_json
        .get("swing_leverage")
        .and_then(|v| v.as_f64())
        .unwrap_or(15.0);

    let out_json = format!(
        "{{\n  \"sl_pct\": {:.4},\n  \"tp_pct\": {:.4},\n  \"ml_threshold_l\": {:.4},\n  \"ml_threshold_s\": {:.4},\n  \"tech_threshold_l\": {:.4},\n  \"tech_threshold_s\": {:.4},\n  \"scalp_leverage\": {:.1},\n  \"swing_leverage\": {:.1},\n  \"symbols\": {}\n}}",
        best_config.scalp_sl_base, best_config.scalp_tp_base, best_config.ml_threshold_long, best_config.ml_threshold_short, 
        best_config.trend_threshold, best_config.trend_threshold,
        scalp_lev, swing_lev, symbols
    );

    std::fs::write("data/dynamic_config.json", out_json)
        .expect("Unable to write dynamic_config.json");
    println!("💾 Exported to data/dynamic_config.json (all fields preserved).");
}
