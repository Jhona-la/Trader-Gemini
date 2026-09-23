use god_engine_core::stateful_engine::StatefulEngine;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct BinTick {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

fn main() {
    println!("============================================================");
    println!("🌌 RUST QUANTUM FEATURE EXPORTER (1:1 ALIGNMENT)");
    println!("============================================================");

    let symbol = std::env::args().nth(1).unwrap_or_else(|| "BTCUSDT".to_string());
    let input_path = std::env::args().nth(2).unwrap_or_else(|| {
        let aug_real = format!("data/{}_AUG_REAL.bin", symbol);
        let ticks_real = format!("data/{}_ticks_REAL.bin", symbol);
        let ticks_legacy = format!("data/{}_ticks.bin", symbol);
        if std::path::Path::new(&aug_real).exists() {
            aug_real
        } else if std::path::Path::new(&ticks_real).exists() {
            ticks_real
        } else {
            ticks_legacy
        }
    });
    let max_ticks: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var("MAX_TICKS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(1_000_000);

    let file = match File::open(&input_path) {
        Ok(f) => f,
        Err(e) => {
            println!("⚠️ Failed to open {}: {}", input_path, e);
            return;
        }
    };

    let mmap = match unsafe { memmap2::MmapOptions::new().map(&file) } {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Failed to mmap file {}: {}", input_path, e);
            return;
        }
    };

    let bytes_len = mmap.len();
    let (origen, header_len) = backtest_engine::tick_replayer::TickOrigin::from_header(
        &mmap[..bytes_len.min(8)],
    );
    let payload_len = bytes_len - header_len;
    let tick_size = std::mem::size_of::<BinTick>();
    let total_file_ticks = payload_len / tick_size;
    let num_ticks = total_file_ticks.min(max_ticks);

    if num_ticks == 0 {
        println!("❌ No data loaded or file is empty.");
        return;
    }

    println!("📦 Origen de los datos: {}", origen.descripcion());
    let ticks = unsafe {
        std::slice::from_raw_parts(
            mmap.as_ptr().add(header_len) as *const BinTick,
            num_ticks,
        )
    };
    println!("✅ Loaded {}/{} BinTicks from {} for {}", num_ticks, total_file_ticks, input_path, symbol);

    let out_path = format!("data/{}_FEATURES.csv", symbol);
    let out_file = match File::create(&out_path) {
        Ok(f) => f,
        Err(e) => {
            println!("❌ Failed to create {}: {}", out_path, e);
            return;
        }
    };
    let mut out_file = BufWriter::new(out_file);

    // Escribir cabeceras para 54 variables unificadas
    let mut header = String::from("target_5m");
    for i in 0..54 {
        header.push_str(&format!(",feature_{}", i));
    }
    if let Err(e) = writeln!(out_file, "{}", header) {
        println!("❌ Failed to write header to {}: {}", out_path, e);
        return;
    }

    let mut feature_engine = StatefulEngine::new();
    let mut written = 0;

    for i in 0..num_ticks {
        let t = &ticks[i];
        let mid_price = (t.bid_price + t.ask_price) / 2.0;
        let total_vol = t.bid_qty + t.ask_qty;
        let pseudo_maker = t.bid_qty > t.ask_qty;

        feature_engine.process_tick(mid_price, total_vol, t.timestamp);
        feature_engine.update_trade_flow(total_vol, pseudo_maker);
        let _ = feature_engine.update_ofi(t.bid_price, t.ask_price, t.bid_qty, t.ask_qty);

        // Triple Barrier Labeling Method (Marcos López de Prado)
        // Alineado exactamente a los pisos institucionales de GodEngineCore: TP = 0.36%, SL = 0.18%
        if i >= 100 && i + 500 < num_ticks {
            let tp_pct = 0.0036;
            let sl_pct = 0.0018;
            let long_tp = mid_price * (1.0 + tp_pct);
            let long_sl = mid_price * (1.0 - sl_pct);
            let short_tp = mid_price * (1.0 - tp_pct);
            let short_sl = mid_price * (1.0 + sl_pct);

            let mut barrier_label: f64 = 0.5; // 0.5 = Neutral

            for f in 1..=500 {
                let fut_mid = (ticks[i + f].bid_price + ticks[i + f].ask_price) / 2.0;

                // D-329: Causalidad estricta Triple Barrier (López de Prado)
                // 1) Si toca el Stop Loss de Long (-0.15%), la hipótesis alcista fracasa inmediatamente
                if fut_mid <= long_sl {
                    barrier_label = 0.0; // Pérdida en Long / Victoria en Short
                    break;
                }
                // 2) Si toca el Stop Loss de Short (+0.15%), la hipótesis bajista fracasa inmediatamente
                if fut_mid >= short_sl {
                    barrier_label = 1.0; // Victoria en Long / Pérdida en Short
                    break;
                }
                // 3) Si toca Take Profit de Long (+0.30%) sin haber tocado SL previo
                if fut_mid >= long_tp {
                    barrier_label = 1.0;
                    break;
                }
                // 4) Si toca Take Profit de Short (-0.30%) sin haber tocado SL previo
                if fut_mid <= short_tp {
                    barrier_label = 0.0;
                    break;
                }
            }

            // Si ninguna barrera horizontal se tocó en 500 ticks (barrera vertical), usar retorno terminal
            if barrier_label == 0.5 {
                let end_mid = (ticks[i + 500].bid_price + ticks[i + 500].ask_price) / 2.0;
                let end_ret = (end_mid - mid_price) / mid_price;
                if end_ret > 0.0004 {
                    barrier_label = 1.0;
                } else if end_ret < -0.0004 {
                    barrier_label = 0.0;
                }
            }

            // Descartar zonas puramente laterales para entrenamiento discriminativo puro
            if (barrier_label - 0.5).abs() < 0.1 {
                continue;
            }

            // Generar features idénticas a GodEngineCore (34 micro+omni + 20 macro/cross-exchange proxies)
            let stateful_feats = feature_engine.get_universal_features();

            let mut features = [0.0; 54];
            for j in 0..34 {
                features[j] = stateful_feats[j] as f64;
            }

            let delta = feature_engine.v_t;

            // Proxies para variables macro/on-chain 34..54 derivadas de la serie histórica
            features[34] = (delta / mid_price).clamp(-0.1, 0.1); // Log return
            features[35] = feature_engine.get_atr_pct(); // Volatility ATR
            features[36] = (total_vol / 1000.0).tanh(); // Relative Volume
            features[37] = if total_vol > 0.0 {
                (t.bid_qty - t.ask_qty) / total_vol
            } else {
                0.0
            }; // Realized OBI
            features[38] = (features[34] * 10.0).tanh(); // Momentum Proxy
            features[39] = (features[35] * 100.0).min(5.0); // Parkinson Vol Proxy
            // Sincronización 1:1 estricta con build_54d_tensor (crates/god-engine-core/src/lib.rs:433-485)
            features[40] = 0.0; // MUERTA (sin productor vivo)
            features[41] = 1.04; // DXY Index Baseline (~104.0 / 100.0)
            features[42] = 1.02; // SP500 Index Baseline (~5100.0 / 5000.0)
            features[43] = 1.00; // Nasdaq Baseline (~18000.0 / 18000.0)
            features[44] = 0.75; // VIX Baseline (~15.0 / 20.0)
            features[45] = 0.0; // MUERTA (us10y sin productor vivo)
            features[46] = 0.0; // Gold PAXG fallback (0.0 "sin dato")
            features[47] = 0.0; // MUERTA (oil WTI sin productor vivo)
            features[48] = 0.0; // Funding / Econ Impact
            features[49] = 0.0; // MUERTA (fed_rate sin productor vivo)
            features[50] = 1.0; // Taker buy/sell baseline
            features[51] = 0.0; // MUERTA (basis premium sin productor vivo)
            features[52] = 0.0; // MUERTA (liq cluster short sin productor vivo)
            features[53] = 0.0; // MUERTA (liq cluster long sin productor vivo)

            let mut row = format!("{:.1}", barrier_label);
            for f in &features {
                let safe_f = if f.is_finite() { *f } else { 0.0 };
                row.push_str(&format!(",{:.6}", safe_f));
            }
            if writeln!(out_file, "{}", row).is_ok() {
                written += 1;
            }
        }
    }

    println!(
        "🚀 Exporter finished! Wrote {} clean rows to {}",
        written, out_path
    );
}
