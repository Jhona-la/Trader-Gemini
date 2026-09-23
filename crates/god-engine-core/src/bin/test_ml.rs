use god_engine_core::ml_inference::NanoForest;

fn main() {
    let path = if std::path::Path::new("models/BTCUSDT_MOTOR.json").exists() {
        "models/BTCUSDT_MOTOR.json"
    } else {
        "../../models/BTCUSDT_MOTOR.json"
    };
    let forest = NanoForest::load_model(path).unwrap();
    println!("Loaded forest.");

    // Test with all zeros
    let features = vec![0.0; 54];
    let prob = forest.predict(&features).unwrap();
    println!("Prob (zeros): {:.4}", prob);

    // Test with all ones
    let features = vec![1.0; 54];
    let prob = forest.predict(&features).unwrap();
    println!("Prob (ones): {:.4}", prob);

    let mut features = vec![0.0; 54];
    for i in 0..54 {
        features[i] = (i as f32) / 54.0 - 0.5;
    }
    let prob = forest.predict(&features).unwrap();
    println!("Prob (range): {:.4}", prob);

    // Test with StatefulEngine
    let mut engine = god_engine_core::stateful_engine::StatefulEngine::new();
    let mut price = 65000.0;
    for i in 0..200 {
        price += ((i % 5) as f64 - 2.0) * 10.0;
        engine.update_ofi(price - 0.5, price + 0.5, 10.0, 10.0);
        engine.update_trade_flow(1.0, i % 2 == 0);
        engine.process_kline(price - 5.0, price + 10.0, price - 10.0, price, 100.0);
    }
    let sf = engine.get_universal_features();
    let sp = engine.get_spectral_ml_features();
    let mut real_feat = [0.0f32; 54];
    real_feat[..34].copy_from_slice(&sf);
    real_feat[34..44].copy_from_slice(&sp);
    for (i, &f) in real_feat.iter().enumerate() {
        if f.abs() > 1e-6 {
            println!("  feat[{}]: {:.6}", i, f);
        }
    }
    let p_real = forest.predict(&real_feat);
    println!("Prob (real engine features - BTCUSDT_MOTOR): {:?}", p_real);

    if let Ok(cand_forest) = NanoForest::load_model("models/BTCUSDT_MOTOR_CANDIDATE.json") {
        println!("Prob (real engine features - CANDIDATE): {:?}", cand_forest.predict(&real_feat));
    }

    let nn_path = if std::path::Path::new("models/DarkAlpha_BTCUSDT.json").exists() {
        "models/DarkAlpha_BTCUSDT.json"
    } else {
        "../../models/DarkAlpha_BTCUSDT.json"
    };
    if let Ok(json_data) = std::fs::read_to_string(nn_path) {
        if let Ok(mut model) = serde_json::from_str::<dark_alpha_engine::DarkAlphaEngine>(&json_data) {
            model.init_buffers();
            model.sanitize_denormals();
            
            println!("\n🧠 [DARK ALPHA DEEP DIAGNOSTIC]");
            println!("  Layer1: in={}, out={}", model.layer1.in_features, model.layer1.out_features);
            println!("  Layer2: in={}, out={}", model.layer2.in_features, model.layer2.out_features);
            println!("  Layer3: in={}, out={}", model.layer3.in_features, model.layer3.out_features);
            println!("  Layer3 bias: {:?}", model.layer3.biases);
            println!("  Scaler present: {}", model.scaler.is_some());
            if let Some(ref sc) = model.scaler {
                println!("    Scaler mean len: {}, std_dev len: {}", sc.mean.len(), sc.std_dev.len());
                for idx in 0..sc.mean.len() {
                    println!("    feat[{:02}]: mean={:+.6}, std={:.6}", idx, sc.mean[idx], sc.std_dev[idx]);
                }
            }
            println!("  Channel normalizers len: {}", model.channel_normalizers.len());
            if !model.channel_normalizers.is_empty() {
                println!("    Chan 0 count: {}, mean: {}, std: {}", 
                    model.channel_normalizers[0].count,
                    model.channel_normalizers[0].mean,
                    model.channel_normalizers[0].std_dev());
            }

            println!("  Layer3 weights (32):");
            for k in 0..32 {
                println!("    w3[{:02}]: {:+.6}", k, model.layer3.weights[k]);
            }

            model.freeze();
            let p_zero = model.predict_for_coin(0, &[0.0; 54]);
            println!("  predict_for_coin(0, zeros): {:?}", p_zero);
            let p_ones = model.predict_for_coin(0, &[1.0; 54]);
            println!("  predict_for_coin(0, ones): {:?}", p_ones);
            let mut range_feats = [0.0; 54];
            for i in 0..54 {
                range_feats[i] = (i as f64) / 54.0 - 0.5;
            }
            let p_range = model.predict_for_coin(0, &range_feats);
            println!("  predict_for_coin(0, range): {:?}", p_range);

            // Test with real engine 54D features:
            let mut real_54 = [0.0; 54];
            for i in 0..34 {
                real_54[i] = sf[i] as f64;
            }
            let p_real_engine = model.predict_for_coin(0, &real_54);
            println!("  predict_for_coin(0, real_engine_features): {:?}", p_real_engine);

            // Now test reading real ticks from BTCUSDT_AUG_REAL.bin using std::fs
            use std::io::Read;
            if let Ok(mut file) = std::fs::File::open("data/BTCUSDT_AUG_REAL.bin") {
                let mut header = [0u8; 8];
                let _ = file.read_exact(&mut header);
                #[repr(C)]
                #[derive(Clone, Copy)]
                struct BinTick {
                    pub timestamp: u64,
                    pub bid_price: f64,
                    pub ask_price: f64,
                    pub bid_qty: f64,
                    pub ask_qty: f64,
                }
                let tick_size = std::mem::size_of::<BinTick>();
                let n_ticks = 20000;
                let mut raw_bytes = vec![0u8; n_ticks * tick_size];
                if file.read_exact(&mut raw_bytes).is_ok() {
                    let ticks: &[BinTick] = unsafe {
                        std::slice::from_raw_parts(raw_bytes.as_ptr() as *const BinTick, n_ticks)
                    };
                    println!("\n🔍 [REAL TICKS SIMULATION - WITH SCALER]");
                    let mut real_engine = god_engine_core::stateful_engine::StatefulEngine::new();
                    let mut predictions = Vec::new();
                    for (i, t) in ticks.iter().enumerate() {
                        let mid = (t.bid_price + t.ask_price) / 2.0;
                        let vol = t.bid_qty + t.ask_qty;
                        real_engine.process_tick(mid, vol, t.timestamp);
                        real_engine.update_trade_flow(vol, t.ask_qty > t.bid_qty);
                        let _ = real_engine.update_ofi(t.bid_price, t.ask_price, t.bid_qty, t.ask_qty);
                        if i >= 500 {
                            let sf = real_engine.get_universal_features();
                            let mut t54 = [0.0; 54];
                            for j in 0..34 { t54[j] = sf[j] as f64; }
                            let obi = if t.bid_qty + t.ask_qty > 0.0 { (t.bid_qty - t.ask_qty) / (t.bid_qty + t.ask_qty) } else { 0.0 };
                            let atr = real_engine.get_atr_pct();
                            let delta = real_engine.v_t;
                            t54[34] = (delta / mid.max(1.0)).clamp(-0.1, 0.1);
                            t54[35] = atr;
                            t54[36] = (vol / 1000.0).tanh();
                            t54[37] = obi;
                            t54[38] = (t54[34] * 10.0).tanh();
                            t54[39] = (t54[35] * 100.0).min(5.0);
                            // 40..54 as defined in build_54d_tensor
                            t54[40] = 0.0;
                            t54[41] = 1.04;
                            t54[42] = 1.02;
                            t54[43] = 1.00;
                            t54[44] = 0.75;
                            t54[45] = 0.0;
                            t54[46] = 0.0;
                            t54[47] = 0.0;
                            t54[48] = 0.0;
                            t54[49] = 0.0;
                            t54[50] = 1.0;
                            t54[51] = 0.0;
                            t54[52] = 0.0;
                            t54[53] = 0.0;

                            if let Some(p) = model.predict_for_coin(0, &t54) {
                                predictions.push(p);
                            }
                        }
                    }
                    if !predictions.is_empty() {
                        let count_ge_05 = predictions.iter().filter(|&&p| p >= 0.5).count();
                        let pct_ge_05 = (count_ge_05 as f64 / predictions.len() as f64) * 100.0;
                        let min_p = predictions.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max_p = predictions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let mean_p: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;
                        println!("  [With Scaler] >= 0.5: {} ({:.1}%), Min: {:.4}, Max: {:.4}, Mean: {:.4}", count_ge_05, pct_ge_05, min_p, max_p, mean_p);
                    }

                    // Test with Welford (scaler = None, unfreeze)
                    println!("\n🔍 [REAL TICKS SIMULATION - WITH WELFORD ONLINE (scaler=None)]");
                    let mut model_welford = model.clone();
                    model_welford.scaler = None;
                    model_welford.unfreeze();
                    let mut real_engine = god_engine_core::stateful_engine::StatefulEngine::new();
                    let mut predictions_w = Vec::new();
                    for (i, t) in ticks.iter().enumerate() {
                        let mid = (t.bid_price + t.ask_price) / 2.0;
                        let vol = t.bid_qty + t.ask_qty;
                        real_engine.process_tick(mid, vol, t.timestamp);
                        real_engine.update_trade_flow(vol, t.ask_qty > t.bid_qty);
                        let _ = real_engine.update_ofi(t.bid_price, t.ask_price, t.bid_qty, t.ask_qty);
                        if i >= 500 {
                            let sf = real_engine.get_universal_features();
                            let mut t54 = [0.0; 54];
                            for j in 0..34 { t54[j] = sf[j] as f64; }
                            let obi = if t.bid_qty + t.ask_qty > 0.0 { (t.bid_qty - t.ask_qty) / (t.bid_qty + t.ask_qty) } else { 0.0 };
                            let atr = real_engine.get_atr_pct();
                            let delta = real_engine.v_t;
                            t54[34] = (delta / mid.max(1.0)).clamp(-0.1, 0.1);
                            t54[35] = atr;
                            t54[36] = (vol / 1000.0).tanh();
                            t54[37] = obi;
                            t54[38] = (t54[34] * 10.0).tanh();
                            t54[39] = (t54[35] * 100.0).min(5.0);
                            t54[40] = 0.0;
                            t54[41] = 1.04;
                            t54[42] = 1.02;
                            t54[43] = 1.00;
                            t54[44] = 0.75;
                            t54[45] = 0.0;
                            t54[46] = 0.0;
                            t54[47] = 0.0;
                            t54[48] = 0.0;
                            t54[49] = 0.0;
                            t54[50] = 1.0;
                            t54[51] = 0.0;
                            t54[52] = 0.0;
                            t54[53] = 0.0;

                            if let Some(p) = model_welford.predict_for_coin(0, &t54) {
                                predictions_w.push(p);
                            }
                        }
                    }
                    if !predictions_w.is_empty() {
                        let count_ge_05 = predictions_w.iter().filter(|&&p| p >= 0.5).count();
                        let pct_ge_05 = (count_ge_05 as f64 / predictions_w.len() as f64) * 100.0;
                        let min_p = predictions_w.iter().cloned().fold(f64::INFINITY, f64::min);
                        let max_p = predictions_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let mean_p: f64 = predictions_w.iter().sum::<f64>() / predictions_w.len() as f64;
                        println!("  [With Welford] >= 0.5: {} ({:.1}%), Min: {:.4}, Max: {:.4}, Mean: {:.4}", count_ge_05, pct_ge_05, min_p, max_p, mean_p);
                        println!("  Sample Welford P[0..10]: {:?}", &predictions_w[..predictions_w.len().min(10)]);
                        println!("  Sample Welford P[last 10]: {:?}", &predictions_w[predictions_w.len().saturating_sub(10)..]);
                    }
                }
            }
        } else {
            println!("Failed to parse DarkAlpha_BTCUSDT.json");
        }
    } else {
        println!("Failed to read DarkAlpha_BTCUSDT.json");
    }
}
