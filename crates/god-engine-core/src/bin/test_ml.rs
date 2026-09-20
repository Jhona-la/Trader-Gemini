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
            
            println!("Testing predict_for_coin with fixed scaler:");
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
        } else {
            println!("Failed to parse DarkAlpha_BTCUSDT.json");
        }
    } else {
        println!("Failed to read DarkAlpha_BTCUSDT.json");
    }
}
