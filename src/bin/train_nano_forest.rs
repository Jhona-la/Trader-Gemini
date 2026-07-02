use quantum_engine::stateful_engine::StatefulEngine;
use quantum_engine::ml_inference::{NanoForestData, NanoForest};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

struct Sample {
    features: Vec<f32>,
    label: f32,
    weight: f32,
}

struct TreeNode {
    feature: i32,
    threshold: f32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    value: f32,
}

fn calculate_gini(labels: &[f32], weights: &[f32]) -> f32 {
    if labels.is_empty() { return 0.0; }
    let mut w1 = 0.0;
    let mut w_minus1 = 0.0;
    let mut w0 = 0.0;
    let mut total_weight = 0.0;
    for (i, &l) in labels.iter().enumerate() {
        total_weight += weights[i];
        if l == 1.0 { w1 += weights[i]; }
        else if l == -1.0 { w_minus1 += weights[i]; }
        else { w0 += weights[i]; }
    }
    if total_weight == 0.0 { return 0.0; }
    let p1 = w1 / total_weight;
    let p_minus1 = w_minus1 / total_weight;
    let p0 = w0 / total_weight;
    1.0 - (p1 * p1 + p_minus1 * p_minus1 + p0 * p0)
}

fn split_data<'a>(samples: &'a [&'a Sample], feature_idx: usize, threshold: f32) -> (Vec<&'a Sample>, Vec<&'a Sample>) {
    let mut left = Vec::with_capacity(samples.len());
    let mut right = Vec::with_capacity(samples.len());
    for &s in samples {
        if s.features[feature_idx] <= threshold {
            left.push(s);
        } else {
            right.push(s);
        }
    }
    (left, right)
}

fn build_tree(samples: &[&Sample], depth: u32, max_depth: u32, min_samples_split: usize) -> Box<TreeNode> {
    let mut w1 = 0.0;
    let mut w_minus1 = 0.0;
    let mut total_weight = 0.0;
    for s in samples {
        total_weight += s.weight;
        if s.label == 1.0 { w1 += s.weight; }
        else if s.label == -1.0 { w_minus1 += s.weight; }
    }
    let expected_value = if total_weight == 0.0 { 0.0 } else { (w1 - w_minus1) / total_weight };
    let value = expected_value;

    if depth >= max_depth || samples.len() < min_samples_split || expected_value.abs() == 1.0 || total_weight == 0.0 {
        return Box::new(TreeNode {
            feature: -1,
            threshold: 0.0,
            left: None,
            right: None,
            value,
        });
    }

    let mut best_gini = f32::MAX;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_splits = (Vec::new(), Vec::new());

    let num_features = samples[0].features.len();
    
    // Stochastic Quantile Sketching (O(N log N) equivalent speedup)
    for feat in 0..num_features {
        let step = if samples.len() > 20 { samples.len() / 20 } else { 1 };
        for i in (0..samples.len()).step_by(step) {
            let threshold = samples[i].features[feat];
            let (left, right) = split_data(samples, feat, threshold);
            
            if left.is_empty() || right.is_empty() { continue; }

            let left_labels: Vec<f32> = left.iter().map(|s| s.label).collect();
            let right_labels: Vec<f32> = right.iter().map(|s| s.label).collect();
            let left_weights: Vec<f32> = left.iter().map(|s| s.weight).collect();
            let right_weights: Vec<f32> = right.iter().map(|s| s.weight).collect();

            let gini_left = calculate_gini(&left_labels, &left_weights);
            let gini_right = calculate_gini(&right_labels, &right_weights);

            let weight_left = left_weights.iter().sum::<f32>() / total_weight;
            let weight_right = right_weights.iter().sum::<f32>() / total_weight;

            let gini = weight_left * gini_left + weight_right * gini_right;

            if gini < best_gini {
                best_gini = gini;
                best_feature = feat;
                best_threshold = threshold;
                best_splits = (left, right);
            }
        }
    }

    if best_splits.0.is_empty() || best_splits.1.is_empty() {
        return Box::new(TreeNode {
            feature: -1,
            threshold: 0.0,
            left: None,
            right: None,
            value,
        });
    }

    Box::new(TreeNode {
        feature: best_feature as i32,
        threshold: best_threshold,
        left: Some(build_tree(&best_splits.0, depth + 1, max_depth, min_samples_split)),
        right: Some(build_tree(&best_splits.1, depth + 1, max_depth, min_samples_split)),
        value,
    })
}

fn flatten_tree(
    node: &TreeNode,
    data: &mut NanoForestData,
) -> i32 {
    let current_idx = data.value.len() as i32;
    
    data.children_left.push(-1);
    data.children_right.push(-1);
    data.feature.push(node.feature);
    data.threshold.push(node.threshold);
    
    let eps = 1e-6;
    // Map expected_value [-1, 1] to probability space [0, 1]
    let mut p = (node.value + 1.0) / 2.0;
    if p < eps { p = eps; }
    if p > 1.0 - eps { p = 1.0 - eps; }
    let log_odds = (p / (1.0 - p)).ln();
    
    data.value.push(log_odds * 0.1);

    if let Some(ref left) = node.left {
        let left_idx = flatten_tree(left, data);
        data.children_left[current_idx as usize] = left_idx;
    }
    
    if let Some(ref right) = node.right {
        let right_idx = flatten_tree(right, data);
        data.children_right[current_idx as usize] = right_idx;
    }

    current_idx
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut symbol = "BTCUSDT".to_string();
    let mut horizon = "SCALP".to_string();

    for i in 1..args.len() {
        if args[i] == "--symbol" && i + 1 < args.len() {
            symbol = args[i+1].clone();
        } else if args[i] == "--horizon" && i + 1 < args.len() {
            horizon = args[i+1].clone();
        }
    }

    println!("============================================================");
    println!("🌳 TRAINING NANO FOREST: {} | HORIZON: {}", symbol, horizon);
    println!("============================================================");
    
    let mut engine = StatefulEngine::new();
    let mut all_samples = Vec::new();
    let mut closes_history: Vec<f64> = Vec::new();

    let file_path = format!("data/{}_ticks.bin", symbol);
    
    let mut file = match std::fs::File::open(&file_path) {
        Ok(f) => f,
        Err(_) => {
            println!("❌ Failed to open {}. Make sure data is converted.", file_path);
            return;
        }
    };

    println!("📥 Loading historical data from {}...", file_path);
    let start_load = Instant::now();
    
    use std::io::Read;
    // Fast memory mapped or read-all approach
    let file_size = std::fs::metadata(&file_path).map(|m| m.len() as usize).unwrap_or(0);
    let mut buf = Vec::with_capacity(file_size);
    file.read_to_end(&mut buf).unwrap();
    let num_ticks = buf.len() / (4 * 8); // 4 f64s per tick: timestamp, price, volume, is_buyer_maker
    
    if num_ticks == 0 {
        println!("❌ No data found.");
        return;
    }
    
    let ptr = buf.as_ptr() as *const f64;
    // Tick binary layout: timestamps, prices, quantities, is_buyer_maker
    let timestamps = unsafe { std::slice::from_raw_parts(ptr.add(num_ticks * 0), num_ticks) };
    let closes = unsafe { std::slice::from_raw_parts(ptr.add(num_ticks * 1), num_ticks) };
    let volumes = unsafe { std::slice::from_raw_parts(ptr.add(num_ticks * 2), num_ticks) };
    let is_buyer_maker = unsafe { std::slice::from_raw_parts(ptr.add(num_ticks * 3), num_ticks) };
    
    for i in 0..num_ticks {
        let c = closes[i];
        let v = volumes[i];
        
        // Exact same pseudo_maker logic as unified_engine.rs
        let pseudo_maker = if i > 0 { c < closes[i - 1] } else { false };
        let (bid_qty, ask_qty) = if pseudo_maker { 
            (v * 0.95, v * 0.05) 
        } else { 
            (v * 0.05, v * 0.95) 
        };
        
        engine.process_tick(c, v);
        engine.update_trade_flow(v, pseudo_maker);
        
        let obi = if v > 0.0 { (bid_qty - ask_qty) / v } else { 0.0 };
        engine.update_macro_features(obi, 0.0, 0.0, 0); // Funding rate and DEX severity zeroed for training
        
        closes_history.push(c);
        if closes_history.len() > 50 {
            let features = engine.get_features();
            // Weight samples heavily if volatility (features[1]), momentum (features[5]) and acceleration (features[11]) are high
            let weight = 1.0 + (features[1] * features[5]).abs() as f32 + features[11].abs() as f32 * 100.0;
            all_samples.push(Sample {
                features: features.to_vec(),
                label: 0.0,
                weight,
            });
        }
    }
    
    if all_samples.is_empty() { return; }

    let (lookahead, target_pct, stop_loss_pct): (usize, f64, f64) = if horizon == "SWING" {
        println!("🏷️ Labeling data (Lookahead: 50,000 Ticks, Target: +1.5% growth)...");
        (50000, 0.015, -0.0075)
    } else {
        println!("🏷️ Labeling data (Lookahead: 2000 Ticks, Target: +0.05% growth)...");
        (2000, 0.0005, -0.00025) // Target +0.05% (5 bps), SL at -0.025% (2.5 bps)
    };
    
    let mut num_buy_labels = 0;
    let mut num_sell_labels = 0;
    let mut num_flat_labels = 0;
    
    for i in 0..(all_samples.len() - lookahead) {
        let entry_price = closes_history[i + 50]; // all_samples index aligns with closes_history offset by 50
        let future_slice = &closes_history[i + 50 + 1 .. i + 50 + lookahead];
        
        let max_price = future_slice.iter().cloned().fold(0.0, f64::max);
        let min_price = future_slice.iter().cloned().fold(f64::MAX, f64::min);
        
        let up_move = (max_price - entry_price) / entry_price;
        let down_move = (entry_price - min_price) / entry_price;
        
        let sl_mag = stop_loss_pct.abs();
        
        if up_move > target_pct && down_move < sl_mag { // hit target up, never hit stop loss down
            all_samples[i].label = 1.0;
            num_buy_labels += 1;
        } else if down_move > target_pct && up_move < sl_mag { // hit target down, never hit stop loss up
            all_samples[i].label = -1.0;
            num_sell_labels += 1;
        } else {
            all_samples[i].label = 0.0;
            num_flat_labels += 1;
        }
    }
    
    println!("📊 Label Distribution -> BUY: {}, SELL: {}, FLAT: {}", num_buy_labels, num_sell_labels, num_flat_labels);

    let total_valid_samples = all_samples.len() - lookahead;
    let train_size = (total_valid_samples as f64 * 0.7) as usize;
    
    let mut train_samples: Vec<&Sample> = Vec::with_capacity(train_size);
    let mut test_samples: Vec<&Sample> = Vec::with_capacity(total_valid_samples - train_size);
    
    for i in 0..train_size {
        train_samples.push(&all_samples[i]);
    }
    for i in train_size..total_valid_samples {
        test_samples.push(&all_samples[i]);
    }
    
    println!("🌳 Data loaded: {} total valid samples. Train: {}, Test: {}", total_valid_samples, train_samples.len(), test_samples.len());
    let train_start = Instant::now();
    
    let n_trees = 50;
    let max_depth = 8;
    
    let mut forest_data = NanoForestData {
        children_left: Vec::new(),
        children_right: Vec::new(),
        feature: Vec::new(),
        threshold: Vec::new(),
        value: Vec::new(),
        tree_offsets: Vec::new(),
        init_score: 0.0,
    };
    
    let mut ones = 0;
    for s in &train_samples {
        if s.label == 1.0 { ones += 1; }
    }
    let p = ones as f32 / train_samples.len() as f32;
    forest_data.init_score = (p / (1.0 - p)).ln();
    
    println!("🌲 Growing {} trees in parallel...", n_trees);
    let trees: Vec<Box<TreeNode>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for i in 0..n_trees {
            let samples_ref = &train_samples;
            handles.push(s.spawn(move || {
                let mut subset = Vec::with_capacity(samples_ref.len() / 2);
                for (idx, &sample) in samples_ref.iter().enumerate() {
                    if (idx + i * 997) % 2 == 0 {
                        subset.push(sample);
                    }
                }
                build_tree(&subset, 0, max_depth, 20)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    for root in trees {
        forest_data.tree_offsets.push(forest_data.value.len() as i32);
        flatten_tree(&root, &mut forest_data);
    }
    forest_data.tree_offsets.push(forest_data.value.len() as i32);
    
    let elapsed = train_start.elapsed();
    println!("✅ Forest trained in {:.2}s!", elapsed.as_secs_f64());
    
    println!("🧪 Evaluating Out-Of-Sample (OOS) Performance...");
    let forest = NanoForest::from_data(forest_data.clone());
    
    let mut oos_correct = 0;
    let mut oos_signals = 0;
    let mut oos_hits = 0;
    
    for s in &test_samples {
        let prob = forest.predict(&s.features);
        let pred_label = if prob > 0.5 { 1.0 } else { 0.0 };
        if pred_label == s.label {
            oos_correct += 1;
        }
        
        // Simulating GodEngineCore threshold (0.6 for Long)
        if prob > 0.6 {
            oos_signals += 1;
            if s.label == 1.0 {
                oos_hits += 1;
            }
        }
    }
    
    let oos_accuracy = oos_correct as f64 / test_samples.len() as f64;
    let oos_precision = if oos_signals > 0 { oos_hits as f64 / oos_signals as f64 } else { 0.0 };
    
    println!("📊 OOS Global Accuracy: {:.2}%", oos_accuracy * 100.0);
    println!("📊 OOS High Confidence Precision (>60% Prob): {:.2}% ({} signals)", oos_precision * 100.0, oos_signals);
    
    if oos_precision < 0.50 && oos_signals > 0 {
        println!("⚠️ WARNING: OOS Precision is below 50%. The model might be unprofitable.");
    }
    
    std::fs::create_dir_all("models").unwrap();
    let json_str = serde_json::to_string_pretty(&forest_data).unwrap();
    let out_path = format!("models/{}_{}.json", symbol, horizon);
    let tmp_out_path = format!("{}.tmp", out_path);
    std::fs::write(&tmp_out_path, json_str).unwrap();
    std::fs::rename(&tmp_out_path, &out_path).unwrap();
    println!("💾 Saved NanoForest to {}", out_path);
    
    // Auto compile to bin
    let bin_path = out_path.replace(".json", ".bin");
    let tmp_bin_path = format!("{}.tmp", bin_path);
    if let Ok(encoded) = bincode::serialize(&forest_data) {
        let _ = std::fs::write(&tmp_bin_path, encoded);
        std::fs::rename(&tmp_bin_path, &bin_path).unwrap();
        println!("🗜️ Compiled NanoForest to {}", bin_path);
    }

    println!("🚀 The ML Engine is now 100% Rust Native!");
}
