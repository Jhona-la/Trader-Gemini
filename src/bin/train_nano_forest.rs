use quantum_engine::stateful_engine::StatefulEngine;
use quantum_engine::ml_inference::NanoForestData;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

struct Sample {
    features: Vec<f32>,
    label: f32,
}

struct TreeNode {
    feature: i32,
    threshold: f32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    value: f32,
}

fn calculate_gini(labels: &[f32]) -> f32 {
    if labels.is_empty() { return 0.0; }
    let mut ones = 0;
    for &l in labels {
        if l == 1.0 { ones += 1; }
    }
    let p1 = ones as f32 / labels.len() as f32;
    let p0 = 1.0 - p1;
    1.0 - (p1 * p1 + p0 * p0)
}

fn split_data<'a>(samples: &'a [&'a Sample], feature_idx: usize, threshold: f32) -> (Vec<&'a Sample>, Vec<&'a Sample>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
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
    let mut ones = 0;
    for s in samples {
        if s.label == 1.0 { ones += 1; }
    }
    let value = if samples.is_empty() { 0.0 } else { ones as f32 / samples.len() as f32 };

    if depth >= max_depth || samples.len() < min_samples_split || ones == 0 || ones == samples.len() {
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
    
    // Simplistic split search
    for feat in 0..num_features {
        let step = if samples.len() > 1000 { samples.len() / 100 } else { 1 };
        for i in (0..samples.len()).step_by(step) {
            let threshold = samples[i].features[feat];
            let (left, right) = split_data(samples, feat, threshold);
            
            if left.is_empty() || right.is_empty() { continue; }

            let left_labels: Vec<f32> = left.iter().map(|s| s.label).collect();
            let right_labels: Vec<f32> = right.iter().map(|s| s.label).collect();

            let gini_left = calculate_gini(&left_labels);
            let gini_right = calculate_gini(&right_labels);

            let weight_left = left.len() as f32 / samples.len() as f32;
            let weight_right = right.len() as f32 / samples.len() as f32;

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
    let mut p = node.value;
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
    println!("============================================================");
    println!("🧠 RUST QUANTUM ML TRAINER (NATIVE NANO-FOREST)");
    println!("============================================================");
    
    let sym = "BTCUSDT";
    let file_path = format!("data/historical/{}_1m.csv", sym);
    
    let mut engine = StatefulEngine::new();
    let mut all_samples = Vec::new();
    let mut closes_history = Vec::new();

    println!("📥 Loading historical data from {}...", file_path);
    let start_load = Instant::now();
    
    if let Ok(file) = File::open(&file_path) {
        let reader = BufReader::new(file);
        for line in reader.lines().filter_map(|l| l.ok()).skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 6 {
                if let (Ok(c), Ok(v)) = (parts[4].parse::<f64>(), parts[5].parse::<f64>()) {
                    engine.process_tick(c, v);
                    closes_history.push(c);
                    if closes_history.len() > 50 {
                        let features = engine.get_features();
                        all_samples.push(Sample {
                            features: features.to_vec(),
                            label: 0.0,
                        });
                    }
                }
            }
        }
    } else {
        println!("❌ Failed to open data. Run data_loader first.");
        return;
    }
    
    if all_samples.is_empty() { return; }

    println!("🏷️ Labeling data (Lookahead: 15 mins, Target: +0.2% growth)...");
    let lookahead = 15;
    let target_pct = 0.002;
    
    for i in 0..(all_samples.len() - lookahead) {
        let current_close = closes_history[i + 50];
        let mut hit = 0.0;
        
        for j in 1..=lookahead {
            let future_close = closes_history[i + 50 + j];
            let change = (future_close - current_close) / current_close;
            if change >= target_pct {
                hit = 1.0;
                break;
            } else if change <= -target_pct * 0.5 {
                break;
            }
        }
        
        all_samples[i].label = hit;
    }

    let mut valid_samples: Vec<&Sample> = Vec::new();
    for i in 0..(all_samples.len() - lookahead) {
        valid_samples.push(&all_samples[i]);
    }
    
    println!("🌳 Data loaded: {} samples. Starting Forest Generation...", valid_samples.len());
    let train_start = Instant::now();
    
    let n_trees = 10;
    let max_depth = 5;
    
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
    for s in &valid_samples {
        if s.label == 1.0 { ones += 1; }
    }
    let p = ones as f32 / valid_samples.len() as f32;
    forest_data.init_score = (p / (1.0 - p)).ln();
    
    for i in 0..n_trees {
        println!("🌲 Growing Tree {}/{}...", i + 1, n_trees);
        
        let mut subset = Vec::with_capacity(valid_samples.len() / 2);
        for (idx, &sample) in valid_samples.iter().enumerate() {
            if (idx + i * 997) % 2 == 0 {
                subset.push(sample);
            }
        }
        
        let root = build_tree(&subset, 0, max_depth, 20);
        
        forest_data.tree_offsets.push(forest_data.value.len() as i32);
        flatten_tree(&root, &mut forest_data);
    }
    forest_data.tree_offsets.push(forest_data.value.len() as i32);
    
    let elapsed = train_start.elapsed();
    println!("✅ Forest trained in {:.2}s!", elapsed.as_secs_f64());
    
    std::fs::create_dir_all("models").unwrap();
    let json_str = serde_json::to_string_pretty(&forest_data).unwrap();
    std::fs::write("models/nano_forest.json", json_str).unwrap();
    println!("💾 Saved NanoForest to models/nano_forest.json");
    println!("🚀 The ML Engine is now 100% Rust Native!");
}
