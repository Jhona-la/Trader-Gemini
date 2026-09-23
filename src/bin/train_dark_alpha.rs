use dark_alpha_engine::DarkAlphaEngine;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

// Adam Optimizer state for a DenseLayer
struct AdamState {
    m_w: Vec<f64>,
    v_w: Vec<f64>,
    m_b: Vec<f64>,
    v_b: Vec<f64>,
}

impl AdamState {
    fn new(in_feat: usize, out_feat: usize) -> Self {
        Self {
            m_w: vec![0.0; in_feat * out_feat],
            v_w: vec![0.0; in_feat * out_feat],
            m_b: vec![0.0; out_feat],
            v_b: vec![0.0; out_feat],
        }
    }
}

struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

fn shuffle(indices: &mut [usize], prng: &mut XorShift) {
    let len = indices.len();
    if len < 2 {
        return;
    }
    for i in (1..len).rev() {
        let j = (prng.next() as usize) % (i + 1);
        indices.swap(i, j);
    }
}

fn main() {
    let symbol = std::env::args().nth(1).unwrap_or_else(|| "BTCUSDT".to_string());
    let input_csv = std::env::args().nth(2).unwrap_or_else(|| format!("data/{}_FEATURES.csv", symbol));

    println!("============================================================");
    println!("🧠 RUST NATIVE TRAINER: DARK ALPHA ENGINE");
    println!("============================================================");
    println!("📥 Loading {} for symbol {}...", input_csv, symbol);

    let file = match File::open(&input_csv) {
        Ok(f) => f,
        Err(_) => {
            println!("❌ Could not open CSV file. Run feature_exporter first.");
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut inputs = Vec::new();
    let mut targets = Vec::new(); // 1.0 for Long opportunity, 0.0 for Short/Flat

    let mut skip_header = true;
    for line in reader.lines() {
        if let Ok(l) = line {
            if skip_header {
                skip_header = false;
                continue;
            }
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() >= 26 {
                if let Ok(target_return) = parts[0].parse::<f64>() {
                    let max_cols = (parts.len() - 1).min(54);
                    let mut feat = vec![0.0; max_cols];
                    for i in 0..max_cols {
                        if let Ok(val) = parts[i + 1].parse::<f64>() {
                            if val.is_nan() || val.is_infinite() {
                                feat[i] = 0.0;
                            } else {
                                feat[i] = val;
                            }
                        } else {
                            feat[i] = 0.0;
                        }
                    }
                    // Target label directo del método Triple Barrier (1.0 = Long ganador, 0.0 = Short ganador)
                    let label = if target_return > 0.5 { 1.0 } else { 0.0 };
                    inputs.push(feat);
                    targets.push(label);
                }
            }
        }
    }

    let num_samples = inputs.len();
    if num_samples == 0 {
        println!("❌ No valid samples found.");
        return;
    }

    let long_count = targets.iter().filter(|&&t| t > 0.5).count();
    let short_count = num_samples - long_count;
    println!(
        "📊 Class Balance: {} Long (Bullish, {:.2}%) vs {} Short (Bearish, {:.2}%)",
        long_count,
        (long_count as f64 / num_samples as f64) * 100.0,
        short_count,
        (short_count as f64 / num_samples as f64) * 100.0
    );

    let input_dim = inputs[0].len();

    // --- ETL: COMPUTE MEAN AND STD DEV ---
    println!(
        "🧹 Calculating Means and StdDevs for Normalization ({} features)...",
        input_dim
    );
    let mut mean = vec![0.0; input_dim];
    for x in &inputs {
        for i in 0..input_dim {
            mean[i] += x[i];
        }
    }
    for i in 0..input_dim {
        mean[i] /= num_samples as f64;
    }

    let mut std_dev = vec![0.0; input_dim];
    for x in &inputs {
        for i in 0..input_dim {
            let diff = x[i] - mean[i];
            std_dev[i] += diff * diff;
        }
    }
    for i in 0..input_dim {
        std_dev[i] = (std_dev[i] / num_samples as f64).sqrt();
        if std_dev[i] < 1e-8 {
            std_dev[i] = 1e-8; // Prevent division by zero
        }
    }

    let scaler = dark_alpha_engine::Scaler::new(mean.clone(), std_dev.clone());

    // Normalize inputs
    for x in &mut inputs {
        scaler.scale(x);
    }

    println!(
        "✅ Loaded and Normalized {} valid samples ({}D). Starting Adam Optimization with L2 Regularization...",
        num_samples, input_dim
    );

    let mut engine = DarkAlphaEngine::new(input_dim, 64, 32);
    engine.scaler = Some(scaler);
    for i in 0..input_dim {
        engine.channel_normalizers[i].count = num_samples as f64;
        engine.channel_normalizers[i].mean = mean[i];
        let var = std_dev[i] * std_dev[i];
        engine.channel_normalizers[i].m2 = var * (num_samples as f64 - 1.0).max(1.0);
    }
    if !engine.per_coin_normalizers.is_empty() {
        for i in 0..input_dim {
            engine.per_coin_normalizers[0][i] = engine.channel_normalizers[i];
        }
    }

    // Initialize Adam States
    let mut adam1 = AdamState::new(input_dim, 64);
    let mut adam2 = AdamState::new(64, 32);
    let mut adam3 = AdamState::new(32, 1);

    let epochs = 8;
    let batch_size = 1024;
    let learning_rate = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    let l2_reg = 0.0001; // L2 weight decay to prevent weight explosion and sigmoid saturation

    let mut indices: Vec<usize> = (0..num_samples).collect();
    let mut prng = XorShift::new(123456789);

    let mut t = 0; // Adam time step

    let start_time = Instant::now();

    for epoch in 0..epochs {
        shuffle(&mut indices, &mut prng);
        let mut epoch_loss = 0.0;

        for batch_start in (0..num_samples).step_by(batch_size) {
            let end = (batch_start + batch_size).min(num_samples);
            let b_size = end - batch_start;

            // Gradients accumulation
            let mut g_w1 = vec![0.0; 64 * input_dim];
            let mut g_b1 = vec![0.0; 64];
            let mut g_w2 = vec![0.0; 32 * 64];
            let mut g_b2 = vec![0.0; 32];
            let mut g_w3 = vec![0.0; 32];
            let mut g_b3 = vec![0.0; 1];

            for b in 0..b_size {
                let idx = indices[batch_start + b];
                let x = &inputs[idx];
                let y = targets[idx];

                // --- FORWARD PASS ---
                let mut z1 = [0.0; 64];
                let mut a1 = [0.0; 64];
                for i in 0..64 {
                    let mut sum = engine.layer1.biases[i];
                    for j in 0..input_dim {
                        sum += engine.layer1.weights[i * input_dim + j] * x[j];
                    }
                    z1[i] = sum;
                    a1[i] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
                }

                let mut z2 = [0.0; 32];
                let mut a2 = [0.0; 32];
                for i in 0..32 {
                    let mut sum = engine.layer2.biases[i];
                    for j in 0..64 {
                        sum += engine.layer2.weights[i * 64 + j] * a1[j];
                    }
                    z2[i] = sum;
                    a2[i] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
                }

                let mut z3 = engine.layer3.biases[0];
                for j in 0..32 {
                    z3 += engine.layer3.weights[j] * a2[j];
                }
                let pred = 1.0 / (1.0 + (-z3.clamp(-700.0, 700.0)).exp());

                // Binary Cross Entropy Loss
                let loss =
                    -(y * (pred.max(1e-15)).ln() + (1.0 - y) * ((1.0 - pred).max(1e-15)).ln());
                epoch_loss += loss;

                // --- BACKWARD PASS ---
                let d_loss = pred - y; // BCE + Sigmoid derivative simplifies to (pred - y)

                // Layer 3 Gradients
                g_b3[0] += d_loss;
                let mut d_a2 = [0.0; 32];
                for j in 0..32 {
                    g_w3[j] += d_loss * a2[j];
                    d_a2[j] = d_loss * engine.layer3.weights[j];
                }

                // Layer 2 Gradients
                let mut d_z2 = [0.0; 32];
                for j in 0..32 {
                    d_z2[j] = if z2[j] > 0.0 { d_a2[j] } else { 0.0 };
                    g_b2[j] += d_z2[j];
                }

                let mut d_a1 = [0.0; 64];
                for i in 0..32 {
                    for j in 0..64 {
                        g_w2[i * 64 + j] += d_z2[i] * a1[j];
                        d_a1[j] += d_z2[i] * engine.layer2.weights[i * 64 + j];
                    }
                }

                // Layer 1 Gradients
                let mut d_z1 = [0.0; 64];
                for i in 0..64 {
                    d_z1[i] = if z1[i] > 0.0 { d_a1[i] } else { 0.0 };
                    g_b1[i] += d_z1[i];
                    for j in 0..input_dim {
                        g_w1[i * input_dim + j] += d_z1[i] * x[j];
                    }
                }
            }

            // --- ADAM UPDATE ---
            t += 1;
            let scale = 1.0 / b_size as f64;

            let apply_adam =
                |w: &mut Vec<f64>, g: &Vec<f64>, m: &mut Vec<f64>, v: &mut Vec<f64>| {
                    for i in 0..w.len() {
                        let grad = g[i] * scale + l2_reg * w[i];
                        m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
                        v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;

                        let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
                        let v_hat = v[i] / (1.0 - beta2.powi(t as i32));

                        w[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                    }
                };

            apply_adam(
                &mut engine.layer3.weights,
                &g_w3,
                &mut adam3.m_w,
                &mut adam3.v_w,
            );
            apply_adam(
                &mut engine.layer3.biases,
                &g_b3,
                &mut adam3.m_b,
                &mut adam3.v_b,
            );

            apply_adam(
                &mut engine.layer2.weights,
                &g_w2,
                &mut adam2.m_w,
                &mut adam2.v_w,
            );
            apply_adam(
                &mut engine.layer2.biases,
                &g_b2,
                &mut adam2.m_b,
                &mut adam2.v_b,
            );

            apply_adam(
                &mut engine.layer1.weights,
                &g_w1,
                &mut adam1.m_w,
                &mut adam1.v_w,
            );
            apply_adam(
                &mut engine.layer1.biases,
                &g_b1,
                &mut adam1.m_b,
                &mut adam1.v_b,
            );
        }

        println!(
            "Epoch {}/{} - BCE Loss: {:.6}",
            epoch + 1,
            epochs,
            epoch_loss / num_samples as f64
        );
    }

    println!(
        "⏱️ Training finished in {:.2}s",
        start_time.elapsed().as_secs_f64()
    );

    // Sanitizar números subnormales antes de guardar el modelo
    engine.sanitize_denormals();

    // FIX #1481: Creación de directorio y guardado resiliente de modelo JSON
    if let Err(e) = std::fs::create_dir_all("models") {
        println!("❌ Failed to create models directory: {}", e);
        return;
    }
    let out_path = format!("models/DarkAlpha_{}.json", symbol);
    match serde_json::to_string_pretty(&engine) {
        Ok(json_str) => {
            if let Err(e) = std::fs::write(&out_path, json_str) {
                println!("❌ Failed to write model file {}: {}", out_path, e);
            } else {
                println!("💾 Dark Alpha Model Saved: {}", out_path);
            }
        }
        Err(e) => {
            println!("❌ Failed to serialize Dark Alpha model: {}", e);
        }
    }
}
