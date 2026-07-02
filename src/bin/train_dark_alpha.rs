use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use dark_alpha_engine::{DarkAlphaEngine, DenseLayer};

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
    if len < 2 { return; }
    for i in (1..len).rev() {
        let j = (prng.next() as usize) % (i + 1);
        indices.swap(i, j);
    }
}

fn main() {
    let symbol = "BTCUSDT";
    let input_csv = format!("data/{}_FEATURES.csv", symbol);
    
    println!("============================================================");
    println!("🧠 RUST NATIVE TRAINER: DARK ALPHA ENGINE");
    println!("============================================================");
    println!("📥 Loading {}...", input_csv);
    
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
            if skip_header { skip_header = false; continue; }
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() == 26 {
                if let Ok(target_return) = parts[0].parse::<f64>() {
                    let mut feat = vec![0.0; 54];
                    let mut valid = true;
                    for i in 0..25 {
                        if let Ok(val) = parts[i+1].parse::<f64>() {
                            feat[i] = val;
                        } else {
                            valid = false;
                        }
                    }
                    if valid {
                        // Swing label: If return in 5 periods > 0.05% -> 1.0 (Long)
                        // If return < -0.05% -> 0.0 (Short/Avoid)
                        let label = if target_return > 0.0005 { 1.0 } else { 0.0 };
                        inputs.push(feat);
                        targets.push(label);
                    }
                }
            }
        }
    }
    
    let num_samples = inputs.len();
    if num_samples == 0 {
        println!("❌ No valid samples found.");
        return;
    }
    println!("✅ Loaded {} samples. Starting Adam Optimization...", num_samples);
    
    let mut engine = DarkAlphaEngine::new(54, 64, 32);
    
    // Initialize Adam States
    let mut adam1 = AdamState::new(54, 64);
    let mut adam2 = AdamState::new(64, 32);
    let mut adam3 = AdamState::new(32, 1);
    
    let epochs = 50;
    let batch_size = 256;
    let learning_rate = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    
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
            let mut g_w1 = vec![0.0; 64 * 54];
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
                let mut z1 = vec![0.0; 64];
                let mut a1 = vec![0.0; 64];
                for i in 0..64 {
                    let mut sum = engine.layer1.biases[i];
                    for j in 0..54 { sum += engine.layer1.weights[i * 54 + j] * x[j]; }
                    z1[i] = sum;
                    a1[i] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
                }
                
                let mut z2 = vec![0.0; 32];
                let mut a2 = vec![0.0; 32];
                for i in 0..32 {
                    let mut sum = engine.layer2.biases[i];
                    for j in 0..64 { sum += engine.layer2.weights[i * 64 + j] * a1[j]; }
                    z2[i] = sum;
                    a2[i] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
                }
                
                let mut z3 = engine.layer3.biases[0];
                for j in 0..32 { z3 += engine.layer3.weights[j] * a2[j]; }
                
                let clamped: f64 = z3.clamp(-15.0, 15.0);
                let a3: f64 = 1.0 / (1.0 + (-clamped).exp()); // Sigmoid
                
                // BCE Loss: - (y * log(a3) + (1-y) * log(1-a3))
                let a3_clamped: f64 = a3.clamp(1e-7, 1.0 - 1e-7);
                epoch_loss -= y * a3_clamped.ln() + (1.0 - y) * (1.0 - a3_clamped).ln();
                
                // --- BACKWARD PASS ---
                // dL/dz3 for BCE + Sigmoid is just (a3 - y)
                let d_z3 = a3 - y;
                
                // Layer 3 Gradients
                g_b3[0] += d_z3;
                for j in 0..32 {
                    g_w3[j] += d_z3 * a2[j];
                }
                
                // Backprop to Layer 2
                let mut d_a2 = vec![0.0; 32];
                for j in 0..32 {
                    d_a2[j] = d_z3 * engine.layer3.weights[j];
                }
                let mut d_z2 = vec![0.0; 32];
                for j in 0..32 {
                    d_z2[j] = if z2[j] > 0.0 { d_a2[j] } else { 0.0 }; // ReLU derivative
                }
                
                // Layer 2 Gradients
                for i in 0..32 {
                    g_b2[i] += d_z2[i];
                    for j in 0..64 {
                        g_w2[i * 64 + j] += d_z2[i] * a1[j];
                    }
                }
                
                // Backprop to Layer 1
                let mut d_a1 = vec![0.0; 64];
                for i in 0..32 {
                    for j in 0..64 {
                        d_a1[j] += d_z2[i] * engine.layer2.weights[i * 64 + j];
                    }
                }
                let mut d_z1 = vec![0.0; 64];
                for j in 0..64 {
                    d_z1[j] = if z1[j] > 0.0 { d_a1[j] } else { 0.0 };
                }
                
                // Layer 1 Gradients
                for i in 0..64 {
                    g_b1[i] += d_z1[i];
                    for j in 0..54 {
                        g_w1[i * 54 + j] += d_z1[i] * x[j];
                    }
                }
            }
            
            // --- ADAM UPDATE ---
            t += 1;
            let scale = 1.0 / b_size as f64;
            
            let mut apply_adam = |w: &mut Vec<f64>, g: &Vec<f64>, m: &mut Vec<f64>, v: &mut Vec<f64>| {
                for i in 0..w.len() {
                    let grad = g[i] * scale;
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
                    
                    let m_hat = m[i] / (1.0 - beta1.powi(t as i32));
                    let v_hat = v[i] / (1.0 - beta2.powi(t as i32));
                    
                    w[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                }
            };
            
            apply_adam(&mut engine.layer3.weights, &g_w3, &mut adam3.m_w, &mut adam3.v_w);
            apply_adam(&mut engine.layer3.biases, &g_b3, &mut adam3.m_b, &mut adam3.v_b);
            
            apply_adam(&mut engine.layer2.weights, &g_w2, &mut adam2.m_w, &mut adam2.v_w);
            apply_adam(&mut engine.layer2.biases, &g_b2, &mut adam2.m_b, &mut adam2.v_b);
            
            apply_adam(&mut engine.layer1.weights, &g_w1, &mut adam1.m_w, &mut adam1.v_w);
            apply_adam(&mut engine.layer1.biases, &g_b1, &mut adam1.m_b, &mut adam1.v_b);
        }
        
        println!("Epoch {}/{} - BCE Loss: {:.6}", epoch + 1, epochs, epoch_loss / num_samples as f64);
    }
    
    println!("⏱️ Training finished in {:.2}s", start_time.elapsed().as_secs_f64());
    
    // Save model
    std::fs::create_dir_all("models").unwrap();
    let out_path = format!("models/DarkAlpha_{}.json", symbol);
    let json_str = serde_json::to_string_pretty(&engine).unwrap();
    std::fs::write(&out_path, json_str).unwrap();
    
    println!("💾 Dark Alpha Model Saved: {}", out_path);
}
