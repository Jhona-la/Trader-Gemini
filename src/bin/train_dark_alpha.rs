use dark_alpha_engine::DarkAlphaEngine;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
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

const LEGACY_FEATURE_DIM: usize = 54;

#[derive(Debug)]
struct TrainingData {
    inputs: Vec<Vec<f64>>,
    targets: Vec<f64>,
}

/// Syntax/numeric validation of the legacy exporter contract only.
/// The name target_5m does NOT certify a clock horizon (FMT-195).
fn read_training_csv(reader: impl BufRead) -> Result<TrainingData, String> {
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or("missing CSV header")?
        .map_err(|e| format!("header I/O: {e}"))?;
    let expected = std::iter::once("target_5m".to_string())
        .chain((0..LEGACY_FEATURE_DIM).map(|i| format!("feature_{i}")))
        .collect::<Vec<_>>()
        .join(",");
    if header.trim_end_matches('\r') != expected {
        return Err(
            "unsupported CSV schema: require legacy target_5m,feature_0..feature_53".into(),
        );
    }
    let mut data = TrainingData {
        inputs: Vec::new(),
        targets: Vec::new(),
    };
    for (offset, line) in lines.enumerate() {
        let row_number = offset + 2;
        let line = line.map_err(|e| format!("CSV row {row_number}: I/O error: {e}"))?;
        let parts: Vec<_> = line.split(',').collect();
        if parts.len() != LEGACY_FEATURE_DIM + 1 {
            return Err(format!(
                "CSV row {row_number}: expected 55 columns, found {}",
                parts.len()
            ));
        }
        let number = |column: usize| -> Result<f64, String> {
            let value = parts[column]
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("CSV row {row_number}, column {column}: not numeric"))?;
            if !value.is_finite() {
                return Err(format!("CSV row {row_number}, column {column}: nonfinite"));
            }
            Ok(value)
        };
        let target = number(0)?;
        if target != 0.0 && target != 1.0 {
            return Err(format!(
                "CSV row {row_number}: target must be exactly 0 or 1"
            ));
        }
        let features = (1..=LEGACY_FEATURE_DIM)
            .map(number)
            .collect::<Result<Vec<_>, _>>()?;
        data.inputs.push(features);
        data.targets.push(target);
    }
    if data.inputs.is_empty() {
        return Err("CSV contains no observations".into());
    }
    Ok(data)
}

/// Fit on declared training rows. No claim of held-out validation here.
fn fit_scaler(inputs: &[Vec<f64>]) -> Result<dark_alpha_engine::Scaler, String> {
    if inputs.is_empty() {
        return Err("cannot fit scaler without data".into());
    }
    let dim = inputs[0].len();
    let mut mean = vec![0.0; dim];
    let mut m2 = vec![0.0; dim];
    for (row, x) in inputs.iter().enumerate() {
        if x.len() != dim || dim == 0 || x.iter().any(|v| !v.is_finite()) {
            return Err("invalid scaler training row".into());
        }
        let count = (row + 1) as f64;
        for i in 0..dim {
            let delta = x[i] - mean[i];
            mean[i] += delta / count;
            m2[i] += delta * (x[i] - mean[i]);
            if !mean[i].is_finite() || !m2[i].is_finite() || m2[i] < 0.0 {
                return Err("scaler fitting overflow".into());
            }
        }
    }
    // Retained serialization floor, not a learned noise scale.
    let std_dev = m2
        .iter()
        .map(|v| (v / inputs.len() as f64).sqrt().max(1e-8))
        .collect();
    Ok(dark_alpha_engine::Scaler::new(mean, std_dev))
}

fn research_candidate_path(symbol: &str, run_id: u128) -> Result<std::path::PathBuf, String> {
    if symbol.is_empty()
        || !symbol
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
    {
        return Err("symbol must contain uppercase ASCII letters/digits only".into());
    }
    // Outside the operational models namespace: legacy CSV has no temporal
    // provenance, so this binary cannot authorize live model publication.
    Ok(std::path::Path::new("artifacts/training")
        .join(format!("DarkAlpha_{symbol}_CANDIDATE_{run_id}.json")))
}

fn main() -> Result<(), String> {
    let symbol = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "BTCUSDT".to_string());
    let input_csv = std::env::args()
        .nth(2)
        .unwrap_or_else(|| format!("data/{}_FEATURES.csv", symbol));
    let run_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let out_path = research_candidate_path(&symbol, run_id)?;

    println!("============================================================");
    println!("🧠 RUST NATIVE TRAINER: DARK ALPHA ENGINE");
    println!("============================================================");
    println!("📥 Loading {} for symbol {}...", input_csv, symbol);

    let file = File::open(&input_csv).map_err(|e| format!("cannot open training CSV: {e}"))?;
    let TrainingData {
        mut inputs,
        targets,
    } = read_training_csv(BufReader::new(file))?;
    let num_samples = inputs.len();

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

    let scaler = fit_scaler(&inputs)?;
    let mean = scaler.mean.clone();
    let std_dev = scaler.std_dev.clone();
    for x in &mut inputs {
        scaler.scale_checked(x).map_err(str::to_string)?;
    }
    println!(
        "   {} rows / {} features; TRAINING evidence only, no independent evaluation",
        num_samples, input_dim
    );
    let mut engine = DarkAlphaEngine::new(input_dim, 64, 32);
    engine.scaler = Some(scaler);
    for i in 0..input_dim {
        engine.channel_normalizers[i].count = num_samples as f64;
        engine.channel_normalizers[i].mean = mean[i];
        engine.channel_normalizers[i].m2 = std_dev[i].powi(2) * (num_samples as f64 - 1.0).max(1.0);
    }
    // No claim that numeric asset slot zero identifies this symbol.
    // Scaler is the fixed coordinate system used by this trainer.

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

    // No implicit pruning and no operational publication. The current CSV
    // cannot support interval purging or an auditable independent test.
    engine.validate()?;
    engine.freeze();
    let json = serde_json::to_vec_pretty(&engine).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(out_path.parent().unwrap()).map_err(|e| e.to_string())?;
    let mut destination = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out_path)
        .map_err(|e| format!("candidate create: {e}"))?;
    destination
        .write_all(&json)
        .map_err(|e| format!("candidate write: {e}"))?;
    destination
        .sync_all()
        .map_err(|e| format!("candidate sync: {e}"))?;
    println!(
        "Research candidate only: {}. NOT approved for trading.",
        out_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use std::io::Cursor;

    fn header() -> String {
        std::iter::once("target_5m".to_string())
            .chain((0..54).map(|i| format!("feature_{i}")))
            .collect::<Vec<_>>()
            .join(",")
    }
    fn row(target: &str, width: usize) -> String {
        std::iter::once(target.to_string())
            .chain((0..width).map(|i| i.to_string()))
            .collect::<Vec<_>>()
            .join(",")
    }
    #[test]
    fn csv_retains_numeric_values_and_labels_for_exact_legacy_schema() {
        let csv = format!("{}\r\n{}\r\n{}\r\n", header(), row("1", 54), row("0", 54));
        let data = read_training_csv(Cursor::new(csv)).unwrap();
        assert_eq!(data.targets, vec![1.0, 0.0]);
        assert_eq!(data.inputs[0][53], 53.0);
    }
    #[test]
    fn mixed_short_long_and_extra_columns_are_rejected() {
        for width in [25, 53, 55] {
            let csv = format!("{}\n{}\n{}", header(), row("1", 54), row("0", width));
            assert!(read_training_csv(Cursor::new(csv))
                .unwrap_err()
                .contains("row 3"));
        }
    }
    #[test]
    fn nonfinite_ambiguous_or_out_of_range_targets_are_rejected() {
        for target in ["NaN", "inf", "-inf", "0.5", "-1", "2", "bad"] {
            let csv = format!("{}\n{}", header(), row(target, 54));
            assert!(read_training_csv(Cursor::new(csv)).is_err(), "{target}");
        }
    }
    #[test]
    fn invalid_features_are_not_silently_imputed() {
        for value in ["NaN", "inf", "-inf", "", "broken"] {
            let mut parts = vec!["1"; 55];
            parts[17] = value;
            let csv = format!("{}\n{}", header(), parts.join(","));
            assert!(read_training_csv(Cursor::new(csv))
                .unwrap_err()
                .contains("column 17"));
        }
    }
    #[test]
    fn schema_and_empty_data_are_rejected() {
        for csv in [
            "".to_string(),
            header(),
            format!("wrong_header\n{}", row("1", 54)),
        ] {
            assert!(read_training_csv(Cursor::new(csv)).is_err());
        }
    }
    #[test]
    fn input_read_errors_are_not_silent_skips() {
        let bytes = [format!("{}\n", header()).as_bytes(), &[0xff, 0xfe, b'\n']].concat();
        assert!(read_training_csv(Cursor::new(bytes))
            .unwrap_err()
            .contains("I/O error"));
    }
    #[test]
    fn scaler_matches_population_moments_and_rejects_overflow() {
        let s = fit_scaler(&[vec![1.0, 7.0], vec![3.0, 7.0]]).unwrap();
        assert_eq!(s.mean, vec![2.0, 7.0]);
        assert_eq!(s.std_dev, vec![1.0, 1e-8]);
        assert!(fit_scaler(&[vec![f64::MAX], vec![-f64::MAX]]).is_err());
        assert!(fit_scaler(&[vec![1.0], vec![]]).is_err());
    }
    #[test]
    fn publication_path_is_versioned_outside_operational_namespace() {
        let a = research_candidate_path("BTCUSDT", 1).unwrap();
        let b = research_candidate_path("BTCUSDT", 2).unwrap();
        assert_ne!(a, b);
        assert!(a.starts_with("artifacts/training"));
        assert!(!a.starts_with("models"));
        for symbol in ["", "../BTCUSDT", "btcUSDT", "BTC/USDT", "BTC:USDT"] {
            assert!(research_candidate_path(symbol, 1).is_err());
        }
    }
}
