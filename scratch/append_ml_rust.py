import os

math_kernels_path = 'core/rust_engine/src/math_kernels.rs'
with open(math_kernels_path, 'a', encoding='utf-8') as f:
    f.write('''
// =====================================================================
// MACHINE LEARNING INFERENCE KERNELS (Nano-Latency)
// =====================================================================

pub fn predict_rf(
    x: &[f64],
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    value: &[f64],
    tree_offsets: &[i64],
) -> f64 {
    let n_trees = tree_offsets.len().saturating_sub(1);
    if n_trees == 0 { return 0.0; }
    let mut total_prob = 0.0;

    for i in 0..n_trees {
        let mut node = tree_offsets[i] as usize;
        while children_left[node] != -1 {
            let f_idx = feature[node] as usize;
            if x[f_idx] <= threshold[node] {
                node = children_left[node] as usize;
            } else {
                node = children_right[node] as usize;
            }
        }
        total_prob += value[node];
    }
    total_prob / (n_trees as f64)
}

pub fn predict_gb(
    x: &[f64],
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    value: &[f64],
    tree_offsets: &[i64],
    init_score: f64,
    learning_rate: f64,
) -> f64 {
    let n_trees = tree_offsets.len().saturating_sub(1);
    let mut score = init_score;

    for i in 0..n_trees {
        let mut node = tree_offsets[i] as usize;
        while children_left[node] != -1 {
            let f_idx = feature[node] as usize;
            if x[f_idx] <= threshold[node] {
                node = children_left[node] as usize;
            } else {
                node = children_right[node] as usize;
            }
        }
        score += learning_rate * value[node];
    }

    // Sigmoid
    if score >= 0.0 {
        1.0 / (1.0 + (-score).exp())
    } else {
        let exp_s = score.exp();
        exp_s / (1.0 + exp_s)
    }
}

pub fn fused_compute_step(
    closes: &[f64],
    volumes: &[f64],
    portfolio_state: &[f64; 3], // [has_pos, pnl_norm, dur_norm]
    gene_params: &[f64; 2],     // [sl_norm, tp_norm]
    brain_weights: &[f64; 100], // 25 * 4 = 100 flattened
    l2_state: &[f64; 2],        // [ofi, microprice_divergence]
    window: usize,
    out_scores: &mut [f64; 4]
) {
    let n = closes.len();
    if n < 30 {
        out_scores.fill(0.0);
        return;
    }

    let mut state_tensor = [0.0f64; 25];

    // 1A. Market Data (20 Features)
    // Returns (5)
    for i in 0..window {
        let idx = n - window + i;
        let val = (closes[idx] - closes[idx - 1]) / closes[idx - 1];
        state_tensor[i] = val;
    }

    // Volatility (5)
    let mut vol_sum = 0.0;
    for i in (n - 20)..n {
        vol_sum += volumes[i];
    }
    let mut mean_vol = vol_sum / 20.0;
    if mean_vol < 1e-8 {
        mean_vol = 1.0;
    }

    for i in 0..window {
        let idx = n - window + i;
        state_tensor[5 + i] = volumes[idx] / mean_vol;
    }

    // Momentum / Custom (5)
    for i in 0..window {
        let idx = n - window + i;
        let mom = if idx >= 2 {
            (closes[idx] / closes[idx - 2]) - 1.0
        } else {
            0.0
        };
        state_tensor[10 + i] = mom;
        state_tensor[15 + i] = 0.0;
    }

    // Inject L2 Data
    state_tensor[18] = l2_state[0];
    state_tensor[19] = l2_state[1];

    // 2. Add Portfolio & Gene (5 Features)
    state_tensor[20] = portfolio_state[0];
    state_tensor[21] = portfolio_state[1];
    state_tensor[22] = portfolio_state[2];
    state_tensor[23] = gene_params[0];
    state_tensor[24] = gene_params[1];

    // 3. Neural Inference Dot Product
    for act in 0..4 {
        let mut score = 0.0;
        let base_idx = act * 25;
        for j in 0..25 {
            score += state_tensor[j] * brain_weights[base_idx + j];
        }
        out_scores[act] = score;
    }
}
''')

lib_path = 'core/rust_engine/src/lib.rs'
with open(lib_path, 'a', encoding='utf-8') as f:
    f.write('''
// =====================================================================
// FFI C-ABI EXPORTS: MACHINE LEARNING INFERENCE
// =====================================================================

#[no_mangle]
pub unsafe extern "C" fn ffi_predict_rf(
    x_ptr: *const f64,
    x_len: usize,
    cl_ptr: *const i64,
    cr_ptr: *const i64,
    feat_ptr: *const i64,
    thresh_ptr: *const f64,
    val_ptr: *const f64,
    nodes_len: usize,
    to_ptr: *const i64,
    to_len: usize,
) -> f64 {
    let x = std::slice::from_raw_parts(x_ptr, x_len);
    let cl = std::slice::from_raw_parts(cl_ptr, nodes_len);
    let cr = std::slice::from_raw_parts(cr_ptr, nodes_len);
    let feat = std::slice::from_raw_parts(feat_ptr, nodes_len);
    let thresh = std::slice::from_raw_parts(thresh_ptr, nodes_len);
    let val = std::slice::from_raw_parts(val_ptr, nodes_len);
    let to = std::slice::from_raw_parts(to_ptr, to_len);

    math_kernels::predict_rf(x, cl, cr, feat, thresh, val, to)
}

#[no_mangle]
pub unsafe extern "C" fn ffi_predict_gb(
    x_ptr: *const f64,
    x_len: usize,
    cl_ptr: *const i64,
    cr_ptr: *const i64,
    feat_ptr: *const i64,
    thresh_ptr: *const f64,
    val_ptr: *const f64,
    nodes_len: usize,
    to_ptr: *const i64,
    to_len: usize,
    init_score: f64,
    learning_rate: f64,
) -> f64 {
    let x = std::slice::from_raw_parts(x_ptr, x_len);
    let cl = std::slice::from_raw_parts(cl_ptr, nodes_len);
    let cr = std::slice::from_raw_parts(cr_ptr, nodes_len);
    let feat = std::slice::from_raw_parts(feat_ptr, nodes_len);
    let thresh = std::slice::from_raw_parts(thresh_ptr, nodes_len);
    let val = std::slice::from_raw_parts(val_ptr, nodes_len);
    let to = std::slice::from_raw_parts(to_ptr, to_len);

    math_kernels::predict_gb(x, cl, cr, feat, thresh, val, to, init_score, learning_rate)
}

#[no_mangle]
pub unsafe extern "C" fn ffi_fused_compute_step(
    closes_ptr: *const f64,
    closes_len: usize,
    volumes_ptr: *const f64,
    portfolio_ptr: *const f64, // len 3
    gene_ptr: *const f64,      // len 2
    brain_ptr: *const f64,     // len 100
    l2_ptr: *const f64,        // len 2
    window: usize,
    out_ptr: *mut f64          // len 4
) {
    let closes = std::slice::from_raw_parts(closes_ptr, closes_len);
    let volumes = std::slice::from_raw_parts(volumes_ptr, closes_len);
    let port = &*(portfolio_ptr as *const [f64; 3]);
    let gene = &*(gene_ptr as *const [f64; 2]);
    let brain = &*(brain_ptr as *const [f64; 100]);
    let l2 = &*(l2_ptr as *const [f64; 2]);
    let out = &mut *(out_ptr as *mut [f64; 4]);

    math_kernels::fused_compute_step(closes, volumes, port, gene, brain, l2, window, out);
}
''')
