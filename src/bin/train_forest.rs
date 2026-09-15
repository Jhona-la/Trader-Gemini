//! ENTRENADOR RUST DEL FOREST `_SCALP` (B2.2) — Gradient Boosting.
//!
//! POR QUÉ: los models/*_SCALP fueron entrenados por el pipeline Python
//! archivado (directriz: cero Python) y llevan semanas congelados — las
//! predicciones cuantizadas (0.337/0.476 constantes) congelan al SA.
//! `NanoForest::predict` es ESPACIO-LOGIT (init_score + Σárboles →
//! sigmoid): es gradient boosting, no promedio de forest. Este entrenador
//! replica ese contrato EXACTO.
//!
//! PARIDAD 1:1 con inferencia: los features salen del MISMO
//! `StatefulEngine::get_swing_features()` que alimenta al motor en vivo
//! (misma secuencia process_tick/update_trade_flow/update_ofi que
//! feature_exporter). Etiquetas: triple-barrera (López de Prado) con los
//! pisos institucionales TP 0.36% / SL 0.18%; los neutros se DESCARTAN
//! (promediarlos hacia 0.5 es una causa documentada de forests '~0.5').
//!
//! HONESTIDAD (F4.2): split TEMPORAL 80/20 + early stopping por logloss de
//! validación + GATE — el modelo sólo se guarda si mejora la logloss de
//! validación del baseline (tasa base constante). Sin gate: jamás
//! sobrescribir el modelo vivo con ruido.
//!
//! Uso: train_forest BTCUSDT [--in data/BTCUSDT_SEP26.bin] [--max-samples 400000]
//!      [--horizon 500] [--trees 300] [--lr 0.1] [--depth 5] [--promote]

use god_engine_core::ml_inference::NanoForestData;
use god_engine_core::stateful_engine::StatefulEngine;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::fs::File;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct BinTick {
    ts: u64,
    bid: f64,
    ask: f64,
    bq: f64,
    aq: f64,
}

// ── GBDT núcleo ──────────────────────────────────────────────────────────

struct TreeNode {
    feature: i32,
    threshold: f32,
    left: i32,
    right: i32,
    value: f32,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).clamp(-50.0, 50.0).exp())
}

/// Un árbol de regresión sobre gradientes (ganancia estilo XGBoost con
/// grad/hess). Split por cuantiles con submuestreo de filas del nodo.
fn build_tree(
    data: &mut Vec<(Vec<f32>, f64, f64)>, // (features, grad, hess) por muestra IN PLACE
    idx: &mut Vec<usize>,                  // índices del nodo (se particiona in situ)
    depth: u32,
    max_depth: u32,
    min_child: usize,
    lambda: f64,
    feat_subset: &[usize],
    n_quantiles: usize,
    rng: &mut StdRng,
    nodes: &mut Vec<TreeNode>,
) {
    let g_sum: f64 = idx.iter().map(|&i| data[i].1).sum();
    let h_sum: f64 = idx.iter().map(|&i| data[i].2).sum();
    let leaf_value = (g_sum / (h_sum + lambda)) as f32;
    if depth >= max_depth || idx.len() < 2 * min_child {
        nodes.push(TreeNode { feature: -1, threshold: 0.0, left: -1, right: -1, value: leaf_value });
        return;
    }
    // Submuestreo de filas para buscar splits (hasta 2048): velocidad sin
    // perder orden de magnitud de la ganancia.
    let probe: Vec<usize> = if idx.len() > 2048 {
        let mut p: Vec<usize> = Vec::with_capacity(2048);
        for _ in 0..2048 {
            p.push(idx[(rng.random::<f64>() * idx.len() as f64) as usize]);
        }
        p
    } else {
        idx.clone()
    };
    let parent_score = g_sum * g_sum / (h_sum + lambda + 1e-12);
    let mut best = (f64::MIN, 0usize, 0.0f32); // (ganancia, feature, threshold)
    for &f in feat_subset {
        let mut vals: Vec<f32> = probe.iter().map(|&i| data[i].0[f]).collect();
        if vals.windows(2).all(|w| w[0] == w[1]) {
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Umbrales candidatos por cuantiles (excluye extremos)
        for q in 1..n_quantiles {
            let thr = vals[q * vals.len() / n_quantiles];
            let mut gl = 0.0f64;
            let mut hl = 0.0f64;
            for &i in &probe {
                if data[i].0[f] <= thr {
                    gl += data[i].1;
                    hl += data[i].2;
                }
            }
            let gr = g_sum - gl;
            let hr = h_sum - hl;
            let nl = probe.iter().filter(|&&i| data[i].0[f] <= thr).count();
            let nr = probe.len() - nl;
            if nl < min_child || nr < min_child {
                continue;
            }
            let gain = gl * gl / (hl + lambda + 1e-12) + gr * gr / (hr + lambda + 1e-12)
                - parent_score;
            if gain > best.0 {
                best = (gain, f, thr);
            }
        }
    }
    if best.0 <= 1e-9 {
        nodes.push(TreeNode { feature: -1, threshold: 0.0, left: -1, right: -1, value: leaf_value });
        return;
    }
    let (f, thr) = (best.1, best.2);
    let mut left_idx: Vec<usize> = Vec::with_capacity(idx.len() / 2);
    let mut right_idx: Vec<usize> = Vec::with_capacity(idx.len() / 2);
    for &i in idx.iter() {
        if data[i].0[f] <= thr {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }
    let me = nodes.len();
    nodes.push(TreeNode { feature: f as i32, threshold: thr, left: 0, right: 0, value: 0.0 });
    // DFS pre-orden: subárbol izquierdo en [me+1, …]; la raíz del derecho
    // se captura ANTES de construirlo (nodes.len()-1 tras el derecho sería
    // su última hoja, no su raíz).
    build_tree(data, &mut left_idx, depth + 1, max_depth, min_child, lambda, feat_subset,
               n_quantiles, rng, nodes);
    nodes[me].left = (me + 1) as i32;
    let right_root = nodes.len();
    build_tree(data, &mut right_idx, depth + 1, max_depth, min_child, lambda, feat_subset,
               n_quantiles, rng, nodes);
    nodes[me].right = right_root as i32;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let symbol = if args.len() > 1 && !args[1].starts_with('-') {
        args[1].clone()
    } else {
        eprintln!("Uso: train_forest <SYMBOL> [--in FILE] [--max-samples N] ...");
        std::process::exit(1);
    };
    let arg = |name: &str, dflt: &str| -> String {
        args.iter().position(|a| a == name)
            .and_then(|p| args.get(p + 1))
            .map(|s| s.to_string())
            .unwrap_or_else(|| dflt.to_string())
    };
    let default_in = format!("data/{}_SEP26.bin", symbol);
    let in_path = arg("--in", &default_in);
    let max_samples: usize = arg("--max-samples", "200000").parse().unwrap();
    // HORIZONTE DE RELOJ, no de ticks: en datos densos (~75ms/tick) 500
    // ticks ≈ 2s de mercado y las barreras 0.36/0.18% jamás se tocan
    // (hallazgo real: 99.8% neutros, entrenamiento abortado). 5 minutos
    // cubre la escala de los pisos institucionales; stride 50s = solape 6x.
    let horizon_ms: u64 = arg("--horizon-ms", "300000").parse().unwrap();
    let stride_ms: u64 = arg("--stride-ms", "50000").parse().unwrap();
    let n_rounds: usize = arg("--trees", "300").parse().unwrap();
    let lr: f64 = arg("--lr", "0.1").parse().unwrap();
    let max_depth: u32 = arg("--depth", "5").parse().unwrap();
    let min_child: usize = arg("--min-child", "40").parse().unwrap();
    let lambda: f64 = arg("--lambda", "1.0").parse().unwrap();
    let patience: usize = arg("--patience", "40").parse().unwrap();
    let promote = args.iter().any(|a| a == "--promote");

    println!("🌲 [TRAIN-FOREST] {} ← {}", symbol, in_path);
    println!("   muestras≤{} horizonte={}ms stride={}ms árboles≤{} lr={} depth={} λ={}",
             max_samples, horizon_ms, stride_ms, n_rounds, lr, max_depth, lambda);

    // ── 1. Cargar ticks ──────────────────────────────────────────────────
    let file = File::open(&in_path).unwrap_or_else(|e| {
        eprintln!("❌ no pude abrir {}: {}", in_path, e);
        std::process::exit(1);
    });
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.unwrap();
    let header_off = if mmap.len() >= 8 && &mmap[..8] == b"TGMTICK1" { 8usize } else { 0 };
    let sz = std::mem::size_of::<BinTick>();
    let n_total = (mmap.len() - header_off) / sz;
    if n_total < 50_000 {
        eprintln!("❌ datos insuficientes: {} ticks", n_total);
        std::process::exit(1);
    }
    let ptr = unsafe { mmap.as_ptr().add(header_off) } as *const BinTick;
    let raw = unsafe { std::slice::from_raw_parts(ptr, n_total) };
    println!("   {} ticks crudos", n_total);

    // ── 2. Features + etiquetas (misma secuencia que la inferencia) ──────
    let last_ts = raw[n_total - 1].ts;
    // El stride se adapta si el archivo cubriera tan poco calendario que
    // las muestras no alczan max_samples (nunca menor a 1s de separación).
    let span_ms = last_ts.saturating_sub(raw[0].ts);
    let stride_ms_eff = if span_ms / stride_ms.saturating_sub(0).max(1) < (max_samples as u64) {
        (span_ms / (max_samples as u64)).max(1_000)
    } else {
        stride_ms
    };
    println!("   span {:.1} días · stride efectivo {}ms", span_ms as f64 / 86_400_000.0, stride_ms_eff);
    let mut engine = StatefulEngine::new();
    let mut feats: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();
    let mut neutrals = 0usize;
    let tp_pct = 0.0036;
    let sl_pct = 0.0018;
    let mut next_sample_ts: u64 = 0;
    let mut warmup = 0usize;
    for i in 0..n_total {
        let t = &raw[i];
        if t.bid <= 0.0 || t.ask <= 0.0 || t.bid > t.ask || t.ts == 0 {
            continue;
        }
        let mid = (t.bid + t.ask) / 2.0;
        let vol = t.bq + t.aq;
        let pseudo_maker = t.bq > t.aq;
        engine.process_tick(mid, vol, t.ts);
        engine.update_trade_flow(vol, pseudo_maker);
        let _ = engine.update_ofi(t.bid, t.ask, t.bq, t.aq);
        warmup += 1;
        if warmup >= 100 && t.ts >= next_sample_ts && t.ts + horizon_ms <= last_ts {
            next_sample_ts = t.ts + stride_ms_eff;
            let sf = engine.get_swing_features();
            if sf.iter().all(|f| f.is_finite()) {
                // Triple barrera por TIEMPO DE RELOJ dentro de horizon_ms
                let deadline = t.ts + horizon_ms;
                let long_tp = mid * (1.0 + tp_pct);
                let long_sl = mid * (1.0 - sl_pct);
                let mut label = 0.5f64;
                'barrier: for f in (i + 1)..n_total {
                    let ft = &raw[f];
                    if ft.ts > deadline {
                        break 'barrier;
                    }
                    if ft.bid <= 0.0 || ft.ask <= 0.0 {
                        continue;
                    }
                    let fut_mid = (ft.bid + ft.ask) / 2.0;
                    if fut_mid <= long_sl {
                        label = 0.0;
                        break 'barrier;
                    }
                    if fut_mid >= mid * (1.0 + sl_pct) {
                        label = 1.0; // la hipótesis corta fracasó primero
                        break 'barrier;
                    }
                    if fut_mid >= long_tp {
                        label = 1.0;
                        break 'barrier;
                    }
                    if fut_mid <= mid * (1.0 - tp_pct) {
                        label = 0.0;
                        break 'barrier;
                    }
                }
                if (label - 0.5).abs() < 1e-9 {
                    neutrals += 1;
                } else {
                    feats.push(sf.to_vec());
                    labels.push(label);
                }
            }
        }
    }
    let n = labels.len();
    if n < 5_000 {
        eprintln!("❌ muestras decisivas insuficientes: {} (+{} neutros)", n, neutrals);
        std::process::exit(1);
    }
    let pos_rate = labels.iter().filter(|&&y| y > 0.5).count() as f64 / n as f64;
    println!("   {} muestras decisivas ({} neutros descartados, stride {}) · largo {:.1}%",
             n, neutrals, stride_ms_eff, pos_rate * 100.0);

    // ── 3. Split temporal 80/20 ───────────────────────────────────────────
    let split = n * 8 / 10;
    let (tr_feats, va_feats) = (&feats[..split], &feats[split..]);
    let (tr_y, va_y) = (&labels[..split], &labels[split..]);

    // ── 4. GBDT con early stopping ───────────────────────────────────────
    let p_bar = tr_y.iter().sum::<f64>() / tr_y.len() as f64;
    let init_score = (p_bar / (1.0 - p_bar)).ln() as f32;
    let logloss = |fs: &[Vec<f32>], ys: &[f64], f_pred: &[f64]| -> f64 {
        fs.iter()
            .zip(ys.iter())
            .zip(f_pred.iter())
            .map(|((_, &y), &f)| {
                let p = sigmoid(f).clamp(1e-7, 1.0 - 1e-7);
                -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
            })
            .sum::<f64>()
            / ys.len() as f64
    };
    let mut f_train = vec![init_score as f64; split];
    let mut f_val = vec![init_score as f64; n - split];
    let mut trees: Vec<Vec<TreeNode>> = Vec::new();
    let mut best_val = f64::INFINITY;
    let mut best_rounds = 0usize;
    let mut since_improve = 0usize;
    let mut rng = StdRng::seed_from_u64(42);
    let n_feat = tr_feats[0].len();

    for round in 0..n_rounds {
        // grad/hess de la logloss por muestra de train (XGBoost-style)
        let mut data: Vec<(Vec<f32>, f64, f64)> = tr_feats
            .iter()
            .zip(f_train.iter())
            .zip(tr_y.iter())
            .map(|((x, &f), &y)| {
                let p = sigmoid(f);
                (x.clone(), y - p, p * (1.0 - p))
            })
            .collect();
        // Subconjunto de features (70%) por árbol para diversidad
        let mut all: Vec<usize> = (0..n_feat).collect();
        use rand::seq::SliceRandom;
        all.shuffle(&mut rng);
        let keep = (n_feat * 7 / 10).max(4);
        let feat_subset: Vec<usize> = all[..keep].to_vec();

        let mut idx: Vec<usize> = (0..split).collect();
        let mut nodes: Vec<TreeNode> = Vec::new();
        // Bootstrap 80% de filas
        let mut boot: Vec<usize> = (0..split / 10 * 8)
            .map(|_| idx[(rng.random::<f64>() * idx.len() as f64) as usize])
            .collect();
        build_tree(&mut data, &mut boot, 0, max_depth, min_child, lambda, &feat_subset, 24,
                   &mut rng, &mut nodes);
        // Aplicar el árbol con shrinkage
        for (i, x) in tr_feats.iter().enumerate() {
            f_train[i] += lr * eval_tree(&nodes, x) as f64;
        }
        for (i, x) in va_feats.iter().enumerate() {
            f_val[i] += lr * eval_tree(&nodes, x) as f64;
        }
        trees.push(nodes);
        if round % 5 == 0 || round == n_rounds - 1 {
            let vl = logloss(va_feats, va_y, &f_val);
            if vl + 1e-6 < best_val {
                best_val = vl;
                best_rounds = trees.len();
                since_improve = 0;
            } else {
                since_improve += 5;
            }
            let tl = logloss(tr_feats, tr_y, &f_train);
            println!("   ronda {:3} train {:.5} · val {:.5} {}", round, tl, vl,
                     if since_improve == 0 { "★" } else { "" });
            if since_improve >= patience {
                println!("   early stopping (paciencia {})", patience);
                break;
            }
        }
    }
    trees.truncate(best_rounds.max(1));

    // Baseline honesto: logloss de validación prediciendo la tasa base
    let base_pred = vec![init_score as f64; va_y.len()];
    let baseline = logloss(va_feats, va_y, &base_pred);
    // Varianza de predicción en val (el diagnóstico del forest congelado)
    let val_preds: Vec<f64> = va_feats.iter().zip(f_val.iter()).map(|(_, &f)| sigmoid(f)).collect();
    let mut sorted_p = val_preds.clone();
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let var = val_preds.iter().map(|p| (p - 0.5) * (p - 0.5)).sum::<f64>() / val_preds.len() as f64;
    println!("═══ VEREDICTO ═══");
    println!("   val logloss modelo: {:.5} · baseline: {:.5} (Δ {:+.5})", best_val, baseline,
             baseline - best_val);
    println!("   p10={:.3} p50={:.3} p90={:.3} · varianza={:.5}",
             sorted_p[sorted_p.len() / 10], sorted_p[sorted_p.len() / 2],
             sorted_p[9 * sorted_p.len() / 10], var);

    // Margen anti-empate: una diferencia de 1e-5 es ruido numérico, no
    // edge (hallazgo real: un empate exacto se coló como 'gate superado').
    let gate_margin: f64 = arg("--gate-margin", "0.001").parse().unwrap();
    if baseline - best_val < gate_margin {
        println!("🚫 GATE: Δ {:+.5} < margen {} — sin evidencia real de edge. El modelo vivo NO se toca.",
                 baseline - best_val, gate_margin);
        if !promote {
            return;
        }
    }

    // ── 5. Serializar al formato NanoForestData ──────────────────────────
    let mut children_left: Vec<i32> = Vec::new();
    let mut children_right: Vec<i32> = Vec::new();
    let mut feature: Vec<i32> = Vec::new();
    let mut threshold: Vec<f32> = Vec::new();
    let mut value: Vec<f32> = Vec::new();
    let mut tree_offsets: Vec<i32> = Vec::new();
    for t in &trees {
        tree_offsets.push(children_left.len() as i32);
        for nd in t {
            children_left.push(nd.left);
            children_right.push(nd.right);
            feature.push(nd.feature);
            threshold.push(nd.threshold);
            value.push(nd.value);
        }
    }
    tree_offsets.push(children_left.len() as i32);
    let model = NanoForestData {
        children_left,
        children_right,
        feature,
        threshold,
        value,
        tree_offsets,
        init_score,
    };
    let out = if promote {
        format!("models/{}_SCALP.json", symbol)
    } else {
        format!("models/{}_SCALP_CANDIDATE.json", symbol)
    };
    let mut f = File::create(&out).unwrap();
    serde_json::to_writer_pretty(&mut f, &model).unwrap();
    println!("💾 {} ({} árboles, init {:.4}){}", out, trees.len(), init_score,
             if best_val >= baseline { " — [gate NO superado, revisar antes de promover]" } else { "" });
    if !promote && best_val < baseline {
        println!("   para promover al vivo: re-ejecuta con --promote (hot-swap lo recoge en ≤10s)");
    }
}

fn eval_tree(nodes: &[TreeNode], x: &[f32]) -> f32 {
    let mut cur = 0usize;
    loop {
        let nd = &nodes[cur];
        if nd.left == -1 && nd.right == -1 {
            return nd.value;
        }
        let v = x.get(nd.feature as usize).copied().unwrap_or(0.0);
        cur = if v <= nd.threshold { nd.left as usize } else { nd.right as usize };
    }
}
