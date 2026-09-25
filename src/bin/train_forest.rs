//! ENTRENADOR RUST POR INSTRUMENTO Y OBJETIVO — Gradient Boosting.
//!
//! POR QUÉ: los models/*_SCALP fueron entrenados por el pipeline Python
//! archivado (directriz: cero Python) y llevan semanas congelados — las
//! predicciones cuantizadas (0.337/0.476 constantes) congelan al SA.
//! `NanoForest::predict` es ESPACIO-LOGIT (init_score + Σárboles →
//! sigmoid): es gradient boosting, no promedio de forest. Este entrenador
//! conserva esa forma aditiva en la exportación con shrinkage e índices globales.
//!
//! CONTRATO COMPARTIDO: los features salen del MISMO
//! `StatefulEngine::get_universal_features()` que alimenta al motor en vivo
//! (secuencia process_tick/update_trade_flow/update_ofi de feature_exporter).
//! Esto no acredita igualdad de fuentes, disponibilidad o causalidad. Etiquetas:
//! primer toque de TP 0.36% / SL 0.18% del largo; timeouts descartados.
//! Son parámetros heredados de una tarea, no leyes del mercado. Se estima
//! un evento condicionado a toque, no PnL neto ni éxito simétrico del corto.
//! Un horizonte por ejecución todavía NO es un modelo temporal continuo.
//!
//! VALIDACIÓN: selección temporal 80/20 (o --val-in), con purga de ventanas
//! de etiqueta. Early stopping usa sólo selección. --promote exige --test-in
//! posterior a TODA la evidencia de ajuste/selección; su gate evalúa el
//! artefacto congelado. Sin test sólo se permite candidato de investigación.
//! Esto no controla reutilización del holdout entre ejecuciones ni prueba PnL.
//!
//! Uso: train_forest BTCUSDT [--in data/BTCUSDT_SEP26.bin] [--max-samples 400000]
//!      [--horizon-ms 300000] [--trees 300] [--lr 0.1] [--depth 5]
//!      [--val-in selection.bin] [--test-in later-test.bin] [--promote]

use god_engine_core::ml_inference::{NanoForest, NanoForestData};
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

fn valid_tick(t: &BinTick) -> bool {
    t.ts > 0
        && t.bid.is_finite()
        && t.ask.is_finite()
        && t.bid > 0.0
        && t.ask >= t.bid
        && t.bq.is_finite()
        && t.aq.is_finite()
        && t.bq >= 0.0
        && t.aq >= 0.0
        && (t.bq + t.aq).is_finite()
}

fn validate_tick_order(raw: &[BinTick]) -> Result<(), String> {
    if raw.is_empty() || raw.iter().any(|t| t.ts == 0) || raw.windows(2).any(|w| w[1].ts < w[0].ts)
    {
        return Err("tick timestamps must be positive and nondecreasing; no silent sorting".into());
    }
    Ok(())
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

fn regression_residual(prediction: f64, target: f64) -> f64 {
    // build_tree stores +G/(H+lambda), then the booster ADDS the tree.
    // G must therefore be the negative gradient of half squared error.
    target - prediction
}

/// Freeze exactly the additive predictor used during training:
/// F(x) = init + sum(lr * tree(x)). The wire format has global children
/// and no separate learning-rate field, so shrinkage belongs in leaf values.
fn serialize_forest(
    trees: &[Vec<TreeNode>],
    init_score: f32,
    lr: f64,
) -> Result<NanoForestData, String> {
    if trees.is_empty() || !init_score.is_finite() || !lr.is_finite() || lr <= 0.0 {
        return Err(
            "forest export requires trees, finite bias and positive finite learning rate".into(),
        );
    }
    let mut model = NanoForestData {
        children_left: Vec::new(),
        children_right: Vec::new(),
        feature: Vec::new(),
        threshold: Vec::new(),
        value: Vec::new(),
        tree_offsets: Vec::new(),
        init_score,
    };
    for tree in trees {
        if tree.is_empty() {
            return Err("cannot export an empty tree".into());
        }
        let base = i32::try_from(model.value.len()).map_err(|_| "too many tree nodes")?;
        model.tree_offsets.push(base);
        let relocate = |child: i32| -> Result<i32, String> {
            if child == -1 {
                return Ok(-1);
            }
            if child < 0 || child as usize >= tree.len() {
                return Err("invalid local child index".into());
            }
            base.checked_add(child)
                .ok_or_else(|| "child index overflow".into())
        };
        for node in tree {
            model.children_left.push(relocate(node.left)?);
            model.children_right.push(relocate(node.right)?);
            model.feature.push(node.feature);
            model.threshold.push(node.threshold);
            model.value.push((node.value as f64 * lr) as f32);
        }
    }
    model
        .tree_offsets
        .push(i32::try_from(model.value.len()).map_err(|_| "too many tree nodes")?);
    NanoForest::from_data(model.clone())?;
    Ok(model)
}

/// Gate and diagnostics must evaluate the frozen, retained artifact, not
/// f_val from a later early-stopping iteration that has been discarded.
fn serving_predictions(model: &NanoForestData, features: &[Vec<f32>]) -> Result<Vec<f64>, String> {
    let forest = NanoForest::from_data(model.clone())?;
    features
        .iter()
        .map(|x| {
            forest
                .predict_raw_checked(x)
                .map(|(raw, _)| raw as f64)
                .ok_or_else(|| "invalid serving prediction in validation set".into())
        })
        .collect()
}

/// The probe proposes thresholds only. Score every candidate on the same
/// bootstrap multiset as its parent and children (including multiplicities).
/// Sorting once and sweeping prefixes/suffixes avoids O(rows * thresholds).
fn candidate_gains(
    data: &[(Vec<f32>, f64, f64)],
    idx: &[usize],
    feature: usize,
    thresholds: &[f32],
    lambda: f64,
    min_child: usize,
) -> Vec<Option<f64>> {
    let mut gains = vec![None; thresholds.len()];
    if !lambda.is_finite()
        || lambda < 0.0
        || min_child == 0
        || idx.iter().any(|&i| {
            !data[i].0[feature].is_finite()
                || !data[i].1.is_finite()
                || !data[i].2.is_finite()
                || data[i].2 < 0.0
        })
    {
        return gains;
    }
    let mut ordered = idx.to_vec();
    ordered.sort_unstable_by(|&a, &b| data[a].0[feature].total_cmp(&data[b].0[feature]));
    let mut suffix = vec![(0.0, 0.0); ordered.len() + 1];
    for j in (0..ordered.len()).rev() {
        suffix[j] = (
            suffix[j + 1].0 + data[ordered[j]].1,
            suffix[j + 1].1 + data[ordered[j]].2,
        );
    }
    let score = |g: f64, h: f64| g * g / (h + lambda + 1e-12);
    let parent_score = score(suffix[0].0, suffix[0].1);
    let mut queries: Vec<_> = thresholds
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, t)| t.is_finite())
        .collect();
    queries.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let (mut nl, mut gl, mut hl) = (0usize, 0.0, 0.0);
    for (q, thr) in queries {
        while nl < ordered.len() && data[ordered[nl]].0[feature] <= thr {
            gl += data[ordered[nl]].1;
            hl += data[ordered[nl]].2;
            nl += 1;
        }
        if nl >= min_child && ordered.len() - nl >= min_child {
            let (gr, hr) = suffix[nl];
            let gain = score(gl, hl) + score(gr, hr) - parent_score;
            if gain.is_finite() {
                gains[q] = Some(gain);
            }
        }
    }
    gains
}

/// For an inclusive span S and a budget B, floor(S / B) + 1 ensures
/// floor(S / stride) + 1 <= B. Saturation is backed by an explicit counter.
fn effective_stride(span: u64, requested: u64, cap: usize) -> Result<u64, String> {
    if requested == 0 || cap == 0 {
        return Err("stride and sample budget must be positive".into());
    }
    let cap = u64::try_from(cap).map_err(|_| "sample budget does not fit timestamp domain")?;
    Ok(requested.max((span / cap).saturating_add(1)))
}

struct SamplingBudget {
    stride: u64,
    limit: usize,
    attempts: usize,
    next: Option<u64>,
}

impl SamplingBudget {
    fn new(span: u64, stride: u64, limit: usize) -> Result<Self, String> {
        Ok(Self {
            stride: effective_stride(span, stride, limit)?,
            limit,
            attempts: 0,
            next: Some(0),
        })
    }
    /// A reserved attempt consumes budget even if its label is later rejected.
    fn reserve(&mut self, ts: u64) -> bool {
        if self.attempts >= self.limit || self.next.is_none_or(|next| ts < next) {
            return false;
        }
        self.attempts += 1;
        self.next = ts.checked_add(self.stride);
        true
    }
}

#[derive(Clone, Copy, Debug)]
struct LabelInterval {
    start: u64,
    end: u64,
}

#[derive(Default)]
struct TrainingSamples {
    features: Vec<Vec<f32>>,
    labels: Vec<f64>,
    intervals: Vec<LabelInterval>,
}

impl TrainingSamples {
    fn push(&mut self, features: Vec<f32>, label: f64, start: u64, end: u64) {
        self.features.push(features);
        self.labels.push(label);
        self.intervals.push(LabelInterval { start, end });
    }
    fn validate(&self) -> Result<(), String> {
        let n = self.labels.len();
        if n == 0 || self.features.len() != n || self.intervals.len() != n {
            return Err("empty or misaligned sample set".into());
        }
        let dim = self.features[0].len();
        if dim == 0
            || self
                .features
                .iter()
                .any(|x| x.len() != dim || x.iter().any(|v| !v.is_finite()))
            || self.labels.iter().any(|v| !v.is_finite())
            || self
                .intervals
                .iter()
                .any(|v| v.start == 0 || v.end < v.start)
            || self.intervals.windows(2).any(|w| w[1].start <= w[0].start)
        {
            return Err("invalid features, labels or chronological intervals".into());
        }
        Ok(())
    }
    fn split_at(&mut self, at: usize) -> Self {
        Self {
            features: self.features.split_off(at),
            labels: self.labels.split_off(at),
            intervals: self.intervals.split_off(at),
        }
    }
    fn max_end(&self) -> u64 {
        self.intervals.iter().map(|v| v.end).max().unwrap_or(0)
    }
}

/// Label windows are closed intervals: equality at the boundary is overlap.
/// End times need not be monotone: filter every row, not only a suffix.
fn purge_training(
    train: TrainingSamples,
    selection: &TrainingSamples,
) -> Result<(TrainingSamples, usize), String> {
    train.validate()?;
    selection.validate()?;
    let boundary = selection.intervals[0].start;
    if train.intervals.last().unwrap().start >= boundary {
        return Err("selection must start after all training feature times".into());
    }
    let mut kept = TrainingSamples::default();
    let mut purged = 0;
    for ((x, y), interval) in train
        .features
        .into_iter()
        .zip(train.labels)
        .zip(train.intervals)
    {
        if interval.end >= boundary {
            purged += 1;
        } else {
            kept.push(x, y, interval.start, interval.end);
        }
    }
    kept.validate()?;
    Ok((kept, purged))
}

fn require_later_holdout(test: &TrainingSamples, evidence_end: u64) -> Result<(), String> {
    test.validate()?;
    if test.intervals[0].start <= evidence_end {
        return Err("test overlaps training/selection information intervals".into());
    }
    Ok(())
}

fn require_promotion_holdout(promote: bool, test_path: &str) -> Result<(), String> {
    if promote && test_path.is_empty() {
        return Err("--promote requires a later, independent --test-in; --val-in is selection evidence, not a final test".into());
    }
    Ok(())
}

fn passes_loss_gate(regression: bool, loss: f64, baseline: f64, margin: f64) -> bool {
    if !loss.is_finite()
        || !baseline.is_finite()
        || !margin.is_finite()
        || loss < 0.0
        || baseline < 0.0
        || margin < 0.0
    {
        return false;
    }
    if regression {
        baseline > 0.0 && (baseline - loss) / baseline >= margin
    } else {
        baseline - loss >= margin
    }
}

/// Quantile candidates may be approximate; their gain uses full-node statistics.
fn build_tree(
    data: &mut Vec<(Vec<f32>, f64, f64)>, // (features, grad, hess) por muestra IN PLACE
    idx: &mut Vec<usize>,                 // índices del nodo (se particiona in situ)
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
        nodes.push(TreeNode {
            feature: -1,
            threshold: 0.0,
            left: -1,
            right: -1,
            value: leaf_value,
        });
        return;
    }
    // Approximate threshold proposals only; no claim of unbiased gain from probe.
    let probe: Vec<usize> = if idx.len() > 2048 {
        let mut p: Vec<usize> = Vec::with_capacity(2048);
        for _ in 0..2048 {
            p.push(idx[(rng.random::<f64>() * idx.len() as f64) as usize]);
        }
        p
    } else {
        idx.clone()
    };
    let mut best = (f64::MIN, 0usize, 0.0f32); // (ganancia, feature, threshold)
    for &f in feat_subset {
        let mut vals: Vec<f32> = probe.iter().map(|&i| data[i].0[f]).collect();
        if vals.windows(2).all(|w| w[0] == w[1]) {
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Umbrales candidatos por cuantiles (excluye extremos)
        let thresholds: Vec<_> = (1..n_quantiles)
            .map(|q| vals[q * vals.len() / n_quantiles])
            .collect();
        let gains = candidate_gains(data, idx, f, &thresholds, lambda, min_child);
        for (thr, gain) in thresholds.into_iter().zip(gains) {
            let Some(gain) = gain else { continue };
            if gain > best.0 {
                best = (gain, f, thr);
            }
        }
    }
    if best.0 <= 1e-9 {
        nodes.push(TreeNode {
            feature: -1,
            threshold: 0.0,
            left: -1,
            right: -1,
            value: leaf_value,
        });
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
    nodes.push(TreeNode {
        feature: f as i32,
        threshold: thr,
        left: 0,
        right: 0,
        value: 0.0,
    });
    // DFS pre-orden: subárbol izquierdo en [me+1, …]; la raíz del derecho
    // se captura ANTES de construirlo (nodes.len()-1 tras el derecho sería
    // su última hoja, no su raíz).
    build_tree(
        data,
        &mut left_idx,
        depth + 1,
        max_depth,
        min_child,
        lambda,
        feat_subset,
        n_quantiles,
        rng,
        nodes,
    );
    nodes[me].left = (me + 1) as i32;
    let right_root = nodes.len();
    build_tree(
        data,
        &mut right_idx,
        depth + 1,
        max_depth,
        min_child,
        lambda,
        feat_subset,
        n_quantiles,
        rng,
        nodes,
    );
    nodes[me].right = right_root as i32;
}

/// B3.4 — carga las series macro reales de data/macro. VIX/SP500/NASDAQ
/// (Yahoo v8, cierres idénticos a FRED) son OBLIGATORIAS: faltan ⇒ aborto,
/// porque una columna de ceros constante es capacidad fantasma. DXY
/// (DTWEXBGS, sólo FRED) se TOLERA ausente con warning: el CDN de FRED
/// abre y cierra ventanas desde esta red y un re-run del sync lo completa;
/// mientras tanto la dim queda en 0 neutro y NINGÚN árbol parte por ella
/// (columna constante = sin splits — no hay ruptura de paridad).
fn load_macro_series() -> (
    Vec<(u64, f64)>,
    Vec<(u64, f64)>,
    Vec<(u64, f64)>,
    Vec<(u64, f64)>,
) {
    let read = |tag: &str, required: bool| -> Vec<(u64, f64)> {
        let path = format!("data/macro/{tag}.csv");
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                if required {
                    eprintln!(
                        "❌ macro {tag}: falta data/macro/{tag}.csv ({e}) — ejecuta `macro_history_sync` primero."
                    );
                    std::process::exit(1);
                }
                eprintln!(
                    "⚠️ macro {tag}: sin data/macro/{tag}.csv ({e}) — dim en 0 neutro, sin splits. Re-ejecuta `macro_history_sync` cuando FRED abra."
                );
                return Vec::new();
            }
        };
        let rows: Vec<(u64, f64)> = content
            .lines()
            .skip(1)
            .filter_map(|l| {
                let mut p = l.split(',');
                let ms = p.next()?.trim().parse::<u64>().ok()?;
                let v = p.next()?.trim().parse::<f64>().ok()?;
                (v.is_finite() && v > 0.0).then_some((ms, v))
            })
            .collect();
        if rows.len() < 100 {
            if required {
                eprintln!("❌ macro {tag}: historia irreal ({} filas)", rows.len());
                std::process::exit(1);
            }
            eprintln!(
                "⚠️ macro {tag}: historia irreal ({} filas) — dim en 0 neutro",
                rows.len()
            );
            return Vec::new();
        }
        println!("   [macro] {tag}: {} días", rows.len());
        rows
    };
    (
        read("VIX", true),
        read("SP500", true),
        read("DXY", false),
        read("NASDAQ", true),
    )
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
        args.iter()
            .position(|a| a == name)
            .and_then(|p| args.get(p + 1))
            .map(|s| s.to_string())
            .unwrap_or_else(|| dflt.to_string())
    };
    let default_in = format!("data/{}_SEP26.bin", symbol);
    let in_path = arg("--in", &default_in);
    let max_samples: usize = arg("--max-samples", "200000").parse().unwrap();
    // Clock horizon, not event count. The inherited 5-minute default is
    // configurable, not a universal scale or a data-derived optimum.
    // Example: 500 events spaced 75ms span about 37.5s, NOT 2s.
    let horizon_ms: u64 = arg("--horizon-ms", "300000").parse().unwrap();
    let stride_ms: u64 = arg("--stride-ms", "50000").parse().unwrap();
    let n_rounds: usize = arg("--trees", "300").parse().unwrap();
    let lr: f64 = arg("--lr", "0.1").parse().unwrap();
    let max_depth: u32 = arg("--depth", "5").parse().unwrap();
    let min_child: usize = arg("--min-child", "40").parse().unwrap();
    let lambda: f64 = arg("--lambda", "1.0").parse().unwrap();
    let patience: usize = arg("--patience", "40").parse().unwrap();
    let promote = args.iter().any(|a| a == "--promote");
    let test_in = arg("--test-in", "");
    require_promotion_holdout(promote, &test_in).expect("promotion contract rejected before I/O");
    let gate_margin: f64 = arg("--gate-margin", "0.001").parse().unwrap();
    assert!(
        horizon_ms > 0
            && stride_ms > 0
            && max_samples > 0
            && n_rounds > 0
            && min_child > 0
            && min_child <= usize::MAX / 2
            && patience > 0
            && lr.is_finite()
            && lr > 0.0
            && lambda.is_finite()
            && lambda > 0.0
            && gate_margin.is_finite()
            && gate_margin >= 0.0,
        "invalid training parameters; no files have been written"
    );
    // P-1/P-2/P-3c — objetivo: dir (primer toque largo) | vol (RMS por evento,
    // regresión) | volu (profundidad media por evento) | oi (ΔOI% a horizonte,
    // regresión con join as-of del histórico horario).
    let label_mode = arg("--label", "dir");
    if !matches!(label_mode.as_str(), "dir" | "vol" | "volu" | "oi") {
        eprintln!("❌ --label inválido: {} (dir|vol|volu|oi)", label_mode);
        std::process::exit(1);
    }

    // P-3c — serie histórica de OI para `--label oi` (join as-of estricto:
    // la última fila con ts ≤ t). El endpoint sólo conserva ~30 días: la
    // validación es DENTRO de la ventana (split temporal del subconjunto
    // con label), documentado en la salida.
    let oi_series: Vec<(u64, f64)> = if label_mode == "oi" {
        let path = format!("data/oihist/{}.csv", symbol);
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!(
                "❌ --label oi sin histórico: {} ({}) — ejecuta oi_history_sync",
                path, e
            );
            std::process::exit(1);
        });
        let mut rows: Vec<(u64, f64)> = content
            .lines()
            .skip(1)
            .filter_map(|ln| {
                let mut it = ln.split(',');
                let ts = it.next()?.trim().parse::<u64>().ok()?;
                let oi = it.next()?.trim().parse::<f64>().ok()?;
                (ts > 0 && oi.is_finite() && oi > 0.0).then_some((ts, oi))
            })
            .collect();
        rows.sort_unstable_by_key(|(ts, _)| *ts);
        if rows.len() < 48 {
            eprintln!("❌ histórico OI demasiado corto: {} filas", rows.len());
            std::process::exit(1);
        }
        println!(
            "   [oi] {} filas · span {:.1} días (validación DENTRO de ventana)",
            rows.len(),
            (rows.last().unwrap().0 - rows[0].0) as f64 / 86_400_000.0
        );
        rows
    } else {
        Vec::new()
    };

    println!("🌲 [TRAIN-FOREST] {} ← {}", symbol, in_path);
    println!(
        "   intentos≤{} por archivo; etiquetas≤intentos; horizonte={}ms stride={}ms árboles≤{} lr={} depth={} λ={}",
        max_samples, horizon_ms, stride_ms, n_rounds, lr, max_depth, lambda
    );

    let oi_series_ref = &oi_series;
    // ── 1+2. Muestras: features+etiquetas desde ticks (reutilizable) ────
    let build = |path: &str| -> TrainingSamples {
        let file = File::open(path).unwrap_or_else(|e| {
            eprintln!("❌ no pude abrir {}: {}", path, e);
            std::process::exit(1);
        });
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.unwrap();
        let header_off = if mmap.len() >= 8 && &mmap[..8] == b"TGMTICK1" {
            8usize
        } else {
            0
        };
        let sz = std::mem::size_of::<BinTick>();
        assert_eq!(
            (mmap.len() - header_off) % sz,
            0,
            "partial tick record in {path}"
        );
        let n_total = (mmap.len() - header_off) / sz;
        if n_total < 50_000 {
            eprintln!("❌ datos insuficientes: {} ticks ({})", n_total, path);
            std::process::exit(1);
        }
        let ptr = unsafe { mmap.as_ptr().add(header_off) } as *const BinTick;
        let raw = unsafe { std::slice::from_raw_parts(ptr, n_total) };
        validate_tick_order(raw).expect("invalid tick chronology");
        println!("   [{}] {} ticks crudos", path, n_total);

        let last_ts = raw[n_total - 1].ts;
        let span_ms = last_ts.saturating_sub(raw[0].ts);
        let mut budget = SamplingBudget::new(span_ms, stride_ms, max_samples).unwrap();
        println!(
            "   [{}] span {:.1} días · stride efectivo {}ms",
            path,
            span_ms as f64 / 86_400_000.0,
            budget.stride
        );
        let mut engine = StatefulEngine::new();
        // B3.4 — bloque MACRO real (dims 44..48): series FRED VIXCLS/SP500/
        // DTWEXBGS/NASDAQCOM de data/macro (macro_history_sync). Join as-of
        // ESTRICTO: último cierre de un día ANTERIOR al tick — el cierre del
        // mismo día aún no existe intradía. Tres series obligatorias; DXY
        // puede faltar y entonces queda en cero según load_macro_series.
        // El join por fecha no demuestra el instante real de publicación.
        let (macro_vix, macro_spx, macro_dxy, macro_ndx) = load_macro_series();
        let mut c_vix = 0usize;
        let mut c_spx = 0usize;
        let mut c_dxy = 0usize;
        let mut c_ndx = 0usize;
        let mut macro_asof = |ts: u64| -> Option<[f32; 4]> {
            let day_start = ts - (ts % 86_400_000);
            // Serie VACÍA (DXY sin FRED) ⇒ 0 neutro — columna sin splits,
            // no descarte de muestras. Serie con datos pero tick anterior a
            // su cobertura ⇒ muestra descartada (honestidad).
            fn adv(series: &[(u64, f64)], cur: &mut usize, day_start: u64) -> Option<f64> {
                if series.is_empty() {
                    return Some(0.0);
                }
                while *cur + 1 < series.len() && series[*cur + 1].0 < day_start {
                    *cur += 1;
                }
                match series.get(*cur) {
                    Some(&(ms, v)) if ms < day_start => Some(v),
                    _ => None,
                }
            }
            let mut omni = [0.0f64; 54];
            omni[24] = adv(&macro_vix, &mut c_vix, day_start)?;
            omni[22] = adv(&macro_spx, &mut c_spx, day_start)?;
            omni[21] = adv(&macro_dxy, &mut c_dxy, day_start)?;
            omni[23] = adv(&macro_ndx, &mut c_ndx, day_start)?;
            Some(god_engine_core::ml_inference::macro_ml_features(&omni))
        };
        let mut samples = TrainingSamples::default();
        let mut neutrals = 0usize;
        let tp_pct = 0.0036;
        let sl_pct = 0.0018;
        let mut warmup = 0usize;
        for i in 0..n_total {
            if budget.attempts >= budget.limit || budget.next.is_none() {
                break;
            }
            let t = &raw[i];
            if !valid_tick(t) {
                continue;
            }
            let mid = 0.5 * t.bid + 0.5 * t.ask;
            let vol = t.bq + t.aq;
            let pseudo_maker = t.bq > t.aq;
            engine.process_tick(mid, vol, t.ts);
            engine.update_trade_flow(vol, pseudo_maker);
            let _ = engine.update_ofi(t.bid, t.ask, t.bq, t.aq);
            // B3.35 — OBI FUERA del trainer: el OBI sintético (reconstruido
            // de aggTrades con is_buyer_maker) tiene DISTRIBUCIÓN INCOMPATIBLE
            // con el OBI del depth L2 real que sirve el vivo. Alimentar
            // obi_accel sólo en train produjo modelos que en vivo predicen
            // ~0.503 constante (v41: señal muerta, cero entradas). Las dims
            // [4][5][10] están zerificadas en AMBOS lados vía
            // FEATURES_DEAD_IN_SERVE — esta llamada NO debe volver
            // hasta que exista un OBI sintético calibrado contra la
            // distribución real del libro.
            warmup += 1;
            let Some(deadline) = t.ts.checked_add(horizon_ms) else {
                break;
            };
            if warmup >= 100 && deadline <= last_ts && budget.reserve(t.ts) {
                // B2.3: universal(34) + spectral(10), same component layout
                // as inference, not proof of identical data distributions.
                // B3.4: ⊕ macro(4) — mismo contrato macro_ml_features del
                // core, as-of t-1 estricto. Muestra sin contexto macro se
                // descarta (honestidad), nunca se rellena.
                let Some(macro_block) = macro_asof(t.ts) else {
                    continue;
                };
                let sf = engine.get_universal_features();
                let sp = engine.get_spectral_ml_features();
                // B3.9: mismo contrato de dimensión que la inferencia viva —
                // un modelo más ancho que el binario sería rechazado al cargar.
                const FULL_DIM: usize = god_engine_core::ml_inference::NanoForest::ML_VECTOR_DIM;
                let mut full = [0f32; FULL_DIM];
                full[..34].copy_from_slice(&sf);
                // C-02 — features MUERTAS en serve ⇒ 0.0 también en train.
                // FEATURES_DEAD_IN_SERVE (stateful_engine) es la única
                // fuente de verdad del mapa vivo/muerto del contrato 34D.
                // Hoy [9] dark_alpha: ya sale 0.0 porque el trainer alimenta
                // dex_severity=0.0; el borrado explícito mantiene el
                // invariante train≡serve aunque alguien cablee después una
                // historia dex que el vivo no sirve.
                for &d in god_engine_core::stateful_engine::FEATURES_DEAD_IN_SERVE {
                    full[d] = 0.0;
                }
                full[34..44].copy_from_slice(&sp);
                full[44..].copy_from_slice(&macro_block);
                if full.iter().all(|f| f.is_finite()) {
                    // P-1/P-2 (MOTOR UNIVERSAL): modos de PREDICCIÓN además de
                    // dirección. `--label vol` ⇒ RMS de retornos de mid sobre
                    // el horizonte (×100, %/evento), NO volatilidad por reloj;
                    // `--label volu` ⇒ promedio por evento de bq+aq, NO volumen
                    // negociado ni promedio ponderado por duración (FMT-199).
                    // REGRESIÓN (residual = y−f, hess = 1, init = media;
                    // serving por predict_raw SIN sigmoid). `dir` ⇒ barrera
                    // triple HOST-010 (clasificación, como siempre).
                    if label_mode != "dir" {
                        if label_mode == "oi" {
                            // P-3c — ΔOI% a horizonte por join as-of de la
                            // serie horaria (última fila con ts ≤ t). Sin
                            // label válido (fuera de ventana o horizonte
                            // incompleto) la muestra se DESCARTA — nunca se
                            // rellena.
                            let oi_asof = |ts: u64| -> Option<f64> {
                                let idx = oi_series_ref
                                    .binary_search_by(|(ots, _)| ots.cmp(&ts))
                                    .unwrap_or_else(|i| i);
                                if idx == 0 {
                                    // ts anterior a la primera fila: sin valor
                                    if oi_series_ref
                                        .first()
                                        .map(|(ots, _)| *ots > ts)
                                        .unwrap_or(true)
                                    {
                                        return None;
                                    }
                                }
                                if idx >= oi_series_ref.len() {
                                    oi_series_ref.last().map(|(_, oi)| *oi)
                                } else if oi_series_ref[idx].0 == ts {
                                    Some(oi_series_ref[idx].1)
                                } else if idx > 0 {
                                    Some(oi_series_ref[idx - 1].1)
                                } else {
                                    None
                                }
                            };
                            let (Some(oi_now), Some(oi_fut)) = (oi_asof(t.ts), oi_asof(deadline))
                            else {
                                continue;
                            };
                            // honestidad: el futuro debe ser una fila REAL del
                            // histórico (deadline ≤ última fila), no el último
                            // valor colado por el as-of del borde.
                            if deadline > oi_series_ref.last().unwrap().0 {
                                continue;
                            }
                            let label = (oi_fut - oi_now) / oi_now * 100.0;
                            if label.is_finite() {
                                samples.push(full.to_vec(), label, t.ts, deadline);
                            }
                        } else {
                            let mut sum_sq = 0.0f64;
                            let mut n_rt = 0usize;
                            let mut qty_sum = 0.0f64;
                            let mut prev = mid;
                            for f in (i + 1)..n_total {
                                let ft = &raw[f];
                                if ft.ts > deadline {
                                    break;
                                }
                                if !valid_tick(ft) {
                                    continue;
                                }
                                let fm = 0.5 * ft.bid + 0.5 * ft.ask;
                                if prev > 0.0 {
                                    let r = (fm - prev) / prev;
                                    sum_sq += r * r;
                                    n_rt += 1;
                                }
                                qty_sum += ft.bq + ft.aq;
                                prev = fm;
                            }
                            let label = if n_rt == 0 {
                                continue;
                            } else if label_mode == "vol" {
                                (sum_sq / n_rt as f64).sqrt() * 100.0
                            } else {
                                qty_sum / n_rt as f64
                            };
                            if label.is_finite() {
                                samples.push(full.to_vec(), label, t.ts, deadline);
                            }
                        }
                    } else {
                        // Triple barrera por TIEMPO DE RELOJ dentro de horizon_ms.
                        // HOST-010 (DECIMOCUARTO): la geometría del label debe ser
                        // la del trade REAL — SL −sl_pct vs TP +tp_pct (RR≥2 por
                        // friction_floors). La versión anterior chequeaba el SL del
                        // corto (`>= mid*(1+sl_pct)`, +0.18%) ANTES del TP largo
                        // (+0.36%): el TP era código muerto y todo toque de +0.18%
                        // se contaba como victoria — el modelo aprendía una barrera
                        // simétrica ±0.18% (coin-flip tras fees) en vez del trade
                        // asimétrico RR 2:1 que el vivo ejecuta. Ahora: primer
                        // toque de SL largo ⇒ 0.0, primer toque de TP largo ⇒ 1.0,
                        // Sin toque antes del deadline: timeout descartado.
                        // 1−p describe SL del LARGO antes de TP, condicionado
                        // a toque; NO es probabilidad de éxito del corto con
                        // otras barreras ni prueba de cierre en BE (FMT-028).
                        let long_tp = mid * (1.0 + tp_pct);
                        let long_sl = mid * (1.0 - sl_pct);
                        let mut label = 0.5f64;
                        'barrier: for f in (i + 1)..n_total {
                            let ft = &raw[f];
                            if ft.ts > deadline {
                                break 'barrier;
                            }
                            if !valid_tick(ft) {
                                continue;
                            }
                            let fut_mid = 0.5 * ft.bid + 0.5 * ft.ask;
                            if fut_mid <= long_sl {
                                label = 0.0;
                                break 'barrier;
                            }
                            if fut_mid >= long_tp {
                                label = 1.0;
                                break 'barrier;
                            }
                        }
                        if (label - 0.5).abs() < 1e-9 {
                            neutrals += 1;
                        } else {
                            samples.push(full.to_vec(), label, t.ts, deadline);
                        }
                    } // fin modo dir
                }
            }
        }
        let labels = &samples.labels;
        let n = labels.len();
        println!(
            "   [{}] {} intentos reservados / {} de presupuesto",
            path, budget.attempts, budget.limit
        );
        if n < 5_000 {
            eprintln!("❌ muestras insuficientes: {} (+{} neutros)", n, neutrals);
            std::process::exit(1);
        }
        if label_mode == "dir" {
            let pos_rate = labels.iter().filter(|&&y| y > 0.5).count() as f64 / n as f64;
            println!(
                "   [{}] {} muestras decisivas ({} neutros) · largo {:.1}%",
                path,
                n,
                neutrals,
                pos_rate * 100.0
            );
        } else {
            let mean_y = labels.iter().sum::<f64>() / n as f64;
            println!(
                "   [{}] {} muestras · label[{}] media {:.6}",
                path, n, label_mode, mean_y
            );
        }
        samples.validate().expect("invalid sample contract");
        samples
    };
    let mut training = build(&in_path);

    // ── 3. Selection and purging by label information intervals ────────
    // 80/20 is a retained policy, not a theorem. Separate files alone do
    // not imply disjoint information; verify chronology and purge overlap.
    let val_in = arg("--val-in", "");
    let validation = if val_in.is_empty() {
        let cut = training.labels.len() * 8 / 10;
        training.split_at(cut)
    } else {
        build(&val_in)
    };
    let (training, purged) =
        purge_training(training, &validation).expect("invalid training/selection separation");
    let evidence_end = training.max_end().max(validation.max_end());
    println!(
        "   purga: {} etiquetas que cruzaban el primer instante de selección",
        purged
    );
    let (tr_feats, tr_y) = (training.features, training.labels);
    let (va_feats, va_y) = (validation.features, validation.labels);
    let split = tr_y.len();
    assert!(
        split >= 2 && split / 10 * 8 >= 2,
        "too few training rows after purging"
    );

    // ── 4. GBDT con early stopping ───────────────────────────────────────
    let is_regression = label_mode != "dir";
    let p_bar = tr_y.iter().sum::<f64>() / tr_y.len() as f64;
    assert!(
        p_bar.is_finite() && (is_regression || (p_bar > 0.0 && p_bar < 1.0)),
        "invalid target mean or single-class training set"
    );
    // Regresión: init = media (predicción cruda); clasificación: logit.
    let init_score = if is_regression {
        p_bar as f32
    } else {
        (p_bar / (1.0 - p_bar)).ln() as f32
    };
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
    // P-1/P-2: MSE sobre la predicción CRUDA (sin sigmoid) — regresión.
    let mse = |ys: &[f64], f_pred: &[f64]| -> f64 {
        ys.iter()
            .zip(f_pred.iter())
            .map(|(&y, &f)| (y - f) * (y - f))
            .sum::<f64>()
            / ys.len().max(1) as f64
    };
    let mut f_train = vec![init_score as f64; split];
    let mut f_val = vec![init_score as f64; va_y.len()];
    let mut trees: Vec<Vec<TreeNode>> = Vec::new();
    let mut best_val = f64::INFINITY;
    let mut best_rounds = 0usize;
    let mut since_improve = 0usize;
    let mut rng = StdRng::seed_from_u64(42);
    let n_feat = tr_feats[0].len();

    for round in 0..n_rounds {
        // grad/hess por muestra de train (XGBoost-style):
        // clasificación: logloss (y−σ(f), σ(1−σ)); regresión: squared loss
        // (y−f, 1): negative gradient for the additive update.
        let mut data: Vec<(Vec<f32>, f64, f64)> = tr_feats
            .iter()
            .zip(f_train.iter())
            .zip(tr_y.iter())
            .map(|((x, &f), &y)| {
                if is_regression {
                    (x.clone(), regression_residual(f, y), 1.0)
                } else {
                    let p = sigmoid(f);
                    (x.clone(), y - p, p * (1.0 - p))
                }
            })
            .collect();
        // Subconjunto de features (70%) por árbol para diversidad
        let mut all: Vec<usize> = (0..n_feat).collect();
        use rand::seq::SliceRandom;
        all.shuffle(&mut rng);
        let keep = (n_feat * 7 / 10).max(4);
        let feat_subset: Vec<usize> = all[..keep].to_vec();

        let idx: Vec<usize> = (0..split).collect();
        let mut nodes: Vec<TreeNode> = Vec::new();
        // Bootstrap 80% de filas
        let mut boot: Vec<usize> = (0..split / 10 * 8)
            .map(|_| idx[(rng.random::<f64>() * idx.len() as f64) as usize])
            .collect();
        build_tree(
            &mut data,
            &mut boot,
            0,
            max_depth,
            min_child,
            lambda,
            &feat_subset,
            24,
            &mut rng,
            &mut nodes,
        );
        // Aplicar el árbol con shrinkage
        for (i, x) in tr_feats.iter().enumerate() {
            f_train[i] += lr * eval_tree(&nodes, x) as f64;
        }
        for (i, x) in va_feats.iter().enumerate() {
            f_val[i] += lr * eval_tree(&nodes, x) as f64;
        }
        trees.push(nodes);
        if round % 5 == 0 || round == n_rounds - 1 {
            let (vl, tl) = if is_regression {
                (mse(&va_y, &f_val), mse(&tr_y, &f_train))
            } else {
                (
                    logloss(&va_feats, &va_y, &f_val),
                    logloss(&tr_feats, &tr_y, &f_train),
                )
            };
            if vl + 1e-9 < best_val {
                best_val = vl;
                best_rounds = trees.len();
                since_improve = 0;
            } else {
                since_improve += 5;
            }
            println!(
                "   ronda {:3} train {:.6} · val {:.6} {}",
                round,
                tl,
                vl,
                if since_improve == 0 { "★" } else { "" }
            );
            if since_improve >= patience {
                println!("   early stopping (paciencia {})", patience);
                break;
            }
        }
    }
    trees.truncate(best_rounds.max(1));

    // FMT-190/191: validate and score the exact serialized predictor before
    // any destination is selected or any model file is created.
    let model = serialize_forest(&trees, init_score, lr)
        .expect("invalid forest export; no model has been written");
    f_val = serving_predictions(&model, &va_feats)
        .expect("invalid validation evidence; no model has been written");
    best_val = if is_regression {
        mse(&va_y, &f_val)
    } else {
        logloss(&va_feats, &va_y, &f_val)
    };

    // Baseline honesto: el modelo debe vencer al predictor constante
    // (tasa base en clasificación; media en regresión).
    let base_pred = vec![init_score as f64; va_y.len()];
    let baseline = if is_regression {
        mse(&va_y, &base_pred)
    } else {
        logloss(&va_feats, &va_y, &base_pred)
    };
    // Diagnóstico: predicciones (sigmoid en clasificación; crudas en regresión)
    let val_preds: Vec<f64> = va_feats
        .iter()
        .zip(f_val.iter())
        .map(|(_, &f)| if is_regression { f } else { sigmoid(f) })
        .collect();
    let mut sorted_p = val_preds.clone();
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p_mean = val_preds.iter().sum::<f64>() / val_preds.len() as f64;
    let var = val_preds
        .iter()
        .map(|p| (p - p_mean) * (p - p_mean))
        .sum::<f64>()
        / val_preds.len() as f64;
    let metric_name = if is_regression { "MSE" } else { "logloss" };
    println!("═══ DIAGNÓSTICO DE SELECCIÓN (no test independiente) ═══");
    println!(
        "   val {} modelo: {:.6} · baseline: {:.6} (mejora {:+.6})",
        metric_name,
        best_val,
        baseline,
        baseline - best_val
    );
    println!(
        "   p10={:.6} p50={:.6} p90={:.6} · varianza={:.6}",
        sorted_p[sorted_p.len() / 10],
        sorted_p[sorted_p.len() / 2],
        sorted_p[9 * sorted_p.len() / 10],
        var
    );

    // Only a frozen artifact reaches the test. Test labels cannot select
    // rounds, fit the baseline or update any parameter in this invocation.
    let (gate_loss, gate_baseline, gate_source) = if test_in.is_empty() {
        println!("   sin --test-in: evidencia de selección solamente; NO apta para promoción");
        (best_val, baseline, "selection_only")
    } else {
        let test = build(&test_in);
        require_later_holdout(&test, evidence_end).expect("invalid final holdout");
        let predictions =
            serving_predictions(&model, &test.features).expect("invalid test predictions");
        let constant = vec![init_score as f64; test.labels.len()];
        let loss = if is_regression {
            mse(&test.labels, &predictions)
        } else {
            logloss(&test.features, &test.labels, &predictions)
        };
        let base = if is_regression {
            mse(&test.labels, &constant)
        } else {
            logloss(&test.features, &test.labels, &constant)
        };
        println!(
            "   test posterior: {} modelo {:.6} · baseline {:.6} · n={}",
            metric_name,
            loss,
            base,
            test.labels.len()
        );
        (loss, base, "later_holdout")
    };
    // Margin remains an operator policy, not statistical significance or
    // proof of economic edge. Regression uses improvement vs TRAIN mean:
    // this is a baseline-relative skill score, not conventional test R².
    // D-720 (DÉCIMA OLA · auditoría integral): EL GATE GOBIERNA EL DESTINO.
    //
    // La condición estaba invertida respecto al docstring de este fichero
    // («Sin gate: jamás sobrescribir el modelo vivo con ruido»): al fallar el
    // gate sólo se retornaba SIN `--promote`, es decir, se retornaba cuando el
    // modelo iba al candidato y se CONTINUABA cuando iba al modelo VIVO. Un
    // `train_forest BTCUSDT --promote` sobre un mes sin edge imprimía «el modelo
    // vivo NO se toca» y acto seguido lo sobrescribía; el watcher de god_engine
    // lo hot-swapea en ≤10 s y, desde B3.18, ese modelo decide TODAS las
    // entradas. Ahora un gate no superado nunca escribe el modelo vivo: va al
    // candidato y el proceso termina con código 2 para que cualquier
    // automatización lo detecte. Vale igual para los predictores de regresión
    // (P-1/P-2: _VOL, _VOLU, _OI), cuyo gate usa mejora relativa al baseline.
    if is_regression {
        let skill = if gate_baseline > 0.0 {
            (gate_baseline - gate_loss) / gate_baseline
        } else {
            f64::NAN // zero loss has no positive improvement to certify
        };
        println!(
            "   skill relativo al baseline de train = {:.6} [{}]",
            skill, gate_source
        );
    }
    let gate_ok = passes_loss_gate(is_regression, gate_loss, gate_baseline, gate_margin);
    if !gate_ok {
        println!(
            "🚫 GATE: evidencia inválida o mejora < margen {}. El modelo vivo NO se toca.",
            gate_margin
        );
    }
    let promote = promote && gate_ok;

    // ── 5. Serializar al formato NanoForestData ──────────────────────────
    // Already serialized and validated before the gate above.
    // P-1/P-2: el sufijo del modelo declara su objetivo — {SYM}_MOTOR
    // (primer toque largo), {SYM}_VOL (RMS por evento), {SYM}_VOLU
    // (profundidad media por evento). El sufijo no sustituye un manifiesto.
    // El watcher del host auto-carga cualquier models/{KEY}.json bajo esa
    // key; los predictores de regresión se sirven con predict_raw (sin
    // sigmoid).
    let suffix = match label_mode.as_str() {
        "vol" => "_VOL",
        "volu" => "_VOLU",
        "oi" => "_OI",
        _ => "_MOTOR",
    };
    let out = if promote {
        format!("models/{}{}.json", symbol, suffix)
    } else {
        format!("models/{}{}_CANDIDATE.json", symbol, suffix)
    };
    let mut f = File::create(&out).unwrap();
    serde_json::to_writer_pretty(&mut f, &model).unwrap();
    println!(
        "💾 {} ({} árboles, init {:.4}){}",
        out,
        trees.len(),
        init_score,
        if !gate_ok {
            " — [gate NO superado, revisar antes de promover]"
        } else {
            ""
        }
    );
    if !promote && gate_ok {
        println!("   candidato solamente; promover exige --promote --test-in <archivo posterior independiente>. No reutilizar el test para ajustar hiperparámetros.");
    }
    // D-720: sin evidencia, salida distinta de cero — el fichero escrito es el
    // candidato, no el vivo.
    if !gate_ok {
        std::process::exit(2);
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
        cur = if v <= nd.threshold {
            nd.left as usize
        } else {
            nd.right as usize
        };
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use god_engine_core::ml_inference::NanoForest;

    #[test]
    fn full_population_split_gain_matches_direct_sum() {
        let data: Vec<_> = (0..4096)
            .map(|i| {
                (
                    vec![(i >= 2048) as u8 as f32],
                    if i < 2048 { 1.0 } else { -1.0 },
                    1.0,
                )
            })
            .collect();
        let idx: Vec<_> = (0..4096).collect();
        let probe: Vec<_> = (0..4096).step_by(2).collect();
        let threshold = data[probe[0]].0[0];
        let gain = candidate_gains(&data, &idx, 0, &[threshold], 1.0, 40)[0].unwrap();
        let expected = 2.0 * 2048.0f64.powi(2) / (2048.0 + 1.0 + 1e-12);
        assert!(
            (gain - expected).abs() < 1e-8,
            "gain={gain} expected={expected}"
        );
    }

    #[test]
    fn child_eligibility_uses_full_population_not_probe() {
        let data: Vec<_> = (0..4096)
            .map(|i| {
                (
                    vec![(i >= 80) as u8 as f32],
                    if i < 80 { 1.0 } else { -1.0 },
                    1.0,
                )
            })
            .collect();
        let idx: Vec<_> = (0..4096).collect();
        let probe: Vec<_> = (0..4096).step_by(4).collect();
        let threshold = data[probe[0]].0[0];
        assert!(candidate_gains(&data, &idx, 0, &[threshold], 1.0, 40)[0].is_some());
    }

    #[test]
    fn sampling_stride_respects_budget_on_long_span() {
        assert_eq!(effective_stride(10_000_000, 1000, 100).unwrap(), 100_001);
    }

    #[test]
    fn sampling_stride_never_densifies_requested_grid() {
        assert_eq!(effective_stride(1_000_000, 50_000, 1000).unwrap(), 50_000);
    }

    #[test]
    fn split_gains_match_bruteforce_with_bootstrap_duplicates() {
        let data = vec![
            (vec![0.0], 2.0, 0.5),
            (vec![1.0], -3.0, 1.0),
            (vec![2.0], 0.25, 0.125),
        ];
        let idx = vec![0, 0, 1, 2, 2, 2];
        let thresholds = [1.5, 0.5, 1.5, -1.0, 2.0];
        let gains = candidate_gains(&data, &idx, 0, &thresholds, 0.3, 1);
        let sum = |rows: &[usize]| -> (f64, f64) {
            (
                rows.iter().map(|&i| data[i].1).sum(),
                rows.iter().map(|&i| data[i].2).sum(),
            )
        };
        let score = |(g, h): (f64, f64)| g * g / (h + 0.3 + 1e-12);
        for (&threshold, gain) in thresholds.iter().zip(gains) {
            let (left, right): (Vec<_>, Vec<_>) = idx
                .iter()
                .copied()
                .partition(|&i| data[i].0[0] <= threshold);
            if left.is_empty() || right.is_empty() {
                assert!(gain.is_none());
            } else {
                assert!(
                    (gain.unwrap() - (score(sum(&left)) + score(sum(&right)) - score(sum(&idx))))
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn invalid_split_statistics_cannot_be_selected() {
        for (gradient, hessian) in [(f64::NAN, 1.0), (1.0, -1.0), (f64::INFINITY, 1.0)] {
            let data = vec![(vec![0.0], gradient, hessian), (vec![1.0], -1.0, 1.0)];
            assert_eq!(
                candidate_gains(&data, &[0, 1], 0, &[0.5], 1.0, 1),
                vec![None]
            );
        }
    }

    #[test]
    fn large_node_tree_uses_valid_full_population_children() {
        let mut data: Vec<_> = (0..4096)
            .map(|i| {
                (
                    vec![(i >= 2048) as u8 as f32],
                    if i < 2048 { 1.0 } else { -1.0 },
                    1.0,
                )
            })
            .collect();
        let mut nodes = Vec::new();
        build_tree(
            &mut data,
            &mut (0..4096).collect(),
            0,
            1,
            40,
            1.0,
            &[0],
            24,
            &mut StdRng::seed_from_u64(42),
            &mut nodes,
        );
        assert_eq!(nodes.len(), 3);
        assert!(eval_tree(&nodes, &[0.0]) > 0.99);
        assert!(eval_tree(&nodes, &[1.0]) < -0.99);
    }

    #[test]
    fn sampling_budget_caps_attempts_inclusive_end_and_irregular_ticks() {
        for span in [0, 1, 9, 100, 1000] {
            for cap in [1, 2, 7, 100] {
                for stride in [1, 3, 50] {
                    let mut plan = SamplingBudget::new(span, stride, cap).unwrap();
                    let selected: Vec<_> = (0..=span).filter(|&t| plan.reserve(t)).collect();
                    assert!(selected.len() <= cap);
                    assert!(selected.windows(2).all(|w| w[1] - w[0] >= stride));
                }
            }
        }
        let mut plan = SamplingBudget::new(1000, 50, 3).unwrap();
        for t in [10, 10, 11, 500, 999, 5000, 10000] {
            plan.reserve(t);
        }
        assert_eq!(plan.attempts, 3); // Counter applies even beyond the planned span.
    }

    #[test]
    fn sampling_budget_rejects_zero_and_handles_timestamp_overflow() {
        assert!(SamplingBudget::new(1000, 0, 10).is_err());
        assert!(SamplingBudget::new(1000, 10, 0).is_err());
        let mut plan = SamplingBudget::new(u64::MAX, 1, 1).unwrap();
        assert!(plan.reserve(u64::MAX));
        assert!(!plan.reserve(u64::MAX));
        let mut plan = SamplingBudget::new(100, 10, 10).unwrap();
        assert!(plan.reserve(u64::MAX - 1));
        assert!(!plan.reserve(u64::MAX));
    }

    fn sample_set(windows: &[(u64, u64)]) -> TrainingSamples {
        let mut samples = TrainingSamples::default();
        for &(start, end) in windows {
            samples.push(vec![start as f32], start as f64, start, end);
        }
        samples
    }

    #[test]
    fn temporal_purge_uses_closed_information_intervals() {
        let train = sample_set(&[(1, 3), (4, 10), (7, 11)]);
        let validation = sample_set(&[(10, 12), (13, 14)]);
        let (kept, purged) = purge_training(train, &validation).unwrap();
        assert_eq!(purged, 2); // equality at 10 also overlaps
        assert_eq!(kept.labels, vec![1.0]);
        assert_eq!(kept.features, vec![vec![1.0]]);
    }

    #[test]
    fn purge_handles_nonmonotone_label_ends_without_losing_alignment() {
        let train = sample_set(&[(1, 100), (2, 3), (4, 50), (5, 6)]);
        let validation = sample_set(&[(10, 12)]);
        let (kept, purged) = purge_training(train, &validation).unwrap();
        assert_eq!(purged, 2);
        assert_eq!(kept.labels, vec![2.0, 5.0]);
        assert_eq!(kept.features, vec![vec![2.0], vec![5.0]]);
        assert_eq!(
            kept.intervals.iter().map(|v| v.end).collect::<Vec<_>>(),
            vec![3, 6]
        );
    }

    #[test]
    fn temporal_split_purges_only_training_and_keeps_selection_intact() {
        let mut training = sample_set(&[(1, 2), (3, 6), (5, 8), (7, 9)]);
        let selection = training.split_at(2);
        let (kept, purged) = purge_training(training, &selection).unwrap();
        assert_eq!(kept.labels, vec![1.0]);
        assert_eq!(purged, 1);
        assert_eq!(selection.labels, vec![5.0, 7.0]);
    }

    #[test]
    fn reversed_identical_and_exhausted_training_windows_fail_closed() {
        assert!(purge_training(sample_set(&[(10, 11)]), &sample_set(&[(10, 11)])).is_err());
        assert!(purge_training(sample_set(&[(10, 11)]), &sample_set(&[(1, 3)])).is_err());
        assert!(purge_training(sample_set(&[(1, 11)]), &sample_set(&[(10, 12)])).is_err());
    }

    #[test]
    fn sample_contract_rejects_misalignment_nonfinite_and_bad_intervals() {
        for windows in [
            vec![],
            vec![(0, 1)],
            vec![(2, 1)],
            vec![(1, 2), (1, 3)],
            vec![(2, 3), (1, 2)],
        ] {
            assert!(sample_set(&windows).validate().is_err());
        }
        let mut bad = sample_set(&[(1, 2)]);
        bad.labels.clear();
        assert!(bad.validate().is_err());
        let mut bad = sample_set(&[(1, 2)]);
        bad.features[0][0] = f32::NAN;
        assert!(bad.validate().is_err());
        let mut bad = sample_set(&[(1, 2)]);
        bad.labels[0] = f64::INFINITY;
        assert!(bad.validate().is_err());
        let mut bad = sample_set(&[(1, 2), (3, 4)]);
        bad.features[1].push(0.0);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn holdout_must_follow_the_latest_label_end_not_the_latest_start() {
        let evidence = sample_set(&[(1, 100), (5, 6)]);
        assert!(require_later_holdout(&sample_set(&[(99, 101)]), evidence.max_end()).is_err());
        assert!(require_later_holdout(&sample_set(&[(100, 101)]), evidence.max_end()).is_err());
        assert!(require_later_holdout(&sample_set(&[(101, 102)]), evidence.max_end()).is_ok());
    }

    #[test]
    fn promotion_requires_a_separate_test_but_research_candidate_does_not() {
        assert!(require_promotion_holdout(true, "").is_err());
        assert!(require_promotion_holdout(false, "").is_ok());
        assert!(require_promotion_holdout(true, "later.bin").is_ok());
    }

    #[test]
    fn loss_gate_rejects_invalid_and_zero_baseline_evidence() {
        for regression in [false, true] {
            for (loss, base, margin) in [
                (f64::NAN, 1.0, 0.1),
                (0.0, f64::INFINITY, 0.1),
                (-1.0, 1.0, 0.1),
                (0.1, 1.0, f64::NAN),
                (0.1, 1.0, -0.1),
            ] {
                assert!(!passes_loss_gate(regression, loss, base, margin));
            }
        }
        assert!(!passes_loss_gate(true, 0.0, 0.0, 0.0));
        assert!(passes_loss_gate(true, 0.8e-20, 1e-20, 0.1));
        assert!(!passes_loss_gate(true, 1.2e-20, 1e-20, 0.1));
        assert!(passes_loss_gate(false, 0.4, 0.5, 0.01));
        assert!(!passes_loss_gate(false, 0.51, 0.5, 0.01));
    }

    #[test]
    fn tick_contract_rejects_bad_chronology_and_nonfinite_market_values() {
        let good = BinTick {
            ts: 10,
            bid: 100.0,
            ask: 101.0,
            bq: 1.0,
            aq: 2.0,
        };
        assert!(valid_tick(&good));
        assert!(validate_tick_order(&[good, good]).is_ok());
        assert!(validate_tick_order(&[good, BinTick { ts: 9, ..good }]).is_err());
        assert!(validate_tick_order(&[BinTick { ts: 0, ..good }]).is_err());
        for bad in [
            BinTick {
                bid: f64::NAN,
                ..good
            },
            BinTick { ask: 99.0, ..good },
            BinTick { bq: -1.0, ..good },
            BinTick {
                aq: f64::INFINITY,
                ..good
            },
        ] {
            assert!(!valid_tick(&bad));
        }
    }

    #[test]
    fn serialized_json_and_binary_preserve_additive_predictions() {
        let trees = vec![stump(1.0, -1.0), stump(4.0, -5.0)];
        for lr in [0.05, 0.1, 0.5, 1.0] {
            let model = serialize_forest(&trees, 0.25, lr).unwrap();
            let json: NanoForestData =
                serde_json::from_slice(&serde_json::to_vec(&model).unwrap()).unwrap();
            let binary: NanoForestData =
                bincode::deserialize(&bincode::serialize(&model).unwrap()).unwrap();
            for data in [json, binary] {
                let forest = NanoForest::from_data(data).unwrap();
                for x in [[0.1], [0.9]] {
                    let expected =
                        0.25 + lr * trees.iter().map(|t| eval_tree(t, &x) as f64).sum::<f64>();
                    let (raw, p) = forest.predict_raw_checked(&x).unwrap();
                    assert!((raw as f64 - expected).abs() < 2e-6);
                    assert!((p as f64 - sigmoid(expected)).abs() < 2e-6);
                }
            }
        }
    }
    #[test]
    fn invalid_exports_are_rejected_without_writing_a_model() {
        for lr in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(serialize_forest(&[stump(1.0, -1.0)], 0.0, lr).is_err());
        }
        assert!(serialize_forest(&[stump(f32::MAX, 0.0)], 0.0, 2.0).is_err());
        assert!(serialize_forest(&[], 0.0, 1.0).is_err());
        let mut invalid = stump(1.0, -1.0);
        invalid[0].left = 3;
        assert!(serialize_forest(&[invalid], 0.0, 1.0).is_err());
    }
    #[test]
    fn retained_ensemble_predictions_exclude_discarded_rounds() {
        let mut trees = vec![stump(1.0, -1.0), stump(4.0, -5.0)];
        trees.truncate(1);
        let model = serialize_forest(&trees, 0.25, 0.1).unwrap();
        let p = serving_predictions(&model, &[vec![0.1], vec![0.9]]).unwrap();
        assert!((p[0] - 0.35).abs() < 1e-6);
        assert!((p[1] - 0.15).abs() < 1e-6);
        assert!(serving_predictions(&model, &[vec![f32::NAN]]).is_err());
    }
    #[test]
    fn regression_residual_is_the_negative_gradient_in_both_directions() {
        for (prediction, target) in [(0.0, 2.0), (2.0, -3.0)] {
            let residual = regression_residual(prediction, target);
            let epsilon = 1e-5;
            let loss = |p: f64| 0.5 * (p - target).powi(2);
            let gradient =
                (loss(prediction + epsilon) - loss(prediction - epsilon)) / (2.0 * epsilon);
            assert!((residual + gradient).abs() < 1e-8);
        }
    }

    fn stump(left: f32, right: f32) -> Vec<TreeNode> {
        vec![
            TreeNode {
                feature: 0,
                threshold: 0.5,
                left: 1,
                right: 2,
                value: 0.0,
            },
            TreeNode {
                feature: -1,
                threshold: 0.0,
                left: -1,
                right: -1,
                value: left,
            },
            TreeNode {
                feature: -1,
                threshold: 0.0,
                left: -1,
                right: -1,
                value: right,
            },
        ]
    }
    #[test]
    fn exported_forest_keeps_learning_rate() {
        let trees = vec![stump(2.0, -3.0)];
        let forest = NanoForest::from_data(serialize_forest(&trees, 0.25, 0.1).unwrap()).unwrap();
        for x in [[0.1], [0.9]] {
            let expected = 0.25 + 0.1 * eval_tree(&trees[0], &x) as f64;
            assert!((forest.predict_raw(&x).0 as f64 - expected).abs() < 1e-6);
        }
    }
    #[test]
    fn exported_forest_keeps_each_tree_children() {
        let trees = vec![stump(1.0, -1.0), stump(4.0, -5.0)];
        let forest = NanoForest::from_data(serialize_forest(&trees, 0.0, 1.0).unwrap()).unwrap();
        for x in [[0.1], [0.9]] {
            let expected: f32 = trees.iter().map(|t| eval_tree(t, &x)).sum();
            assert_eq!(forest.predict_raw(&x).0, expected);
        }
    }
    #[test]
    fn regression_newton_step_reduces_squared_error() {
        let prediction = 0.0;
        let target = 2.0;
        let mut data = vec![(vec![0.0], regression_residual(prediction, target), 1.0)];
        let mut nodes = Vec::new();
        build_tree(
            &mut data,
            &mut vec![0],
            0,
            0,
            1,
            1.0,
            &[0],
            2,
            &mut StdRng::seed_from_u64(1),
            &mut nodes,
        );
        let updated = prediction + 0.1 * eval_tree(&nodes, &[0.0]) as f64;
        assert!((target - updated).powi(2) < (target - prediction).powi(2));
    }
}
