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
//! `StatefulEngine::get_universal_features()` que alimenta al motor en vivo
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
//! ── UNIDADES DE LAS ETIQUETAS DE REGRESIÓN (revisión D-731/D-732) ───────
//!
//! `--label vol` YA NO es «RMS de retornos por tick». Era
//! `sqrt(Σ r² / n_ticks) · 100`, es decir la volatilidad DIVIDIDA por la
//! intensidad de ticks: un tramo con muchos trades pequeños salía MENOS
//! volátil que uno tranquilo con pocos saltos, justo al revés de lo que
//! mide el mercado. Ahora la etiqueta es la VOLATILIDAD REALIZADA DEL
//! HORIZONTE, `sqrt(Σ r²) · 100` sobre (t, t+τ] — es decir σ(τ) en % de
//! precio, la misma magnitud que el VOL-BRAKE compara contra la base del
//! modelo. CONSECUENCIA: los `models/{SYM}_VOL.json` entrenados antes de
//! este cambio quedan OBSOLETOS — su `init_score` (la media del label) y
//! sus hojas están en la escala vieja (÷ sqrt(n_ticks)); hay que
//! reentrenarlos. El serving NO cambia: el freno usa el COCIENTE
//! pronóstico/base y ese cociente es invariante de escala, pero mezclar un
//! modelo viejo con uno nuevo sí compara manzanas con peras.
//!
//! `--label volu` YA NO es «profundidad media del libro». Sumaba
//! `(bid_qty + ask_qty)` y lo dividía por el número de ticks, pero en el
//! tape de aggTrades ese par NO es un libro: `binance_vision_sync` lo
//! fabrica como `qty + base` en el lado del agresor y `base` en el otro,
//! con `base = max(0,25·qty; 0,1)`. La suma valía ≈1,5·qty con un suelo
//! artificial y la media por tick volvía a dividir por la intensidad: la
//! etiqueta era «tamaño medio de trade inflado un 50%», no volumen. Ahora
//! es el NOCIONAL NEGOCIADO del horizonte, `Σ |qty| · precio` sobre
//! (t, t+τ], con `|qty| = |bid_qty − ask_qty|` (la diferencia CANCELA la
//! base fabricada y recupera el `qty` exacto del aggTrade). Si el tape no
//! procede de aggTrades el binario aborta en vez de etiquetar ruido.
//!
//! ── BASELINE DEL GATE EN REGRESIÓN ──────────────────────────────────────
//!
//! El baseline honesto de una magnitud persistente NO es su media: es la
//! PERSISTENCIA (σ futura ≈ σ reciente, volumen futuro ≈ volumen reciente).
//! Contra la media, un R² de 0,1 no demuestra nada porque la persistencia
//! ya explica bastante más. El gate exige ahora batir el MSE de la
//! persistencia medida sobre la MISMA partición de validación; el R² contra
//! la media sigue imprimiéndose, etiquetado como lo que es (informativo).
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

// ── Magnitudes físicas del tape y ventanas CAUSALES ──────────────────────
//
// Todo lo que sigue es aritmética pura sobre ticks: se aísla aquí para que
// las pruebas puedan fijar la geometría de las ventanas (rasgos con
// información ≤ t, etiqueta en (t, t+τ], persistencia en (t−τ, t]) sin
// levantar el motor ni leer un fichero.

/// Muestras construidas desde un tape: rasgos (información ≤ t), etiqueta
/// (ventana (t, t+τ]), baseline de PERSISTENCIA (misma magnitud en (t−τ, t],
/// información ≤ t) y la marca temporal t de cada muestra, que la purga del
/// solape train/val necesita.
struct Samples {
    feats: Vec<Vec<f32>>,
    labels: Vec<f64>,
    persist: Vec<f64>,
    ts: Vec<u64>,
}

/// Acumuladores de UNA ventana temporal de ticks.
#[derive(Default, Clone, Copy, PartialEq, Debug)]
struct WindowStats {
    /// Σ r² de los retornos de mid dentro de la ventana — varianza realizada.
    sum_sq: f64,
    /// Σ |qty| · precio — nocional negociado dentro de la ventana.
    notional: f64,
    /// Nº de retornos observados (0 ⇒ ventana sin información de precio).
    n_ret: usize,
}

/// Cantidad NEGOCIADA de un tick del tape de aggTrades.
///
/// POR QUÉ: `binance_vision_sync` no tiene libro, así que fabrica el par
/// (bid_qty, ask_qty) a partir del `qty` del aggTrade: un lado recibe
/// `qty + base` y el otro `base`. La SUMA arrastra `2·base` (≈ ×1,5 el qty,
/// con un suelo absoluto), pero la DIFERENCIA cancela la base y devuelve el
/// `qty` exacto, sea cual sea la fórmula de `base`. Ésta es la única
/// cantidad negociada realmente medible en este tape.
fn agg_trade_qty(bq: f64, aq: f64) -> f64 {
    (bq - aq).abs()
}

/// ¿El par (bid_qty, ask_qty) procede de la fabricación de aggTrades?
///
/// Espejo EXACTO del contrato de `src/bin/binance_vision_sync.rs`:
/// `base = max(0,25·qty; 0,1)`, lado agresor = `qty + base`. De ahí la
/// identidad `min = max(0,25·(max − min); 0,1)`, que se cumple tick a tick
/// en ese tape y NO se cumple en el tape sintético de klines
/// (`parquet_to_bin`, donde ambos lados son fracciones del volumen de vela).
/// Si el productor cambiara su fórmula esta comprobación debe romperse
/// RUIDOSAMENTE: etiquetar «nocional» sobre un par que no es de aggTrades
/// sería inventar la magnitud.
fn is_aggtrades_pair(bq: f64, aq: f64) -> bool {
    if !bq.is_finite() || !aq.is_finite() {
        return false;
    }
    let (lo, hi) = if bq <= aq { (bq, aq) } else { (aq, bq) };
    let qty = hi - lo;
    if !(qty.is_finite() && qty > 0.0) {
        return false;
    }
    let expected = (qty * 0.25).max(0.1);
    // Tolerancia NUMÉRICA, no de decisión: 1e-9 relativo queda siete órdenes
    // por encima del épsilon de f64 (2,2e-16) que acumula el round-trip
    // cálculo→fichero→mmap, y varios órdenes por debajo de cualquier
    // desviación estructural (el tape de klines da directamente qty = 0).
    (lo - expected).abs() <= expected * 1e-9
}

/// σ REALIZADA de la ventana, en % de precio: `sqrt(Σ r²) · 100`.
///
/// No se divide por el número de ticks: eso convertiría la volatilidad en
/// «volatilidad por tick» y haría que un tramo con mucha actividad pareciera
/// más tranquilo. Con τ fijo (el horizonte del muestreo) esto ES σ(τ), la
/// magnitud que compara el VOL-BRAKE.
fn realized_vol_pct(sum_sq: f64) -> f64 {
    sum_sq.max(0.0).sqrt() * 100.0
}

/// Recorre `ticks` acumulando retornos² y nocional, anclando el primer
/// retorno en `anchor_mid` (el mid del borde izquierdo de la ventana).
/// `anchor_mid ≤ 0` ⇒ el primer tick sólo fija el ancla y no genera retorno.
fn accumulate_window(ticks: &[BinTick], anchor_mid: f64) -> WindowStats {
    let mut st = WindowStats::default();
    let mut prev = anchor_mid;
    for tk in ticks {
        if tk.bid <= 0.0 || tk.ask <= 0.0 || tk.bid > tk.ask {
            continue;
        }
        let m = (tk.bid + tk.ask) / 2.0;
        if prev > 0.0 {
            let r = (m - prev) / prev;
            st.sum_sq += r * r;
            st.n_ret += 1;
        }
        st.notional += agg_trade_qty(tk.bq, tk.aq) * m;
        prev = m;
    }
    st
}

/// Fin EXCLUSIVO de la ventana futura: primer índice > `i` con `ts > deadline`.
/// La ventana resultante `i+1 .. fin` es estrictamente (t, t+τ]: nunca incluye
/// el propio tick t ni nada posterior al vencimiento.
fn forward_window_end(raw: &[BinTick], i: usize, deadline: u64) -> usize {
    let mut hi = i + 1;
    while hi < raw.len() && raw[hi].ts <= deadline {
        hi += 1;
    }
    hi
}

/// Inicio de la ventana pasada (t−τ, t] y mid de anclaje (último mid válido
/// con `ts ≤ start`). La ventana `inicio ..= i` sólo contiene información
/// disponible en t — es el baseline de persistencia, no puede mirar futuro.
/// Ancla 0.0 ⇒ el tape no cubre τ hacia atrás desde `i`.
fn trailing_window_start(raw: &[BinTick], i: usize, start: u64) -> (usize, f64) {
    let mut lo = i;
    while lo > 0 && raw[lo - 1].ts > start {
        lo -= 1;
    }
    let mut a = lo;
    let mut anchor = 0.0;
    while a > 0 {
        a -= 1;
        let tk = &raw[a];
        if tk.bid > 0.0 && tk.ask > 0.0 && tk.bid <= tk.ask {
            anchor = (tk.bid + tk.ask) / 2.0;
            break;
        }
    }
    (lo, anchor)
}

/// PURGA (López de Prado) del solape entre train y validación.
///
/// POR QUÉ: con τ = 300 s y stride = 50 s las muestras se solapan 6×, así que
/// la etiqueta de las últimas muestras de train se resuelve DENTRO del tramo
/// de validación — el modelo «ya vio» ese futuro y la métrica de validación
/// sale optimista. Devuelve el fin (exclusivo) del train: la última muestra
/// cuya ventana de etiqueta (t, t+τ] termina antes de la primera muestra de
/// validación. El número de muestras descartadas no es una constante: lo fija
/// el propio horizonte τ contra la densidad real del muestreo.
fn purge_end(ts: &[u64], split: usize, horizon_ms: u64) -> usize {
    if split == 0 || split >= ts.len() {
        return split.min(ts.len());
    }
    let first_val_ts = ts[split];
    ts[..split]
        .iter()
        .position(|&t| t.saturating_add(horizon_ms) > first_val_ts)
        .unwrap_or(split)
}

/// Veredicto del gate de REGRESIÓN. Devuelve
/// `(R² contra la media, skill contra la persistencia, pasa)`.
///
/// POR QUÉ: la media constante es un baseline de juguete para magnitudes
/// persistentes (σ, volumen). Un R² de 0,1 contra la media puede convivir con
/// un modelo PEOR que repetir el valor reciente. El gate exige batir la
/// persistencia por `margin` (fracción del MSE de la persistencia); el R²
/// contra la media queda como diagnóstico. Persistencia degenerada (MSE nulo
/// o no finito) ⇒ no hay baseline comparable ⇒ NO pasa.
fn regression_gate(mse_model: f64, mse_mean: f64, mse_persist: f64, margin: f64) -> (f64, f64, bool) {
    let r2_mean = if mse_mean.is_finite() && mse_mean > 0.0 {
        (mse_mean - mse_model) / mse_mean
    } else {
        0.0
    };
    if !(mse_persist.is_finite() && mse_persist > 0.0) || !mse_model.is_finite() {
        return (r2_mean, 0.0, false);
    }
    let skill = (mse_persist - mse_model) / mse_persist;
    (r2_mean, skill, skill >= margin)
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

/// B3.4 — carga las series macro reales de data/macro. VIX/SP500/NASDAQ
/// (Yahoo v8, cierres idénticos a FRED) son OBLIGATORIAS: faltan ⇒ aborto,
/// porque una columna de ceros constante es capacidad fantasma. DXY
/// (DTWEXBGS, sólo FRED) se TOLERA ausente con warning: el CDN de FRED
/// abre y cierra ventanas desde esta red y un re-run del sync lo completa;
/// mientras tanto la dim queda en 0 neutro y NINGÚN árbol parte por ella
/// (columna constante = sin splits — no hay ruptura de paridad).
fn load_macro_series() -> (Vec<(u64, f64)>, Vec<(u64, f64)>, Vec<(u64, f64)>, Vec<(u64, f64)>) {
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
            eprintln!("⚠️ macro {tag}: historia irreal ({} filas) — dim en 0 neutro", rows.len());
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
    // P-1/P-2/P-3c — objetivo: dir (barrera triple HOST-010) | vol (σ futura,
    // regresión) | volu (profundidad media, regresión) | oi (ΔOI% a horizonte,
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
            eprintln!("❌ --label oi sin histórico: {} ({}) — ejecuta oi_history_sync", path, e);
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
        println!("   [oi] {} filas · span {:.1} días (validación DENTRO de ventana)", rows.len(),
            (rows.last().unwrap().0 - rows[0].0) as f64 / 86_400_000.0);
        rows
    } else {
        Vec::new()
    };

    println!("🌲 [TRAIN-FOREST] {} ← {}", symbol, in_path);
    println!("   muestras≤{} horizonte={}ms stride={}ms árboles≤{} lr={} depth={} λ={}",
             max_samples, horizon_ms, stride_ms, n_rounds, lr, max_depth, lambda);

    let oi_series_ref = &oi_series;
    // ── 1+2. Muestras: features+etiquetas desde ticks (reutilizable) ────
    let build = |path: &str| -> Samples {
        let file = File::open(path).unwrap_or_else(|e| {
            eprintln!("❌ no pude abrir {}: {}", path, e);
            std::process::exit(1);
        });
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.unwrap();
        let header_off = if mmap.len() >= 8 && &mmap[..8] == b"TGMTICK1" { 8usize } else { 0 };
        let sz = std::mem::size_of::<BinTick>();
        let n_total = (mmap.len() - header_off) / sz;
        if n_total < 50_000 {
            eprintln!("❌ datos insuficientes: {} ticks ({})", n_total, path);
            std::process::exit(1);
        }
        let ptr = unsafe { mmap.as_ptr().add(header_off) } as *const BinTick;
        let raw = unsafe { std::slice::from_raw_parts(ptr, n_total) };
        println!("   [{}] {} ticks crudos", path, n_total);

        let last_ts = raw[n_total - 1].ts;
        let span_ms = last_ts.saturating_sub(raw[0].ts);
        let stride_ms_eff = if stride_ms.max(1) as u64 * (max_samples as u64) < span_ms {
            stride_ms
        } else {
            (span_ms / (max_samples as u64)).max(1_000)
        };
        println!("   [{}] span {:.1} días · stride efectivo {}ms", path,
                 span_ms as f64 / 86_400_000.0, stride_ms_eff);
        // D-732 — `--label volu` etiqueta NOCIONAL NEGOCIADO, y eso exige que
        // (bid_qty, ask_qty) venga de la fabricación de aggTrades: sólo ahí la
        // diferencia de los dos lados recupera el `qty` del trade. En un tape
        // sintético de klines la diferencia es una fracción del volumen de vela
        // y la etiqueta mediría una invención. Una pasada aritmética sobre los
        // ticks válidos (despreciable frente al O(n·ventana) del bucle
        // principal) decide: si ALGÚN tick incumple la identidad del productor,
        // se aborta en vez de entrenar sobre ruido.
        if label_mode == "volu" {
            let (mut n_valid, mut n_bad) = (0usize, 0usize);
            for tk in raw.iter() {
                if tk.bid <= 0.0 || tk.ask <= 0.0 || tk.bid > tk.ask || tk.ts == 0 {
                    continue;
                }
                n_valid += 1;
                if !is_aggtrades_pair(tk.bq, tk.aq) {
                    n_bad += 1;
                }
            }
            if n_valid == 0 || n_bad > 0 {
                eprintln!(
                    "❌ --label volu sobre un tape que NO es de aggTrades: {}/{} ticks incumplen \
                     la identidad min = max(0,25·qty; 0,1) de binance_vision_sync ({}). \
                     El nocional negociado no es medible en este fichero — usa el tape de \
                     aggTrades o no entrenes este predictor.",
                    n_bad, n_valid, path
                );
                std::process::exit(1);
            }
            println!("   [{}] tape aggTrades verificado: qty recuperable en {} ticks", path, n_valid);
        }
        let mut engine = StatefulEngine::new();
        // B3.4 — bloque MACRO real (dims 44..48): series FRED VIXCLS/SP500/
        // DTWEXBGS/NASDAQCOM de data/macro (macro_history_sync). Join as-of
        // ESTRICTO: último cierre de un día ANTERIOR al tick — el cierre del
        // mismo día aún no existe intradía (lookahead). Sin las cuatro series
        // reales no se entrena: ceros harían columnas muertas, y las columnas
        // muertas son capacidad fantasma (directriz del operador).
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
        let mut feats: Vec<Vec<f32>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();
        // D-733 — baseline de PERSISTENCIA: la misma magnitud de la etiqueta
        // medida en la ventana ANTERIOR (t−τ, t]. Se guarda por muestra para
        // poder evaluarla EXACTAMENTE sobre la partición de validación.
        let mut persist: Vec<f64> = Vec::new();
        // Marca temporal de cada muestra: la purga del solape train/val
        // necesita saber cuándo se resuelve cada etiqueta.
        let mut sample_ts: Vec<u64> = Vec::new();
        let mut neutrals = 0usize;
        // Sin contexto temporal COMPLETO (ventana (t−τ, t] dentro del tape, o
        // fila de OI anterior a t−τ) no hay persistencia medible ⇒ la muestra
        // se descarta en los modos de regresión, nunca se rellena con un
        // sustituto inventado.
        let mut sin_contexto = 0usize;
        let tp_pct = 0.0036;
        let sl_pct = 0.0018;
        // Primer tick UTILIZABLE del tape (no `raw[0].ts`: la cabecera puede ir
        // seguida de ticks inválidos con ts = 0). Marca el borde por debajo del
        // cual la ventana (t−τ, t] dejaría de estar cubierta por datos.
        let mut first_ts: u64 = 0;
        let mut next_sample_ts: u64 = 0;
        let mut warmup = 0usize;
        for i in 0..n_total {
            let t = &raw[i];
            if t.bid <= 0.0 || t.ask <= 0.0 || t.bid > t.ask || t.ts == 0 {
                continue;
            }
            if first_ts == 0 {
                first_ts = t.ts;
            }
            let mid = (t.bid + t.ask) / 2.0;
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
            if warmup >= 100 && t.ts >= next_sample_ts && t.ts + horizon_ms <= last_ts {
                next_sample_ts = t.ts + stride_ms_eff;
                // B2.3: vector swing(34) ⊕ espectral(10) — IDÉNTICO al
                // input de inferencia en god-engine-core/lib.rs. El bloque
                // espectral (FFT, Hurst multifractal, momentum de EMAs de
                // kline) responde al resultado "34 features sin edge".
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
                    // dirección, todos funciones del MISMO horizonte τ =
                    // horizon_ms (no hay bandas ni pares de valores: τ es el
                    // único parámetro de escala).
                    //   `--label vol`  ⇒ σ(τ) realizada = sqrt(Σ r²)·100  [% de
                    //                    precio sobre (t, t+τ]]  (D-731)
                    //   `--label volu` ⇒ nocional negociado Σ |qty|·precio en
                    //                    (t, t+τ]                        (D-732)
                    //   `--label oi`   ⇒ ΔOI% en (t, t+τ] por join as-of
                    // REGRESIÓN (grad = f−y, hess = 1, init = media; serving por
                    // predict_raw SIN sigmoid). `dir` ⇒ barrera triple HOST-010
                    // (clasificación, como siempre).
                    //
                    // Cada muestra de regresión lleva además su PERSISTENCIA:
                    // la misma magnitud medida en (t−τ, t], información
                    // estrictamente ≤ t. Es el baseline que el gate exige batir.
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
                                    if oi_series_ref.first().map(|(ots, _)| *ots > ts).unwrap_or(true) {
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
                            let deadline = t.ts + horizon_ms;
                            // D-733 — persistencia del ΔOI: el ΔOI% del
                            // horizonte ANTERIOR (t−τ, t], todo con ts ≤ t.
                            let past = t.ts.saturating_sub(horizon_ms);
                            let (Some(oi_now), Some(oi_fut), Some(oi_past)) =
                                (oi_asof(t.ts), oi_asof(deadline), oi_asof(past))
                            else {
                                sin_contexto += 1;
                                continue;
                            };
                            // honestidad: el futuro debe ser una fila REAL del
                            // histórico (deadline ≤ última fila), no el último
                            // valor colado por el as-of del borde.
                            if deadline > oi_series_ref.last().unwrap().0 {
                                continue;
                            }
                            // …y el pasado debe estar DENTRO del histórico: si
                            // t−τ precede a la primera fila, el as-of no tiene
                            // valor que devolver y la persistencia sería una
                            // invención.
                            if past < oi_series_ref[0].0 || oi_past <= 0.0 {
                                sin_contexto += 1;
                                continue;
                            }
                            let label = (oi_fut - oi_now) / oi_now * 100.0;
                            let pers = (oi_now - oi_past) / oi_past * 100.0;
                            if label.is_finite() && pers.is_finite() {
                                feats.push(full.to_vec());
                                labels.push(label);
                                persist.push(pers);
                                sample_ts.push(t.ts);
                            }
                        } else {
                        let deadline = t.ts + horizon_ms;
                        // Ventana FUTURA (t, t+τ]: arranca en i+1 (nunca en el
                        // propio tick t) y corta en el primer ts > vencimiento.
                        let fwd_end = forward_window_end(raw, i, deadline);
                        let fwd = accumulate_window(&raw[i + 1..fwd_end], mid);
                        if fwd.n_ret == 0 {
                            continue;
                        }
                        // Ventana PASADA (t−τ, t]: mismo ancho τ, información
                        // estrictamente ≤ t. Sin τ completo hacia atrás (arranque
                        // del fichero) no hay persistencia y la muestra se cae.
                        let past = t.ts.saturating_sub(horizon_ms);
                        if past < first_ts {
                            sin_contexto += 1;
                            continue;
                        }
                        let (lo, anchor) = trailing_window_start(raw, i, past);
                        let back = accumulate_window(&raw[lo..=i], anchor);
                        if back.n_ret == 0 {
                            sin_contexto += 1;
                            continue;
                        }
                        // D-731/D-732 — unidades: σ(τ) en % de precio y nocional
                        // negociado del horizonte. Ninguna de las dos se divide
                        // por el número de ticks: esa división convertía la
                        // magnitud en «por tick» y la hacía DECRECER cuando el
                        // mercado se activaba, justo al revés de lo que mide.
                        let (label, pers) = if label_mode == "vol" {
                            (realized_vol_pct(fwd.sum_sq), realized_vol_pct(back.sum_sq))
                        } else {
                            (fwd.notional, back.notional)
                        };
                        if label.is_finite() && pers.is_finite() {
                            feats.push(full.to_vec());
                            labels.push(label);
                            persist.push(pers);
                            sample_ts.push(t.ts);
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
                    // tocar ±sl_pct sin llegar al TP ⇒ timeout neutral
                    // (descartado — coincide con la escalera trailing: un
                    // trade que toca +0.18% y vuelve cierra en BE). El corto
                    // consume 1−p en serve (ml_thr_short): con esta definición
                    // es P(SL largo primero) — proxy honesto de la hipótesis
                    // corta.
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
                        if fut_mid >= long_tp {
                            label = 1.0;
                            break 'barrier;
                        }
                    }
                    if (label - 0.5).abs() < 1e-9 {
                        neutrals += 1;
                    } else {
                        feats.push(full.to_vec());
                        labels.push(label);
                        // Clasificación: el baseline es la tasa base del propio
                        // modelo, no una persistencia — se rellena con la tasa
                        // base más adelante y NUNCA se usa en el gate `dir`.
                        persist.push(0.5);
                        sample_ts.push(t.ts);
                    }
                    } // fin modo dir
                }
            }
        }
        let n = labels.len();
        if n < 5_000 {
            eprintln!("❌ muestras insuficientes: {} (+{} neutros, +{} sin contexto temporal)",
                      n, neutrals, sin_contexto);
            std::process::exit(1);
        }
        if label_mode == "dir" {
            let pos_rate = labels.iter().filter(|&&y| y > 0.5).count() as f64 / n as f64;
            println!("   [{}] {} muestras decisivas ({} neutros) · largo {:.1}%",
                     path, n, neutrals, pos_rate * 100.0);
        } else {
            let mean_y = labels.iter().sum::<f64>() / n as f64;
            let mean_p = persist.iter().sum::<f64>() / n as f64;
            println!("   [{}] {} muestras ({} sin contexto temporal) · label[{}] media {:.6} \
                      · persistencia media {:.6}",
                     path, n, sin_contexto, label_mode, mean_y, mean_p);
        }
        Samples { feats, labels, persist, ts: sample_ts }
    };
    let s_train = build(&in_path);
    let (feats, labels, persist_all, sample_ts) =
        (s_train.feats, s_train.labels, s_train.persist, s_train.ts);
    let n = labels.len();

    // ── 3. Split: temporal 80/20 CON PURGA, o CRUZADO por archivo ────────
    // El split cruzado (train AGO → val SEP) es la validación MÁS dura y
    // honesta disponible: el edge debe sobrevivir un mes de mercado nuevo.
    //
    // D-734 — PURGA DEL SOLAPE (López de Prado). Con τ = 300 s y stride
    // efectivo de 50 s las ventanas de etiqueta se solapan 6×: las últimas
    // muestras de train resuelven su barrera/su σ DENTRO del tramo de
    // validación. Eso es fuga: la validación deja de medir mercado nuevo. Se
    // descartan del train las muestras cuya ventana (t, t+τ] alcanza la
    // primera muestra de validación. El recorte no es un número elegido: lo
    // fija τ contra la densidad real del muestreo.
    let val_in = arg("--val-in", "");
    let (tr_feats, va_feats, tr_y, va_y, va_persist): (
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) = if val_in.is_empty() {
        let split = n * 8 / 10;
        let tr_end = purge_end(&sample_ts, split, horizon_ms);
        // Mínimo FÍSICO del entrenador, no un número elegido: un árbol no
        // puede partir con menos de 2·min_child filas, así que por debajo de
        // ese tamaño el GBDT sólo puede devolver la hoja raíz (= la media).
        if tr_end < 2 * min_child {
            eprintln!(
                "❌ tras purgar el solape de horizonte quedan {} muestras de train \
                 (de {}), menos de 2·min_child={}: el horizonte {}ms cubre casi toda \
                 la partición — reduce --horizon-ms o amplía el tape.",
                tr_end, split, 2 * min_child, horizon_ms
            );
            std::process::exit(1);
        }
        println!(
            "   split temporal {}/{} · purga de solape: −{} muestras de train (τ={}ms)",
            tr_end,
            n - split,
            split - tr_end,
            horizon_ms
        );
        (
            feats[..tr_end].to_vec(),
            feats[split..].to_vec(),
            labels[..tr_end].to_vec(),
            labels[split..].to_vec(),
            persist_all[split..].to_vec(),
        )
    } else {
        // Validación en OTRO fichero (otro mes): no hay solape que purgar —
        // ninguna etiqueta de train se resuelve dentro del tape de validación.
        let s_val = build(&val_in);
        (feats, s_val.feats, labels, s_val.labels, s_val.persist)
    };
    let split = tr_y.len();

    // ── 4. GBDT con early stopping ───────────────────────────────────────
    let is_regression = label_mode != "dir";
    let p_bar = tr_y.iter().sum::<f64>() / tr_y.len() as f64;
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
        // (f−y, 1) — P-1/P-2.
        let mut data: Vec<(Vec<f32>, f64, f64)> = tr_feats
            .iter()
            .zip(f_train.iter())
            .zip(tr_y.iter())
            .map(|((x, &f), &y)| {
                if is_regression {
                    (x.clone(), f - y, 1.0)
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
            let (vl, tl) = if is_regression {
                (mse(&va_y, &f_val), mse(&tr_y, &f_train))
            } else {
                (logloss(&va_feats, &va_y, &f_val), logloss(&tr_feats, &tr_y, &f_train))
            };
            if vl + 1e-9 < best_val {
                best_val = vl;
                best_rounds = trees.len();
                since_improve = 0;
            } else {
                since_improve += 5;
            }
            println!("   ronda {:3} train {:.6} · val {:.6} {}", round, tl, vl,
                     if since_improve == 0 { "★" } else { "" });
            if since_improve >= patience {
                println!("   early stopping (paciencia {})", patience);
                break;
            }
        }
    }
    trees.truncate(best_rounds.max(1));

    // Baseline CONSTANTE: tasa base en clasificación, media del train en
    // regresión. En clasificación es el baseline correcto (no hay «clase
    // reciente» que repetir). En regresión es sólo DIAGNÓSTICO — ver abajo.
    let base_pred = vec![init_score as f64; va_y.len()];
    let baseline = if is_regression {
        mse(&va_y, &base_pred)
    } else {
        logloss(&va_feats, &va_y, &base_pred)
    };
    // D-733 — BASELINE DE PERSISTENCIA (sólo regresión): predecir la magnitud
    // futura con la MISMA magnitud medida en la ventana anterior (t−τ, t].
    // σ futura ≈ σ reciente y volumen futuro ≈ volumen reciente son ciertos
    // en cualquier tape, así que la media constante es un rival de paja: un
    // R² de 0,1 contra la media puede esconder un modelo PEOR que repetir el
    // último valor. Se evalúa sobre la MISMA partición de validación y en las
    // MISMAS unidades que la etiqueta.
    let mse_persist = if is_regression { mse(&va_y, &va_persist) } else { f64::NAN };
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
    println!("═══ VEREDICTO ═══");
    println!("   val {} modelo: {:.6} · baseline constante: {:.6} (mejora {:+.6})",
             metric_name, best_val, baseline, baseline - best_val);
    println!("   p10={:.6} p50={:.6} p90={:.6} · varianza={:.6}",
             sorted_p[sorted_p.len() / 10], sorted_p[sorted_p.len() / 2],
             sorted_p[9 * sorted_p.len() / 10], var);

    // Margen anti-empate. Clasificación: Δ absoluto de logloss ≥ 0.001.
    // Regresión: el margen se interpreta RELATIVO — fracción del MSE de la
    // PERSISTENCIA que el modelo debe recortar (skill score). Default 0.001 =
    // 0,1% de mejora sobre repetir el valor reciente.
    let gate_margin: f64 = arg("--gate-margin", "0.001").parse().unwrap();
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
    // automatización lo detecte.
    //
    // D-733 — para los predictores de regresión (P-1/P-2: _VOL, _VOLU, _OI)
    // el gate YA NO es el R² contra la media: es el SKILL contra la
    // PERSISTENCIA. Batir a la media no demuestra nada en magnitudes que se
    // autocorrelacionan; batir al «mañana será como hoy» sí. El R² contra la
    // media se sigue imprimiendo, etiquetado como informativo.
    let gate_pass = if is_regression {
        let (r2_mean, skill, pass) = regression_gate(best_val, baseline, mse_persist, gate_margin);
        println!("   MSE persistencia (σ/volumen recientes en (t−τ,t]): {:.6}", mse_persist);
        println!("   skill vs PERSISTENCIA = {:.4}  ← ESTE es el gate (margen {})",
                 skill, gate_margin);
        println!("   R² vs media constante = {:.4}  (informativo: NO es el gate)", r2_mean);
        pass
    } else {
        baseline - best_val >= gate_margin
    };
    let gate_ok = gate_pass;
    if !gate_ok {
        if is_regression {
            println!("🚫 GATE: el modelo no bate a la PERSISTENCIA por el margen {} — \
                      repetir el valor reciente es igual o mejor. El modelo vivo NO se toca.",
                     gate_margin);
        } else {
            println!("🚫 GATE: mejora < margen {} — sin evidencia real. El modelo vivo NO se toca.",
                     gate_margin);
        }
    }
    let promote = promote && gate_ok;

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
    // P-1/P-2: el sufijo del modelo declara su objetivo — {SYM}_MOTOR
    // (dirección), {SYM}_VOL (σ(τ) realizada del horizonte, % de precio),
    // {SYM}_VOLU (nocional negociado del horizonte, en quote). El watcher del
    // host auto-carga cualquier models/{KEY}.json bajo esa key; los
    // predictores de regresión se sirven con predict_raw (sin sigmoid).
    //
    // D-731 — CAMBIO DE ESCALA: los {SYM}_VOL anteriores a esta revisión
    // están en la escala vieja (σ dividida por sqrt(n_ticks)) y quedan
    // OBSOLETOS: su `init_score` y sus hojas no son comparables con los
    // nuevos. Hay que reentrenar antes de promover. El VOL-BRAKE del host usa
    // el COCIENTE pronóstico/base, que es invariante de escala, así que el
    // serving no necesita cambios — pero un models/ con modelos de las dos
    // épocas mezcladas compararía magnitudes distintas entre símbolos.
    //
    // D-732 — {SYM}_VOLU no tiene HOY ningún consumidor en el host (sólo
    // _VOL alimenta el freno de sizing). Se conserva porque el nocional del
    // horizonte SÍ es una magnitud física medible en el tape, pero mientras
    // nadie lo lea sigue siendo un predictor sin uso: pasar su gate no
    // cambia ninguna decisión del motor.
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
    println!("💾 {} ({} árboles, init {:.4}){}", out, trees.len(), init_score,
             if !gate_ok { " — [gate NO superado, revisar antes de promover]" } else { "" });
    if !promote && gate_ok {
        println!("   para promover al vivo: re-ejecuta con --promote (hot-swap lo recoge en ≤10s)");
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
        cur = if v <= nd.threshold { nd.left as usize } else { nd.right as usize };
    }
}

// ── PRUEBAS ──────────────────────────────────────────────────────────────
//
// Cada prueba fija un invariante que el código ANTERIOR violaba: la etiqueta
// de volatilidad dividida por la intensidad de ticks, la etiqueta `volu`
// construida sobre un par fabricado, el gate contra la media constante y el
// solape de horizonte entre train y validación.
#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduce la fabricación de `binance_vision_sync`: un aggTrade de
    /// `qty` al precio `price` se escribe como (bid_qty, ask_qty) con la
    /// base `max(0,25·qty; 0,1)` en el lado pasivo.
    fn agg_tick(ts: u64, price: f64, qty: f64, is_maker: bool) -> BinTick {
        let base = (qty * 0.25).max(0.1);
        let (bq, aq) = if is_maker { (base, qty + base) } else { (qty + base, base) };
        let half = price * 0.00005;
        BinTick { ts, bid: price - half, ask: price + half, bq, aq }
    }

    /// D-731 — la σ del horizonte NO puede depender de cuántos ticks lo
    /// partan. Dos caminos con la misma varianza realizada acumulada sobre la
    /// MISMA ventana deben etiquetarse igual. La fórmula vieja
    /// `sqrt(Σr²/n_ticks)·100` daba a la ruta de 8 pasos la MITAD de σ que a
    /// la de 2 pasos: un mercado más activo parecía más tranquilo.
    #[test]
    fn vol_es_sigma_del_horizonte_no_media_por_tick() {
        // Camino A: 2 retornos de +r. Camino B: 8 retornos de +r/2.
        // Sigma r^2 (A) = 2r^2;  (B) = 8*(r/2)^2 = 2r^2. Misma varianza realizada.
        let p0 = 100.0f64;
        let r = 0.002f64;
        let mut a = Vec::new();
        let mut p = p0;
        for k in 0..2u64 {
            p *= 1.0 + r;
            a.push(agg_tick(1_000 + k * 1_000, p, 1.0, false));
        }
        let mut b = Vec::new();
        let mut q = p0;
        for k in 0..8u64 {
            q *= 1.0 + r / 2.0;
            b.push(agg_tick(1_000 + k * 250, q, 1.0, false));
        }
        let sa = accumulate_window(&a, p0);
        let sb = accumulate_window(&b, p0);
        assert_eq!(sa.n_ret, 2);
        assert_eq!(sb.n_ret, 8);

        let vol_a = realized_vol_pct(sa.sum_sq);
        let vol_b = realized_vol_pct(sb.sum_sq);
        // Tolerancia: los retornos compuestos no son exactamente r, así que
        // se comparan en relativo con holgura de capitalización (2e-3).
        assert!(
            ((vol_a - vol_b) / vol_a).abs() < 2e-3,
            "sigma(tau) debe ser independiente de la intensidad de ticks: A={vol_a} B={vol_b}"
        );

        // Y la fórmula VIEJA (media por tick) las separaba un factor ~2.
        let viejo_a = (sa.sum_sq / sa.n_ret as f64).sqrt() * 100.0;
        let viejo_b = (sb.sum_sq / sb.n_ret as f64).sqrt() * 100.0;
        assert!(
            viejo_a / viejo_b > 1.9,
            "la etiqueta vieja hacía parecer MENOS volátil al tramo más activo: {viejo_a} vs {viejo_b}"
        );
    }

    /// D-732 — `|bid_qty − ask_qty|` recupera el `qty` EXACTO del aggTrade,
    /// también cuando el suelo absoluto de la base (0,1) domina.
    #[test]
    fn agg_trade_qty_recupera_la_cantidad_exacta() {
        for &q in &[0.05f64, 0.4, 1.0, 37.25, 1_234.5] {
            let tk = agg_tick(0, 100.0, q, false);
            assert!(
                (agg_trade_qty(tk.bq, tk.aq) - q).abs() <= q * 1e-12,
                "qty={q} no recuperado: {}",
                agg_trade_qty(tk.bq, tk.aq)
            );
            let tk2 = agg_tick(0, 100.0, q, true);
            assert!((agg_trade_qty(tk2.bq, tk2.aq) - q).abs() <= q * 1e-12);
            assert!(is_aggtrades_pair(tk.bq, tk.aq));
        }
        // Tape sintético de klines (parquet_to_bin): los dos lados son
        // fracciones del volumen de vela, no un aggTrade.
        assert!(!is_aggtrades_pair(2.5, 2.5), "bq==aq no es un aggTrade");
        assert!(
            !is_aggtrades_pair(1.15 * 3.0, 0.85 * 3.0),
            "el par de klines no cumple la identidad del productor"
        );
    }

    /// D-732 — `volu` es el NOCIONAL negociado de la ventana, no la media
    /// inflada de `(bq + aq)`. La etiqueta vieja valía ≈1,5× la cantidad
    /// MEDIA por tick (sin precio y con suelo artificial): ni volumen, ni
    /// profundidad, ni nocional.
    #[test]
    fn volu_es_nocional_negociado_no_media_inflada() {
        let qtys = [2.0f64, 5.0, 0.05, 11.0];
        let precios = [100.0f64, 100.5, 99.5, 101.0];
        let ticks: Vec<BinTick> = qtys
            .iter()
            .zip(precios.iter())
            .enumerate()
            .map(|(k, (&q, &p))| agg_tick(1_000 + k as u64 * 100, p, q, k % 2 == 0))
            .collect();
        let st = accumulate_window(&ticks, precios[0]);

        let esperado: f64 = qtys
            .iter()
            .zip(ticks.iter())
            .map(|(&q, tk)| q * ((tk.bid + tk.ask) / 2.0))
            .sum();
        assert!(
            (st.notional - esperado).abs() <= esperado * 1e-12,
            "nocional {} != suma qty*precio {}",
            st.notional,
            esperado
        );

        // La etiqueta vieja: media de (bq+aq) por tick — sin precio, con la
        // base fabricada dentro. Debe diferir en ÓRDENES de magnitud.
        let viejo: f64 = ticks.iter().map(|tk| tk.bq + tk.aq).sum::<f64>() / ticks.len() as f64;
        assert!(
            st.notional > viejo * 100.0,
            "la etiqueta vieja ({viejo}) no medía nocional ({})",
            st.notional
        );
    }

    /// D-733 — un modelo puede batir holgadamente a la MEDIA y aun así ser
    /// peor que repetir el valor reciente. Con el gate viejo (R² vs media)
    /// ese modelo se promovía al vivo.
    #[test]
    fn el_gate_exige_batir_la_persistencia_no_la_media() {
        let margen = 0.001;
        // MSE media 1.0, modelo 0.85 ⇒ R² = 0.15 (el gate viejo pasaba),
        // pero la persistencia da 0.50: el modelo es PEOR que no hacer nada.
        let (r2, skill, pasa) = regression_gate(0.85, 1.0, 0.50, margen);
        assert!((r2 - 0.15).abs() < 1e-12, "R2 vs media informativo: {r2}");
        assert!(skill < 0.0, "skill vs persistencia debe ser negativo: {skill}");
        assert!(!pasa, "un modelo peor que la persistencia NO puede pasar el gate");

        // Modelo que sí bate a la persistencia.
        let (_, skill2, pasa2) = regression_gate(0.40, 1.0, 0.50, margen);
        assert!(skill2 > margen && pasa2, "skill {skill2} debería pasar");

        // Persistencia degenerada ⇒ sin baseline comparable ⇒ no pasa.
        assert!(!regression_gate(0.1, 1.0, 0.0, margen).2);
        assert!(!regression_gate(0.1, 1.0, f64::NAN, margen).2);
    }

    /// Las ventanas son CAUSALES: la de etiqueta vive en (t, t+τ] y la de
    /// persistencia en (t−τ, t]. Ninguna puede tocar el otro lado de t.
    #[test]
    fn las_ventanas_no_cruzan_el_instante_t() {
        // Ticks cada 100 ms durante 10 s.
        let raw: Vec<BinTick> = (0..100u64)
            .map(|k| agg_tick(k * 100, 100.0 + k as f64 * 0.01, 1.0, false))
            .collect();
        let i = 50usize; // t = 5000 ms
        let tau = 2_000u64;
        let t = raw[i].ts;

        let end = forward_window_end(&raw, i, t + tau);
        for tk in &raw[i + 1..end] {
            assert!(
                tk.ts > t && tk.ts <= t + tau,
                "ventana futura fuera de (t, t+tau]: {}",
                tk.ts
            );
        }
        assert!(end < raw.len() && raw[end].ts > t + tau, "el corte es el primer ts > t+tau");

        let (lo, ancla) = trailing_window_start(&raw, i, t - tau);
        for tk in &raw[lo..=i] {
            assert!(
                tk.ts > t - tau && tk.ts <= t,
                "ventana pasada fuera de (t-tau, t]: {}",
                tk.ts
            );
        }
        assert!(lo > 0 && raw[lo - 1].ts <= t - tau, "el inicio es el primer ts > t-tau");
        assert!(ancla > 0.0, "el ancla existe cuando el tape cubre tau hacia atrás");
        // El ancla es un mid ANTERIOR a la ventana, nunca uno futuro.
        let mid_ancla = (raw[lo - 1].bid + raw[lo - 1].ask) / 2.0;
        assert!((ancla - mid_ancla).abs() < 1e-12);
    }

    /// D-734 — las muestras solapan τ/stride veces: sin purga, la etiqueta de
    /// las últimas muestras de train se resuelve DENTRO de la validación.
    #[test]
    fn la_purga_elimina_el_solape_de_horizonte() {
        let stride = 50_000u64;
        let tau = 300_000u64;
        let ts: Vec<u64> = (0..100u64).map(|k| k * stride).collect();
        let split = 80usize;
        let tr_end = purge_end(&ts, split, tau);

        // Con tau/stride = 6, la muestra situada EXACTAMENTE tau antes del
        // borde resuelve su etiqueta en el instante de la primera muestra de
        // validación, y ese instante todavía no pertenece a la ventana de
        // etiqueta de la validación —que es (t_val, t_val+tau]—, así que no
        // invade: se purgan las 5 posteriores, no 6. El número lo fija la
        // geometría de las ventanas, no una constante elegida.
        assert_eq!(split - tr_end, 5, "purgadas {} muestras", split - tr_end);
        for &t in &ts[..tr_end] {
            assert!(t + tau <= ts[split], "muestra de train que resuelve en validación: {t}");
        }
        assert!(ts[tr_end] + tau > ts[split], "la primera purgada debe invadir");

        // Sin solape (tau < stride) no se purga nada.
        assert_eq!(purge_end(&ts, split, 10_000), split);
    }
}
