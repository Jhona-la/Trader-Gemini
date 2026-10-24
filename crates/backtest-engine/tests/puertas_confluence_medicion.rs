//! MEDICIÓN de las TRES PUERTAS del gate de FlowExcitationConfluence
//! (GLM 130, Fase B — la auditoría de integración de las olas Ω136-143
//! mostró que TODO el eje matemático nuevo cuelga de este motor, censado
//! con 0 votos en 2M ticks por GLM-129).
//!
//! El gate (flow_excitation_confluence.rs:285-490) exige SIMULTÁNEOS:
//!   (H) hawkes_intensity ≥ STEADY_STATE_RATIO + (gene−0.5)·2   [~1.7]
//!   (O) |order_book_imbalance| ≥ max(obi_p80_medido·gene, 0.02)
//!   (L) lift ML: ml_prob_motor ≥ ml_model_base + 0.05 (long) o
//!       ≤ base − 0.05 (short) — con early-returns CL-15 si falta
//!       ml_prob_motor o ml_model_base.
//!
//! Este test muestrea, por evento (observer pre-evento = tick anterior;
//! la publicación ocurre antes de la encuesta en el mismo tick), el
//! estado de las tres puertas y produce: tasa de apertura de cada una
//! EN SOLITARIO, presencia de las claves ML, distribución del lift, y
//! la conjunción — CUÁL sub-gate mata el voto.
//!
//! #[ignore]: investigación local con tape REAL (walk-up compartido).

use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay_with_observer};

#[test]
#[ignore = "investigación local: requiere tape REAL de 239 MB en data/"]
fn puertas_confluence_cual_gate_domina_btc_sep() {
    // LG-23: el núcleo ya no carga modelos al construirse; este vehículo mide
    // con los del roster, así que los carga explícitamente (misma conducta
    // que con la precarga implícita de Ω144).
    let modelos = god_engine_core::model_reload::preload_models_from_disk();
    println!("[LG-23] modelos cargados de disco: {modelos}");
    let nombre = "BTCUSDT_2026-09-14_REAL.bin";
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidatos = [
        std::env::var("TG_DATA_ROOT").map(std::path::PathBuf::from).ok(),
        Some(manifest.join("../..").join("data").join(nombre)),
        Some(manifest.join("../../..").join("data").join(nombre)),
    ];
    let Some(path) = candidatos.into_iter().flatten().find(|p| p.exists()) else {
        eprintln!("[puertas] tape ausente — skip");
        return;
    };
    let raw = backtest_engine::tick_replayer::load_binary_ticks_prefix(&path, 0, 2_000_000)
        .expect("tape legible");
    let n = raw.len().min(2_000_000);
    let ticks: Vec<ReplayTick> = raw[..n]
        .iter()
        .map(|t| ReplayTick {
            ts_ms: t.timestamp,
            bid: t.bid_price,
            ask: t.ask_price,
            bid_qty: t.bid_qty,
            ask_qty: t.ask_qty,
        })
        .collect();
    drop(raw);

    backtest_engine::asegurar_spec_nativo("BTCUSDT");
    let genome = quantum_arena::genome_store::GenomeEnvelope::load_active()
        .map(|e| e.genome)
        .unwrap_or_else(|| quantum_arena::genome::SuperGenotype::new_baseline(0.0002, 0.0005));
    let cfg = ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 200,
        trade_only: false,
        shift_atr_frac: 0.10,
    };

    // Contadores de puertas (muestreo por evento, coin 0 / BTCUSDT).
    let mut muestras: u64 = 0;
    let (mut h_ok, mut o_ok, mut l_ok, mut hl, mut ol, mut hl_and_o, mut todas) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut ml_prob_presente, mut ml_base_presente) = (0u64, 0u64);
    let mut lift_abs_p90: Vec<f64> = Vec::new();
    let mut hawkes_vals_p90: Vec<f64> = Vec::new();

    let _stats = run_booktick_replay_with_observer(
        &ticks,
        &genome,
        None,
        &cfg,
        |_, core| {
            let reg = &core.arena.registry;
            let g = |k: &str| reg.get_scoped_value_or("BTCUSDT", k, f64::NAN);
            muestras += 1;

            // (H) hawkes
            let hawkes = g("hawkes_intensity");
            let h_gene = g("hawkes_excitation_gene");
            let h_gene = if h_gene.is_finite() && (0.50..=0.95).contains(&h_gene) {
                h_gene
            } else {
                0.55
            };
            let h_thr = signal_engine::hawkes_bessel::STEADY_STATE_RATIO
                + (h_gene - 0.50).max(0.0) * 2.0;
            let h = hawkes.is_finite() && hawkes >= h_thr;
            if h {
                h_ok += 1;
            }
            if hawkes.is_finite() {
                hawkes_vals_p90.push(hawkes);
            }

            // (O) OBI vs p80·gene
            let obi = g("order_book_imbalance");
            let p80 = g("obi_p80_medido");
            let o_gene = g("obi_zscore_gene");
            let p80 = if p80.is_finite() && p80 >= 0.0 { p80 } else { 0.15 };
            let o_gene = if o_gene.is_finite() && (0.1..=3.0).contains(&o_gene) {
                o_gene
            } else {
                1.0
            };
            let o_floor = (p80 * o_gene).max(0.02);
            let o = obi.is_finite() && obi.abs() >= o_floor;
            if o {
                o_ok += 1;
            }

            // (L) lift ML (con presencia de claves)
            let ml_prob = g("ml_prob_motor");
            let ml_base = g("ml_model_base");
            let prob_ok = ml_prob.is_finite();
            let base_ok = ml_base.is_finite() && ml_base > 0.0 && ml_base < 1.0;
            if prob_ok {
                ml_prob_presente += 1;
            }
            if base_ok {
                ml_base_presente += 1;
            }
            let l = prob_ok
                && base_ok
                && (ml_prob >= ml_base + 0.05 || ml_prob <= ml_base - 0.05);
            if l {
                l_ok += 1;
            }
            if prob_ok && base_ok {
                lift_abs_p90.push((ml_prob - ml_base).abs());
            }

            if h && l {
                hl += 1;
            }
            if o && l {
                ol += 1;
            }
            if h && o {
                hl_and_o += 1;
            }
            if h && o && l {
                todas += 1;
            }
        },
    );

    let pct = |x: u64| 100.0 * x as f64 / muestras.max(1) as f64;
    let q = |v: &mut Vec<f64>, p: f64| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let i = ((p / 100.0) * (v.len() - 1) as f64).floor() as usize;
        v[i.min(v.len() - 1)]
    };

    eprintln!("==================== PUERTAS DEL CONFLUENCE (GLM 130-B) ====================");
    eprintln!(
        "tape: BTCUSDT 2026-09-14 | muestras: {} (observer pre-evento)",
        muestras
    );
    eprintln!("(H) hawkes ≥ umbral (~1.7):   {:>9} ({:6.3}%)", h_ok, pct(h_ok));
    eprintln!(
        "    hawkes p50={:.3} p90={:.3} p99={:.3}",
        q(&mut hawkes_vals_p90, 50.0),
        q(&mut hawkes_vals_p90, 90.0),
        q(&mut hawkes_vals_p90, 99.0)
    );
    eprintln!("(O) |OBI| ≥ p80·gene:          {:>9} ({:6.3}%)", o_ok, pct(o_ok));
    eprintln!(
        "(L) lift ML ±0.05:             {:>9} ({:6.3}%)  [ml_prob presente: {} ({:.2}%), base presente: {} ({:.2}%)]",
        l_ok,
        pct(l_ok),
        ml_prob_presente,
        pct(ml_prob_presente),
        ml_base_presente,
        pct(ml_base_presente)
    );
    if !lift_abs_p90.is_empty() {
        eprintln!(
        "    |lift| p50={:.4} p90={:.4} p99={:.4} — el gate exige ≥ 0.0500",
        q(&mut lift_abs_p90, 50.0),
        q(&mut lift_abs_p90, 90.0),
        q(&mut lift_abs_p90, 99.0)
    );
    }
    eprintln!("conjunciones: H∧L={:.3}% O∧L={:.3}% H∧O={:.3}% TODAS={:.3}%", pct(hl), pct(ol), pct(hl_and_o), pct(todas));
    if todas == 0 {
        let (culpable, solo) = if l_ok == 0 {
            ("L (lift ML)", pct(l_ok))
        } else if h_ok < o_ok {
            ("H (hawkes)", pct(h_ok))
        } else {
            ("O (OBI)", pct(o_ok))
        };
        eprintln!(
            "HALLAZGO: la conjunción JAMÁS abre ({:.3}%) — el cuello de botella es la puerta {} (sola abre al {:.3}%).",
            pct(todas),
            culpable,
            solo
        );
    } else {
        eprintln!("HALLAZGO: la conjunción abre {} veces — el motor NO está estructuralmente mudo en esta tape.", todas);
    }
    eprintln!("==========================================================================");
}
