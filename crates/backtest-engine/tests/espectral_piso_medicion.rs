//! MEDICIÓN XIV §14 del PISO ESPECTRAL (GLM 128, 2026-10-16).
//!
//! Contexto: el análisis de aceptación del ensamble (GLM 127,
//! docs/audit/ANALISIS_ACEPTACION_ENSAMBLE_2026-10-16.md) encontró que
//! el piso 0.70 de la vía ESPECTRAL sigue vivo
//! (orchestrator.rs:590-594: net = v_dom·(0.70+0.30·coherencia)).
//! Drenarlo NO es no-op — XIV §14 exige «comparar tasas de aceptación,
//! costes y exposición con datos apropiados».
//!
//! ESTE test ES esa medición: corre el core de producción (replay con
//! paridad 1:1) sobre una tape REAL y muestrea el estado espectral por
//! evento (ADR-0010 — el observer como vehículo de export). Produce:
//!   - distribución de |v_dom| cuando el veredicto espectral está ACTIVO
//!     (|v|>1e-9, τ operable ≥ 30 s),
//!   - el cutoff de señal vigente (genoma activo),
//!   - la comparación de tasas: cuántas señales cruzan el corte CON la
//!     modulación (factor 0.70 con coherencia neutra) vs cuántas abriría
//!     SIN modulación (v directo). NOTA: el factor 0.70+0.30·coh ∈
//!     [0.40,1.00] ATENÚA por defecto — el análisis GLM 127 temía un
//!     inflador y la medición rectifica: es una compuerta represora.
//!
//! #[ignore]: requiere data/BTCUSDT_2026-09-14_REAL.bin (239 MB) —
//! corrida explícita local:
//!   cargo test -p backtest-engine --test espectral_piso_medicion -- \
//!     --include-ignored --nocapture
//! La salida es el recibo (p50/p90/p99 + banda crítica) que el consejo
//! necesita para decidir el drenaje.

use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay_with_observer};

#[test]
#[ignore = "investigación local: requiere tape REAL de 239 MB en data/"]
fn piso_espectral_distribucion_vdom_y_banda_critica_btc_sep() {
    // Cargo corre los tests con CWD = directorio del paquete. La tape
    // vive en la raíz del WORKSPACE (dos niveles arriba del paquete);
    // desde un worktree, además, la raíz real está tres niveles arriba
    // (las tapes NO se duplican en worktrees — regla del repo). Orden:
    // env TG_DATA_ROOT, workspace, worktree→raíz.
    let nombre = "BTCUSDT_2026-09-14_REAL.bin";
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidatos = [
        std::env::var("TG_DATA_ROOT").map(std::path::PathBuf::from).ok(),
        Some(manifest.join("../..").join("data").join(nombre)),
        Some(manifest.join("../../..").join("data").join(nombre)),
    ];
    let Some(path) = candidatos
        .into_iter()
        .flatten()
        .find(|p| p.exists())
    else {
        eprintln!("[medición] tape ausente (probé TG_DATA_ROOT, workspace y raíz) — skip");
        return;
    };
    let raw = backtest_engine::tick_replayer::load_binary_ticks(&path, 0)
        .expect("tape legible");
    // Muestreo: primeros 2M ticks — suficiente para la distribución sin
    // pagar el replay completo de la tape (horas).
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
    eprintln!("[medición] {} ticks muestreados", ticks.len());

    backtest_engine::asegurar_spec_nativo("BTCUSDT");

    // Genoma ACTIVO del sistema (el que producción usa) con fallback a
    // baseline — el cutoff debe ser el vigente.
    let genome = quantum_arena::genome_store::GenomeEnvelope::load_active()
        .map(|e| e.genome)
        .unwrap_or_else(|| quantum_arena::genome::SuperGenotype::new_baseline(0.0002, 0.0005));
    let cfg = ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 200,
        trade_only: false,
        shift_atr_frac: 0.10,
    };

    // Acumuladores del muestreo (observer secuencial FnMut — sin sync).
    let mut vdom_activo: Vec<f64> = Vec::new();
    let mut muestras_totales: u64 = 0;
    let mut min_conf_vigente: f64 = 0.0;

    let stats = run_booktick_replay_with_observer(
        &ticks,
        &genome,
        None,
        &cfg,
        |_, core| {
            muestras_totales += 1;
            let reg = &core.arena.registry;
            let v = reg.get_for_coin_or(0, "consenso_espectral_dominante", 0.0);
            let tau = reg.get_for_coin_or(0, "consenso_espectral_tau", 0.0);
            min_conf_vigente = core
                .arena
                .config
                .min_confidence_btc
                .load(std::sync::atomic::Ordering::Relaxed);
            const TAU_MIN_MS: f64 = 30_000.0;
            if v.is_finite() && v.abs() > 1e-9 && tau.is_finite() && tau >= TAU_MIN_MS {
                vdom_activo.push(v.abs());
            }
        },
    );

    // ---- Recibo de la medición ----
    eprintln!("==================== MEDICIÓN XIV §14 — PISO ESPECTRAL ====================");
    eprintln!(
        "tape: BTCUSDT 2026-09-14 | ticks: {} | muestras observer: {} | trades del replay: {}",
        ticks.len(),
        muestras_totales,
        stats.trades
    );
    eprintln!(
        "cutoff vigente (min_confidence del genoma activo): {:.4}",
        min_conf_vigente
    );
    eprintln!(
        "eventos con veredicto espectral ACTIVO (|v|>1e-9, τ≥30s): {} ({:.2}% de muestras)",
        vdom_activo.len(),
        100.0 * vdom_activo.len() as f64 / muestras_totales.max(1) as f64
    );

    if vdom_activo.is_empty() {
        eprintln!("HALLAZGO: el veredicto espectral JAMÁS estuvo activo en la tape ⇒ el piso es INOPERANTE aquí (drenaje = no-op empírico para esta muestra).");
        return;
    }

    vdom_activo.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| -> f64 {
        let i = ((p / 100.0) * (vdom_activo.len() - 1) as f64).floor() as usize;
        vdom_activo[i.min(vdom_activo.len() - 1)]
    };
    eprintln!(
        "|v_dom| activo: p10={:.4} p50={:.4} p90={:.4} p99={:.4} máx={:.4}",
        q(10.0),
        q(50.0),
        q(90.0),
        q(99.0),
        vdom_activo[vdom_activo.len() - 1]
    );

    // Banda de decisión de la MODULACIÓN espectral. net = v·(0.70+0.30·coh)
    // con coh ∈ [-1,1] ⇒ factor ∈ [0.40, 1.00]: con coherencia NEUTRA el
    // factor es 0.70 (ATENÚA, no infla); con coherencia +1 es 1.00 (pasa
    // v directo). Por tanto:
    //   - pasan CON la modulación (coh=0):  {|v| ≥ cutoff/0.70}
    //   - pasarían SIN modulación (v directo): {|v| ≥ cutoff}
    // El "piso" NO fabrica convicción: es una compuerta que hoy, con la
    // distribución real de |v_dom|, mantiene la vía espectral bajo el
    // corte; quitar la modulación la ABRIRÍA.
    let cutoff = min_conf_vigente.max(0.0);
    let borde_con_modulacion = if cutoff > 0.0 { cutoff / 0.70 } else { f64::INFINITY };
    let pasan_con_modulacion = vdom_activo
        .iter()
        .filter(|v| **v >= borde_con_modulacion)
        .count();
    let abriria_sin_modulacion = vdom_activo
        .iter()
        .filter(|v| **v >= cutoff && **v < borde_con_modulacion)
        .count();
    let total_act = vdom_activo.len();
    eprintln!(
        "DECISIÓN: con la modulación (coh=0 ⇒ factor 0.70), pasan el corte |v| ≥ {:.4}: {} de {} activos ({:.3}%)",
        borde_con_modulacion,
        pasan_con_modulacion,
        total_act,
        100.0 * pasan_con_modulacion as f64 / total_act as f64
    );
    eprintln!(
        "SIN modulación (v directo), |v| ∈ [{:.4}, {:.4}) abriría {} señales-evento ({:.2}% de activos) que hoy la modulación reprime",
        cutoff,
        borde_con_modulacion,
        abriria_sin_modulacion,
        100.0 * abriria_sin_modulacion as f64 / total_act as f64
    );
    if pasan_con_modulacion == 0 && abriria_sin_modulacion > 0 {
        eprintln!("HALLAZGO: la vía espectral pura NO cruza el corte NI UNA VEZ en la muestra (la modulación la mantiene cerrada); quitarla abriría {} señales-evento que los gates aguas abajo filtrarían. La preocupación «el piso fabrica convicción» es INOPERANTE aquí: el factor reprime, no infla.", abriria_sin_modulacion);
    } else {
        eprintln!("HALLAZGO: {} señales cruzan hoy por vía espectral — el impacto del drenaje depende de su coste/exposición (paso 2 con PnL por decisión).", pasan_con_modulacion);
    }
    eprintln!("==========================================================================");
}
