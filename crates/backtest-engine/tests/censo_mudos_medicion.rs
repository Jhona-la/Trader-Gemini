//! MEDICIÓN de VOTANTES MUDOS en el ensamble (GLM 129, 2026-10-16).
//!
//! Contexto: el mapa C-F3 :717 y el análisis GLM 127 documentaron que
//! `avg_conviction = Σ|v| / n_fuentes` (tras el 127) divide entre TODOS
//! los votantes registrados, incluidos los estructuralmente mudos
//! (coaxial: 0 en precios ≥ 0,2236 USD; choque: 0 salvo saltos). La
//! calibración de «eff» que el consejo tiene pendiente necesita saber
//! la magnitud REAL: tasa de voto no-cero por fuente.
//!
//! Vehículo: el mismo del 128 (observer sobre tape REAL, core de
//! producción, ADR-0010). El orchestrator expone `censo_snapshot()`
//! (nombre, evaluaciones, votos no-cero) y `GodEngineCore.
//! tensor_orchestrator` es público — el observer lee el censo
//! directamente sin tocar el registry.
//!
//! #[ignore]: investigación local con tape REAL (walk-up compartido
//! con espectral_piso_medicion: TG_DATA_ROOT / workspace / raíz).

use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay_with_observer};

#[test]
#[ignore = "investigación local: requiere tape REAL de 239 MB en data/"]
fn censo_mudos_tasa_de_voto_por_fuente_btc_sep() {
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
        eprintln!("[censo] tape ausente — skip");
        return;
    };
    let raw = backtest_engine::tick_replayer::load_binary_ticks(&path, 0)
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

    // Snapshots del censo en tres puntos (1/3, 2/3, final) para ver
    // estabilidad; el censo es acumulado desde el arranque.
    let tercio = ticks.len() / 3;
    let mut snapshots: Vec<(usize, Vec<(&'static str, u64, u64)>)> = Vec::new();
    let mut i_visto: usize = 0;

    let _stats = run_booktick_replay_with_observer(
        &ticks,
        &genome,
        None,
        &cfg,
        |i, core| {
            if (i + 1 == tercio || i + 1 == 2 * tercio || i + 1 == ticks.len())
                && i > i_visto
            {
                i_visto = i;
                snapshots.push((i + 1, core.tensor_orchestrator.censo_snapshot()));
            }
        },
    );

    eprintln!("==================== CENSO DE MUDOS (GLM 129) ====================");
    eprintln!("tape: BTCUSDT 2026-09-14 | ticks: {}", ticks.len());
    for (hasta, censo) in &snapshots {
        eprintln!("--- censo acumulado hasta el tick {} ---", hasta);
        let mut hablantes = 0usize;
        let mut total_fuentes = 0usize;
        // Orden por tasa descendente: los mudos quedan al final,
        // visibles de un vistazo.
        let mut filas: Vec<(f64, &str, u64, u64)> = censo
            .iter()
            .map(|(nombre, total, no_cero)| {
                let tasa = if *total > 0 { *no_cero as f64 / *total as f64 } else { 0.0 };
                (tasa, *nombre, *total, *no_cero)
            })
            .collect();
        filas.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (tasa, nombre, total, no_cero) in &filas {
            let marca = if *tasa < 0.01 {
                " ◄ MUDO (<1%)"
            } else if *tasa < 0.10 {
                " ◄ casi-mudo (<10%)"
            } else {
                ""
            };
            eprintln!(
                "  {:<28} no_cero {:>9}/{} tasa {:>6.2}%{}",
                nombre,
                no_cero,
                total,
                tasa * 100.0,
                marca
            );
            total_fuentes += 1;
            if *tasa >= 0.01 {
                hablantes += 1;
            }
        }
        eprintln!(
            "  ⇒ hablantes (tasa ≥1%): {} de {} fuentes registradas — avg_conviction divide entre {}",
            hablantes, total_fuentes, total_fuentes
        );
    }
    eprintln!("==================================================================");
}
