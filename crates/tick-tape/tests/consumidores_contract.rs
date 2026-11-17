//! TGB-E — contrato de anclas: los lectores directos de cintas de ticks del pedido CODEX-CLI (Parte 1) abren la
//! cinta LOGICA por `tick_tape::logica` y ya no abren, mapean ni miden el `.bin` fisico por su cuenta; las
//! barreras de las pruebas de medicion reconocen `X.bin.tgb`; ninguna corrupcion se convierte en omision.
//!
//! Es un escaneo de texto (como las listas blancas): fija que la migracion no se deshaga en silencio. La
//! conducta la prueban `tgb_reader_contract`, `tgb_selector_contract` y las pruebas de este crate.

struct Consumidor {
    ruta: &'static str,
    fuente: &'static str,
    /// Lo que tiene que aparecer.
    exige: &'static [&'static str],
    /// La apertura vieja, que ya no puede aparecer.
    prohibe: &'static [&'static str],
}

macro_rules! fuente {
    ($r:literal) => {
        ($r, include_str!(concat!("../../../", $r)))
    };
}

fn consumidores() -> Vec<Consumidor> {
    let c = |(ruta, fuente): (&'static str, &'static str), exige: &'static [&'static str], prohibe: &'static [&'static str]| Consumidor { ruta, fuente, exige, prohibe };
    vec![
        c(fuente!("src/bin/backtest_windows.rs"), &["tick_tape::logica::open_logical_tape"], &["std::fs::File::open(&file)"]),
        c(fuente!("src/bin/audit_forensic_backtest.rs"), &["tick_tape::logica::open_logical_tape"], &["std::fs::File::open(&data_path)"]),
        c(fuente!("src/bin/train_forest.rs"), &["tick_tape::logica::open_logical_tape"], &["File::open(path).unwrap_or_else"]),
        c(fuente!("src/bin/feature_exporter.rs"), &["tick_tape::logica::open_logical_tape", "Presupuesto::hasta(budget)"], &["File::open(&options.input)"]),
        c(fuente!("src/bin/votes_export.rs"), &["tick_tape::logica::leer_logica"], &["std::fs::read(ruta)"]),
        c(fuente!("src/bin/copulas_manifest.rs"), &["tick_tape::logica::leer_logica", "ErrorKind::NotFound => return None"], &["std::fs::read(ruta)", "leer_logica(ruta).ok()?"]),
        c(fuente!("src/bin/ml_path_probe.rs"), &["tick_tape::logica::open_logical_tape"], &["std::fs::File::open(\"data/BTCUSDT_ticks_REAL.bin\")"]),
        c(fuente!("src/bin/spectral_bench.rs"), &["tick_tape::logica::open_logical_tape"], &["File::open(path).expect"]),
        c(fuente!("crates/god-engine-core/src/bin/test_ml.rs"), &["tick_tape::logica::open_logical_tape"], &["std::fs::File::open(\"data/BTCUSDT_AUG_REAL.bin\")"]),
        c(fuente!("src/bin/walkforward_evolver.rs"), &["tick_tape::logica::largo_logico"], &["std::fs::metadata(&data)"]),
        c(fuente!("src/bin/oi_history_sync.rs"), &["strip_suffix(\".tgb\")"], &[]),
        c(fuente!("crates/backtest-engine/src/bin/continuous_evolution_backtest.rs"), &["tick_replayer::elegir_cinta", "tick_replayer::cintas_presentes", "tick_tape::logica::hay_cinta(btc_path)"], &["real.exists()", "p.exists()", "btc_path.exists()"]),
        c(fuente!("crates/backtest-engine/tests/cinta_real/mod.rs"), &["tick_tape::logica::open_logical_tape"], &["std::fs::File::open(&ruta)"]),
        c(fuente!("crates/backtest-engine/tests/bt_vivo_parity_audit.rs"), &["ErrorKind::NotFound"], &["let Ok(events) = load_binary_ticks(&path, 0) else"]),
        c(fuente!("crates/backtest-engine/tests/mps_entrelazamiento_medicion.rs"), &["ruta_tgb(p)", "tick_tape::tgb::fallo_de(&e)"], &[]),
        c(fuente!("crates/backtest-engine/tests/censo_mudos_medicion.rs"), &["ruta_tgb(p)"], &[]),
        c(fuente!("crates/backtest-engine/tests/espectral_piso_medicion.rs"), &["ruta_tgb(p)"], &[]),
        c(fuente!("crates/backtest-engine/tests/fases_ising_medicion.rs"), &["ruta_tgb(p)"], &[]),
        c(fuente!("crates/backtest-engine/tests/firmas_camino_medicion.rs"), &["ruta_tgb(p)"], &[]),
        c(fuente!("crates/backtest-engine/tests/koopman_observables_medicion.rs"), &["ruta_tgb(p)"], &[]),
        c(fuente!("crates/backtest-engine/tests/ortogonalidad_voces_medicion.rs"), &["ruta_tgb(p)"], &[]),
        c(fuente!("crates/backtest-engine/tests/puertas_confluence_medicion.rs"), &["ruta_tgb(p)"], &[]),
        c(fuente!("crates/backtest-engine/tests/tsallis_entrelazamiento_medicion.rs"), &["ruta_tgb(p)"], &[]),
        c(fuente!("crates/feature-engine/tests/copula_real.rs"), &["tick_tape::logica::leer_logica", "ErrorKind::NotFound => return None"], &["std::fs::read(ruta)"]),
        c(fuente!("crates/feature-engine/tests/transfer_entropy_real.rs"), &["tick_tape::logica::leer_logica", "ErrorKind::NotFound => return None"], &["std::fs::read(ruta)"]),
    ]
}

#[test]
fn e_los_lectores_directos_abren_la_cinta_logica() {
    let mut mal = Vec::new();
    for k in consumidores() {
        let texto = k.fuente.replace("\r\n", "\n");
        for e in k.exige {
            if !texto.contains(e) {
                mal.push(format!("{}: falta `{e}`", k.ruta));
            }
        }
        for p in k.prohibe {
            if texto.contains(p) {
                mal.push(format!("{}: sigue la apertura directa `{p}`", k.ruta));
            }
        }
    }
    assert!(mal.is_empty(), "lectores sin migrar a la cinta logica:\n{}", mal.join("\n"));
}
