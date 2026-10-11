//! LC-F1-01 (CL-61): la carpeta del estado aprendido sigue al MODO de
//! operación, no sólo al venue.
//!
//! El host decide el modo con la puerta de mainnet: sin `--force-live` y
//! `config_dir/MAINNET_ARMED` opera en papel y fija `TG_GENOME_ENV=demo`
//! (src/bin/god_engine.rs, «E3 — ENTORNO DE GENOMA por modo de operación»).
//! La envolvente Kelly, el fee-breaker y los diarios se guardan con
//! `env_data_path`, que hasta CL-61 sólo miraba el venue (`is_demo_env`): un
//! arranque en papel sin banderas aprendía de cierres simulados en
//! `data/prod` y la sesión armada siguiente dimensionaba con ese posterior.

use quantum_arena::paths::env_data_path;
use std::sync::Mutex;

/// Las variables de entorno son del proceso: un test a la vez.
static ENTORNO: Mutex<()> = Mutex::new(());

const VARIABLES: [&str; 4] = ["SHADOW_MODE", "USE_TESTNET", "BINANCE_USE_DEMO", "TG_GENOME_ENV"];

fn ruta_con(modo: Option<&str>, venue_de_prueba: bool) -> String {
    let _g = ENTORNO.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: el Mutex serializa los tests de este binario, que no lanza hilos.
    unsafe {
        for v in VARIABLES {
            std::env::remove_var(v);
        }
        if venue_de_prueba {
            std::env::set_var("USE_TESTNET", "true");
        }
        if let Some(m) = modo {
            std::env::set_var("TG_GENOME_ENV", m);
        }
    }
    env_data_path("kelly_envelope.json")
}

#[test]
fn cl61_papel_sin_banderas_escribe_en_demo() {
    assert_eq!(ruta_con(Some("demo"), false), "data/demo/kelly_envelope.json");
}

#[test]
fn cl61_produccion_solo_con_dinero_real_en_el_venue_real() {
    assert_eq!(ruta_con(Some("prod"), false), "data/prod/kelly_envelope.json");
    // Armado contra testnet: los cierres no son de mercado real.
    assert_eq!(ruta_con(Some("prod"), true), "data/demo/kelly_envelope.json");
    assert_eq!(ruta_con(Some("demo"), true), "data/demo/kelly_envelope.json");
    assert_eq!(ruta_con(Some("backtest"), false), "data/demo/kelly_envelope.json");
}

#[test]
fn cl61_sin_modo_declarado_no_se_presume_produccion() {
    assert_eq!(ruta_con(None, false), "data/demo/kelly_envelope.json");
}
