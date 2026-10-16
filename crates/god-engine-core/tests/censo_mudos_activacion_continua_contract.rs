//! Formal Contract Test for Ola Ω144: Censo Forense de Motores Silenciosos y Activación Continua.
//!
//! Verifica:
//! 1. `resolve_models_directory()` resuelve canónicamente `models/`.
//! 2. `preload_models_from_disk()` puebla `GLOBAL_FORESTS` de manera atómica y lock-free.
//! 3. `GodEngineCore::new(arena)` inicializa el núcleo con modelos disponibles (`BTCUSDT_MOTOR`),
//!    eliminando la divergencia histórica entre producción y simulación/backtest.
//! 4. `FlowExcitationConfluenceEngine` recibe probabilidades reales y base honesta, produciendo
//!    votos continuos no nulos y eliminando la mudez en el consenso tensorial.
//! 5. `voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz` propaga los 9
//!    campos físicos hacia las 32 escalas espectrales continuas.

use god_engine_core::GodEngineCore;
use god_engine_core::ml_inference::{NanoForest, GLOBAL_FORESTS};
use god_engine_core::model_reload::{resolve_models_directory, preload_models_from_disk};
use omniscient_registry::OmniscientRegistry;
use quantum_arena::GlobalArena;
use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

#[test]
fn contract_model_directory_resolution_and_preload() {
    let dir_opt = resolve_models_directory();
    assert!(dir_opt.is_some(), "Debe resolver canónicamente el directorio models/");
    let dir = dir_opt.unwrap();
    assert!(dir.is_dir(), "La ruta resuelta debe ser un directorio válido");

    let count = preload_models_from_disk();
    assert!(count > 0, "Debe precargar al menos 1 modelo desde disk (e.g. BTCUSDT_MOTOR, etc.)");

    let map = GLOBAL_FORESTS.load();
    assert!(map.contains_key("BTCUSDT_MOTOR"), "GLOBAL_FORESTS debe contener BTCUSDT_MOTOR");
}

#[test]
fn contract_god_engine_core_initialization_paridad_modelos() {
    let arena = GlobalArena::build_in_own_stack(1000.0);
    let core = GodEngineCore::new(arena);

    // Verificar que el bosque activo de BTCUSDT se puede obtener por símbolo y sufijo
    let btc_forest = NanoForest::get_global_suffixed("BTCUSDT", "_MOTOR");
    assert!(btc_forest.is_some(), "GodEngineCore::new debe asegurar BTCUSDT_MOTOR en GLOBAL_FORESTS");

    let forest = btc_forest.unwrap();
    let base_p = forest.base_prob();
    assert!(
        base_p > 0.05 && base_p < 0.95,
        "La base honesta del bosque debe estar acotada entre 5% y 95%, valor: {base_p}"
    );

    // Verificar que swing_nn (DarkAlpha) se cargó si el archivo está presente
    if resolve_models_directory().map(|d| d.join("DarkAlpha_BTCUSDT.json").is_file()).unwrap_or(false) {
        assert!(core.swing_nn.is_some(), "DarkAlpha_BTCUSDT.json debe inicializar swing_nn congelado");
    }
}

#[test]
fn contract_flow_excitation_confluence_despertar_continuo() {
    let reg = Arc::new(OmniscientRegistry::new());
    reg.set("hawkes_intensity", 2.2);
    reg.set("order_book_imbalance", 0.40);
    reg.set("ml_prob_motor", 0.48);
    reg.set("ml_model_base", 0.30); // Base honesta con lift positivo de +0.18

    let mut engine = FlowExcitationConfluenceEngine::new();
    assert!(engine.init(Arc::clone(&reg)).is_ok());

    let vote = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(vote.is_finite(), "El voto debe ser finito");
    assert!(vote > 0.0, "Flujo comprador con lift ML positivo y excitación Hawkes debe votar long: {vote}");
    assert!(vote <= 1.0, "El voto debe estar acotado superiormente en 1.0: {vote}");

    // Probar lado opuesto (vendedor)
    reg.set("order_book_imbalance", -0.40);
    reg.set("ml_prob_motor", 0.12); // Lift vendedor: 0.12 < 0.30 - 0.05
    let vote_short = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(vote_short.is_finite(), "El voto short debe ser finito");
    assert!(vote_short < 0.0, "Flujo vendedor con lift ML negativo debe votar short: {vote_short}");
    assert!(vote_short >= -1.0, "El voto short debe estar acotado inferiormente en -1.0: {vote_short}");
}

#[test]
fn contract_shadow_spectral_confluence_nine_fields_alive() {
    let mut desplazamientos = [0.0f64; 32];
    for (k, d) in desplazamientos.iter_mut().enumerate() {
        *d = 0.25 * (k as f64 - 15.5) / 16.0;
    }

    let voto = FlowExcitationConfluenceEngine::voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
        &desplazamientos,
        2.5,   // ratio hawkes / mu
        0.30,  // hidden liq pressure
        0.85,  // laminar fraction
        0.20,  // yang-mills current
        0.15,  // kahler current
        1.10,  // riemann GUE conviction
        0.12,  // bsd restoring current
        1.05,  // bsd conviction
        0.10,  // dirac chiral current
        1.05,  // dirac conviction
        0.15,  // wasserstein velocity
        1.10,  // wasserstein conviction
    );

    assert!(voto.dominante().is_some(), "El voto espectral continuo con 9 campos debe tener escala dominante");
    let media = voto.media_banda(0, 31).unwrap();
    assert!(media.is_finite(), "La media de banda espectral debe ser finita");
    assert!(media.abs() > 1e-4, "La media de banda espectral debe ser no nula bajo flujo activo: {media}");
}
