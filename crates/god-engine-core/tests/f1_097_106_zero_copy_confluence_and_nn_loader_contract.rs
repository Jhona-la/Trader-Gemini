//! Contrato F1-097 y F1-106 (Ola Ω164): Optimización Zero-Copy en FlowExcitationConfluence
//! y Erradicación de la Lectura Síncrona de Disco de DarkAlpha NN en el Constructor del Núcleo.
//!
//! # Fundamento Teórico y Rendimiento (Nanosegundos y 16 GB RAM)
//! - **F1-106 (Doctrina LG-23 / Prevención de Saturación de RAM y E/S)**:
//!   `GodEngineCore::new` no debe leer jamás el disco de forma síncrona ni alocar el árbol JSON
//!   de la red neuronal en memoria durante la construcción. El host de producción inyecta el modelo
//!   explícitamente antes de arrancar, y los evaluadores de mutantes (Darwin/online_daemon) crean
//!   núcleos en microsegundos sin recargar archivos ni fugar memoria heap transitoria.
//! - **F1-097 (Erradicación de ~1,500 RMWs Atómicos por Tick)**:
//!   `FlowExcitationConfluenceEngine::evaluate_for_coin` utiliza accesores `get_scoped_val_or`
//!   que leen atómicos de punto flotante directamente del registro de forma lock-free y zero-alloc,
//!   eliminando por completo la clonación y liberación de punteros `Arc<Parameter>`.

use god_engine_core::model_reload::{load_dark_alpha_model, resolve_models_directory};
use god_engine_core::GodEngineCore;
use omniscient_registry::OmniscientRegistry;
use quantum_arena::state::GlobalArena;
use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use strategy_core::QuantumStrategy;
use std::sync::Arc;

#[test]
fn test_f1_106_core_constructor_zero_disk_io_and_on_demand_nn_loading() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena);

    // 1. Verificar cumplimiento estricto de LG-23: swing_nn debe ser None tras new()
    assert!(
        core.swing_nn.is_none(),
        "F1-106: GodEngineCore::new() NO debe cargar swing_nn de disco automáticamente (LG-23)"
    );

    // 2. Si el archivo models/DarkAlpha_BTCUSDT.json está presente, verificar carga explícita
    let has_model = resolve_models_directory()
        .map(|d| d.join("DarkAlpha_BTCUSDT.json").is_file())
        .unwrap_or(false);

    if has_model {
        let loaded = load_dark_alpha_model("BTCUSDT");
        assert!(
            loaded.is_some(),
            "load_dark_alpha_model debe cargar y deserializar el modelo si existe en models/"
        );
        let model = loaded.unwrap();
        assert!(model.freeze_normalizers, "El modelo cargado debe estar congelado (T-04)");

        // 3. Probar builder with_swing_nn
        core = core.with_swing_nn(model);
        assert!(core.swing_nn.is_some(), "with_swing_nn debe asignar el modelo correctamente");
    }
}

#[test]
fn test_f1_097_flow_excitation_confluence_zero_copy_and_cl15_discipline() {
    let reg = Arc::new(OmniscientRegistry::new());
    let mut engine = FlowExcitationConfluenceEngine::new();
    assert!(engine.init(Arc::clone(&reg)).is_ok());

    // 1. CL-15: Sin probabilidad ML ni base medida, el motor NO OPINA (debe retornar 0.0)
    let vote_absent = engine.evaluate_for_coin(0, "BTCUSDT");
    assert_eq!(
        vote_absent, 0.0,
        "F1-097: Sin ml_prob_motor ni ml_model_base, debe retornar exactamente 0.0 (CL-15)"
    );

    // 2. Configurar valores en registro para disparar confluencia Zero-Copy
    reg.set("BTCUSDT_hawkes_intensity", 2.0);
    reg.set("BTCUSDT_order_book_imbalance", 0.35);
    reg.set("BTCUSDT_ml_prob_motor", 0.52);
    reg.set("BTCUSDT_ml_model_base", 0.30); // lift = 0.22 > 0.05
    reg.set("BTCUSDT_ml_model_lift", 0.05);

    let vote_active = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(
        vote_active > 0.0,
        "F1-097: Con lift positivo sobre la base honesta, el voto debe ser alcista (>0.0), valor: {}",
        vote_active
    );
    assert!(
        vote_active <= 1.0,
        "F1-097: El voto de confluencia debe estar acotado en [-1.0, 1.0], valor: {}",
        vote_active
    );
}
