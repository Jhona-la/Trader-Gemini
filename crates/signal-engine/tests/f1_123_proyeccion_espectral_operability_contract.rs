//! Contract test for F1-123 (Ola Ω153):
//!
//! Contract verifying that `ProyeccionEspectralEngine::is_operable_for_coin` returns `false`
//! when `espectral_masa_resuelta` is missing or below `masa_minima`.
//! This prevents the engine from entering `n_fuentes` in `SignalOrchestrator` when it has no
//! observable mass, avoiding dilution of scalar consensus conviction.

use std::sync::Arc;
use quantum_arena::GlobalArena;
use signal_engine::proyeccion_espectral::{ProyeccionEspectralEngine, ProyeccionEspectralParams};
use strategy_core::QuantumStrategy;

#[test]
fn test_f1_123_proyeccion_espectral_operability_contract() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let params = ProyeccionEspectralParams {
        masa_minima: 0.25,
        masa_plena: 0.85,
    };
    let mut engine = ProyeccionEspectralEngine::with_params(params);
    engine.init(Arc::clone(&arena.registry)).expect("Init must succeed");

    let coin_id = 0;
    let symbol = "BTCUSDT";

    // 1. In cold start, no `espectral_masa_resuelta` exists in registry
    assert!(
        !engine.is_operable_for_coin(coin_id, symbol),
        "Cold engine with no spectral mass telemetry must be inoperable"
    );

    // 2. Publish insufficient spectral mass (below masa_minima 0.25)
    arena.registry.set_for_coin(coin_id, "espectral_masa_resuelta", 0.15);
    assert!(
        !engine.is_operable_for_coin(coin_id, symbol),
        "Engine with mass 0.15 < 0.25 must be inoperable (weight = 0.0)"
    );

    // 3. Publish sufficient spectral mass (above masa_minima 0.25)
    arena.registry.set_for_coin(coin_id, "espectral_masa_resuelta", 0.50);
    assert!(
        engine.is_operable_for_coin(coin_id, symbol),
        "Engine with mass 0.50 >= 0.25 must be operable"
    );

    // 4. Publish full spectral mass (above masa_plena 0.85)
    arena.registry.set_for_coin(coin_id, "espectral_masa_resuelta", 0.95);
    assert!(
        engine.is_operable_for_coin(coin_id, symbol),
        "Engine with full mass 0.95 must be operable with full weight"
    );
}
