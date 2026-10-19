//! #652/H2 en el orquestador VIVO, el que arma `GodEngineCore::new` con todas
//! sus estrategias. Con la escala dominante inoperable (τ < 30 s) el veredicto
//! espectral no se consume: el consenso escalar queda bit a bit igual que sin
//! dominante (crates/signal-engine/src/orchestrator.rs, #652/H2). Los tests
//! del orquestador lo comprueban con estrategias de prueba; éste, con las vivas.
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;

/// τ del dominante: 1 s, por debajo de la banda operable.
const TAU_INOPERABLE_MS: f64 = 1_000.0;

fn decision_con_dominante(v_dom: f64) -> (String, u64, u64) {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    // La sombra de Feynman publica la coherencia por moneda en cada evento de libro.
    arena.registry.set_for_coin(0, "feynman_coherence", 0.9);
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_dominante", v_dom);
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_tau", TAU_INOPERABLE_MS);
    let d = core
        .tensor_orchestrator
        .evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    (
        format!("{:?}", d.signal),
        d.net_confidence.to_bits(),
        d.expected_lifetime_ms,
    )
}

#[test]
fn dominante_inoperable_no_mueve_el_consenso_vivo() {
    assert!(TAU_INOPERABLE_MS < quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS);
    let sin_dominante = decision_con_dominante(0.0);
    let con_dominante = decision_con_dominante(0.8);
    assert_eq!(
        con_dominante, sin_dominante,
        "con τ inoperable el dominante espectral no debe llegar al consenso escalar"
    );
}
