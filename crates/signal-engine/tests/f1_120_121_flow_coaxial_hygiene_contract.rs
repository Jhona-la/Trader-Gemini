use quantum_arena::genome::SuperGenotype;
use quantum_arena::GlobalArena;
use signal_engine::flow_impulse::FlowImpulseEngine;
use std::sync::atomic::Ordering;

#[test]
fn test_f1_120_flow_impulse_duration_preserves_genome_under_15s_and_fails_closed() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // Configurar umbrales para que el flow_impulse emita señal válida
    arena.config.weight_obi.store(1.0, Ordering::Relaxed);
    arena.config.weight_ofi.store(1.0, Ordering::Relaxed);
    arena.config.turbo_coherence_threshold.store(0.1, Ordering::Relaxed);
    arena.config.turbo_z_score_stdev.store(0.5, Ordering::Relaxed);
    arena.config.tensor_poly_a.store(0.01, Ordering::Relaxed);
    arena.config.tensor_poly_b.store(0.001, Ordering::Relaxed);

    // 1. Probar valor genómico en [10_000, 15_000): ej. 11_500 ms
    arena.config.base_duration_ms.store(11_500.0, Ordering::Relaxed);
    let intent = FlowImpulseEngine::evaluate_flow_impulse(
        &arena,
        0.8,  // obi
        0.6,  // ofi
        3.5,  // hawkes_ratio (superior al SS de 1.6 para inducir z-score positivo)
        0.05, // entropy baja
        60_000.0,
        0.002,
        1_000_000,
    );
    assert!(intent.is_some(), "FlowImpulseEngine debe emitir con microestructura favorable");
    let intent = intent.unwrap();
    assert_eq!(
        intent.expected_duration_ms, 11_500,
        "F1-120: La duración esperada debe respetar el genoma (11_500 ms) sin suelo literal de 15_000 ms"
    );

    // 2. Probar valor inválido (NaN o <= 0.0) -> debe fallar cerrado (None)
    arena.config.base_duration_ms.store(f64::NAN, Ordering::Relaxed);
    let intent_nan = FlowImpulseEngine::evaluate_flow_impulse(
        &arena,
        0.8,
        0.6,
        3.5,
        0.05,
        60_000.0,
        0.002,
        1_000_000,
    );
    assert!(
        intent_nan.is_none(),
        "F1-120: Duración base no finita (NaN) debe fallar cerrado (None) sin inventar 15_000 ms"
    );

    arena.config.base_duration_ms.store(0.0, Ordering::Relaxed);
    let intent_zero = FlowImpulseEngine::evaluate_flow_impulse(
        &arena,
        0.8,
        0.6,
        3.5,
        0.05,
        60_000.0,
        0.002,
        1_000_000,
    );
    assert!(
        intent_zero.is_none(),
        "F1-120: Duración base <= 0 debe fallar cerrado (None)"
    );
}

#[test]
fn test_f1_121_coaxial_squeeze_threshold_is_frozen_across_mutations() {
    let mut genome = SuperGenotype::default();
    let initial_val = genome.coaxial_squeeze_threshold;
    assert!(
        initial_val > 0.0 && initial_val.is_finite(),
        "El gen coaxial_squeeze_threshold debe tener un valor inicial finito y positivo"
    );

    // Mutar 100 veces consecutivas y verificar invariancia estricta (gen congelado sin deriva)
    for _ in 0..100 {
        genome = genome.mutate_cmaes(0.2);
        assert_eq!(
            genome.coaxial_squeeze_threshold, initial_val,
            "F1-121: coaxial_squeeze_threshold debe permanecer estrictamente congelado durante la mutación"
        );
    }
}
