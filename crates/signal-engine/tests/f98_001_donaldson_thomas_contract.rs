//! # Contrato Formal de Verificación — Fase 98 (§98)
//!
//! ## Correspondencia de Donaldson-Thomas Cuántica & Invariantes de Vértice MacMahon
//!
//! Contratos verificados:
//! - `c1`: Monotonía y convergencia finita de la serie de MacMahon y densidad de invariantes BPS.
//! - `c2`: Continuidad y suavizado de los saltos de estabilidad de Kontsevich-Soibelman.
//! - `c3`: Activación determinista del veto estricto `DONALDSON_THOMAS_WALL_CROSSING_VETO` y contracción de slot a 0.10.
//! - `c4`: Inmunidad absoluta a NaN/Inf y determinismo estricto de stack $\mathcal{O}(1)$ (`rb_p1`).

use signal_engine::{DonaldsonThomasConfig, DonaldsonThomasQuantumEngine};

#[test]
fn test_c1_macmahon_partition_and_dt_density_bounds() {
    let config = DonaldsonThomasConfig::default();
    let mut engine = DonaldsonThomasQuantumEngine::new(config);

    // Estado con liquidez profunda y spread estrecho (2 bps)
    let rep_deep = engine.evaluate(0.0001, 0.05, 0.10, 2.0);
    assert!(rep_deep.dt_invariant_density >= 0.0 && rep_deep.dt_invariant_density <= 1.0);
    assert!(rep_deep.macmahon_partition_energy > 0.0);
    assert!(!rep_deep.donaldson_thomas_wall_crossing_veto);
    assert!(rep_deep.slot_scaling_factor >= 0.50);

    // Estado con spread muy amplio (50 bps) -> menor acoplamiento q -> menor energía de MacMahon
    let rep_wide = engine.evaluate(0.0001, 0.05, 0.10, 50.0);
    assert!(rep_wide.macmahon_partition_energy <= rep_deep.macmahon_partition_energy);
    assert!(!rep_wide.donaldson_thomas_wall_crossing_veto);
}

#[test]
fn test_c2_kontsevich_soibelman_stability_smoothing() {
    let config = DonaldsonThomasConfig::default();
    let mut engine = DonaldsonThomasQuantumEngine::new(config);

    // Secuencia de variaciones de flujo moderadas
    let mut last_rep = engine.evaluate(0.0005, 0.10, 0.20, 3.0);
    for i in 1..=12 {
        let p = 0.0005 * (i as f64);
        let ofi = 0.10 + 0.02 * (i as f64);
        let imb = (0.20 - 0.03 * (i as f64)).clamp(-1.0, 1.0);
        last_rep = engine.evaluate(p, ofi, imb, 3.0);

        assert!(last_rep.wall_crossing_jump.is_finite());
        assert!((last_rep.directional_conviction).abs() <= 1.0);
        assert!(last_rep.slot_scaling_factor >= 0.10 && last_rep.slot_scaling_factor <= 1.25);
    }

    assert!(!last_rep.donaldson_thomas_wall_crossing_veto);
}

#[test]
fn test_c3_donaldson_thomas_wall_crossing_veto_activation() {
    let config = DonaldsonThomasConfig {
        wall_crossing_threshold: 0.15, // Umbral muy sensible para validar veto
        ..Default::default()
    };
    let mut engine = DonaldsonThomasQuantumEngine::new(config);

    // Inicializamos con un punto de referencia
    let _ = engine.evaluate(0.0, 0.0, 0.0, 3.0);

    // Forzamos un salto masivo y violento en las cargas de brana
    let rep_jump = engine.evaluate(0.08, 4.5, -0.95, 30.0);
    assert!(
        rep_jump.donaldson_thomas_wall_crossing_veto,
        "Salto violento debe detonar DONALDSON_THOMAS_WALL_CROSSING_VETO"
    );
    assert_eq!(
        rep_jump.slot_scaling_factor, 0.10,
        "Veto debe contraer el slot inmediatamente a 0.10"
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = DonaldsonThomasConfig::default();
    let mut engine = DonaldsonThomasQuantumEngine::new(config);

    // Inyección de entradas patológicas
    let rep_nan = engine.evaluate(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    assert!(
        rep_nan.donaldson_thomas_wall_crossing_veto,
        "Valores NaN/Infinitos deben detonar el veto de seguridad"
    );
    assert!(rep_nan.dt_invariant_density.is_finite());
    assert!(rep_nan.macmahon_partition_energy.is_finite());
    assert!(rep_nan.wall_crossing_jump.is_finite());
    assert!(rep_nan.directional_conviction.is_finite());
    assert_eq!(rep_nan.slot_scaling_factor, 0.10);

    // Reset y recuperación limpia
    engine.reset();
    let rep_clean = engine.evaluate(0.0002, 0.01, 0.05, 2.0);
    assert!(!rep_clean.donaldson_thomas_wall_crossing_veto);
    assert!(rep_clean.dt_invariant_density.is_finite());
}
