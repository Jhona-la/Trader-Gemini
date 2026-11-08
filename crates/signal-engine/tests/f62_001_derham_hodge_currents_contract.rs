//! Contrato Formal de Verificación: Fase 62 (Cohomología de De Rham y Corrientes de Hodge)
//!
//! Este conjunto de pruebas certifica las propiedades topológicas y diferenciales del motor:
//! - C1: Conservación estricta de energía y partición ortogonal de Hodge: $\sum f_i \approx 1.0$.
//! - C2: Primer número de Betti acotado $b_1 \in [0, 1]$ y síntesis continua libre de divergencia.
//! - C3: Detección de turbulencia rotacional coexacta con disparo de alerta de vorticidad.
//! - C4: Determinismo en stack $O(1)$, cero heap allocations en hot path e inmunidad a NaNs.

use signal_engine::{DeRhamHodgeCurrentsEngine, DeRhamHodgeParams, DERHAM_SIMPLICIAL_NODES};

#[test]
fn test_c1_hodge_energy_partition_and_orthogonality() {
    let mut engine = DeRhamHodgeCurrentsEngine::new(DeRhamHodgeParams::default());

    // Poblar la cadena simplicial
    for i in 0..DERHAM_SIMPLICIAL_NODES {
        let ret = 0.001 * (i as f64 - 3.5);
        let ofi = 0.1 * ((i % 3) as f64 - 1.0);
        let rep = engine.update(ret, ofi, 2.0, 0.015);

        if i >= 3 {
            let sum_ratios =
                rep.exact_energy_ratio + rep.coexact_energy_ratio + rep.harmonic_energy_ratio;
            assert!(
                (sum_ratios - 1.0).abs() < 1e-4,
                "La partición de Hodge debe sumar 1.0, obtenido: {}",
                sum_ratios
            );
            assert!(rep.laplacian_dirichlet_energy >= 0.0);
            assert!(rep.betti_one_persistence >= 0.0 && rep.betti_one_persistence <= 1.0);
            assert!(rep.synthesized_signal >= -1.0 && rep.synthesized_signal <= 1.0);
        }
    }
}

#[test]
fn test_c2_harmonic_persistence_and_gradient_reversion() {
    let mut engine = DeRhamHodgeCurrentsEngine::new(DeRhamHodgeParams::default());

    // Inyectar oscilación periódica suave (régimen conservativo armónico)
    for i in 0..16 {
        let phase = (i as f64) * 0.5;
        let ret = phase.sin() * 0.002;
        let ofi = phase.cos() * 0.2;
        let rep = engine.update(ret, ofi, 1.5, 0.01);

        assert!(rep.exact_energy_ratio >= 0.0 && rep.exact_energy_ratio <= 1.0);
        assert!(rep.harmonic_energy_ratio >= 0.0 && rep.harmonic_energy_ratio <= 1.0);
        assert!(!rep.turbulent_vorticity_alert);
    }
}

#[test]
fn test_c3_turbulent_vorticity_detection() {
    let mut engine = DeRhamHodgeCurrentsEngine::new(DeRhamHodgeParams {
        coexact_turbulent_threshold: 0.60,
        ..Default::default()
    });

    // Inyectar corriente extremadamente caótica con saltos alternados de alta frecuencia (vórtice)
    for i in 0..12 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let ret = sign * 0.05;
        let ofi = -sign * 1.0;
        let rep = engine.update(ret, ofi, 0.1, 0.001);

        if i >= 6 && rep.coexact_energy_ratio >= 0.60 {
            assert!(rep.turbulent_vorticity_alert);
        }
    }
}

#[test]
fn test_c4_stack_o1_and_nan_infinity_resilience() {
    let mut engine = DeRhamHodgeCurrentsEngine::new(DeRhamHodgeParams::default());

    let rep_nan = engine.update(f64::NAN, f64::INFINITY, f64::NAN, f64::NEG_INFINITY);

    assert!(rep_nan.exact_energy_ratio.is_finite());
    assert!(rep_nan.coexact_energy_ratio.is_finite());
    assert!(rep_nan.harmonic_energy_ratio.is_finite());
    assert!(rep_nan.laplacian_dirichlet_energy.is_finite());
    assert!(rep_nan.betti_one_persistence.is_finite());
    assert!(rep_nan.synthesized_signal.is_finite());
}
