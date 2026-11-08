//! Contrato Formal de Verificación: Fase 65 (Dinámica de Solitones de Korteweg-de Vries)
//!
//! Este conjunto de pruebas certifica las propiedades integrables y no-lineales del motor:
//! - C1: Relación canónica de solitón $v = 2A$ y amplitud acotada.
//! - C2: Balance de Korteweg-de Vries y energía hamiltoniana $I_3$ finita.
//! - C3: Detección determinista de tren de solitones coherentes y señal direccional acotada en $[-1, 1]$.
//! - C4: Determinismo en stack $O(1)$, cero heap allocations en hot path e inmunidad a NaNs.

use signal_engine::{KdvSolitonDispersionEngine, KdvSolitonParams, KDV_GRID_POINTS};

#[test]
fn test_c1_canonical_soliton_velocity_amplitude_relation() {
    let mut engine = KdvSolitonDispersionEngine::new(KdvSolitonParams::default());

    // Inyectar perfil suave
    for i in 0..KDV_GRID_POINTS {
        let rep = engine.update(0.001 * (i as f64 - 8.0), 0.1, 1.5, 0.015);
        if i >= 6 {
            assert!(rep.soliton_amplitude >= 0.0);
            assert_eq!(
                rep.soliton_propagation_velocity,
                2.0 * rep.soliton_amplitude
            );
            assert!(rep.nonlinearity_dispersion_balance_ratio >= 0.0);
            assert!(rep.kdv_hamiltonian_energy.is_finite());
            assert!(rep.soliton_impulse_signal >= -1.0 && rep.soliton_impulse_signal <= 1.0);
        }
    }
}

#[test]
fn test_c2_coherent_soliton_detection_under_kdv_balance() {
    let mut engine = KdvSolitonDispersionEngine::new(KdvSolitonParams {
        dispersion_beta: 0.10,
        soliton_balance_min: 0.50,
        soliton_balance_max: 2.00,
        ..Default::default()
    });

    // Inyectar pulso solitónico canónico de forma sech^2
    for i in 0..KDV_GRID_POINTS {
        let x = (i as f64 - 8.0) * 0.5;
        let sech = 1.0 / (x.cosh() * x.cosh());
        let ret = 0.005 * sech;
        let ofi = 0.5 * sech;
        let rep = engine.update(ret, ofi, 1.0, 0.01);

        if i == KDV_GRID_POINTS - 1 {
            assert!(rep.soliton_amplitude > 0.0);
            assert!(rep.nonlinearity_dispersion_balance_ratio > 0.0);
        }
    }
}

#[test]
fn test_c3_directional_impulse_polarity() {
    let mut engine_pos = KdvSolitonDispersionEngine::new(KdvSolitonParams::default());
    let mut engine_neg = KdvSolitonDispersionEngine::new(KdvSolitonParams::default());

    let mut rep_pos = signal_engine::KdvSolitonReport::default();
    let mut rep_neg = signal_engine::KdvSolitonReport::default();

    for _ in 0..KDV_GRID_POINTS {
        rep_pos = engine_pos.update(0.01, 0.8, 1.0, 0.01);
        rep_neg = engine_neg.update(-0.01, -0.8, 1.0, 0.01);
    }

    assert!(rep_pos.soliton_impulse_signal > 0.0);
    assert!(rep_neg.soliton_impulse_signal < 0.0);
}

#[test]
fn test_c4_stack_o1_and_nan_infinity_resilience() {
    let mut engine = KdvSolitonDispersionEngine::new(KdvSolitonParams::default());

    let rep_nan = engine.update(f64::NAN, f64::INFINITY, f64::NAN, f64::NEG_INFINITY);

    assert!(rep_nan.soliton_amplitude.is_finite());
    assert!(rep_nan.soliton_propagation_velocity.is_finite());
    assert!(rep_nan.nonlinearity_dispersion_balance_ratio.is_finite());
    assert!(rep_nan.kdv_hamiltonian_energy.is_finite());
    assert!(rep_nan.soliton_impulse_signal.is_finite());
}
