//! Contrato Formal de Certificación: Fase 110 (§110) — Modelos Cuánticos Dobles de Kitaev $\mathcal{D}(S_3)$
//!
//! Verifica en silicio:
//! 1. Invariantes del vacío topológico $\mathcal{D}(S_3)$ en estado nominal.
//! 2. Consistencia algebraica exacta de la tabla de Cayley de $S_3$ (axiomas de grupo).
//! 3. Respuesta direccional de operadores de cinta anyónica ante OFI.
//! 4. Activación determinista fail-closed del veto ante des-confinamiento gauge.
//! 5. Robustez numérica absoluta e inmunidad a NaN/Inf (invariante `rb_p1`).

use signal_engine::kitaev_quantum_double::{
    KitaevQuantumDoubleConfig, KitaevQuantumDoubleEngine, S3_INVERSES, S3_MULTIPLICATION_TABLE,
    S3_ORDER,
};

#[test]
fn test_c1_nominal_kitaev_quantum_double_invariants() {
    let mut engine = KitaevQuantumDoubleEngine::new(KitaevQuantumDoubleConfig::default());

    let rep = engine.update(0.0001, 0.05, 0.50, 1.5);

    assert!(rep.ground_state_fidelity >= 0.50);
    assert!(rep.topological_order_parameter > 0.0);
    assert!(rep.star_operator_expectation > 0.0);
    assert!(rep.plaquette_operator_expectation > 0.0);
    assert!(!rep.deconfinement_veto);
    assert!(rep.slot_scale >= 0.10 && rep.slot_scale <= 1.25);
}

#[test]
fn test_c2_s3_group_algebra_cayley_table_invariants() {
    let engine = KitaevQuantumDoubleEngine::new(KitaevQuantumDoubleConfig::default());
    let table = engine.multiplication_table();

    // 1. Identidad: e * g = g * e = g para todo g
    for g in 0..S3_ORDER {
        assert_eq!(table[0][g], g, "0 debe actuar como identidad izquierda");
        assert_eq!(table[g][0], g, "0 debe actuar como identidad derecha");
    }

    // 2. Inversos: g * g^{-1} = e = 0
    for g in 0..S3_ORDER {
        let inv = S3_INVERSES[g];
        assert_eq!(table[g][inv], 0, "g * g^{{-1}} debe ser la identidad");
        assert_eq!(table[inv][g], 0, "g^{{-1}} * g debe ser la identidad");
    }

    // 3. Asociatividad estricta para todos los 6^3 = 216 tripletes: (a * b) * c = a * (b * c)
    for a in 0..S3_ORDER {
        for b in 0..S3_ORDER {
            for c in 0..S3_ORDER {
                let ab = S3_MULTIPLICATION_TABLE[a][b];
                let lhs = S3_MULTIPLICATION_TABLE[ab][c];

                let bc = S3_MULTIPLICATION_TABLE[b][c];
                let rhs = S3_MULTIPLICATION_TABLE[a][bc];

                assert_eq!(lhs, rhs, "Fallo de asociatividad en triplete ({}, {}, {})", a, b, c);
            }
        }
    }
}

#[test]
fn test_c3_non_abelian_ribbon_directional_signal_responsiveness() {
    let mut engine_bull = KitaevQuantumDoubleEngine::new(KitaevQuantumDoubleConfig::default());
    let mut engine_bear = KitaevQuantumDoubleEngine::new(KitaevQuantumDoubleConfig::default());

    // Calentamiento con flujo direccional persistente
    for _ in 0..5 {
        engine_bull.update(0.001, 0.70, 0.50, 1.0);
        engine_bear.update(-0.001, -0.70, 0.50, 1.0);
    }

    let rep_bull = engine_bull.update(0.001, 0.70, 0.50, 1.0);
    let rep_bear = engine_bear.update(-0.001, -0.70, 0.50, 1.0);

    assert!(
        rep_bull.directional_signal > rep_bear.directional_signal,
        "Flujo alcista debe producir señal direccional superior al flujo bajista"
    );
}

#[test]
fn test_c4_deconfinement_veto_propagation() {
    let mut config = KitaevQuantumDoubleConfig::default();
    config.min_ground_state_fidelity = 0.95; // Umbral ultraestricto para inducir veto
    let mut engine = KitaevQuantumDoubleEngine::new(config);

    // Shock turbulento con spread extremo que dispersa la función de onda gauge
    let rep = engine.update(-0.02, 0.90, 8.0, 45.0);

    assert!(rep.deconfinement_veto);
    assert_eq!(rep.slot_scale, 0.10);
}

#[test]
fn test_c5_zero_panic_rb_p1_and_numerical_guards() {
    let mut engine = KitaevQuantumDoubleEngine::new(KitaevQuantumDoubleConfig::default());

    let degenerate_inputs = [
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, 1e25, -100.0),
        (-1.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 50.0, 200.0),
    ];

    for &(ret, ofi, vol, spr) in &degenerate_inputs {
        let rep = engine.update(ret, ofi, vol, spr);
        assert!(rep.ground_state_fidelity.is_finite());
        assert!(rep.topological_order_parameter.is_finite());
        assert!(rep.star_operator_expectation.is_finite());
        assert!(rep.plaquette_operator_expectation.is_finite());
        assert!(rep.directional_signal.is_finite());
        assert!(rep.slot_scale >= 0.10 && rep.slot_scale <= 1.25);
    }
}
