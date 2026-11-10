//! # Contrato Formal de Certificación: Fase 100 (§100) — Teoría M en 11D & Variedades Joyce $G_2$
//!
//! ## Modo Profesor: 7 Pilares de Verificación Cuántica
//! - QUÉ: Validación formal del motor `MTheoryJoyceG2Engine` y sus propiedades de geometría asociativa.
//! - POR QUÉ: Asegurar que la 3-forma asociativa de Fano, la 4-forma coasociativa dual y el veto de torsión operen sin pánicos.
//! - PARA QUÉ: Blindar la cota de ruina Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ y regular el slot continuo.
//! - CÓMO: 5 contratos exhaustivos testeando álgebra de Fano, derivadas exteriores discretas, veto de torsión y resiliencia `rb_p1`.
//! - CUÁNDO: En cada ejecución de pruebas de integración continua de `feature-engine`.
//! - DÓNDE: `crates/feature-engine/tests/f100_001_joyce_g2_contract.rs`.
//! - QUIÉN: Rol 5 (QA Engineer) y Rol 8 (Físico Teórico de Teoría M).

use feature_engine::m_theory_joyce_g2::{JoyceG2Config, MTheoryJoyceG2Engine, JOYCE_DIM};

#[test]
fn test_c1_joyce_g2_initialization_and_purity() {
    let engine = MTheoryJoyceG2Engine::default();
    let report = engine.clone().update([0.0; JOYCE_DIM]);

    assert!(
        !report.joyce_g2_singularity_veto,
        "No debe haber veto en inicialización vacía"
    );
    assert!(
        (report.slot_scaling_factor - 1.0).abs() < 1e-6,
        "Factor de escala debe ser 1.0"
    );
    assert!(
        (report.metric_determinant - 1.0).abs() < 1e-6,
        "Determinante inicial debe ser 1.0"
    );
    assert_eq!(report.torsion_norm, 0.0, "Torsión inicial debe ser cero");
}

#[test]
fn test_c2_fano_plane_associative_3form_properties() {
    // Base canónica e_0, e_1, ..., e_6 en R^7
    let mut e = [[0.0; JOYCE_DIM]; JOYCE_DIM];
    for (i, row) in e.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    // 1. Triple positiva del plano de Fano: (0, 1, 2)
    let phi_012 = MTheoryJoyceG2Engine::compute_associative_3form(&e[0], &e[1], &e[2]);
    assert!(
        (phi_012 - 1.0).abs() < 1e-12,
        "Phi(e0, e1, e2) debe ser +1.0"
    );

    // 2. Triple negativa del plano de Fano: (1, 4, 6)
    let phi_146 = MTheoryJoyceG2Engine::compute_associative_3form(&e[1], &e[4], &e[6]);
    assert!(
        (phi_146 - (-1.0)).abs() < 1e-12,
        "Phi(e1, e4, e6) debe ser -1.0"
    );

    // 3. Propiedad de antisimetría bajo permutación impar: Phi(u, v, w) = -Phi(v, u, w)
    let phi_102 = MTheoryJoyceG2Engine::compute_associative_3form(&e[1], &e[0], &e[2]);
    assert!(
        (phi_102 - (-1.0)).abs() < 1e-12,
        "Permutación impar debe invertir el signo"
    );

    // 4. Anulación sobre vectores linealmente dependientes
    let phi_002 = MTheoryJoyceG2Engine::compute_associative_3form(&e[0], &e[0], &e[2]);
    assert!(
        phi_002.abs() < 1e-12,
        "Phi con dos vectores idénticos debe ser cero"
    );
}

#[test]
fn test_c3_coassociative_4form_properties() {
    let mut e = [[0.0; JOYCE_DIM]; JOYCE_DIM];
    for (i, row) in e.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    // Cuadrupla complementaria canónica (3, 4, 5, 6)
    let psi_3456 = MTheoryJoyceG2Engine::compute_coassociative_4form(&e[3], &e[4], &e[5], &e[6]);
    assert!(
        (psi_3456 - 1.0).abs() < 1e-12,
        "*Phi(e3, e4, e5, e6) debe ser +1.0"
    );

    // Cuadrupla con signo negativo: (0, 2, 3, 5)
    let psi_0235 = MTheoryJoyceG2Engine::compute_coassociative_4form(&e[0], &e[2], &e[3], &e[5]);
    assert!(
        (psi_0235 - (-1.0)).abs() < 1e-12,
        "*Phi(e0, e2, e3, e5) debe ser -1.0"
    );
}

#[test]
fn test_c4_joyce_g2_torsion_and_singularity_veto_activation() {
    let config = JoyceG2Config {
        torsion_threshold: 1.50,
        metric_det_floor: 1e-7,
        associative_scale: 1.0,
    };
    let mut engine = MTheoryJoyceG2Engine::new(config);

    // Inicializar con flujo suave y variado (7 dimensiones distintas)
    for i in 1..=4 {
        let base_state = [
            (i as f64) * 0.05,
            (i as f64) * -0.03,
            (i as f64) * 0.08,
            (i as f64) * 0.02,
            (i as f64) * -0.06,
            (i as f64) * 0.04,
            (i as f64) * 0.01,
        ];
        engine.update(base_state);
    }

    // Inyectar un choque caótico multidimensional no armónico (violenta rotación fuera de G2)
    let mut violent_state = [0.0; JOYCE_DIM];
    violent_state[0] = 50.0;
    violent_state[1] = -45.0;
    violent_state[3] = 120.0;
    violent_state[6] = -80.0;

    let report = engine.update(violent_state);

    assert!(
        report.torsion_norm > 0.0,
        "La torsión debe ser estrictamente positiva"
    );
    assert!(
        report.ricci_scalar_proxy <= 0.0,
        "Curvatura de Ricci debe ser no positiva"
    );

    // Inyectar segundo choque desestabilizador masivo que supere el umbral
    let mut second_shock = [0.0; JOYCE_DIM];
    second_shock[0] = -500.0;
    second_shock[2] = 2000.0;
    second_shock[4] = -1500.0;
    second_shock[5] = 900.0;

    let report_shock = engine.update(second_shock);

    assert!(
        report_shock.torsion_norm >= config.torsion_threshold,
        "Torsión ({}) debe superar el umbral ({}) tras choque masivo",
        report_shock.torsion_norm,
        config.torsion_threshold
    );
    assert!(
        report_shock.joyce_g2_singularity_veto,
        "Debe detonar JOYCE_G2_SINGULARITY_VETO ante torsión extrema"
    );
    assert!(
        (report_shock.slot_scaling_factor - 0.10).abs() < 1e-6,
        "Slot factor debe colapsar a 0.10 bajo veto"
    );
}

#[test]
fn test_c5_rb_p1_invariants_and_numerical_resilience() {
    let mut engine = MTheoryJoyceG2Engine::default();

    // Inyectar secuencias con NaN, Inf y valores fuera de escala
    let bad_state1 = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e30,
        -1e30,
        0.0,
        1.0,
    ];
    let rep1 = engine.update(bad_state1);

    assert!(
        rep1.associative_calibration_energy.is_finite(),
        "Energía asociativa debe ser finita"
    );
    assert!(
        rep1.coassociative_volume.is_finite(),
        "Volumen coasociativo debe ser finito"
    );
    assert!(
        rep1.torsion_norm.is_finite(),
        "Norma de torsión debe ser finita"
    );
    assert!(
        rep1.metric_determinant > 0.0,
        "Determinante métrico debe ser estrictamente positivo"
    );
    assert!(
        rep1.slot_scaling_factor >= 0.10 && rep1.slot_scaling_factor <= 1.25,
        "Slot factor acotado en [0.10, 1.25]"
    );
}
