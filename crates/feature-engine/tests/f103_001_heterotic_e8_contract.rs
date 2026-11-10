//! # Contrato Formal de Certificación: Fase 103 (§103) — Teoría de Cuerdas Heterótica $E_8 \times E_8$
//!
//! ## Modo Profesor: 7 Pilares de Verificación Cuántica
//! - QUÉ: Validación formal del motor `HeteroticE8CalabiYauEngine`.
//! - POR QUÉ: Comprobar que la cancelación de Green-Schwarz $dH = \text{tr}(R \wedge R) - \text{tr}(F \wedge F)$ opere sin pánicos.
//! - PARA QUÉ: Proteger la cota de ruina Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ y regular el slot continuo.
//! - CÓMO: 5 contratos exhaustivos testeando álgebra de Cartan, cancelación de Bianchi, detonación de veto y resiliencia `rb_p1`.
//! - CUÁNDO: En cada ejecución de pruebas de integración continua de `feature-engine`.
//! - DÓNDE: `crates/feature-engine/tests/f103_001_heterotic_e8_contract.rs`.
//! - QUIÉN: Rol 5 (QA Engineer) y Rol 8 (Físico Teórico).

use feature_engine::heterotic_e8_calabi_yau::{
    HeteroticE8CalabiYauEngine, HeteroticE8Config, E8_DIM,
};

#[test]
fn test_c1_heterotic_e8_initialization_and_purity() {
    let mut engine = HeteroticE8CalabiYauEngine::default();
    let report = engine.update([0.0; E8_DIM]);

    assert!(
        !report.heterotic_anomaly_veto,
        "Inicialización vacía no debe detonar veto"
    );
    assert!(
        (report.slot_scaling_factor - 1.0).abs() < 1e-6,
        "Factor de escala inicial debe ser 1.0"
    );
    assert_eq!(
        report.green_schwarz_anomaly, 0.0,
        "Anomalía inicial debe ser cero"
    );
    assert_eq!(
        report.e8_instanton_number, 0.0,
        "Número de instantones inicial debe ser cero"
    );
}

#[test]
fn test_c2_cartan_matrix_e8_unimodular_positive_definiteness() {
    // Para cualquier vector no nulo en R^8, la forma cuadrática de Cartan de E_8 debe ser estrictamente positiva
    let mut v = [0.0; E8_DIM];
    for (i, val) in v.iter_mut().enumerate() {
        *val = (i as f64 + 1.0) * 0.10;
    }

    let q = HeteroticE8CalabiYauEngine::cartan_quadratic_form(&v);
    assert!(
        q > 0.0,
        "La forma cuadrática de Cartan de E_8 debe ser estrictamente definida positiva: q = {}",
        q
    );

    // Vector canónico unitario e_0: x^T C x = C_00 = 2.0
    let mut e0 = [0.0; E8_DIM];
    e0[0] = 1.0;
    let q0 = HeteroticE8CalabiYauEngine::cartan_quadratic_form(&e0);
    assert!(
        (q0 - 2.0).abs() < 1e-12,
        "La diagonal de la matriz de Cartan de E_8 debe ser exactamente 2.0"
    );
}

#[test]
fn test_c3_green_schwarz_anomaly_cancellation_under_exact_conformal_flow() {
    let mut engine = HeteroticE8CalabiYauEngine::default();

    // Alimentar 4 pasos de flujo suave y conmutativo
    for step in 1..=4 {
        let state = [step as f64 * 0.02; E8_DIM];
        let rep = engine.update(state);
        if step == 4 {
            assert!(
                !rep.heterotic_anomaly_veto,
                "Flujo proporcional regular debe satisfacer la cancelación de anomalías"
            );
            assert!(
                rep.slot_scaling_factor >= 0.50,
                "Factor de escala debe ser saludable en régimen armónico"
            );
        }
    }
}

#[test]
fn test_c4_heterotic_anomaly_veto_activation_on_severe_asymmetry() {
    let config = HeteroticE8Config {
        anomaly_threshold: 0.50,
        curvature_scale: 1.0,
        gauge_coupling: 0.10,
    };
    let mut engine = HeteroticE8CalabiYauEngine::new(config);

    // Warmup inicial suave
    for step in 1..=3 {
        let mut st = [0.0; E8_DIM];
        st[0] = step as f64 * 0.01;
        engine.update(st);
    }

    // Shock de altísima torsión no conmutativa y desbalance en dimensiones ortogonales
    let mut shock_state = [0.0; E8_DIM];
    shock_state[0] = 50.0;
    shock_state[1] = -40.0;
    shock_state[2] = 30.0;
    shock_state[7] = -60.0;

    let rep = engine.update(shock_state);
    assert!(
        rep.heterotic_anomaly_veto,
        "Desgarro de torsión extrema debe detonar HETEROTIC_ANOMALY_CANCELLATION_VETO"
    );
    assert_eq!(
        rep.slot_scaling_factor, 0.10,
        "Veto de anomalía debe contraer el slot a 0.10"
    );
    assert!(
        rep.green_schwarz_anomaly > 0.50,
        "Magnitud de anomalía dH debe exceder el umbral"
    );
}

#[test]
fn test_c5_heterotic_e8_rb_p1_resilience_and_nan_guards() {
    let mut engine = HeteroticE8CalabiYauEngine::default();

    let poisoned_state = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e308,
        -1e308,
        f64::MIN_POSITIVE,
        0.0,
        -0.0,
    ];

    let report = engine.update(poisoned_state);
    assert!(
        report.riemann_curvature_sq.is_finite(),
        "Riemann curvature debe ser finita"
    );
    assert!(
        report.yang_mills_curvature_sq.is_finite(),
        "Yang-Mills curvature debe ser finita"
    );
    assert!(
        report.green_schwarz_anomaly.is_finite(),
        "Green-Schwarz anomaly debe ser finita"
    );
    assert!(
        report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.25,
        "Slot scaling factor debe estar acotado"
    );
}
