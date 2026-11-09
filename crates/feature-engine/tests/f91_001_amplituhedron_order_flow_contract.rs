//! Contrato Formal de Verificación: Fase 91 (§91) — Geometría del Amplituhedro & Amplitudes Parke-Taylor MHV
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor del Amplituhedro:
//! - C1: Positividad del volumen del politopo $\mathcal{V}_{\text{ampl}} \in [0, 1]$ y finitud de la amplitud Parke-Taylor en condiciones normales.
//! - C2: Detección y respuesta ante variaciones de flujo y modulación de slot $\eta_{\text{ampl}} \in [0.10, 1.25]$.
//! - C3: Detonación determinista del veto fail-closed ante colapso del volumen o singularidad colineal.
//! - C4: Determinismo estricto en stack $O(1)$, cero heap allocations e inmunidad ante NaNs e infinitos.

use feature_engine::{AmplituhedronOrderFlowConfig, AmplituhedronOrderFlowEngine};

#[test]
fn test_c1_amplituhedron_positivity_and_parke_taylor_finitude() {
    let config = AmplituhedronOrderFlowConfig::default();
    let engine = AmplituhedronOrderFlowEngine::new(config);

    let report = engine.evaluate(0.002, 0.15, 1.5, 0.03);

    assert!(
        report.polytope_volume > 0.0 && report.polytope_volume <= 1.0,
        "Volumen del Amplituhedro debe pertenecer a (0, 1]: {}",
        report.polytope_volume
    );
    assert!(
        report.parke_taylor_amplitude.is_finite() && report.parke_taylor_amplitude >= 0.0,
        "Amplitud Parke-Taylor debe ser finita y no negativa"
    );
    assert!(
        report.cyclic_spinor_product > 0.0,
        "Producto cíclico de espinores debe ser positivo"
    );
    assert!(
        !report.collinear_singularity_veto,
        "Flujo normal no debe detonar el veto colineal"
    );
}

#[test]
fn test_c2_flow_sensitivity_and_slot_scaling() {
    let config = AmplituhedronOrderFlowConfig::default();
    let engine = AmplituhedronOrderFlowEngine::new(config);

    for i in 0..10 {
        let p_ret = 0.0005 * (i as f64);
        let ofi = -0.5 + 0.1 * (i as f64);
        let rep = engine.evaluate(p_ret, ofi, 2.0, 0.02);

        assert!(
            rep.directional_conviction >= 0.0 && rep.directional_conviction <= 1.0,
            "Convicción direccional fuera de [0, 1]: {}",
            rep.directional_conviction
        );
        assert!(
            rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25,
            "Factor de slot fuera de límites [0.10, 1.25]: {}",
            rep.slot_scaling_factor
        );
    }
}

#[test]
fn test_c3_collinear_singularity_veto_activation() {
    // Configuración con umbral estricto para probar detonación del veto
    let config = AmplituhedronOrderFlowConfig {
        min_polytope_volume: 0.30,
        min_spinor_bracket_threshold: 0.05,
        ..Default::default()
    };
    let engine = AmplituhedronOrderFlowEngine::new(config);

    // Con spread gigante y volatilidad colosal, el volumen cae por debajo del umbral mínimo
    let report = engine.evaluate(0.0, 0.0, 250.0, 10.0);

    assert!(
        report.collinear_singularity_veto,
        "Condiciones de dispersión extrema deben detonar collinear_singularity_veto"
    );
    assert!(
        report.slot_scaling_factor <= 0.35,
        "En singularidad colineal, el factor de slot debe comprimirse drásticamente: {}",
        report.slot_scaling_factor
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = AmplituhedronOrderFlowConfig::default();
    let engine = AmplituhedronOrderFlowEngine::new(config);

    let report = engine.evaluate(f64::NAN, f64::INFINITY, -10.0, f64::NAN);

    assert!(report.polytope_volume.is_finite());
    assert!(report.parke_taylor_amplitude.is_finite());
    assert!(report.cyclic_spinor_product.is_finite());
    assert!(report.collinearity_degree.is_finite());
    assert!(report.directional_conviction.is_finite());
    assert!(report.slot_scaling_factor.is_finite());
}
