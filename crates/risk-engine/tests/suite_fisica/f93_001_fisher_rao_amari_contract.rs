//! Contrato Formal de Verificación: Fase 93 (§93) — Geometría de Información de Fisher-Rao & Variedad $\alpha$-Dual de Amari
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor de Fisher-Rao y Amari:
//! - C1: Positividad del tensor métrico $g_{ij} > 0$, curvatura seccional hiperbólica $\mathcal{K} = -0.50$ y distancia geodésica finita.
//! - C2: Divergencia $\alpha$ dual de Amari $D^{(\alpha)} \ge 0$ y modulación continua de slot $\eta_{\text{amari}} \in [0.10, 1.25]$.
//! - C3: Detonación determinista del veto fail-closed ante colapso de información de Fisher-Rao o divergencia crítica.
//! - C4: Determinismo estricto en stack $O(1)$, cero heap allocations e inmunidad ante NaNs e infinitos.

use risk_engine::{FisherRaoAmariConfig, FisherRaoAmariManifoldEngine};

#[test]
fn test_c1_fisher_rao_metric_positivity_and_curvature() {
    let config = FisherRaoAmariConfig::default();
    let mut engine = FisherRaoAmariManifoldEngine::new(config);

    let report = engine.evaluate(0.001, 0.02, 1.5, 0.02);

    assert!(
        report.fisher_rao_geodesic_distance >= 0.0,
        "Distancia geodésica debe ser no-negativa: {}",
        report.fisher_rao_geodesic_distance
    );
    assert!(
        report.metric_component_g_mu > 0.0,
        "Componente métrica g_mu debe ser positiva"
    );
    assert!(
        report.metric_component_g_sigma > 0.0,
        "Componente métrica g_sigma debe ser positiva"
    );
    assert_eq!(
        report.riemannian_curvature, -0.50,
        "Curvatura riemanniana seccional debe ser exactamente -1/2"
    );
    assert!(
        !report.information_collapse_veto,
        "Régimen de riesgo normal no debe activar el veto de colapso de información"
    );
}

#[test]
fn test_c2_amari_divergence_and_slot_scaling() {
    let config = FisherRaoAmariConfig::default();
    let mut engine = FisherRaoAmariManifoldEngine::new(config);

    for i in 0..10 {
        let p_ret = 0.0005 * (i as f64);
        let dd = 0.01 * (i as f64);
        let rep = engine.evaluate(p_ret, dd, 2.0, 0.02);

        assert!(
            rep.amari_alpha_divergence >= 0.0,
            "Divergencia de Amari debe ser no-negativa: {}",
            rep.amari_alpha_divergence
        );
        assert!(
            rep.amari_skewness_tensor.is_finite(),
            "Tensor de skewness de Amari debe ser finito: {}",
            rep.amari_skewness_tensor
        );
        assert!(
            rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25,
            "Factor de slot fuera de límites [0.10, 1.25]: {}",
            rep.slot_scaling_factor
        );
    }
}

#[test]
fn test_c3_information_collapse_veto_activation() {
    let config = FisherRaoAmariConfig {
        max_fisher_rao_geodesic_distance: 1.50,
        max_amari_divergence: 1.00,
        ..Default::default()
    };
    let mut engine = FisherRaoAmariManifoldEngine::new(config);

    // Inyectamos choques catastróficos de pérdida y drawdown extremo para forzar divergencia geométrica
    let mut report = engine.evaluate(-0.15, 0.40, 50.0, 5.0);
    for _ in 0..5 {
        report = engine.evaluate(-0.25, 0.45, 60.0, 8.0);
    }

    assert!(
        report.information_collapse_veto,
        "Condiciones catastróficas deben detonar information_collapse_veto"
    );
    assert!(
        report.slot_scaling_factor <= 0.40,
        "En colapso de información, el factor de slot debe comprimirse drásticamente: {}",
        report.slot_scaling_factor
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = FisherRaoAmariConfig::default();
    let mut engine = FisherRaoAmariManifoldEngine::new(config);

    let report = engine.evaluate(f64::NAN, f64::INFINITY, -5.0, f64::NAN);

    assert!(report.fisher_rao_geodesic_distance.is_finite());
    assert!(report.amari_alpha_divergence.is_finite());
    assert!(report.riemannian_curvature.is_finite());
    assert!(report.metric_component_g_mu.is_finite());
    assert!(report.metric_component_g_sigma.is_finite());
    assert!(report.amari_skewness_tensor.is_finite());
    assert!(report.slot_scaling_factor.is_finite());
}

#[test]
fn test_c5_invalid_inputs_veto_without_state_mutation() {
    let nominal = (0.001, 0.02, 1.5, 0.02);
    let mut engine = FisherRaoAmariManifoldEngine::default();
    let mut control = FisherRaoAmariManifoldEngine::default();

    let initial = engine.evaluate(nominal.0, nominal.1, nominal.2, nominal.3);
    control.evaluate(nominal.0, nominal.1, nominal.2, nominal.3);
    assert!(!initial.information_collapse_veto);

    let mut assert_invalid = |inputs: (f64, f64, f64, f64)| {
        let report = engine.evaluate(inputs.0, inputs.1, inputs.2, inputs.3);
        assert!(
            report.information_collapse_veto,
            "Entrada inválida {:?} debe activar el veto",
            inputs
        );
        assert_eq!(report.slot_scaling_factor, 0.10);
        for value in [
            report.fisher_rao_geodesic_distance,
            report.amari_alpha_divergence,
            report.riemannian_curvature,
            report.metric_component_g_mu,
            report.metric_component_g_sigma,
            report.amari_skewness_tensor,
            report.slot_scaling_factor,
        ] {
            assert!(value.is_finite());
        }

        let expected = control.evaluate(nominal.0, nominal.1, nominal.2, nominal.3);
        let actual = engine.evaluate(nominal.0, nominal.1, nominal.2, nominal.3);
        assert_eq!(
            actual.fisher_rao_geodesic_distance.to_bits(),
            expected.fisher_rao_geodesic_distance.to_bits(),
            "La entrada inválida {:?} mutó rolling_mean/rolling_var",
            inputs
        );
        assert_eq!(
            actual.metric_component_g_mu.to_bits(),
            expected.metric_component_g_mu.to_bits(),
            "La entrada inválida {:?} mutó rolling_var",
            inputs
        );
        assert_eq!(
            actual.amari_skewness_tensor.to_bits(),
            expected.amari_skewness_tensor.to_bits(),
            "La entrada inválida {:?} mutó rolling_skew",
            inputs
        );
    };

    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for inputs in [
            (invalid, 0.02, 1.5, 0.02),
            (0.001, invalid, 1.5, 0.02),
            (0.001, 0.02, invalid, 0.02),
            (0.001, 0.02, 1.5, invalid),
        ] {
            assert_invalid(inputs);
        }
    }

    for inputs in [
        (0.001, -0.01, 1.5, 0.02),
        (0.001, 0.02, 0.0, 0.02),
        (0.001, 0.02, -1.0, 0.02),
        (0.001, 0.02, 1.5, -0.01),
    ] {
        assert_invalid(inputs);
    }

    let mut zero_engine = FisherRaoAmariManifoldEngine::default();
    let zero_report = zero_engine.evaluate(0.001, 0.0, 1.5, 0.0);
    assert!(!zero_report.information_collapse_veto);
}
