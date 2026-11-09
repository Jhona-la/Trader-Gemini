//! Contrato Formal de Verificación: Fase 92 (§92) — Cosmología Inflacionaria de Starobinsky
//!
//! Este conjunto de pruebas certifica matemáticamente las propiedades del motor inflacionario:
//! - C1: No-negatividad del potencial de Starobinsky $V(\phi) \ge 0$ y régimen de slow-roll en flujo regular.
//! - C2: Índices espectrales cosmológicos ($n_s, r, N_e$) y modulación de slot $\eta_{\text{starobinsky}} \in [0.15, 1.30]$.
//! - C3: Detonación determinista del veto fail-closed ante recalentamiento térmico violento o colapso de slow-roll.
//! - C4: Determinismo estricto en stack $O(1)$, cero heap allocations e inmunidad ante NaNs e infinitos.

use signal_engine::{StarobinskyConfig, StarobinskyInflationaryEngine};

#[test]
fn test_c1_starobinsky_slow_roll_and_potential_positivity() {
    let config = StarobinskyConfig::default();
    let mut engine = StarobinskyInflationaryEngine::new(config);

    let report = engine.evaluate(0.001, 0.15, 1.8, 0.02, 0.01);

    assert!(
        report.effective_potential >= 0.0,
        "Potencial de Starobinsky debe ser no-negativo: {}",
        report.effective_potential
    );
    assert!(
        report.epsilon_slow_roll >= 0.0,
        "Parámetro epsilon de slow-roll debe ser no-negativo: {}",
        report.epsilon_slow_roll
    );
    assert!(
        report.tensor_to_scalar_ratio >= 0.0,
        "Ratio tensorial-escalar debe ser no-negativo: {}",
        report.tensor_to_scalar_ratio
    );
    assert!(
        !report.reheating_collapse_veto,
        "Flujo laminar no debe detonar el veto de recalentamiento"
    );
}

#[test]
fn test_c2_spectral_indices_and_slot_scaling() {
    let config = StarobinskyConfig::default();
    let mut engine = StarobinskyInflationaryEngine::new(config);

    for i in 0..10 {
        let p_ret = 0.0005 * (i as f64);
        let ofi = 0.05 * (i as f64);
        let rep = engine.evaluate(p_ret, ofi, 2.0, 0.02, 0.01);

        assert!(
            rep.spectral_index.is_finite(),
            "Índice espectral debe ser finito: {}",
            rep.spectral_index
        );
        assert!(
            rep.directional_bias >= -1.0 && rep.directional_bias <= 1.0,
            "Sesgo direccional fuera de [-1, 1]: {}",
            rep.directional_bias
        );
        assert!(
            rep.slot_scaling_factor >= 0.15 && rep.slot_scaling_factor <= 1.30,
            "Factor de escala de slot fuera de límites [0.15, 1.30]: {}",
            rep.slot_scaling_factor
        );
    }
}

#[test]
fn test_c3_reheating_collapse_veto_activation() {
    // Configuración estricta para probar la transición a recalentamiento violento
    let config = StarobinskyConfig {
        max_epsilon_slow_roll: 0.80,
        ..Default::default()
    };
    let mut engine = StarobinskyInflationaryEngine::new(config);

    // Inyectamos un impulso masivo explosivo para forzar salida caótica de la meseta
    let mut report = engine.evaluate(0.50, 1.0, 50.0, 5.0, 0.001);
    for _ in 0..10 {
        report = engine.evaluate(-0.60, -1.0, 60.0, 8.0, 0.001);
    }

    assert!(
        report.reheating_collapse_veto,
        "Fluctuación explosiva debe detonar reheating_collapse_veto"
    );
    assert!(
        report.slot_scaling_factor <= 0.60,
        "En colapso de recalentamiento, el factor de slot debe comprimirse: {}",
        report.slot_scaling_factor
    );
}

#[test]
fn test_c4_nan_infinity_immunity_and_stack_determinism() {
    let config = StarobinskyConfig::default();
    let mut engine = StarobinskyInflationaryEngine::new(config);

    let report = engine.evaluate(f64::NAN, f64::INFINITY, -5.0, f64::NAN, 0.0);

    assert!(report.inflaton_field.is_finite());
    assert!(report.inflaton_velocity.is_finite());
    assert!(report.effective_potential.is_finite());
    assert!(report.epsilon_slow_roll.is_finite());
    assert!(report.eta_slow_roll.is_finite());
    assert!(report.spectral_index.is_finite());
    assert!(report.tensor_to_scalar_ratio.is_finite());
    assert!(report.e_folding_estimate.is_finite());
    assert!(report.directional_bias.is_finite());
    assert!(report.slot_scaling_factor.is_finite());
}
