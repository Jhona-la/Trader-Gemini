//! CONTRATO FORMAL DE ADAPTACIÓN Y DESACOPLAMIENTO YANG-MILLS (OLA Ω122)
//!
//! Verifica la validación de parámetros, actualización dinámica en caliente,
//! adaptación LMS acelerada, límites de beta y escalado de corrientes de gauge.

use strategy_core::yang_mills_gauge::{YangMillsGaugeEngine, YangMillsParams};

#[test]
fn test_yang_mills_params_validation_and_defaults() {
    let def = YangMillsParams::default();
    assert!(def.validate().is_ok());
    assert!((def.adaptation_rate - 0.005).abs() < 1e-9);
    assert!((def.beta_min - 0.01).abs() < 1e-9);
    assert!((def.beta_max - 100.0).abs() < 1e-9);
    assert!((def.current_scale - 1.0).abs() < 1e-9);
    assert!((def.gauge_coupling_g - 0.05).abs() < 1e-9);

    let mut invalid = def;
    invalid.adaptation_rate = -0.01;
    assert!(invalid.validate().is_err());

    let mut invalid_beta = def;
    invalid_beta.beta_min = 10.0;
    invalid_beta.beta_max = 5.0; // max <= min
    assert!(invalid_beta.validate().is_err());

    let mut invalid_scale = def;
    invalid_scale.current_scale = 0.0;
    assert!(invalid_scale.validate().is_err());

    let mut invalid_g = def;
    invalid_g.gauge_coupling_g = -0.1;
    assert!(invalid_g.validate().is_err());
}

#[test]
fn test_yang_mills_engine_with_params_and_update_params() {
    let custom = YangMillsParams {
        adaptation_rate: 0.02,
        beta_min: 0.05,
        beta_max: 50.0,
        current_scale: 2.0,
        gauge_coupling_g: 0.10,
    };
    let mut engine = YangMillsGaugeEngine::new(3).with_params(custom);
    assert_eq!(engine.params, custom);
    assert!((engine.adaptation_rate - 0.02).abs() < 1e-9);

    let updated = YangMillsParams {
        adaptation_rate: 0.001,
        beta_min: 0.02,
        beta_max: 80.0,
        current_scale: 0.5,
        gauge_coupling_g: 0.02,
    };
    engine.update_params(updated);
    assert_eq!(engine.params, updated);
    assert!((engine.adaptation_rate - 0.001).abs() < 1e-9);
}

#[test]
fn test_yang_mills_dynamic_lms_rate_affects_beta_convergence() {
    // Motor con baja tasa de adaptación vs motor con alta tasa de adaptación
    let mut engine_slow = YangMillsGaugeEngine::new(3).with_params(YangMillsParams {
        adaptation_rate: 0.001,
        ..Default::default()
    });
    let mut engine_fast = YangMillsGaugeEngine::new(3).with_params(YangMillsParams {
        adaptation_rate: 0.05,
        ..Default::default()
    });

    let prices_0 = [100.0, 100.0, 100.0];
    engine_slow.update_and_calculate_curvature(&prices_0);
    engine_fast.update_and_calculate_curvature(&prices_0);

    // Salto asimétrico de paridad entre activo 0 y 1 con retorno no nulo en ambos activos
    let prices_1 = [120.0, 105.0, 100.0];
    engine_slow.update_and_calculate_curvature(&prices_1);
    engine_fast.update_and_calculate_curvature(&prices_1);

    // El motor rápido debe haber adaptado beta[0][1] mucho más que el motor lento
    let delta_slow = (engine_slow.beta_matrix[0][1] - 1.0).abs();
    let delta_fast = (engine_fast.beta_matrix[0][1] - 1.0).abs();

    assert!(
        delta_fast > delta_slow,
        "Fast adaptation rate ({delta_fast}) must exceed slow adaptation rate ({delta_slow})"
    );
}

#[test]
fn test_yang_mills_restoring_current_scale() {
    let mut engine_unit = YangMillsGaugeEngine::new(3).with_params(YangMillsParams {
        current_scale: 1.0,
        ..Default::default()
    });
    let mut engine_boosted = YangMillsGaugeEngine::new(3).with_params(YangMillsParams {
        current_scale: 3.0,
        ..Default::default()
    });

    let prices_0 = [100.0, 100.0, 100.0];
    engine_unit.update_and_calculate_curvature(&prices_0);
    engine_boosted.update_and_calculate_curvature(&prices_0);

    // Generar dislocación triangular: P_0 sube un 10%
    let prices_1 = [110.0, 100.0, 100.0];
    let (_, curr_unit) = engine_unit.update_and_calculate_curvature(&prices_1);
    let (_, curr_boosted) = engine_boosted.update_and_calculate_curvature(&prices_1);

    assert!(
        curr_boosted[0].abs() >= curr_unit[0].abs(),
        "Boosted current ({}) must have magnitude >= unit current ({})",
        curr_boosted[0],
        curr_unit[0]
    );
}
