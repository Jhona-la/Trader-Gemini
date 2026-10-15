//! CONTRATO FORMAL DE ADAPTACIÓN Y DESACOPLAMIENTO HELMHOLTZ-HODGE (OLA Ω122)
//!
//! Verifica la validación de parámetros, actualización en caliente,
//! sensibilidad adaptativa de retornos cruzados L2/L3 y umbral de energía mínima.

use feature_engine::hodge_flow::{HelmholtzHodgeFlowEngine, HelmholtzHodgeParams};

#[test]
fn test_hodge_params_validation_and_defaults() {
    let def = HelmholtzHodgeParams::default();
    assert!(def.validate().is_ok());
    assert!((def.return_sensitivity - 100.0).abs() < 1e-9);
    assert!((def.min_flow_energy_threshold - 1e-15).abs() < 1e-20);

    let mut invalid = def;
    invalid.return_sensitivity = 0.5; // < 1.0
    assert!(invalid.validate().is_err());

    let mut invalid_thresh = def;
    invalid_thresh.min_flow_energy_threshold = -1.0;
    assert!(invalid_thresh.validate().is_err());
}

#[test]
fn test_hodge_engine_with_params_and_update_params() {
    let custom = HelmholtzHodgeParams {
        return_sensitivity: 250.0,
        min_flow_energy_threshold: 1e-12,
    };
    let mut engine = HelmholtzHodgeFlowEngine::new(4).with_params(custom);
    assert_eq!(engine.params, custom);

    let updated = HelmholtzHodgeParams {
        return_sensitivity: 50.0,
        min_flow_energy_threshold: 1e-14,
    };
    engine.update_params(updated);
    assert_eq!(engine.params, updated);
}

#[test]
fn test_hodge_cross_microstructure_return_sensitivity() {
    let ofis = [10.0, -8.0, 5.0];
    let rets = [0.002, -0.003, 0.001];

    let params_low = HelmholtzHodgeParams {
        return_sensitivity: 10.0,
        ..Default::default()
    };
    let params_high = HelmholtzHodgeParams {
        return_sensitivity: 500.0,
        ..Default::default()
    };

    let (mat_low, n_low) = HelmholtzHodgeFlowEngine::build_cross_microstructure_flow_matrix_with_params(
        &ofis,
        &rets,
        &params_low,
    ).expect("valid matrix low");

    let (mat_high, n_high) = HelmholtzHodgeFlowEngine::build_cross_microstructure_flow_matrix_with_params(
        &ofis,
        &rets,
        &params_high,
    ).expect("valid matrix high");

    assert_eq!(n_low, 3);
    assert_eq!(n_high, 3);

    // Alta sensibilidad produce mayor saturación en la función de flujo cruzado
    let flow_01_low = mat_low[0][1].abs();
    let flow_01_high = mat_high[0][1].abs();
    assert!(
        flow_01_high >= flow_01_low,
        "High sensitivity flow ({flow_01_high}) must be >= low sensitivity flow ({flow_01_low})"
    );
}

#[test]
fn test_hodge_min_energy_threshold_filtering() {
    let engine_strict = HelmholtzHodgeFlowEngine::new(3).with_params(HelmholtzHodgeParams {
        min_flow_energy_threshold: 1e-6,
        ..Default::default()
    });

    // Flujo muy débil pero no nulo (energía ~ 1e-8 < 1e-6)
    let (vortex_matrix, n) = HelmholtzHodgeFlowEngine::build_pure_vortex_matrix(1e-4);
    let decomp = engine_strict.decompose(&vortex_matrix, n);
    assert!(
        decomp.is_none(),
        "Flow energy below min_flow_energy_threshold should be rejected cleanly"
    );
}
