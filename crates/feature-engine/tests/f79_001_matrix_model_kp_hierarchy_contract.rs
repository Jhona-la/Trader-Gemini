use feature_engine::{MatrixModelKpConfig, MatrixModelKpHierarchyEngine, KP_MATRIX_DIM};

#[test]
fn test_matrix_model_kp_hierarchy_basic_flow_and_moments() {
    let config = MatrixModelKpConfig::default();
    let mut engine = MatrixModelKpHierarchyEngine::new(config);

    // Matriz de identidad escalada 4x4
    let m = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let report = engine.evaluate(&m, 0.1, 0.0, 0.0);

    // Tr(I_4) = 4, t1 = 4
    assert!((report.moments_tk[0] - 4.0).abs() < 1e-6);
    // Tr(I_4^2) = 4, t2 = 0.5 * 4 = 2.0
    assert!((report.moments_tk[1] - 2.0).abs() < 1e-6);
    // Tr(I_4^3) = 4, t3 = (1/3) * 4 = 1.333333
    assert!((report.moments_tk[2] - 4.0 / 3.0).abs() < 1e-6);
    // Tr(I_4^4) = 4, t4 = 0.25 * 4 = 1.0
    assert!((report.moments_tk[3] - 1.0).abs() < 1e-6);

    assert!(report.tau_function_val.is_finite());
    assert!(report.tau_function_val > 0.0);
    assert!(report.soliton_potential_u.is_finite());
    assert!(report.kp_curvature_uxx.is_finite());
    assert!(report.dispersion_residual.is_finite());
    assert!(report.slot_scale_factor >= 0.20 && report.slot_scale_factor <= 1.35);
}

#[test]
fn test_matrix_model_kp_hierarchy_shock_detection() {
    let config = MatrixModelKpConfig {
        critical_residual_threshold: 5.0, // Umbral estricto para forzar detección de shock
        ..Default::default()
    };
    let mut engine = MatrixModelKpHierarchyEngine::new(config);

    // Matriz con correlaciones asimétricas extremas para generar gran dispersión
    let m = [
        [5.0, 4.0, 3.0, 2.0],
        [4.0, 5.0, 2.0, 1.0],
        [3.0, 2.0, 6.0, 3.0],
        [2.0, 1.0, 3.0, 7.0],
    ];

    let mut shock_detected = false;
    for step in 1..=10 {
        let report = engine.evaluate(&m, 0.05, 1.5 * (step as f64), 0.5 * (step as f64));
        if report.kp_soliton_shock_detected {
            shock_detected = true;
            assert_eq!(report.slot_scale_factor, 0.20);
            break;
        }
    }

    assert!(
        shock_detected,
        "El choque KP solitónico debe detectarse ante fuertes gradientes y curvatura"
    );
}

#[test]
fn test_matrix_model_kp_hierarchy_stability_under_nan_and_zeros() {
    let config = MatrixModelKpConfig::default();
    let mut engine = MatrixModelKpHierarchyEngine::new(config);

    let m_zero = [[0.0; KP_MATRIX_DIM]; KP_MATRIX_DIM];
    let report_zero = engine.evaluate(&m_zero, 0.0, 0.0, 0.0);

    assert!(report_zero.soliton_potential_u.is_finite());
    assert!(report_zero.slot_scale_factor >= 0.20 && report_zero.slot_scale_factor <= 1.35);

    let m_nan = [
        [f64::NAN, 0.0, 0.0, 0.0],
        [0.0, f64::INFINITY, 0.0, 0.0],
        [0.0, 0.0, f64::NEG_INFINITY, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let report_nan = engine.evaluate(&m_nan, f64::NAN, f64::NAN, f64::NAN);
    assert!(report_nan.soliton_potential_u.is_finite());
    assert!(report_nan.slot_scale_factor >= 0.20 && report_nan.slot_scale_factor <= 1.35);
}

#[test]
fn test_matrix_model_kp_hierarchy_deterministic_replay() {
    let config = MatrixModelKpConfig::default();
    let mut engine_a = MatrixModelKpHierarchyEngine::new(config);
    let mut engine_b = MatrixModelKpHierarchyEngine::new(config);

    let m = [
        [1.2, 0.3, 0.1, 0.05],
        [0.3, 0.9, 0.2, 0.1],
        [0.1, 0.2, 1.5, 0.4],
        [0.05, 0.1, 0.4, 0.8],
    ];

    let rep_a1 = engine_a.evaluate(&m, 0.1, 0.5, 0.2);
    let rep_b1 = engine_b.evaluate(&m, 0.1, 0.5, 0.2);

    assert_eq!(rep_a1.moments_tk, rep_b1.moments_tk);
    assert_eq!(rep_a1.tau_function_val, rep_b1.tau_function_val);
    assert_eq!(rep_a1.soliton_potential_u, rep_b1.soliton_potential_u);
    assert_eq!(rep_a1.slot_scale_factor, rep_b1.slot_scale_factor);
}
