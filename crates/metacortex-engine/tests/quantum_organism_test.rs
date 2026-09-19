use metacortex_engine::{
    read_epigenoma_gene, set_epigenoma_gene, CazadorConstantes, ConsejoDeliberacion,
    FaseAutonomous, FaseAutonomousManager, HealthMetrics, MarketSnapshotPayload,
};

#[test]
fn test_quantum_organism_components() {
    // 1. Test Council of Seniors Deliberation
    // MOD2/7-006: payload diverso — cada asiento recibe SU dato independiente
    // (ml/espectro/ATR/racha aparte del OBI; graph_correlation retirado).
    let consejo = ConsejoDeliberacion::new();
    let healthy_payload = MarketSnapshotPayload {
        horizon: metacortex_engine::consejo_seniors::TradingHorizon::Continuous,
        book_imbalance: 0.1,
        hurst_exponent: 0.7,
        ml_prob: 0.62,
        fused_score: 0.45,
        persistence: 0.55,
        atr_pct: 0.0012,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.05,
        causal_veto_threshold: 0.80,
        current_drawdown_pct: 0.02,
        estimated_slippage_bps: 0.001,
        dominant_tau_ms: 1_138_000.0,
            whale_burst_z: 0.0,
            liquidation_severity: 0.0,
            open_interest_norm: 0.0,
            spoof_score: 0.0,
            crowd_ls_ratio: 1.0,
            crowd_taker_ratio: 1.0,
    };
    let result = consejo.deliberar(&healthy_payload, 0.65);
    assert!(
        result.approved,
        "Council should approve healthy market snapshot"
    );
    assert!(result.vetoed_by.is_none());

    // Test Risk Veto (Drawdown 85% > 75% bootstrap limit)
    let risky_payload = MarketSnapshotPayload {
        current_drawdown_pct: 0.99,
        ..healthy_payload
    };
    let veto_result = consejo.deliberar(&risky_payload, 0.65);
    assert!(!veto_result.approved);
    assert_eq!(
        veto_result.vetoed_by,
        Some(metacortex_engine::SeniorRole::Riesgo)
    );

    // Test Data Integrity Veto (Corrupt NaN payload)
    let corrupt_payload = MarketSnapshotPayload {
        book_imbalance: f64::NAN,
        ..healthy_payload
    };
    let corrupt_result = consejo.deliberar(&corrupt_payload, 0.65);
    assert!(!corrupt_result.approved);

    // 2. Test Constant Hunter AST Transformation with Unique Keys
    let sample_rust_code = r#"
        pub fn compute_threshold() -> f64 {
            let limit = 0.05;
            let multiplier = 2.5;
            limit * multiplier
        }
    "#;

    let (transformed, count) = CazadorConstantes::strip_constants(sample_rust_code).unwrap();
    assert_eq!(count, 2, "Both 0.05 and 2.5 must be replaced");
    assert!(transformed.contains("gene_global_idx_1"));
    assert!(transformed.contains("gene_global_idx_2"));

    // Verify Epigenoma store read/write
    set_epigenoma_gene("gene_global_idx_1", 0.08);
    assert_eq!(read_epigenoma_gene("gene_global_idx_1", 0.05), 0.08);

    // QO-M2.1: QuantumEvolver DELETED — era teatro puro (el error residual
    // era idéntico para todos los candidatos: sin SPSA, sin Hamiltoniano,
    // sin evaluación real). El test que lo ejercitaba se retiró con él.
}
