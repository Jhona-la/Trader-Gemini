use metacortex_engine::{
    read_epigenoma_gene, set_epigenoma_gene, CazadorConstantes, ConsejoDeliberacion,
    FaseAutonomous, FaseAutonomousManager, HealthMetrics, MarketSnapshotPayload, QuantumEvolver,
};

#[test]
fn test_quantum_organism_components() {
    // 1. Test Council of Seniors Deliberation
    // MOD2/7-006: payload diverso — cada asiento recibe SU dato independiente
    // (ml/espectro/ATR/racha aparte del OBI; graph_correlation retirado).
    let consejo = ConsejoDeliberacion::new();
    let healthy_payload = MarketSnapshotPayload {
        horizon: metacortex_engine::consejo_seniors::TradingHorizon::Scalping,
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

    // 3. Test Quantum State Annealing
    let evolver = QuantumEvolver::new();
    use metacortex_engine::consejo_seniors::TradingHorizon;
    let best_state = evolver.anneal_and_collapse(42, 0.02, TradingHorizon::Scalping);
    assert!(best_state.energy < f64::MAX);
    assert!(best_state.window_size >= 16);

    // 4. Test Autonomous Phase Machine Transitions
    let mut phase_manager = FaseAutonomousManager::new();
    assert_eq!(phase_manager.current_phase, FaseAutonomous::Fase0Genesis);

    let metrics = HealthMetrics {
        concept_drift_score: 0.80, // High drift -> Mutacion
        real_drawdown_pct: 0.01,
        sharpe_30d: 1.5,
        execution_latency_us: 100,
        data_checksum_ok: true,
        compilation_success: true,
        immune_tests_pass: true,
        auditor_discrepancy_pct: 0.0,
        self_deception_detected: false,
    };

    // Transition Genesis -> Exploracion
    let phase1 = phase_manager.evaluate_transition(100, &metrics);
    assert_eq!(phase1, FaseAutonomous::Fase2Exploracion);
}
