use metacortex_engine::{
    ConsejoDeliberacion, CazadorConstantes, QuantumEvolver, FaseAutonomousManager, FaseAutonomous, HealthMetrics, MarketSnapshotPayload, set_epigenoma_gene, read_epigenoma_gene
};

#[test]
fn test_quantum_organism_components() {
    // 1. Test Council of Seniors Deliberation
    let consejo = ConsejoDeliberacion::new();
    let healthy_payload = MarketSnapshotPayload {
        book_imbalance: 0.5,
        hurst_exponent: 0.65,
        graph_correlation: 0.4,
        do_calculus_risk: 0.1,
        current_drawdown_pct: 0.05,
        estimated_slippage_bps: 0.001,
    };
    let result = consejo.deliberar(&healthy_payload, 0.65);
    assert!(result.approved, "Council should approve healthy market snapshot");
    assert!(result.vetoed_by.is_none());

    // Test Risk Veto (Drawdown 25% > 15% limit)
    let risky_payload = MarketSnapshotPayload {
        current_drawdown_pct: 0.25,
        ..healthy_payload
    };
    let veto_result = consejo.deliberar(&risky_payload, 0.65);
    assert!(!veto_result.approved);
    assert_eq!(veto_result.vetoed_by, Some(metacortex_engine::SeniorRole::Riesgo));

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
            let count = 14;
            limit * count as f64
        }
    "#;

    let (transformed, count) = CazadorConstantes::strip_constants(sample_rust_code).unwrap();
    assert_eq!(count, 2, "Both 0.05 and 14 must be replaced");
    assert!(transformed.contains("gene_float_idx_1"));
    assert!(transformed.contains("gene_int_idx_2"));

    // Verify Epigenoma store read/write
    set_epigenoma_gene("gene_float_idx_1", 0.08);
    assert_eq!(read_epigenoma_gene("gene_float_idx_1", 0.05), 0.08);

    // 3. Test Quantum State Annealing
    let evolver = QuantumEvolver::new();
    let best_state = evolver.anneal_and_collapse(42, 0.02);
    assert!(best_state.energy < f64::MAX);
    assert!(best_state.window_size >= 64);

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
