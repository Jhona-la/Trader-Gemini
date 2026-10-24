// ============================================================================
// CONTRATO FORMAL DE ADAPTACIÓN DINÁMICA: SENIOR AUDITOR INTERNO & TRACKER
// OLA Ω119 — DESACOPLAMIENTO DE AUTORIDAD DISIDENTE, AUTOENGAÑO Y ADAPTACIÓN ONLINE
// ============================================================================

use metacortex_engine::{
    consejo_seniors::{
        evaluate_auditor_interno_opinion, ConsejoDeliberacion, CouncilParams,
        MarketSnapshotPayload, SeniorAgent, SeniorAuditorInterno, SeniorOpinion, SeniorRole,
        DEFAULT_DEVIL_ADVOCATE_WEIGHT, DEFAULT_DISCREPANCY_VETO_THRESHOLD,
        SeniorPerformanceTracker,
    },
    online_learning::OnlineLearningModule,
};

#[test]
fn test_01_canonical_parity_auditor_interno_opinion() {
    let payload = MarketSnapshotPayload::default();
    let (dir, conf, weight, is_veto, justification) = evaluate_auditor_interno_opinion(&payload);

    assert_eq!(dir, 0.0, "Auditor Interno debe ser neutral en dirección propia (sin sesgo direccional)");
    assert_eq!(conf, 1.0, "Auditor Interno mantiene convicción analítica 1.0");
    assert_eq!(weight, DEFAULT_DEVIL_ADVOCATE_WEIGHT, "Peso canónico debe ser exactamente DEFAULT_DEVIL_ADVOCATE_WEIGHT (3.0)");
    assert!(!is_veto, "En payload por defecto (discrepancy=0.0 <= 0.20), no debe haber veto");
    assert!(justification.contains("Auditor Interno OK"), "Justificación debe reflejar estado OK");

    let senior = SeniorAuditorInterno;
    let opinion: SeniorOpinion = senior.evaluate(&payload, 0.65);
    assert_eq!(opinion.role, SeniorRole::AuditorInterno);
    assert_eq!(opinion.weight, DEFAULT_DEVIL_ADVOCATE_WEIGHT);
    assert!(!opinion.is_veto);
}

#[test]
fn test_02_dynamic_devil_advocate_weight_adaptation() {
    let senior = SeniorAuditorInterno;

    // Caso A: Peso reforzado a 5.5
    let mut payload_boosted = MarketSnapshotPayload::default();
    payload_boosted.devil_advocate_weight = 5.5;
    let op_boosted = senior.evaluate(&payload_boosted, 0.50);
    assert_eq!(op_boosted.weight, 5.5, "El peso del Abogado del Diablo debe adaptarse a 5.5");

    // Caso B: Peso mínimo de autoridad a 0.8
    let mut payload_min = MarketSnapshotPayload::default();
    payload_min.devil_advocate_weight = 0.8;
    let op_min = senior.evaluate(&payload_min, 0.50);
    assert_eq!(op_min.weight, 0.8, "El peso del Abogado del Diablo debe adaptarse a 0.8");

    // Caso C: Fallback ante peso no finito o fuera de cota
    let mut payload_corrupt = MarketSnapshotPayload::default();
    payload_corrupt.devil_advocate_weight = f64::NAN;
    let op_corrupt = senior.evaluate(&payload_corrupt, 0.50);
    assert_eq!(op_corrupt.weight, DEFAULT_DEVIL_ADVOCATE_WEIGHT, "Peso no finito debe hacer fallback al default");
}

#[test]
fn test_03_discrepancy_veto_firing_on_self_deception() {
    let senior = SeniorAuditorInterno;

    // Caso A: Discrepancia dentro de tolerancia (0.15 <= 0.20) -> OK
    let mut payload_healthy = MarketSnapshotPayload::default();
    payload_healthy.internal_discrepancy = 0.15;
    payload_healthy.discrepancy_veto_threshold = 0.20;
    let op_healthy = senior.evaluate(&payload_healthy, 0.60);
    assert!(!op_healthy.is_veto, "Discrepancia 0.15 <= 0.20 no debe vetar");
    assert!(op_healthy.justification.contains("Auditor Interno OK"));

    // Caso B: Discrepancia superando umbral (0.22 > 0.20) -> VETO POR AUTOENGAÑO
    let mut payload_divergent = MarketSnapshotPayload::default();
    payload_divergent.internal_discrepancy = 0.22;
    payload_divergent.discrepancy_veto_threshold = 0.20;
    let op_divergent = senior.evaluate(&payload_divergent, 0.60);
    assert!(op_divergent.is_veto, "Discrepancia 0.22 > 0.20 debe disparar VETO de Auditor Interno");
    assert!(op_divergent.justification.contains("Auditor Interno VETO: autoengaño/discrepancia"));

    // Caso C: Umbral dinámico adaptativo más tolerante (umbral = 0.35, discrepancia = 0.30) -> OK
    let mut payload_tolerant = MarketSnapshotPayload::default();
    payload_tolerant.internal_discrepancy = 0.30;
    payload_tolerant.discrepancy_veto_threshold = 0.35;
    let op_tolerant = senior.evaluate(&payload_tolerant, 0.60);
    assert!(!op_tolerant.is_veto, "Bajo umbral dinámico 0.35, discrepancia 0.30 debe permitirse");
}

#[test]
fn test_04_dynamic_tracker_weight_bounds_adaptation() {
    let mut tracker = SeniorPerformanceTracker::new(500);

    // Sembrar 10 resultados para que supere min_trials
    let signals = [1.0; 11];
    for _ in 0..10 {
        tracker.record_outcome(&signals, 0.05);
    }

    // Evaluación con parámetros desacoplados: piso 0.30, techo 3.0, min_trials 5
    let custom_weights = tracker.compute_weights_with_params(5, 0.30, 3.0);
    for &w in custom_weights.iter() {
        assert!(w >= 0.30 - 1e-6, "Peso tracker {} no debe caer bajo piso 0.30", w);
        assert!(w <= 3.0 + 1e-6, "Peso tracker {} no debe exceder techo 3.0", w);
    }

    // Evaluación sin historial suficiente: min_trials 20 > 10 ensayos -> debe devolver [1.0; 11]
    let neutral_weights = tracker.compute_weights_with_params(20, 0.30, 3.0);
    for &w in neutral_weights.iter() {
        assert_eq!(w, 1.0, "Bajo min_trials insatisfecho, tracker debe devolver 1.0 neutral");
    }
}

#[test]
fn test_05_online_learning_module_hyperparameter_adaptation() {
    let mut learner = OnlineLearningModule::new(0.001, 0.90);

    // Actualización válida de hiperparámetros
    learner.update_hyperparameters(
        Some(0.005),
        Some(0.85),
        Some(1e-3),
        Some(5e-2),
        Some(0.98),
    );

    // Comprobación de funcionamiento tras adaptación de hiperparámetros
    let features = [0.1f32; 64];
    let pred_before = learner.predict(&features);
    assert!(pred_before.is_finite());

    // inst_vol = 0,55: el proxy que calculaba el envoltorio retirado (RET-20).
    learner.update_weights_with_kalman_adaptive_vol(&features, 1.0, 0.0, 0.55);
    let pred_after = learner.predict(&features);
    assert!(pred_after.is_finite());
}

#[test]
fn test_06_council_params_builder_and_cow_propagation() {
    // 1. Validación de cotas de CouncilParams builder
    let valid_params = CouncilParams::default()
        .with_auditor_and_tracker_bounds(4.5, 0.15, 8, 0.40, 2.5)
        .expect("Parámetros válidos");
    assert_eq!(valid_params.devil_advocate_weight, 4.5);
    assert_eq!(valid_params.discrepancy_veto_threshold, 0.15);
    assert_eq!(valid_params.tracker_min_trials, 8);
    assert_eq!(valid_params.tracker_weight_floor, 0.40);
    assert_eq!(valid_params.tracker_weight_cap, 2.5);

    // Cotas inválidas deben fallar
    assert!(CouncilParams::default().with_auditor_and_tracker_bounds(0.1, 0.15, 8, 0.40, 2.5).is_err()); // devil_weight < 0.5
    assert!(CouncilParams::default().with_auditor_and_tracker_bounds(4.5, 0.01, 8, 0.40, 2.5).is_err()); // disc < 0.05
    assert!(CouncilParams::default().with_auditor_and_tracker_bounds(4.5, 0.15, 0, 0.40, 2.5).is_err()); // min_trials == 0
    assert!(CouncilParams::default().with_auditor_and_tracker_bounds(4.5, 0.15, 8, 2.5, 0.40).is_err()); // floor >= cap

    // 2. Propagación Cow en deliberar_traced
    let consejo = ConsejoDeliberacion::with_params(valid_params).expect("Consejo inicializado");

    // Payload con discrepancy = 0.18. En default (umbral 0.20) pasaría, pero en este consejo (umbral 0.15) DEBE VETAR
    let payload = MarketSnapshotPayload {
        internal_discrepancy: 0.18,
        discrepancy_veto_threshold: DEFAULT_DISCREPANCY_VETO_THRESHOLD, // 0.20 en payload
        devil_advocate_weight: DEFAULT_DEVIL_ADVOCATE_WEIGHT,           // 3.0 en payload
        intended_direction: 1.0,
        ..Default::default()
    };

    let trace = consejo.deliberar_traced(&payload, 0.65, None);
    let auditor_op = trace
        .opinions
        .iter()
        .find(|op| op.role == SeniorRole::AuditorInterno)
        .expect("AuditorInterno opinion must exist");

    // Verificamos que CouncilParams.discrepancy_veto_threshold (0.15) sobreescribió el default del payload (0.20)
    assert!(auditor_op.is_veto, "Auditor Interno debió vetar porque discrepancy=0.18 > consejo.params.threshold=0.15");
    assert_eq!(auditor_op.weight, 4.5, "El peso del Auditor Interno debe ser 4.5 según consejo.params");
    assert!(!trace.consensus.approved, "La deliberación debe ser rechazada ante veto de Auditor Interno");
}
