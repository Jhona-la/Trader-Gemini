//! Contratos Formales de Verificación de Fase 6 (F6-001)
//!
//! Suite exhaustiva de pruebas y contratos matemáticos para:
//! - Mesa de Trading Stanford Discovery (Ciclo 4E: SOL PROPOSE -> OPUS CYPHER -> ISOLATED EVAL -> META DISTILL).
//! - Invarianza proyectiva sobre el simplex de 6 pilares teóricos (Duchi et al., 2008).
//! - Cota analítica de ruina de Cramér-Lundberg P(ruin) <= exp(-theta * u) < 1e-6.
//! - Sandbox determinista con deducción de comisiones Maker/Taker reales y slippage no lineal.
//! - Clasificación multiobjetivo Pareto NSGA-II y selección de crecimiento de Kelly.
//! - Cumplimiento estricto del mandato U-ERR-6 (cero sufijos de banda scalp/swing).
//! - Resiliencia numérica total contra NaNs, infinitos y valores subnormales IEEE-754.

use evolution_engine::stanford_discovery::{
    CypherVerdict, QuantitativeHypothesis, SandboxEvaluationMetrics, StanfordDiscoveryEngine,
    PILLAR_COUNT, PILLAR_GAUGE_YANG_MILLS, PILLAR_HYDRO_NAVIER_STOKES, PILLAR_KOOPMAN_DMD,
    PILLAR_MICROSTRUCTURE_L2L3, PILLAR_QUANTUM_MPS, PILLAR_TOPOLOGICAL_TDA,
};

/// Contrato C1: Invarianza del Simplex de 6 Pilares Cuánticos-Espectrales
#[test]
fn test_c1_simplex_projection_and_sum_invariance() {
    let test_vectors = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.0, -10.0, 5.0, -2.0, 8.0, 1.0],
    ];

    for (case_idx, raw_v) in test_vectors.iter().enumerate() {
        let projected = StanfordDiscoveryEngine::project_to_simplex(raw_v);

        let sum_w: f64 = projected.iter().sum();
        assert!(
            (sum_w - 1.0).abs() < 1e-8,
            "Caso {}: Simplex debe sumar exactamente 1.0 (fue {})",
            case_idx,
            sum_w
        );

        for (p_idx, &w) in projected.iter().enumerate() {
            assert!(
                w >= 0.0 && w.is_finite(),
                "Caso {}, pilar {}: peso debe ser finito y >= 0.0 (fue {})",
                case_idx,
                p_idx,
                w
            );
        }
    }
}

/// Contrato C2: Cota de Ruina de Cramér-Lundberg & Veto Preventivo de OPUS CYPHER
#[test]
fn test_c2_cramer_lundberg_ruin_bound_and_cypher_veto() {
    let hyp_safe = QuantitativeHypothesis::default();

    // 1. Escenario seguro: capital u = 100 USD, deriva mu = 0.03, varianza sigma^2 = 0.01
    // theta* = 2 * 0.03 / 0.01 = 6.0. P(ruin) <= exp(-6 * 100) = exp(-600) ~ 0.0
    let res_safe = StanfordDiscoveryEngine::cypher_validate(&hyp_safe, 100.0, 0.03, 0.01);
    assert!(res_safe.is_causally_valid);
    assert_eq!(res_safe.verdict, CypherVerdict::Approved);
    assert!(res_safe.ruin_probability_bound < 1e-6);

    // 2. Escenario peligroso: capital u = 0.50 USD, deriva mu = 0.001, varianza sigma^2 = 0.08
    // theta* = 2 * 0.001 / 0.08 = 0.025. P(ruin) <= exp(-0.025 * 0.5) ~ 0.9875
    let res_ruin = StanfordDiscoveryEngine::cypher_validate(&hyp_safe, 0.50, 0.001, 0.08);
    assert!(!res_ruin.is_causally_valid);
    assert_eq!(res_ruin.verdict, CypherVerdict::VetoCramerLundbergRuin);
    assert!(res_ruin.ruin_probability_bound > 1e-4);

    // 3. Escenario fuera de límites de Kelly
    let mut hyp_bad_kelly = hyp_safe;
    hyp_bad_kelly.fractional_kelly = 0.85; // Superior al límite seguro de 0.40
    let res_kelly = StanfordDiscoveryEngine::cypher_validate(&hyp_bad_kelly, 100.0, 0.03, 0.01);
    assert!(!res_kelly.is_causally_valid);
    assert_eq!(res_kelly.verdict, CypherVerdict::VetoKellyOutOfRange);
}

/// Contrato C3: Deducción de Fricciones Reales (Maker/Taker & Slippage) en Sandbox Aislado
#[test]
fn test_c3_isolated_eval_friction_drag_deduction() {
    let hyp = QuantitativeHypothesis::default();

    let returns = vec![0.005; 50]; // 50 retornos positivos de +0.5%
    let vols = vec![0.01; 50];

    // Evaluación con cero fricción
    let eval_zero_fee = StanfordDiscoveryEngine::isolated_evaluate(&hyp, &returns, &vols, 0.0, 0.0);

    // Evaluación con fricción realista del exchange (0.04% fee roundtrip + funding)
    let eval_real_fee =
        StanfordDiscoveryEngine::isolated_evaluate(&hyp, &returns, &vols, 0.0004, 0.0001);

    assert!(
        eval_real_fee.net_realized_pnl < eval_zero_fee.net_realized_pnl,
        "El PnL neto con comisiones debe ser estrictamente menor que sin comisiones"
    );
    assert!(
        eval_real_fee.fee_drag_usd > 0.0,
        "El drag de comisiones debe ser positivo y medido"
    );
    assert!(
        eval_real_fee.slippage_drag_usd > 0.0,
        "El drag de slippage debe ser positivo y medido"
    );
    assert_eq!(eval_real_fee.total_trades, 50);
}

/// Contrato C4: Clasificación No Dominada Pareto NSGA-II & Meta-Destilación
#[test]
fn test_c4_pareto_nsga2_non_dominated_fronts_and_distillation() {
    let mut engine = StanfordDiscoveryEngine::new(12345);

    let candidates = [
        QuantitativeHypothesis {
            id: 1,
            pillar_weights: [0.30, 0.20, 0.10, 0.10, 0.15, 0.15],
            fractional_kelly: 0.20,
            conformal_alpha: 0.05,
            tp_sl_ratio: 2.0,
            ofi_sensitivity: 1.0,
            yang_mills_coupling: 0.8,
        },
        QuantitativeHypothesis {
            id: 2,
            pillar_weights: [0.10, 0.10, 0.10, 0.10, 0.30, 0.30],
            fractional_kelly: 0.10,
            conformal_alpha: 0.04,
            tp_sl_ratio: 2.5,
            ofi_sensitivity: 1.5,
            yang_mills_coupling: 0.6,
        },
    ];

    let metrics = [
        SandboxEvaluationMetrics {
            hypothesis_id: 1,
            sharpe_ratio: 2.40,
            win_rate: 0.62,
            max_drawdown: 0.06,
            net_realized_pnl: 35.0,
            kelly_growth_rate: 0.003,
            total_trades: 120,
            fee_drag_usd: 1.2,
            slippage_drag_usd: 0.8,
        },
        SandboxEvaluationMetrics {
            hypothesis_id: 2,
            sharpe_ratio: 1.80,
            win_rate: 0.54,
            max_drawdown: 0.08,
            net_realized_pnl: 18.0,
            kelly_growth_rate: 0.0015,
            total_trades: 110,
            fee_drag_usd: 1.0,
            slippage_drag_usd: 0.6,
        },
    ];

    let distilled = engine.meta_distill(&candidates, &metrics, None);

    // El candidato 1 domina estrictamente al candidato 2 en Sharpe, WR, DD y PnL
    assert_eq!(distilled.champion_hypothesis.id, 1);
    assert_eq!(distilled.champion_metrics.hypothesis_id, 1);
    assert_eq!(distilled.pareto_rank, 0);
    assert!(distilled.is_promoted_to_active);
}

/// Contrato C5: Cumplimiento Estricto U-ERR-6 (Cero Sufijos Scalp/Swing)
#[test]
fn test_c5_u_err_6_zero_scalp_swing_in_discovery_structures() {
    let hyp = QuantitativeHypothesis::default();
    let serialized = serde_json::to_string(&hyp).expect("serializacion json");

    let lower = serialized.to_lowercase();
    assert!(
        !lower.contains("scalp"),
        "Violacion U-ERR-6: la hipotesis contiene el termino prohibido 'scalp'"
    );
    assert!(
        !lower.contains("swing"),
        "Violacion U-ERR-6: la hipotesis contiene el termino prohibido 'swing'"
    );

    // Verificar nombres de los 6 pilares canónicos
    assert_eq!(PILLAR_QUANTUM_MPS, 0);
    assert_eq!(PILLAR_GAUGE_YANG_MILLS, 1);
    assert_eq!(PILLAR_HYDRO_NAVIER_STOKES, 2);
    assert_eq!(PILLAR_KOOPMAN_DMD, 3);
    assert_eq!(PILLAR_MICROSTRUCTURE_L2L3, 4);
    assert_eq!(PILLAR_TOPOLOGICAL_TDA, 5);
    assert_eq!(PILLAR_COUNT, 6);
}

/// Contrato C6: Resiliencia Numérica IEEE-754 Frente a NaNs e Infinitos
#[test]
fn test_c6_ieee754_resilience_and_defensive_sanitization() {
    let mut bad_hyp = QuantitativeHypothesis::default();
    bad_hyp.fractional_kelly = f64::NAN;

    let res_nan = StanfordDiscoveryEngine::cypher_validate(&bad_hyp, 100.0, 0.02, 0.04);
    assert!(!res_nan.is_causally_valid);
    assert_eq!(res_nan.verdict, CypherVerdict::VetoNonFiniteParameter);

    // Evaluación con retornos que contienen NaN o Infinitos
    let bad_returns = vec![0.01, f64::NAN, 0.02, f64::INFINITY, -0.01, f64::NEG_INFINITY, 0.005, 0.01, 0.02, 0.01];
    let bad_vols = vec![0.02; bad_returns.len()];

    let hyp_safe = QuantitativeHypothesis::default();
    let m = StanfordDiscoveryEngine::isolated_evaluate(&hyp_safe, &bad_returns, &bad_vols, 0.0004, 0.0001);

    // Métricas deben ser finitas y no entrar en pánico
    assert!(m.sharpe_ratio.is_finite());
    assert!(m.win_rate.is_finite());
    assert!(m.max_drawdown.is_finite());
    assert!(m.net_realized_pnl.is_finite());
    assert!(m.kelly_growth_rate.is_finite());
}
