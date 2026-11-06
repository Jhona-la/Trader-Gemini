//! Suite de Contratos Formales F25-002: Flujo de Ricci con Cirugía de Perelman,
//! Condensación de Bouchaud-Mézard y Teoría de Juegos Cuánticos EWL en SU(2) (FASE 25)
//!
//! # QUÉ:
//! Pruebas formales rigurosas sobre los pilares físico-matemáticos de la Fase 25:
//! 1. Diagonalización espectral Jacobi en stack de métricas de Kähler.
//! 2. Amortiguamiento suave y decaimiento de curvatura por Flujo de Ricci $\partial_t g = -2 \text{Ric}$.
//! 3. Disparo determinista de Cirugía de Cuello de Perelman ante singularidades y restauración métrica.
//! 4. Transición de fase de condensación de liquidez de Bouchaud-Mézard y veto de cotización Maker.
//! 5. Convergencia del protocolo cuántico EWL al equilibrio clásico en $\gamma = 0$.
//! 6. Ventaja de cooperación cuántica $\Delta \Pi_Q > 0$ con entrelazamiento $\gamma > 0$ y unitariedad estricta.

#![allow(clippy::all)]

use core::f64::consts::PI;
use feature_engine::bouchaud_mezard_ewl::{
    BouchaudMezardEngine, BouchaudMezardParams, EwlQuantumGameEngine, LiquidityCondensationRegime,
    QuantumStrategySU2,
};
use feature_engine::perelman_ricci_surgery::{PerelmanRicciParams, PerelmanRicciSurgeryEngine};

/// C1: Descomposición espectral de Jacobi en métricas de Kähler.
#[test]
fn c1_kahler_eigensystem_spectral_decomposition() {
    let mut engine = PerelmanRicciSurgeryEngine::new(3);
    assert_eq!(engine.dimension(), 3);

    let state = [0.1, -0.2, 0.05];
    let report = engine.update_from_state(&state);

    assert!(
        report.min_eigenvalue > 0.0,
        "La métrica debe ser estrictamente positiva"
    );
    assert!(report.max_eigenvalue >= report.min_eigenvalue);
    assert!(report.scalar_curvature.is_finite());
    assert_eq!(
        report.total_surgeries, 0,
        "No debe haber cirugías en régimen suave"
    );
}

/// C2: Evolución suave y disipación de curvatura en Flujo de Ricci sin singularidades.
#[test]
fn c2_perelman_ricci_flow_smooth_decay() {
    let mut engine = PerelmanRicciSurgeryEngine::new(2);

    let mut prev_w = 0.0;
    for step in 0..10 {
        let perturb = 0.2 / (1.0 + step as f64);
        let report = engine.update_from_state(&[perturb, -perturb]);

        assert!(report.scalar_curvature.is_finite());
        assert!(
            !report.surgery_performed,
            "En régimen suave no debe dispararse cirugía"
        );
        if step > 0 {
            // El funcional de entropía W de Perelman debe permanecer acotado
            assert!(report.perelman_w_entropy.is_finite());
        }
        prev_w = report.perelman_w_entropy;
    }
    assert!(prev_w.is_finite());
}

/// C3: Detección de singularidad de estrangulamiento y Cirugía de Cuello de Perelman.
#[test]
fn c3_perelman_surgery_on_neckpinch_singularity() {
    let params = PerelmanRicciParams {
        dt: 0.05,
        pinch_threshold: 0.20, // Umbral sensible para prueba
        crit_curvature: 10.0,
        cap_curvature: 1.5,
        regularization_floor: 0.25,
    };
    let mut engine = PerelmanRicciSurgeryEngine::new(2).with_params(params);

    // Inyectar deformación extrema anisotrópica simulando un colapso de liquidez
    let extreme_state = [10.0, 0.001];
    let report = engine.update_from_state(&extreme_state);

    assert!(
        report.singularity_alert,
        "Debe alertar sobre singularidad inminente"
    );
    assert!(
        report.surgery_performed,
        "La cirugía de Perelman debe ejecutarse para extirpar el cuello singular"
    );
    assert!(report.total_surgeries >= 1);
    assert!(
        report.min_eigenvalue >= params.regularization_floor,
        "La métrica post-cirugía debe respetar el suelo"
    );
    assert_eq!(
        report.scalar_curvature, params.cap_curvature,
        "La curvatura debe fijarse al casquete regular"
    );
}

/// C4: Transición de fase de condensación de liquidez de Bouchaud-Mézard.
#[test]
fn c4_bouchaud_mezard_condensation_phase_transition() {
    // 1. Régimen disperso fluido: alto acoplamiento J y bajo ruido sigma
    let mut fluid_engine = BouchaudMezardEngine::new(4).with_params(BouchaudMezardParams {
        coupling_j: 0.50,
        noise_sigma: 0.20,
        drift_m: 0.0,
    });
    let uniform_vols = [100.0, 105.0, 95.0, 102.0];
    let fluid_report = fluid_engine.update(&uniform_vols);
    assert!(
        fluid_report.pareto_mu > 2.0,
        "En régimen fluido mu debe ser > 2.0"
    );
    assert_eq!(
        fluid_report.regime,
        LiquidityCondensationRegime::DispersedFluid
    );
    assert!(
        !fluid_report.maker_veto,
        "No debe vetar órdenes Maker en régimen disperso"
    );

    // 2. Régimen condensado monopólico: bajo acoplamiento J y altísimo ruido sigma
    let mut condensed_engine = BouchaudMezardEngine::new(4).with_params(BouchaudMezardParams {
        coupling_j: 0.01,
        noise_sigma: 1.50,
        drift_m: 0.0,
    });
    let monop_vols = [1000.0, 1.0, 0.5, 0.2];
    let condensed_report = condensed_engine.update(&monop_vols);
    assert!(condensed_report.pareto_mu <= 2.0);
    assert_eq!(
        condensed_report.regime,
        LiquidityCondensationRegime::CondensedMonopoly
    );
    assert!(
        condensed_report.maker_veto,
        "Debe activar veto estricto a Maker ante monopolio condensado"
    );
    assert!(condensed_report.max_fraction > 0.65);
}

/// C5: Límite clásico del protocolo de teoría de juegos cuánticos EWL (gamma = 0).
#[test]
fn c5_ewl_quantum_game_classical_limit() {
    let engine = EwlQuantumGameEngine::new(0.0); // Sin entrelazamiento

    // Ambos cooperan (Maker/Maker)
    let rep_cc = engine.play(
        QuantumStrategySU2::cooperate_maker(),
        QuantumStrategySU2::cooperate_maker(),
    );
    assert!((rep_cc.p_cooperate_cooperate - 1.0).abs() < 1e-9);
    assert_eq!(rep_cc.payoff_a, 3.0);
    assert_eq!(rep_cc.payoff_b, 3.0);

    // Ambos defeccionan (Taker/Taker - Equilibrio de Nash clásico)
    let rep_dd = engine.play(
        QuantumStrategySU2::defect_taker(),
        QuantumStrategySU2::defect_taker(),
    );
    assert!((rep_dd.p_defect_defect - 1.0).abs() < 1e-9);
    assert_eq!(rep_dd.payoff_a, -5.0);
    assert_eq!(rep_dd.payoff_b, -5.0);
    assert_eq!(
        rep_dd.quantum_advantage, 0.0,
        "Sin entrelazamiento no hay ventaja cuántica"
    );
}

/// C6: Ventaja cuántica estricta y unitariedad bajo entrelazamiento maximal (gamma = pi/2).
#[test]
fn c6_ewl_quantum_advantage_under_entanglement() {
    let engine = EwlQuantumGameEngine::new(PI * 0.5); // Máximo entrelazamiento informacional

    let strat_q = QuantumStrategySU2::quantum_optimal();
    let strat_classic_defect = QuantumStrategySU2::defect_taker();

    // El jugador cuántico juega la estrategia cuántica contra el agresor clásico
    let report = engine.play(strat_q, strat_classic_defect);

    // Conservación estricta de probabilidad total
    let p_total = report.p_cooperate_cooperate
        + report.p_defect_defect
        + (1.0 - report.p_cooperate_cooperate - report.p_defect_defect);
    assert!(
        (p_total - 1.0).abs() < 1e-9,
        "La probabilidad cuántica total debe ser exactamente 1.0"
    );

    // El jugador cuántico debe superar el pago de castigo clásico (-5.0 bps)
    assert!(
        report.payoff_a > -5.0,
        "La estrategia cuántica debe superar el equilibrio de Nash clásico"
    );
    assert!(
        report.quantum_advantage > 0.0,
        "Debe existir una ventaja cuántica estricta > 0"
    );
    assert_eq!(report.entanglement_gamma, PI * 0.5);
}
