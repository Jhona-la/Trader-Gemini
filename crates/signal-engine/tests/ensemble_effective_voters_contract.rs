//! Contrato formal: Ensamble Efectivo y Eliminación del Sesgo de Votantes Fantasma (Ola Ω103).
//!
//! F3.1-04: El boost de ensamble debe responder exclusivamente a la concurrencia
//! de evidencia activa (|voto| > 1e-9). Estrategias registradas que se abstienen (voto = 0.0)
//! o emiten salidas no finitas no deben conceder amplificación de consenso a un votante solitario.

use omniscient_registry::OmniscientRegistry;
use quantum_arena::GlobalArena;
use signal_engine::orchestrator::{compute_ensemble_boost, TensorVoteOrchestrator};
use std::sync::Arc;
use strategy_core::QuantumStrategy;

struct MockStrategy {
    name: &'static str,
    val: f64,
}

impl QuantumStrategy for MockStrategy {
    fn name(&self) -> &str {
        self.name
    }
    fn init(&mut self, _registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        self.val
    }
}

#[test]
fn test_compute_ensemble_boost_pure_function_properties() {
    // 0 o 1 votante: estrictamente neutro (1.0)
    assert_eq!(compute_ensemble_boost(0, 0), 1.0);
    assert_eq!(compute_ensemble_boost(0, 5), 1.0);
    assert_eq!(compute_ensemble_boost(1, 1), 1.0);
    assert_eq!(compute_ensemble_boost(1, 5), 1.0);
    assert_eq!(compute_ensemble_boost(1, 100), 1.0);

    // Escalamiento monotónico con votantes activos reales
    assert!((compute_ensemble_boost(2, 5) - 1.1).abs() < 1e-12);
    assert!((compute_ensemble_boost(3, 5) - 1.2).abs() < 1e-12);
    assert!((compute_ensemble_boost(4, 5) - 1.3).abs() < 1e-12);
    assert!((compute_ensemble_boost(5, 5) - 1.4).abs() < 1e-12);

    // Saturación en 1.4 para 5 o más votantes
    assert!((compute_ensemble_boost(6, 10) - 1.4).abs() < 1e-12);
    assert!((compute_ensemble_boost(20, 20) - 1.4).abs() < 1e-12);

    // Clamping defensivo si active_voters supera total_registered
    assert_eq!(compute_ensemble_boost(5, 1), 1.0);
    assert!((compute_ensemble_boost(5, 2) - 1.1).abs() < 1e-12);
}

#[test]
fn test_un_votante_solitario_entre_cinco_estrategias_no_recibe_boost_fantasma() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // 1 estrategia activa y 4 silentes/abstenidas
    orch.add_strategy(Box::new(MockStrategy { name: "Activa", val: 0.50 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Silente1", val: 0.0 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Silente2", val: 0.0 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Silente3", val: 0.0 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Silente4", val: 0.0 }));

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Con 1 votante de 0.50 sobre 5 estrategias:
    // avg_conviction = 0.50 / 5 = 0.10.
    // Con el código corregido (Ola Ω103), active_voters = 1 -> boost = 1.0.
    // effective_conviction = 0.10 * 1.0 = 0.10.
    // Con el código previo no corregido, se usaba all.len()=5 -> boost = 1.4 -> 0.14.
    // prob_long = 1.0, prob_short = 0.0 -> net = 1.0 * 0.10 = 0.10.
    assert!(
        (decision.net_confidence - 0.10).abs() < 1e-9,
        "La confianza esperada es 0.10 (sin boost fantasma), pero fue {}",
        decision.net_confidence
    );
}

#[test]
fn test_consenso_unánime_recibe_boost_legítimo_por_concurrencia() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // 5 estrategias activas unánimes en 0.50
    for i in 0..5 {
        orch.add_strategy(Box::new(MockStrategy {
            name: Box::leak(format!("Consensus_{i}").into_boxed_str()),
            val: 0.50,
        }));
    }

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // avg_conviction = (5 * 0.50) / 5 = 0.50.
    // active_voters = 5 -> boost = 1.40.
    // effective_conviction = 0.50 * 1.40 = 0.70.
    // net = 1.0 * 0.70 = 0.70.
    assert!(
        (decision.net_confidence - 0.70).abs() < 1e-9,
        "La confianza esperada para 5 votos unánimes es 0.70, pero fue {}",
        decision.net_confidence
    );
}

#[test]
fn test_horizon_consensus_effective_voters() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);
    orch.add_strategy(Box::new(MockStrategy { name: "Activa", val: 0.50 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Silente1", val: 0.0 }));

    let decision = orch.evaluate_horizon_consensus();
    // avg_conviction = 0.50 / 2 = 0.25.
    // active_voters = 1 -> boost = 1.0.
    // effective_conviction = 0.25 * 1.0 = 0.25.
    // net_confidence = 1.0 * 0.25 = 0.25.
    assert!(
        (decision.net_confidence - 0.25).abs() < 1e-9,
        "Horizon consensus esperado es 0.25 (sin boost fantasma), pero fue {}",
        decision.net_confidence
    );
}

#[test]
fn test_rmt_calibrated_effective_bets_dampens_collinear_clones() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    // GLM 147 measurement: central cluster of 6 collinear sources has only 1.886 effective bets
    // ratio = 1.886 / 6.0 = 0.31433
    arena.registry.set_for_coin(0, "rmt_effective_bets_ratio", 1.886 / 6.0);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // 6 estrategias activas unánimes en 0.50
    for i in 0..6 {
        orch.add_strategy(Box::new(MockStrategy {
            name: Box::leak(format!("Collinear_{i}").into_boxed_str()),
            val: 0.50,
        }));
    }

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Con Kish estándar sin RMT: n_eff = 6.0 -> boost = 1.40 -> effective_conviction = 0.50 * 1.40 = 0.70.
    // Con RMT calibrado: n_eff_rmt = (6.0 * 0.31433).max(1.0) = 1.886.
    // boost = 1.0 + (1.886 - 1.0) * 0.10 = 1.0886.
    // effective_conviction = 0.50 * 1.0886 = 0.5443.
    // net_confidence = 1.0 * 0.5443 = 0.5443.
    assert!(
        decision.net_confidence < 0.60,
        "La confianza no fue atenuada por RMT: {}",
        decision.net_confidence
    );
    assert!(
        (decision.net_confidence - 0.5443).abs() < 1e-3,
        "La confianza esperada RMT-calibrada es ~0.5443, pero fue {}",
        decision.net_confidence
    );
}
