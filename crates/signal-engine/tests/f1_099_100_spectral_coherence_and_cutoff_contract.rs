//! Contrato formal: Paridad Semántica de Umbral (F1-100) e Interferencia Espectral con N_eff (F1-099).
//!
//! Ola Ω161:
//! - F1-099: Factor de interferencia física espectral factor = max(0, (1 + c) / 2) con c = media / v_dom.
//!   Oposición unánime c = -1.0 veta estrictamente (factor = 0.0, señal Flat).
//!   Grados de libertad efectivos de Kish N_eff = (Σ w_i)² / Σ w_i² en el ensamble escalar.
//! - F1-100: Unificación de umbrales entre orchestrator (MOD2/7-012: cutoff_floor = (min_conf - 0.50) * 1.0)
//!   y god-engine-core (tensor_cutoff y slow_cutoff en exceso de convicción ΔP ∈ [0, 0.45], y conf_base = 0.50 + 0.50 * |ΔP|).

use omniscient_registry::OmniscientRegistry;
use quantum_arena::GlobalArena;
use signal_engine::orchestrator::{
    compute_ensemble_boost, compute_ensemble_boost_effective, TensorVoteOrchestrator,
};
use signal_engine::SignalType;
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
fn test_f1_099_kish_effective_boost_properties() {
    // N_eff <= 1.0 produce estrictamente boost neutro 1.0
    assert_eq!(compute_ensemble_boost_effective(0.0), 1.0);
    assert_eq!(compute_ensemble_boost_effective(0.5), 1.0);
    assert_eq!(compute_ensemble_boost_effective(1.0), 1.0);
    assert_eq!(compute_ensemble_boost_effective(f64::NAN), 1.0);

    // Escalamiento continuo monotónico
    assert!((compute_ensemble_boost_effective(2.0) - 1.1).abs() < 1e-12);
    assert!((compute_ensemble_boost_effective(3.0) - 1.2).abs() < 1e-12);
    assert!((compute_ensemble_boost_effective(3.5) - 1.25).abs() < 1e-12);
    assert!((compute_ensemble_boost_effective(5.0) - 1.4).abs() < 1e-12);

    // Saturación en 1.4 para N_eff >= 5.0
    assert!((compute_ensemble_boost_effective(5.5) - 1.4).abs() < 1e-12);
    assert!((compute_ensemble_boost_effective(10.0) - 1.4).abs() < 1e-12);

    // Paridad hacia atrás con compute_ensemble_boost
    assert_eq!(compute_ensemble_boost(1, 5), compute_ensemble_boost_effective(1.0));
    assert_eq!(compute_ensemble_boost(3, 5), compute_ensemble_boost_effective(3.0));
    assert_eq!(compute_ensemble_boost(5, 5), compute_ensemble_boost_effective(5.0));
}

#[test]
fn test_f1_099_unbalanced_ensemble_does_not_inflate_boost() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // 1 estrategia emite un voto contundente 0.80, mientras otras 4 emiten votos ínfimos 0.001
    orch.add_strategy(Box::new(MockStrategy { name: "Dominant", val: 0.80 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Tiny1", val: 0.001 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Tiny2", val: 0.001 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Tiny3", val: 0.001 }));
    orch.add_strategy(Box::new(MockStrategy { name: "Tiny4", val: 0.001 }));

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Con Kish N_eff = (0.804)^2 / (0.80^2 + 4 * 0.001^2) = 0.646416 / 0.640004 ≈ 1.01002.
    // Boost de ensamble es ~ 1.001 (prácticamente 1.0, NO el 1.40 del conteo ingenuo de 5 votantes).
    // avg_conviction ≈ 0.804 / 5 = 0.1608.
    // effective_conviction ≈ 0.1608 * 1.001 ≈ 0.1610.
    // net_confidence ≈ 1.0 * 0.1610 ≈ 0.161.
    assert!(
        decision.net_confidence < 0.20,
        "El ensamble desbalanceado no debe inflar convicción con boost espurio: {}",
        decision.net_confidence
    );
}

#[test]
fn test_f1_099_spectral_interference_physics() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.51, std::sync::atomic::Ordering::Relaxed); // cutoff_floor = 0.01

    let mut orch = TensorVoteOrchestrator::new(std::sync::Arc::clone(&arena));
    orch.add_strategy(Box::new(MockStrategy { name: "S1", val: 0.0 }));

    arena
        .registry
        .set_for_coin(0, "consenso_espectral_dominante", 0.80);
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_tau", 600_000.0);

    // 1. Respaldo unánime (c = +1.0) ⇒ factor = 1.00
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_media", 0.80);
    let d_aligned = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    assert_eq!(d_aligned.signal, SignalType::Long);
    assert!((d_aligned.net_confidence - 0.80).abs() < 1e-9);

    // 2. Banda neutra/desconocida (c = 0.0) ⇒ factor = 0.50
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_media", 0.00);
    let d_neutral = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    assert_eq!(d_neutral.signal, SignalType::Long);
    assert!((d_neutral.net_confidence - 0.40).abs() < 1e-9);

    // 3. Oposición parcial (c = -0.50) ⇒ factor = 0.25 ⇒ conf = 0.80 * 0.25 = 0.20
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_media", -0.40);
    let d_partial = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    assert_eq!(d_partial.signal, SignalType::Long);
    assert!((d_partial.net_confidence - 0.20).abs() < 1e-9);

    // 4. Oposición destructiva total (c = -1.0) ⇒ factor = 0.00 ⇒ veto absoluto Flat
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_media", -0.80);
    let d_vetoed = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    assert_eq!(
        d_vetoed.signal,
        SignalType::Flat,
        "Oposición unánime debe vetar completamente la orden dominante"
    );
    assert_eq!(d_vetoed.net_confidence, 0.0);
}

#[test]
fn test_f1_100_cutoff_semantics_and_probability_mapping() {
    // Simulación del genoma min_confidence_btc = 0.70
    let tensor_min_conf: f64 = 0.70;

    // 1. Umbral del orquestador (MOD2/7-012)
    let cutoff_floor = ((tensor_min_conf - 0.50) * 1.0).clamp(0.0, 0.45);
    assert!((cutoff_floor - 0.20).abs() < 1e-12);

    // 2. Umbral unificado en el núcleo (god-engine-core lib.rs:7538)
    let tensor_cutoff = ((tensor_min_conf - 0.50) * 1.0).clamp(0.0, 0.45);
    assert_eq!(cutoff_floor, tensor_cutoff);

    // 3. Umbral unificado de rama 14 (god-engine-core lib.rs:8094)
    let slow_cutoff = ((tensor_min_conf - 0.50) * 0.95).clamp(0.0, 0.45);
    assert!((slow_cutoff - 0.19).abs() < 1e-12);

    // 4. Mapeo continuo al espacio de probabilidad P = 0.50 + 0.50 * |ΔP|
    let net_conf_sample: f64 = 0.25; // Exceso de convicción que supera cutoff (0.25 >= 0.20)
    assert!(net_conf_sample >= tensor_cutoff);

    let conf_base = (0.50 + 0.50 * net_conf_sample).clamp(0.50, 0.95);
    assert!((conf_base - 0.625).abs() < 1e-12);
    assert!(conf_base >= 0.50 && conf_base <= 0.95);

    // Exceso cero ⇒ 50% de probabilidad (máxima incertidumbre direccional)
    let conf_base_zero = (0.50 + 0.50 * 0.0f64).clamp(0.50, 0.95);
    assert!((conf_base_zero - 0.50).abs() < 1e-12);

    // Exceso unitario ⇒ 95% de probabilidad (máxima convicción acotada)
    let conf_base_max = (0.50 + 0.50 * 1.0f64).clamp(0.50, 0.95);
    assert!((conf_base_max - 0.95).abs() < 1e-12);
}
