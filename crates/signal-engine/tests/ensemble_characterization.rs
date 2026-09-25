//! Open-defect diagnostics, NOT regression tests proving a repair.
use omniscient_registry::OmniscientRegistry;
use signal_engine::{
    micro_scalp_trigger::MicroScalpTriggerEngine, orchestrator::TensorVoteOrchestrator,
    trend_runner::HighPayoffTrendRunner,
};
use std::sync::{atomic::Ordering, Arc};
use strategy_core::{momentum_booster::VolatileMomentumBooster, QuantumStrategy};

struct Constant(f64);
impl QuantumStrategy for Constant {
    fn name(&self) -> &str {
        "constant"
    }
    fn init(&mut self, _: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        self.0
    }
}

fn conviction(value: f64, copies: usize) -> f64 {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut engine = TensorVoteOrchestrator::new(arena);
    for _ in 0..copies {
        engine.add_strategy(Box::new(Constant(value)));
    }
    engine.evaluate_continuous_consensus().net_confidence
}

#[test]
fn diagnostic_tiny_positive_scores_still_have_point_seven_floor() {
    assert!(conviction(1e-12, 1) >= 0.7);
    assert_eq!(conviction(0.0, 1), 0.0);
}

#[test]
fn diagnostic_clones_change_conviction_without_new_evidence() {
    assert!(conviction(0.2, 5) > conviction(0.2, 1));
}

#[test]
fn diagnostic_missing_model_base_changes_directional_gate() {
    let registry = Arc::new(OmniscientRegistry::new());
    registry.set("hawkes_intensity", 2.0);
    registry.set("order_book_imbalance", -0.5);
    registry.set("ml_prob_motor", 0.36);
    let mut trigger = MicroScalpTriggerEngine::new();
    trigger.init(registry.clone()).unwrap();
    assert!(trigger.evaluate() < 0.0); // default baseline 0.5 fabricates a negative lift
    registry.set("ml_model_base", 0.3);
    assert_eq!(trigger.evaluate(), 0.0);
}

#[test]
fn diagnostic_trend_extension_can_shrink_the_base_target() {
    let target = HighPayoffTrendRunner::calculate_expanded_tp(0.02, 0.5, 0.0, 0.01, 0.1);
    assert!(target < 0.02);
    assert!((target - 0.1 * 0.2_f64.tanh()).abs() < 1e-15);
}

#[test]
fn diagnostic_negative_pnl_does_not_disable_momentum_extension() {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .dynamic_ofi_threshold
        .store(0.5, Ordering::Relaxed);
    arena.config.tensor_poly_a.store(0.1, Ordering::Relaxed);
    arena.config.tensor_poly_b.store(0.1, Ordering::Relaxed);
    arena
        .config
        .explosive_leverage_multiplier
        .store(2.0, Ordering::Relaxed);
    assert!(VolatileMomentumBooster::calculate_tp_extension(1.0, -0.01, 3.0, 0.01, &arena) > 1.0);
}
