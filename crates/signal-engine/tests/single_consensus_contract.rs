use omniscient_registry::OmniscientRegistry;
use signal_engine::{orchestrator::TensorVoteOrchestrator, SignalType};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use strategy_core::QuantumStrategy;

struct CountingStrategy {
    calls: Arc<AtomicUsize>,
    fail_init: bool,
}
impl QuantumStrategy for CountingStrategy {
    fn name(&self) -> &str {
        "CountingStrategy"
    }
    fn init(&mut self, _: Arc<OmniscientRegistry>) -> Result<(), String> {
        if self.fail_init {
            Err("intentional init failure".into())
        } else {
            Ok(())
        }
    }
    fn evaluate(&self) -> f64 {
        if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
            0.9
        } else {
            -0.9
        }
    }
}
fn engine(fail_init: bool) -> (TensorVoteOrchestrator, Arc<AtomicUsize>) {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = TensorVoteOrchestrator::new(arena);
    engine.add_strategy(Box::new(CountingStrategy {
        calls: calls.clone(),
        fail_init,
    }));
    (engine, calls)
}

#[test]
fn global_compatibility_path_evaluates_one_universal_snapshot() {
    let (engine, calls) = engine(false);
    let result = engine.evaluate_consensus_for_coin(0, "BTCUSDT");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(result.signal, SignalType::Long);
}

#[test]
fn dual_compatibility_views_are_the_same_single_evaluation() {
    let (engine, calls) = engine(false);
    let (a, b) = engine.evaluate_dual_consensus_for_coin(0, "BTCUSDT");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(a.signal, b.signal);
    assert_eq!(a.net_confidence, b.net_confidence);
    assert_eq!(a.expected_lifetime_ms, b.expected_lifetime_ms);
}

#[test]
fn failed_initialization_never_registers_a_voting_strategy() {
    let (engine, calls) = engine(true);
    assert_eq!(
        engine.evaluate_continuous_consensus().signal,
        SignalType::Flat
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn fallible_registration_exposes_initialization_failure() {
    let (mut engine, calls) = engine(false);
    let error = engine.try_add_strategy(Box::new(CountingStrategy {
        calls: calls.clone(),
        fail_init: true,
    }));
    assert!(error.is_err());
    assert_eq!(
        engine.evaluate_continuous_consensus().signal,
        SignalType::Long
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}
