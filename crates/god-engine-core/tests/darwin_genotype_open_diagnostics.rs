use god_engine_core::darwin::Genotype;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

#[test]
fn open_public_genotype_application_accepts_finite_out_of_domain_values() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut g = Genotype::current_from_arena(&arena);
    g.global_leverage = 2000.0;
    g.min_confidence = 5.0;
    g.capital_split_scalp = -3.0;
    g.apply_to_arena(&arena);
    assert_eq!(arena.config.global_leverage.load(Ordering::Relaxed), 2000.0);
    assert_eq!(arena.config.min_confidence_btc.load(Ordering::Relaxed), 5.0);
    assert_eq!(
        arena.config.capital_split_scalp.load(Ordering::Relaxed),
        -3.0
    );
}

#[test]
fn genotype_preserves_risk_policy_owned_by_the_shared_replay_context() {
    // XXXV: policy is deliberately outside these genes. The common Darwin
    // replay now supplies the same captured drawdown to candidate and baseline.
    let a = GlobalArena::build_in_own_stack(100.0);
    let b = GlobalArena::build_in_own_stack(100.0);
    a.config.global_max_drawdown.store(0.2, Ordering::Relaxed);
    b.config.global_max_drawdown.store(0.95, Ordering::Relaxed);
    let g = Genotype::current_from_arena(&a);
    g.apply_to_arena(&b);
    assert_eq!(a.config.global_max_drawdown.load(Ordering::Relaxed), 0.2);
    assert_eq!(b.config.global_max_drawdown.load(Ordering::Relaxed), 0.95);
}
