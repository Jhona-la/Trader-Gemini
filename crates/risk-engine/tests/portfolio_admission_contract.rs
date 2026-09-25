use quantum_arena::GlobalArena;
use risk_engine::{orchestrator::PortfolioOrchestrator, regime::MarketRegime};
use std::sync::atomic::Ordering;

#[test]
fn unknown_capital_cannot_authorize_margin() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        arena.unified_capital.store(bad, Ordering::Relaxed);
        assert!(
            !PortfolioOrchestrator::new(&arena).allow_trade(false, 1.0, MarketRegime::Range),
            "capital={bad}"
        );
    }
}

#[test]
fn unknown_drawdown_budget_cannot_authorize_margin() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        arena
            .config
            .global_max_drawdown
            .store(bad, Ordering::Relaxed);
        assert!(
            !PortfolioOrchestrator::new(&arena).allow_trade(true, 1.0, MarketRegime::Range),
            "budget={bad}"
        );
    }
}

#[test]
fn valid_margin_boundary_and_crash_policy_are_preserved() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena
        .config
        .global_max_drawdown
        .store(0.1, Ordering::Relaxed);
    let guard = PortfolioOrchestrator::new(&arena);
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range));
    assert!(!guard.allow_trade(true, 90.01, MarketRegime::Range));
    assert!(!guard.allow_trade(true, 1.0, MarketRegime::Crash));
    assert!(guard.allow_trade(false, 1.0, MarketRegime::Crash));
}

#[test]
fn invalid_open_position_margin_cannot_hide_exposure() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let position = arena.coins[0].positions.slots()[0];
    position.is_open.store(true, Ordering::Release);
    for side in [false, true] {
        position.is_long.store(side, Ordering::Relaxed);
        for bad in [f64::NAN, f64::NEG_INFINITY, -1.0] {
            position.margin_used.store(bad, Ordering::Relaxed);
            assert!(
                !PortfolioOrchestrator::new(&arena).allow_trade(true, 1.0, MarketRegime::Range),
                "margin={bad},side={side}"
            );
        }
    }
}

#[test]
fn finite_margins_are_summed_across_spectral_slots_and_sides() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena
        .config
        .global_max_drawdown
        .store(0.1, Ordering::Relaxed);
    for (i, margin) in [40.0, 30.0].into_iter().enumerate() {
        let position = arena.coins[0].positions.slots()[i];
        position.is_long.store(i == 0, Ordering::Relaxed);
        position.margin_used.store(margin, Ordering::Relaxed);
        position.is_open.store(true, Ordering::Release);
    }
    let guard = PortfolioOrchestrator::new(&arena);
    assert!(guard.allow_trade(true, 20.0, MarketRegime::Range));
    assert!(!guard.allow_trade(false, 20.01, MarketRegime::Range));
}
