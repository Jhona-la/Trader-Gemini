//! P2: unknown systemic/spectral pressure cannot authorize new collateral.
//! These contracts use only an in-memory arena, without models or trading.
use quantum_arena::GlobalArena;
use risk_engine::{orchestrator::PortfolioOrchestrator, regime::MarketRegime};
use std::sync::{atomic::Ordering, Arc};

fn arena_with_known_budget(capital: f64) -> Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(capital);
    arena
        .config
        .global_max_drawdown
        .store(0.1, Ordering::Relaxed);
    arena
        .config
        .margin_cushion_pct
        .store(0.90, Ordering::Relaxed);
    arena
}

#[test]
fn nonfinite_systemic_crash_rejects_both_sides() {
    let arena = arena_with_known_budget(100.0);
    let guard = PortfolioOrchestrator::new(&arena);
    let mut admitted = Vec::new();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        arena.regime_p_crash.store(bad, Ordering::Relaxed);
        for is_long in [true, false] {
            if guard.allow_trade(is_long, 1.0, MarketRegime::Range, 5.0) {
                admitted.push((bad, is_long));
            }
        }
    }
    assert!(admitted.is_empty(), "invalid crash admitted: {admitted:?}");
    arena.regime_p_crash.store(0.0, Ordering::Relaxed);
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));
    assert!(guard.allow_trade(false, 90.0, MarketRegime::Range, 5.0));
}

#[test]
fn nonfinite_systemic_bull_rejects_both_sides() {
    let arena = arena_with_known_budget(100.0);
    let guard = PortfolioOrchestrator::new(&arena);
    let mut admitted = Vec::new();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        arena.regime_p_bull.store(bad, Ordering::Relaxed);
        for is_long in [true, false] {
            if guard.allow_trade(is_long, 1.0, MarketRegime::Range, 5.0) {
                admitted.push((bad, is_long));
            }
        }
    }
    assert!(admitted.is_empty(), "invalid bull admitted: {admitted:?}");
    arena.regime_p_bull.store(0.0, Ordering::Relaxed);
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));
    assert!(guard.allow_trade(false, 90.0, MarketRegime::Range, 5.0));
}

#[test]
fn nonfinite_systemic_chaos_rejects_both_sides() {
    let arena = arena_with_known_budget(100.0);
    let guard = PortfolioOrchestrator::new(&arena);
    let mut admitted = Vec::new();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        arena.regime_p_chaos.store(bad, Ordering::Relaxed);
        for is_long in [true, false] {
            if guard.allow_trade(is_long, 1.0, MarketRegime::Range, 5.0) {
                admitted.push((bad, is_long));
            }
        }
    }
    assert!(admitted.is_empty(), "invalid chaos admitted: {admitted:?}");
    arena.regime_p_chaos.store(0.0, Ordering::Relaxed);
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));
    assert!(guard.allow_trade(false, 90.0, MarketRegime::Range, 5.0));
}

#[test]
fn nonfinite_coherence_cannot_hide_behind_direction_or_zero_flux() {
    let arena = arena_with_known_budget(100.0);
    let guard = PortfolioOrchestrator::new(&arena);
    let mut admitted = Vec::new();
    // A different coin, with no open position, must still be checked.
    let coin = &arena.coins[1];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        coin.spectral_coherence.store(bad, Ordering::Relaxed);
        for flux in [0.0, 0.6] {
            coin.spectral_crash_flux.store(flux, Ordering::Relaxed);
            for is_long in [true, false] {
                if guard.allow_trade(is_long, 1.0, MarketRegime::Range, 5.0) {
                    admitted.push((bad, flux, is_long));
                }
            }
        }
    }
    assert!(
        admitted.is_empty(),
        "invalid coherence admitted: {admitted:?}"
    );
    coin.spectral_coherence.store(0.0, Ordering::Relaxed);
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));
    assert!(guard.allow_trade(false, 90.0, MarketRegime::Range, 5.0));
}

#[test]
fn nonfinite_flux_rejects_adverse_favorable_and_neutral_tides() {
    let arena = arena_with_known_budget(100.0);
    let guard = PortfolioOrchestrator::new(&arena);
    let mut admitted = Vec::new();
    let coin = &arena.coins[1];
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        coin.spectral_crash_flux.store(bad, Ordering::Relaxed);
        for tide in [-0.7, -0.0, 0.0, 0.7] {
            coin.spectral_coherence.store(tide, Ordering::Relaxed);
            for is_long in [true, false] {
                if guard.allow_trade(is_long, 1.0, MarketRegime::Range, 5.0) {
                    admitted.push((bad, tide, is_long));
                }
            }
        }
    }
    assert!(admitted.is_empty(), "invalid flux admitted: {admitted:?}");
    coin.spectral_crash_flux.store(0.0, Ordering::Relaxed);
    assert!(guard.allow_trade(true, 90.0, MarketRegime::Range, 5.0));
    assert!(guard.allow_trade(false, 90.0, MarketRegime::Range, 5.0));
}

#[test]
fn valid_cold_state_preserves_standard_and_micro_margin_boundaries() {
    for (capital, limit) in [(100.0, 90.0), (13.0, 13.0 * 0.98)] {
        let arena = arena_with_known_budget(capital);
        let guard = PortfolioOrchestrator::new(&arena);
        // Default systemic probabilities and spectral fields are valid cold data.
        for is_long in [true, false] {
            assert!(guard.allow_trade(is_long, limit, MarketRegime::Range, 5.0));
            assert!(!guard.allow_trade(is_long, limit + 0.01, MarketRegime::Range, 5.0));
        }
    }
}

#[test]
fn finite_pressure_preserves_direction_maximum_and_existing_clamps() {
    let arena = arena_with_known_budget(100.0);
    let guard = PortfolioOrchestrator::new(&arena);
    let coin = &arena.coins[1];
    // Explicit dollar boundaries: adverse spectral/systemic pressure combines
    // by maximum, not sum. Finite values outside [0, 1] retain their clamps.
    for (is_long, tide, flux, crash, bull, limit) in [
        (true, -0.7, 0.6, 0.4, 0.0, 75.0),
        (false, 0.7, 0.6, 0.0, 0.4, 75.0),
        (true, 0.7, 1.0, 0.0, 0.0, 90.0),
        (false, -0.7, 1.0, 0.0, 0.0, 90.0),
        (true, -0.7, -2.0, 0.0, 0.0, 90.0),
        (false, 0.7, -2.0, 0.0, 0.0, 90.0),
        (true, -0.7, 2.0, 0.0, 0.0, 65.0),
        (false, 0.7, 2.0, 0.0, 0.0, 65.0),
        (true, 0.0, 0.0, -2.0, 0.0, 90.0),
        (false, 0.0, 0.0, 0.0, -2.0, 90.0),
        (false, 0.0, 0.0, 0.0, 2.0, 65.0),
        (true, 0.0, 0.0, 0.4, 0.0, 80.0),
        (false, 0.0, 0.0, 0.0, 0.6, 75.0),
    ] {
        coin.spectral_coherence.store(tide, Ordering::Relaxed);
        coin.spectral_crash_flux.store(flux, Ordering::Relaxed);
        arena.regime_p_crash.store(crash, Ordering::Relaxed);
        arena.regime_p_bull.store(bull, Ordering::Relaxed);
        assert!(guard.allow_trade(is_long, limit, MarketRegime::Range, 5.0));
        assert!(!guard.allow_trade(is_long, limit + 0.01, MarketRegime::Range, 5.0));
    }
}

#[test]
fn finite_crash_threshold_and_legacy_regime_veto_are_preserved() {
    let arena = arena_with_known_budget(100.0);
    let guard = PortfolioOrchestrator::new(&arena);
    for (crash, allows_long) in [(0.899, true), (0.90, false), (0.92, false), (2.0, false)] {
        arena.regime_p_crash.store(crash, Ordering::Relaxed);
        assert_eq!(
            guard.allow_trade(true, 1.0, MarketRegime::Range, 5.0),
            allows_long
        );
        assert!(guard.allow_trade(false, 90.0, MarketRegime::Range, 5.0));
    }
    arena.regime_p_crash.store(0.0, Ordering::Relaxed);
    assert!(!guard.allow_trade(true, 1.0, MarketRegime::Crash, 5.0));
    assert!(guard.allow_trade(false, 90.0, MarketRegime::Crash, 5.0));
    assert!(guard.allow_trade(false, 90.0, MarketRegime::BullRun, 5.0));
}
