//! Consumer regressions for FMT-098. In-memory arena only: no tape, exchange,
//! engine process, or journal. This does not certify FMT-113 quantity sizing.
use backtest_engine::booktick_replay::live_envelope_gate;
use quantum_arena::{GlobalArena, position::PositionHorizon};
use risk_engine::kelly_envelope::RiskEnvelope;
use std::sync::{Arc, atomic::Ordering};

fn open_candidate(arena: &Arc<GlobalArena>) {
    arena.coins[0].positions.position.open_with_fee(
        true,
        100.0,
        0.05,
        1.0,
        1_700_000_000_000,
        101.0,
        99.0,
        PositionHorizon::Continuous,
        0.6,
        0.5,
        0.01,
    );
    arena.used_margin.fetch_add(1.0, Ordering::Relaxed);
    arena.unified_capital.fetch_add(-0.01, Ordering::Relaxed);
}

#[test]
fn mature_micro_no_edge_is_vetoed_and_candidate_accounting_is_restored() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut envelope = RiskEnvelope::new();
    for i in 0..500 {
        envelope.record_trade(i % 2 == 0, 1.0, -1.0);
    }
    open_candidate(&arena);
    let mut vetoes = 0;
    assert!(!live_envelope_gate(
        &arena,
        &mut envelope,
        0,
        100.0,
        0.001,
        false,
        &mut vetoes
    ));
    assert_eq!(vetoes, 1);
    assert!(!arena.coins[0].positions.position.is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 0.0);
    assert!((arena.unified_capital.load(Ordering::Relaxed) - 13.0).abs() < 1e-12);
}

#[test]
fn positive_evidence_still_allows_a_feasible_micro_candidate() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut envelope = RiskEnvelope::new();
    for i in 0..500 {
        envelope.record_trade(i % 4 != 0, 2.0, -1.0);
    }
    open_candidate(&arena);
    let mut vetoes = 0;
    assert!(live_envelope_gate(
        &arena,
        &mut envelope,
        0,
        100.0,
        0.001,
        false,
        &mut vetoes
    ));
    assert_eq!(vetoes, 0);
    assert!(arena.coins[0].positions.position.is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 1.0);
}

#[test]
fn invalid_posterior_does_not_admit_a_new_candidate() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut envelope = RiskEnvelope::new();
    envelope.posterior.alpha = f64::NAN;
    open_candidate(&arena);
    let mut vetoes = 0;
    assert!(!live_envelope_gate(
        &arena,
        &mut envelope,
        0,
        100.0,
        0.001,
        false,
        &mut vetoes
    ));
    assert_eq!(vetoes, 1);
    assert!(!arena.coins[0].positions.position.is_open());
    assert!((arena.unified_capital.load(Ordering::Relaxed) - 13.0).abs() < 1e-12);
}
