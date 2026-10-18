//! Contract test for F1-020 (Ola Ω153):
//!
//! Evaluates that `live_envelope_gate_for_slot` gates individual resonant slots
//! without being short-circuited by `prev_open == true`.
//! When slot 0 is already open and a secondary order opens in slot 1:
//! - If the envelope vetoes slot 1, ONLY slot 1 is rolled back.
//! - Slot 0 remains open and untouched.
//! - `veto_counter` is properly incremented.

use std::sync::atomic::Ordering;
use backtest_engine::booktick_replay::live_envelope_gate_for_slot;
use quantum_arena::GlobalArena;
use risk_engine::kelly_envelope::RiskEnvelope;

#[test]
fn test_f1_020_stacked_resonant_slot_gated_and_isolated_rollback() {
    let arena = GlobalArena::build_in_own_stack(13.0); // Small capital: 13 USD
    let mut envelope = RiskEnvelope::new();
    let mut vetoes = 0u64;

    let coin = &arena.coins[0];

    // 1. Manually open slot 0 (representing an existing open resonant position)
    let s0 = coin.positions.get_slot(0);
    s0.entry_price.store(50_000.0, Ordering::Release);
    s0.quantity.store(0.0002, Ordering::Release);
    s0.is_long.store(true, Ordering::Release);
    s0.margin_used.store(2.0, Ordering::Release);
    s0.sl_price.store(49_500.0, Ordering::Release);
    s0.generation.fetch_add(1, Ordering::Release);
    s0.is_open.store(true, Ordering::Release);

    assert!(s0.is_open(), "Slot 0 must be open initially");

    // 2. Open slot 1 with an excessive position that violates risk limits / margin
    let s1 = coin.positions.get_slot(1);
    s1.entry_price.store(50_100.0, Ordering::Release);
    s1.quantity.store(0.005, Ordering::Release); // 250 USD notional on 13 USD account!
    s1.is_long.store(true, Ordering::Release);
    s1.margin_used.store(12.5, Ordering::Release); // Consumes almost all margin
    s1.sl_price.store(45_000.0, Ordering::Release); // Wide SL
    s1.generation.fetch_add(1, Ordering::Release);
    s1.is_open.store(true, Ordering::Release);

    assert!(s1.is_open(), "Slot 1 opened tentatively before envelope check");

    // 3. Evaluate live_envelope_gate_for_slot on slot 1
    let passed = live_envelope_gate_for_slot(
        &arena,
        &mut envelope,
        0,
        1,
        50_100.0,
        0.01,
        &mut vetoes,
    );

    // 4. Assertions:
    // Slot 1 must be vetoed because notional is excessive and envelope has no evidence
    assert!(!passed, "Excessive stacked slot 1 position must be vetoed by envelope/margin-guard");
    assert_eq!(vetoes, 1, "Veto counter must increment on slot 1 rejection");

    // Slot 1 must be rolled back
    assert!(!coin.positions.get_slot(1).is_open(), "Slot 1 must be rolled back");

    // Slot 0 must REMAIN OPEN and undisturbed
    assert!(coin.positions.get_slot(0).is_open(), "Slot 0 must remain open despite slot 1 rollback");
}
