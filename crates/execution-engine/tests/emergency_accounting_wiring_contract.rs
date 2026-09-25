//! Pure arithmetic + static host wiring. No journal, queue, account or network.
use execution_engine::trade_accounting::gross_pnl;

#[test]
fn emergency_host_delegates_to_shared_signed_pnl() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    let body = host
        .split("fn record_emergency_close(")
        .nth(1)
        .unwrap()
        .split("fn compact_position_journal(")
        .next()
        .unwrap();
    assert!(
        body.contains("execution_engine::trade_accounting::checked_gross_pnl("),
        "emergency accounting must not duplicate the reversed-sign formula"
    );
}

#[test]
fn shared_pnl_obeys_long_short_price_direction_and_quantity_units() {
    for (long, entry, exit, expected) in [
        (true, 100.0, 110.0, 20.0),
        (false, 100.0, 110.0, -20.0),
        (true, 100.0, 90.0, -20.0),
        (false, 100.0, 90.0, 20.0),
    ] {
        assert_eq!(gross_pnl(long, entry, exit, 2.0), expected);
        assert_eq!(gross_pnl(long, entry * 10.0, exit * 10.0, 0.2), expected);
    }
}
