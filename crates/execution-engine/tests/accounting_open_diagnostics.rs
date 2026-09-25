//! OPEN: numeric validation cannot stand in for accounting identity or consistency.
use execution_engine::trade_accounting::BracketClose;

#[test]
fn open_numeric_admission_does_not_attest_pnl_consistency_or_exactly_once() {
    let c = BracketClose {
        ts_ms: 1,
        symbol: "TESTUSDT".into(),
        was_long: true,
        qty: 1.0,
        entry_price: 100.0,
        exit_price: 110.0,
        stop_price: 0.0,
        pnl_gross: -10.0,
        fees: 0.0,
        trigger: "X-009",
        slippage_bps: 0.0,
    };
    // Prices imply +10, supplied gross is -10; no fill ID is present.
    assert_eq!(c.checked_numeric_net_pnl(), Ok(-10.0));
    assert_eq!(c.checked_numeric_net_pnl(), Ok(-10.0));
}
