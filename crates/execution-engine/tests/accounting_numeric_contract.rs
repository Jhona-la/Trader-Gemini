//! Isolated arithmetic/source contracts; never start a journal worker or account.
use execution_engine::trade_accounting::{
    checked_gross_pnl, entry_fill_journal_line, gross_pnl, AccountingError as E, BracketClose,
};

fn close() -> BracketClose {
    BracketClose {
        ts_ms: 123,
        symbol: "TESTUSDT".into(),
        was_long: true,
        qty: 2.0,
        entry_price: 100.0,
        exit_price: 110.0,
        stop_price: 109.0,
        pnl_gross: 20.0,
        fees: 0.5,
        trigger: "TP",
        slippage_bps: -0.1,
    }
}

#[test]
fn legacy_pnl_does_not_export_infinite_entry() {
    assert_eq!(gross_pnl(true, f64::INFINITY, 110.0, 1.0), 0.0);
}

#[test]
fn bracket_order_types_are_exact_protocol_tokens() {
    use execution_engine::trade_accounting::is_closing_bracket_order as closing;
    for kind in ["STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"] {
        assert!(closing(kind));
    }
    for kind in [
        "NOT_STOP_MARKET",
        "TAKE_PROFIT_MARKET_FAKE",
        "STOP_MARKET ",
        "",
    ] {
        assert!(!closing(kind), "not a canonical type: {kind}");
    }
}

#[test]
fn invalid_records_are_counted_without_starting_journal_io_or_queueing() {
    use execution_engine::trade_accounting::{
        drain_bracket_closes, record_bracket_close, record_entry_fill, INVALID_RECORDS,
    };
    use std::sync::atomic::Ordering;
    let before = INVALID_RECORDS.load(Ordering::Relaxed);
    let mut c = close();
    c.qty = 0.0;
    record_bracket_close(c);
    record_entry_fill(0, "TEST", true, f64::NAN, 1.0, false, 0.0);
    assert_eq!(INVALID_RECORDS.load(Ordering::Relaxed) - before, 2);
    assert!(drain_bracket_closes().is_empty());
}
#[test]
fn legacy_pnl_does_not_export_infinite_exit() {
    assert_eq!(gross_pnl(false, 100.0, f64::INFINITY, 1.0), 0.0);
}
#[test]
fn legacy_pnl_does_not_export_infinite_quantity() {
    assert_eq!(gross_pnl(true, 100.0, 110.0, f64::INFINITY), 0.0);
}
#[test]
fn legacy_pnl_does_not_export_overflow_from_finite_operands() {
    assert_eq!(gross_pnl(true, 1.0, f64::MAX, 2.0), 0.0);
}
#[test]
fn host_checks_numeric_close_evidence_before_learning() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    let body = host
        .split("for bc in execution_engine::trade_accounting::drain_bracket_closes() {")
        .nth(1)
        .unwrap()
        .split("if !is_trading_allowed {")
        .next()
        .unwrap();
    assert!(body.contains("bc.checked_numeric_net_pnl()"));
}

#[test]
fn checked_arithmetic_reports_field_error_and_keeps_breakeven_valid() {
    assert_eq!(
        checked_gross_pnl(true, 0.0, 2.0, 1.0),
        Err(E::InvalidEntryPrice)
    );
    assert_eq!(
        checked_gross_pnl(true, 1.0, f64::NAN, 1.0),
        Err(E::InvalidExitPrice)
    );
    assert_eq!(
        checked_gross_pnl(false, 1.0, 2.0, -1.0),
        Err(E::InvalidQuantity)
    );
    assert_eq!(
        checked_gross_pnl(true, 1.0, f64::MAX, 2.0),
        Err(E::NonFiniteCalculation)
    );
    assert_eq!(checked_gross_pnl(true, 100.0, 100.0, 2.0), Ok(0.0));
}

#[test]
fn underflow_is_not_a_breakeven_outcome() {
    assert_eq!(
        checked_gross_pnl(true, 1e-200, 2e-200, 1e-200),
        Err(E::Underflow)
    );
}

#[test]
fn net_admission_rejects_invalid_fields_individually() {
    for i in 0..8 {
        let mut c = close();
        match i {
            0 => c.symbol.clear(),
            1 => c.qty = 0.0,
            2 => c.entry_price = f64::INFINITY,
            3 => c.exit_price = f64::NAN,
            4 => c.stop_price = -1.0,
            5 => c.pnl_gross = f64::NAN,
            6 => c.fees = f64::INFINITY,
            _ => c.slippage_bps = f64::NAN,
        }
        assert!(c.checked_numeric_net_pnl().is_err(), "case {i}");
        assert!(c.journal_line().is_err(), "case {i}");
    }
}

#[test]
fn zero_and_signed_fees_keep_their_arithmetic_meaning() {
    let mut c = close();
    assert_eq!(c.checked_numeric_net_pnl(), Ok(19.5));
    c.exit_price = c.entry_price;
    c.pnl_gross = 0.0;
    c.fees = 0.0;
    assert_eq!(c.checked_numeric_net_pnl(), Ok(0.0));
    c.fees = -0.5;
    assert_eq!(c.checked_numeric_net_pnl(), Ok(0.5));
}

#[test]
fn unknown_entry_can_be_journalled_but_cannot_be_used_as_numeric_outcome() {
    let mut c = close();
    c.entry_price = 0.0;
    c.pnl_gross = 0.0;
    assert!(c.journal_line().is_ok());
    assert_eq!(c.checked_numeric_net_pnl(), Err(E::InvalidEntryPrice));
}

#[test]
fn finite_operands_cannot_overflow_net_or_gross_learning_arithmetic() {
    let mut c = close();
    c.pnl_gross = f64::MAX;
    c.fees = -f64::MAX;
    assert_eq!(c.checked_numeric_net_pnl(), Err(E::NonFiniteCalculation));
    assert!(c.journal_line().is_err());
    c = close();
    c.exit_price = f64::MAX;
    c.pnl_gross = 0.0;
    assert_eq!(c.checked_numeric_net_pnl(), Err(E::NonFiniteCalculation));
}

#[test]
fn close_json_escapes_text_and_preserves_sub_micro_values() {
    let mut c = close();
    c.symbol = "TEST\"\\\nUSDT".into();
    c.trigger = "T\"P\n";
    c.qty = 1e-10;
    c.entry_price = 1e-8;
    c.exit_price = 2e-8;
    c.pnl_gross = 1e-18;
    c.fees = 2e-20;
    let line = c.journal_line().unwrap();
    assert_eq!(line.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(value["sym"], c.symbol);
    assert_eq!(value["trig"], c.trigger);
    for key in ["qty", "entry", "exit", "pnl_gross", "fees", "net"] {
        assert!(value[key].as_f64().unwrap() > 0.0, "{key} rounded to zero");
    }
}

#[test]
fn entry_json_validates_before_serialization_and_preserves_precision() {
    let line = entry_fill_journal_line(0, "T\"USDT", true, 1e-10, 1e-8, false, -1e-12).unwrap();
    let value: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(value["sym"], "T\"USDT");
    assert_eq!(value["qty"].as_f64().unwrap(), 1e-10);
    assert_eq!(value["fee"].as_f64().unwrap(), -1e-12);
    assert!(entry_fill_journal_line(0, "T", true, 0.0, 1.0, false, 0.0).is_err());
    assert!(entry_fill_journal_line(0, "T", true, 1.0, f64::NAN, false, 0.0).is_err());
    assert!(entry_fill_journal_line(0, "T", true, 1.0, 1.0, false, f64::NAN).is_err());
}

#[test]
fn numeric_net_scales_with_money_units_and_is_stateless() {
    let c = close();
    let mut scaled = c.clone();
    scaled.entry_price *= 100.0;
    scaled.exit_price *= 100.0;
    scaled.stop_price *= 100.0;
    scaled.pnl_gross *= 100.0;
    scaled.fees *= 100.0;
    assert_eq!(
        scaled.checked_numeric_net_pnl().unwrap(),
        100.0 * c.checked_numeric_net_pnl().unwrap()
    );
    assert_eq!(c.checked_numeric_net_pnl(), c.checked_numeric_net_pnl());
}
