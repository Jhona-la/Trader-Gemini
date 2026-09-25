use quantum_arena::protection_health::{is_exchange_rejection, note_rejection, rejections_of};

#[test]
fn unknown_execution_status_is_not_confirmed_rejection() {
    for msg in [
        "AMBIGUOUS: code=-1007 backend timeout",
        "-1006: Execution status unknown",
        "REJECTED code=-1007 msg=Timeout",
        "Binance API Error: {\"code\":-1000,\"msg\":\"Unknown error\"}",
        "AMBIGUOUS: previous code=-2019; current response unavailable",
    ] {
        assert!(!is_exchange_rejection(msg), "{msg}");
    }
}

#[test]
fn incidental_negative_numbers_are_not_exchange_evidence() {
    for msg in [
        "Network Error: request-id-2019; disconnected",
        "Network Error: retry delta -2019 ms",
        "timeout trace=-4164",
        "-2019.5: observed offset",
        "err -2019",
        "trailing -4182.",
        "Binance API Error: {\"msg\":\"previous -2019: error\"}",
        "REJECTED http=502 body=proxy upstream -2019: unknown",
    ] {
        assert!(!is_exchange_rejection(msg), "{msg}");
    }
}

#[test]
fn current_structured_and_legacy_display_rejections_remain_accepted() {
    for msg in [
        "-2019: Insufficient margin",
        "REJECTED code=-4164 msg=Notional too small",
        "code=-2021: Order would immediately trigger",
        "reject: (-1013) Filter failure",
        "Binance API Error: {\"code\":-4164,\"msg\":\"Notional too small\"}",
        "{\"code\":-2019,\"msg\":\"Insufficient margin\"}",
    ] {
        assert!(is_exchange_rejection(msg), "{msg}");
    }
}

#[test]
fn ambiguous_observation_does_not_increment_escalation_evidence() {
    let symbol = "AUDIT_XXV_AMBIGUOUS";
    let before = rejections_of(symbol);
    assert!(!note_rejection(symbol, "AMBIGUOUS: code=-1007 timeout"));
    assert_eq!(rejections_of(symbol), before);
    assert!(note_rejection(
        symbol,
        "REJECTED code=-2019 msg=Insufficient margin"
    ));
    assert_eq!(rejections_of(symbol), before + 1);
}
