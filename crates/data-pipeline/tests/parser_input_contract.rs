// Same production module, also runnable in an isolated dependency-light harness.
#[path = "../src/parser.rs"]
mod parser;
use parser::{AggTradeEvent, BookTickerEvent, DepthEvent};

#[test]
fn ra_uint_overflow_is_rejected_without_panicking() {
    assert!(
        BookTickerEvent::extract_u64_from(b"{\"E\":18446744073709551616}", 0, b"\"E\":").is_none()
    );
}

#[test]
fn ra_uint_requires_digits() {
    for value in ["", "null", "true", "-1", "\"123\""] {
        let wire = format!("{{\"E\":{value}}}");
        assert!(
            BookTickerEvent::extract_u64_from(wire.as_bytes(), 0, b"\"E\":").is_none(),
            "{wire}"
        );
    }
}

#[test]
fn ra_uint_rejects_numeric_prefixes_and_leading_zeroes() {
    for value in ["12x", "12.5", "12e3", "12false", "01"] {
        let wire = format!("{{\"E\":{value}}}");
        assert!(
            BookTickerEvent::extract_u64_from(wire.as_bytes(), 0, b"\"E\":").is_none(),
            "{wire}"
        );
    }
}

#[test]
fn ra_public_extractors_reject_out_of_range_offsets() {
    assert!(BookTickerEvent::extract_u64_from(b"{}", usize::MAX, b"\"E\":").is_none());
    assert!(BookTickerEvent::extract_f64_from(b"{}", usize::MAX, b"\"b\":\"").is_none());
}

#[test]
fn ra_uint_accepts_json_whitespace_and_exact_maximum() {
    let wire = b"{\"E\": \t\r\n18446744073709551615 \t}";
    assert_eq!(
        BookTickerEvent::extract_u64_from(wire, 0, b"\"E\":")
            .unwrap()
            .0,
        u64::MAX
    );
    assert_eq!(
        BookTickerEvent::extract_u64_from(b"{\"E\":0}", 0, b"\"E\":")
            .unwrap()
            .0,
        0
    );
}

#[test]
fn ra_bad_present_event_time_is_not_absence_or_transaction_fallback() {
    for value in ["null", "-2", "123x", "1.25", "18446744073709551616"] {
        let wire = format!(
            "{{\"b\":\"10\",\"B\":\"1\",\"a\":\"11\",\"A\":\"1\",\"E\":{value},\"T\":456}}"
        );
        assert!(
            BookTickerEvent::parse_from_json(wire.as_bytes()).is_none(),
            "{wire}"
        );
    }
}

#[test]
fn ra_transaction_time_fallback_and_missing_time_contract_are_preserved() {
    let with_t = br#"{"b":"10","B":"1","a":"11","A":"1","T":456}"#;
    assert_eq!(
        BookTickerEvent::parse_from_json(with_t).unwrap().event_time,
        456
    );
    let absent = br#"{"b":"10","B":"1","a":"11","A":"1"}"#;
    assert_eq!(
        BookTickerEvent::parse_from_json(absent).unwrap().event_time,
        0
    );
}

#[test]
fn ra_aggressor_requires_a_complete_boolean_token() {
    for value in [
        "null",
        "0",
        "1",
        "\"true\"",
        "truthy",
        "truex",
        "falsehood",
        "",
    ] {
        let wire = format!("{{\"p\":\"10\",\"q\":\"1\",\"m\":{value}}}");
        assert!(
            AggTradeEvent::parse_from_json(wire.as_bytes()).is_none(),
            "{wire}"
        );
    }
}

#[test]
fn ra_aggressor_accepts_all_json_whitespace() {
    for (value, expected) in [("true", true), ("false", false)] {
        let wire = format!("{{\"p\":\"10\",\"q\":\"1\",\"m\":\r\n\t {value} \r\n}}");
        assert_eq!(
            AggTradeEvent::parse_from_json(wire.as_bytes())
                .unwrap()
                .is_buyer_maker,
            expected
        );
    }
}

#[test]
fn ra_depth_mixed_spacing_preserves_both_sides() {
    for wire in [
        br#"{"bids":[["10","2"]],"asks" : [["11","3"]]}"#.as_slice(),
        br#"{"bids" : [["10","2"]],"asks":[["11","3"]]}"#.as_slice(),
    ] {
        let event = DepthEvent::parse_from_json(wire).unwrap();
        assert_eq!((event.bid_wall, event.ask_wall), (2.0, 3.0));
    }
}

#[test]
fn ra_depth_fallback_does_not_validate_discarded_fast_sums() {
    for wire in [
        br#"{"bids":[["10","1e308"] ],"asks" : [["11","1e308"]]}"#.as_slice(),
        br#"{"asks":[["11","1e308"] ],"bids" : [["10","1e308"]]}"#.as_slice(),
    ] {
        let event = DepthEvent::parse_from_json(wire).unwrap();
        assert_eq!((event.bid_wall, event.ask_wall), (1e308, 1e308));
    }
}

#[test]
fn ra_depth_rejects_nonfinite_aggregate_from_finite_levels() {
    for wire in [
        br#"{"bids":[["10","1e308"],["9","1e308"]],"asks":[["11","1"]]}"#.as_slice(),
        br#"{"bids" : [["10","1e308"],["9","1e308"]],"asks" : [["11","1"]]}"#.as_slice(),
    ] {
        assert!(DepthEvent::parse_from_json(wire).is_none());
    }
}
