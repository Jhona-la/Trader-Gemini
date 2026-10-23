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

#[test]
fn ra_aggressor_rejects_unreadable_price_or_quantity() {
    // F1-092 — la ruta VIVA de ingesta (`AggTradeEvent::parse_from_json`, la que
    // consume `ws_client.rs`) debe ser fail-closed también en precio y
    // cantidad. El contrato cubría el token `m` pero no `p`/`q`: un campo
    // ilegible podría haber entrado como cero. `extract_f64_from` usa
    // `fast_float::parse(...).ok()?` (estricto) y el gate `price <= 0` remata
    // ceros y negativos, así que el tick se descarta entero.
    for price in ["", "abc", "NaN", "inf", "0", "-1", "1e400"] {
        let wire = format!("{{\"p\":\"{price}\",\"q\":\"1\",\"m\":true}}");
        assert!(
            AggTradeEvent::parse_from_json(wire.as_bytes()).is_none(),
            "precio ilegible {price:?} debe descartar el tick: {wire}"
        );
    }
    for qty in ["", "abc", "NaN", "0", "-1"] {
        let wire = format!("{{\"p\":\"10\",\"q\":\"{qty}\",\"m\":false}}");
        assert!(
            AggTradeEvent::parse_from_json(wire.as_bytes()).is_none(),
            "cantidad ilegible {qty:?} debe descartar el tick: {wire}"
        );
    }
    // Y el tick sano sigue pasando con las cifras exactas.
    let wire = br#"{"p":"60050.25","q":"1.500","m":true}"#;
    let event = AggTradeEvent::parse_from_json(wire).expect("tick sano");
    assert_eq!(event.price, 60050.25);
    assert_eq!(event.qty, 1.500);
}

#[test]
fn qra_aggressor_present_corrupt_clock_is_not_missing_clock() {
    for clock in [
        "null",
        "true",
        "\"123\"",
        "-1",
        "+1",
        "1.5",
        "1e3",
        "123x",
        "01",
        "",
        "18446744073709551616",
        "[]",
        "{}",
    ] {
        for key in ["\"T\":", "\"T\" \r\n\t : ", "\"note\":\"T\",\"T\":"] {
            let wire = format!("{{\"p\":\"10\",\"q\":\"1\",\"m\":false,{key}{clock}}}");
            assert!(
                AggTradeEvent::parse_from_json(wire.as_bytes()).is_none(),
                "a corrupt exchange clock must reject the whole event: {wire}"
            );
        }
    }
}

#[test]
fn qra_aggressor_clock_preserves_value_permutation_and_json_whitespace() {
    for wire in [
        br#"{"T":1720000000123,"p":"10","q":"1","m":true}"#.as_slice(),
        b"{\"p\":\"10\",\"q\":\"1\",\"m\":true,\"T\" \r\n\t : \r\n 1720000000123 \t}",
        br#"{"p":"10","q":"1","m":true,"note":"T","T":1720000000123}"#.as_slice(),
    ] {
        let event = AggTradeEvent::parse_from_json(wire).expect("valid exchange clock");
        assert_eq!(event.timestamp_ms, 1720000000123);
        assert_eq!(
            (event.price, event.qty, event.is_buyer_maker),
            (10.0, 1.0, true)
        );
    }
    let max = br#"{"p":"10","q":"1","m":true,"T":18446744073709551615}"#;
    assert_eq!(
        AggTradeEvent::parse_from_json(max).unwrap().timestamp_ms,
        u64::MAX
    );
}

#[test]
fn qra_aggressor_legacy_absence_and_explicit_zero_remain_compatible() {
    for wire in [
        br#"{"p":"10","q":"1","m":false}"#.as_slice(),
        br#"{"p":"10","q":"1","m":false,"T":0}"#.as_slice(),
        br#"{"p":"10","q":"1","m":false,"note":"T"}"#.as_slice(),
    ] {
        assert_eq!(
            AggTradeEvent::parse_from_json(wire).unwrap().timestamp_ms,
            0
        );
    }
}
