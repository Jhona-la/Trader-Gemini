use execution_engine::execution_evidence::parse_commission_rates;

const SYMBOL: &str = "BTCUSDT";

fn body(maker: &str, taker: &str) -> String {
    format!(r#"{{"symbol":"BTCUSDT","makerCommissionRate":{maker},"takerCommissionRate":{taker}}}"#)
}

#[test]
fn official_payload_accepts_the_additional_rpi_field() {
    // Binance User Commission Rate example, consulted 2026-10-22:
    // https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/User-Commission-Rate
    let wire = r#"{
        "symbol": "BTCUSDT",
        "makerCommissionRate": "0.0002",
        "takerCommissionRate": "0.0004",
        "rpiCommissionRate": "0.00005"
    }"#;
    assert_eq!(parse_commission_rates(wire, SYMBOL), Ok((0.0002, 0.0004)));
}

#[test]
fn finite_json_numbers_and_unknown_fields_are_supported() {
    let wire = r#"{
        "symbol":"BTCUSDT",
        "makerCommissionRate":0.0002,
        "takerCommissionRate":0.0004,
        "futureVenueField":{"schema":"unconsumed"}
    }"#;
    assert_eq!(parse_commission_rates(wire, SYMBOL), Ok((0.0002, 0.0004)));
}

#[test]
fn finite_signed_values_are_preserved_without_a_fee_cap() {
    for (maker, taker, expected) in [
        (r#""-0.0001""#, r#""0.0004""#, (-0.0001, 0.0004)),
        (r#""0.0002""#, r#""-0.0004""#, (0.0002, -0.0004)),
        ("-1.5", "2.0", (-1.5, 2.0)),
        (r#""-1e100""#, "1e100", (-1e100, 1e100)),
    ] {
        assert_eq!(
            parse_commission_rates(&body(maker, taker), SYMBOL),
            Ok(expected),
            "maker={maker}, taker={taker}"
        );
    }
}

#[test]
fn zero_is_observed_evidence_and_its_sign_is_preserved() {
    for (maker, taker) in [(r#""0""#, "0"), ("0", r#""0""#)] {
        assert_eq!(
            parse_commission_rates(&body(maker, taker), SYMBOL),
            Ok((0.0, 0.0))
        );
    }
    assert_eq!(
        parse_commission_rates(&body(r#""-0.0""#, "0.0"), SYMBOL)
            .map(|(maker, taker)| (maker.to_bits(), taker.to_bits())),
        Ok(((-0.0_f64).to_bits(), 0.0_f64.to_bits()))
    );
}

#[test]
fn nonfinite_or_overflowing_rates_fail_in_either_field() {
    for raw in [
        r#""NaN""#,
        r#""inf""#,
        r#""-inf""#,
        r#""Infinity""#,
        r#""-Infinity""#,
        r#""1e309""#,
        r#""-1e309""#,
        "1e309",
        "-1e309",
    ] {
        assert!(
            parse_commission_rates(&body(raw, "0.0004"), SYMBOL).is_err(),
            "maker={raw}"
        );
        assert!(
            parse_commission_rates(&body("0.0002", raw), SYMBOL).is_err(),
            "taker={raw}"
        );
    }
}

#[test]
fn each_required_field_must_be_present() {
    for wire in [
        "{}",
        r#"{"makerCommissionRate":"0.0002","takerCommissionRate":"0.0004"}"#,
        r#"{"symbol":"BTCUSDT","takerCommissionRate":"0.0004"}"#,
        r#"{"symbol":"BTCUSDT","makerCommissionRate":"0.0002"}"#,
    ] {
        assert!(parse_commission_rates(wire, SYMBOL).is_err(), "{wire}");
    }
}

#[test]
fn invalid_rate_types_or_unreadable_strings_are_not_defaults() {
    for raw in [
        "null",
        "true",
        "false",
        "[]",
        "{}",
        r#""""#,
        r#""oops""#,
        r#""0.1junk""#,
    ] {
        assert!(
            parse_commission_rates(&body(raw, "0.0004"), SYMBOL).is_err(),
            "maker={raw}"
        );
        assert!(
            parse_commission_rates(&body("0.0002", raw), SYMBOL).is_err(),
            "taker={raw}"
        );
    }
}

#[test]
fn symbol_is_a_required_string_with_exact_request_identity() {
    for raw in [
        "null",
        "42",
        "true",
        "[]",
        "{}",
        r#""""#,
        r#""ETHUSDT""#,
        r#""btcusdt""#,
        r#"" BTCUSDT""#,
        r#""BTCUSDT ""#,
    ] {
        let wire = format!(
            r#"{{"symbol":{raw},"makerCommissionRate":"0.0002","takerCommissionRate":"0.0004"}}"#
        );
        assert!(
            parse_commission_rates(&wire, SYMBOL).is_err(),
            "symbol={raw}"
        );
    }
    let wire = body("0.0002", "0.0004");
    for expected in ["", " ", " BTCUSDT", "BTCUSDT ", "ETHUSDT", "btcusdt"] {
        assert!(
            parse_commission_rates(&wire, expected).is_err(),
            "request={expected}"
        );
    }
    for padded in [" BTCUSDT", "BTCUSDT "] {
        let wire = format!(
            r#"{{"symbol":"{padded}","makerCommissionRate":"0.0002","takerCommissionRate":"0.0004"}}"#
        );
        assert!(parse_commission_rates(&wire, padded).is_err());
    }
    assert!(parse_commission_rates(
        r#"{"symbol":"","makerCommissionRate":0,"takerCommissionRate":0}"#,
        ""
    )
    .is_err());
}

#[test]
fn duplicate_recognized_keys_fail_even_if_values_match() {
    for duplicate in [
        r#""symbol":"BTCUSDT""#,
        r#""symbol":"ETHUSDT""#,
        r#""makerCommissionRate":"0.0002""#,
        r#""makerCommissionRate":"0.5""#,
        r#""takerCommissionRate":"0.0004""#,
        r#""takerCommissionRate":"0.5""#,
        r#""makerCommission\u0052ate":"0.0002""#,
    ] {
        let wire = format!(
            r#"{{{duplicate},"symbol":"BTCUSDT","makerCommissionRate":"0.0002","takerCommissionRate":"0.0004"}}"#
        );
        assert!(parse_commission_rates(&wire, SYMBOL).is_err(), "{wire}");
    }
}

#[test]
fn only_one_complete_json_object_is_commission_evidence() {
    for wire in [
        "",
        "not json",
        "null",
        "true",
        "42",
        r#""BTCUSDT""#,
        "[]",
        r#"["BTCUSDT","0.0002","0.0004"]"#,
        r#"{"code":-2015,"msg":"Invalid API-key"}"#,
        r#"{"symbol":"BTCUSDT","makerCommissionRate":"0.0002","takerCommissionRate":"0.0004""#,
        r#"{"symbol":"BTCUSDT","makerCommissionRate":"0.0002","takerCommissionRate":"0.0004"}{}"#,
        r#"{"symbol":"BTCUSDT","makerCommissionRate":"0.0002","takerCommissionRate":"0.0004","unknown":NaN}"#,
    ] {
        assert!(parse_commission_rates(wire, SYMBOL).is_err(), "{wire}");
    }
    let wire = format!(" \n\t{}\r\n", body("0.0002", "0.0004"));
    assert_eq!(parse_commission_rates(&wire, SYMBOL), Ok((0.0002, 0.0004)));
}
