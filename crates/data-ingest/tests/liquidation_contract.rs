use data_ingest::liquidation::{
    decode_liquidation_snapshots as decode, LiquidationParseError as Error, LiquidationSide,
};
use data_ingest::TensorParser;
use serde_json::{json, Value};

fn snapshot() -> Value {
    json!({"e":"forceOrder","E":1000,"o":{"s":"BTCUSDT","S":"SELL","q":"10","p":"100","ap":"90","z":"2","l":"2","X":"PARTIALLY_FILLED","T":999}})
}

fn parse(v: Value) -> data_ingest::liquidation::LiquidationBatch {
    decode(&serde_json::to_vec(&v).unwrap()).unwrap()
}

#[test]
fn checked_snapshot_preserves_provenance_and_uses_reported_executed_not_requested_notional() {
    let batch = parse(snapshot());
    assert!(batch.rejected.is_empty());
    let s = &batch.snapshots[0];
    assert_eq!(
        (&*s.symbol, s.side, s.event_time_ms, s.trade_time_ms),
        ("BTCUSDT", LiquidationSide::Sell, 1000, 999)
    );
    assert_eq!(
        (
            s.order_quantity,
            s.order_price,
            s.average_price,
            s.filled_quantity,
            s.last_filled_quantity
        ),
        (10.0, 100.0, 90.0, 2.0, 2.0)
    );
    assert_eq!(s.reported_filled_notional, 180.0);
    assert!(s.legacy_um_contract);
}

#[test]
fn raw_and_combined_arrays_retain_each_asset_even_with_escaped_braces() {
    let a = snapshot();
    let mut b = snapshot();
    b["o"]["s"] = json!("ETHUSDT");
    b["o"]["S"] = json!("BUY");
    b["extra"] = json!("escaped \"{ }\" not an event");
    for value in [
        json!([a, b]),
        json!({"stream":"!forceOrder@arr","data":[a,b]}),
        json!({"stream":"btcusdt@forceOrder","data":a}),
    ] {
        let batch = parse(value);
        assert!(batch.rejected.is_empty());
        assert_eq!(batch.snapshots[0].symbol, "BTCUSDT");
        if batch.snapshots.len() == 2 {
            assert_eq!(batch.snapshots[1].symbol, "ETHUSDT");
        }
    }
}

#[test]
fn explicit_coin_margined_contract_is_rejected_without_discarding_um_record() {
    let mut cm = snapshot();
    cm["st"] = json!(2);
    let mut um = snapshot();
    um["st"] = json!(1);
    let batch = parse(json!([cm, um]));
    assert_eq!(batch.rejected, vec![(0, Error::UnsupportedContractType)]);
    assert_eq!(batch.snapshots.len(), 1);
    assert!(!batch.snapshots[0].legacy_um_contract);
    let mut conflicting = snapshot();
    conflicting["st"] = json!(1);
    conflicting["o"]["st"] = json!(2);
    assert_eq!(parse(conflicting).rejected[0].1, Error::InvalidField("st"));
}

#[test]
fn numeric_fields_parse_entire_scientific_tokens_and_reject_nonfinite_or_trailing_junk() {
    let mut v = snapshot();
    v["o"]["ap"] = json!("1e3");
    assert_eq!(parse(v).snapshots[0].reported_filled_notional, 2000.0);
    for value in [
        json!("NaN"),
        json!("inf"),
        json!("-1"),
        json!("100junk"),
        json!(null),
    ] {
        let mut v = snapshot();
        v["o"]["ap"] = value;
        let b = parse(v);
        assert!(b.snapshots.is_empty());
        assert_eq!(b.rejected.len(), 1);
    }
    let mut v = snapshot();
    v["o"]["ap"] = json!("1e308");
    assert_eq!(parse(v).rejected[0].1, Error::InvalidField("ap*z overflow"));
}

#[test]
fn invalid_identity_time_and_execution_bounds_never_emit_snapshots() {
    for (key, value) in [
        ("s", json!("")),
        ("S", json!("UNKNOWN")),
        ("T", json!(1001)),
        ("T", json!(0)),
        ("T", json!("999")),
        ("q", json!("1")),
        ("z", json!("11")),
        ("l", json!("3")),
        ("X", json!("")),
    ] {
        let mut v = snapshot();
        v["o"][key] = value;
        let b = parse(v);
        assert!(b.snapshots.is_empty(), "key={key}");
        assert_eq!(b.rejected.len(), 1);
    }
    let mut v = snapshot();
    v["E"] = json!(0);
    assert!(parse(v).snapshots.is_empty());
}

#[test]
fn zero_executed_quantity_is_observed_zero_not_requested_volume() {
    let mut v = snapshot();
    v["o"]["ap"] = json!("0");
    v["o"]["z"] = json!("0");
    v["o"]["l"] = json!("0");
    assert_eq!(parse(v).snapshots[0].reported_filled_notional, 0.0);
}

#[test]
fn malformed_json_and_invalid_envelope_cannot_emit_partial_records() {
    assert_eq!(
        decode(b"[{\"e\":\"forceOrder\"},").unwrap_err(),
        Error::MalformedJson
    );
    for value in [
        json!(null),
        json!(1),
        json!({"data":snapshot()}),
        json!({"stream":"x","data":snapshot(),"e":"forceOrder"}),
    ] {
        assert_eq!(
            decode(&serde_json::to_vec(&value).unwrap()).unwrap_err(),
            Error::InvalidEnvelope
        );
    }
}

#[test]
fn legacy_number_scanner_still_drops_scientific_exponents() {
    // OPEN: existing generic scanner is outside the new checked liquidation path.
    assert_eq!(TensorParser::fast_parse_f64(b"1e3"), 1.0);
}

#[test]
fn legacy_liquidation_scanner_mixes_requested_with_executed_notional() {
    // OPEN compatibility API: retained but removed from the operational producer.
    let p = br#"{"e":"forceOrder","E":1000,"o":{"s":"BTCUSDT","S":"SELL","q":"10","p":"100","ap":"90","z":"2","l":"2","X":"PARTIALLY_FILLED","T":999}}"#;
    let mut observed = Vec::new();
    TensorParser::parse_force_orders(p, |price, qty| observed.push(price * qty));
    assert_eq!(observed, vec![1000.0]); // reported execution snapshot should be 180
}
