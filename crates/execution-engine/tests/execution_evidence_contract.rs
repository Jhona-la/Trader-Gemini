use execution_engine::{execution_evidence::*, OrderAck, OrderResolution};

fn row(qty: &str) -> String {
    format!(
        r#"{{"symbol":"AUDITUSDT","positionAmt":"{qty}","entryPrice":"100","leverage":"10","positionSide":"BOTH"}}"#
    )
}

#[test]
fn malformed_or_incomplete_positions_cannot_become_empty_success() {
    for body in ["not json", "null", "{}", r#"{"code":-2015}"#, "[{}]"] {
        assert!(parse_active_positions(body).is_err(), "{body}");
    }
    let mixed = format!("[{},{{}}]", row("1"));
    assert!(parse_active_positions(&mixed).is_err());
}

#[test]
fn nonzero_exposure_is_not_removed_by_an_epsilon() {
    for qty in ["0.000000001", "-0.000000001"] {
        let positions = parse_active_positions(&format!("[{}]", row(qty))).unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].qty, 1e-9);
        assert_eq!(positions[0].is_long, !qty.starts_with('-'));
    }
}

#[test]
fn nonfinite_position_values_are_not_exposure_or_flat_evidence() {
    for qty in ["NaN", "inf", "-inf"] {
        assert!(parse_active_positions(&format!("[{}]", row(qty))).is_err());
    }
}

fn ack(status: &str, executed_qty: f64) -> OrderAck {
    OrderAck {
        symbol: "AUDITUSDT".into(),
        client_order_id: "intent-1".into(),
        order_id: 123,
        orig_qty: 1.0,
        executed_qty,
        status: status.into(),
        side: "BUY".into(),
        ..Default::default()
    }
}
fn classify(result: Result<&OrderAck, &str>) -> OrderResolution {
    classify_query_result("AUDITUSDT", "intent-1", result)
}

#[test]
fn not_found_query_is_unknown_not_a_rejection_certificate() {
    assert_eq!(
        classify(Err("-2013 Order does not exist")),
        OrderResolution::Timeout
    );
}

#[test]
fn foreign_or_incomplete_ack_is_unknown_before_registry_mutation() {
    let mut foreign = ack("NEW", 0.0);
    foreign.symbol = "OTHERUSDT".into();
    assert_eq!(classify(Ok(&foreign)), OrderResolution::Timeout);
    assert_eq!(classify(Ok(&OrderAck::default())), OrderResolution::Timeout);
}

#[test]
fn positions_require_each_wire_field_and_validate_domains() {
    let good: serde_json::Value = serde_json::from_str(&row("1")).unwrap();
    for field in [
        "symbol",
        "positionAmt",
        "entryPrice",
        "leverage",
        "positionSide",
    ] {
        let mut missing = good.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(
            parse_active_positions(&format!("[{missing}]")).is_err(),
            "{field}"
        );
    }
    for (field, values) in [
        ("entryPrice", vec!["0", "-1", "NaN", "inf"]),
        ("leverage", vec!["0", "-1", "NaN", "inf", "1.5"]),
        ("symbol", vec!["", " AUDITUSDT"]),
        ("positionSide", vec!["", "UNKNOWN", "SHORT"]),
    ] {
        for value in values {
            let mut invalid = good.clone();
            invalid[field] = value.into();
            assert!(
                parse_active_positions(&format!("[{invalid}]")).is_err(),
                "{field}={value}"
            );
        }
    }
}

#[test]
fn flat_hedge_and_numeric_wire_values_are_valid_without_netting() {
    assert!(parse_active_positions("[]").unwrap().is_empty());
    let flat = row("0").replace("\"100\"", "\"0\"");
    assert!(parse_active_positions(&format!("[{flat}]"))
        .unwrap()
        .is_empty());
    let long = row("2").replace("BOTH", "LONG");
    let short = row("-2").replace("BOTH", "SHORT");
    let result = parse_active_positions(&format!("[{long},{short}]")).unwrap();
    assert_eq!(result.len(), 2); // balanced net is still two gross exposures
    assert!(result[0].is_long);
    assert!(!result[1].is_long);
    let numeric = r#"[{"symbol":"AUDITUSDT","positionAmt":2,"entryPrice":100,"leverage":10,"positionSide":"LONG"}]"#;
    assert_eq!(parse_active_positions(numeric).unwrap()[0].qty, 2.0);
}

#[test]
fn duplicate_or_mixed_legs_reject_the_whole_snapshot() {
    let both = row("1");
    let long = both.replace("BOTH", "LONG");
    for (a, b) in [
        (&both, &both),
        (&long, &long),
        (&both, &long),
        (&long, &both),
    ] {
        assert!(parse_active_positions(&format!("[{a},{b}]")).is_err());
    }
}

#[test]
fn order_states_distinguish_admission_fills_and_terminal_nofill() {
    for status in ["CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"] {
        assert_eq!(classify(Ok(&ack(status, 0.0))), OrderResolution::Rejected);
        assert_eq!(classify(Ok(&ack(status, 0.4))), OrderResolution::Accepted);
    }
    for (status, qty) in [("NEW", 0.0), ("PARTIALLY_FILLED", 0.4), ("FILLED", 1.0)] {
        assert_eq!(classify(Ok(&ack(status, qty))), OrderResolution::Accepted);
    }
    for (status, qty) in [
        ("NEW", 0.4),
        ("PARTIALLY_FILLED", 0.0),
        ("PARTIALLY_FILLED", 1.0),
        ("FILLED", 0.0),
        ("FILLED", 0.4),
        ("UNKNOWN", 0.4),
    ] {
        assert_eq!(classify(Ok(&ack(status, qty))), OrderResolution::Timeout);
    }
}

#[test]
fn invalid_query_quantities_ids_and_transport_failures_remain_unknown() {
    for qty in [-1.0, f64::NAN, f64::INFINITY, 1.1] {
        assert_eq!(
            classify(Ok(&ack("CANCELED", qty))),
            OrderResolution::Timeout
        );
    }
    for qty in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut a = ack("NEW", 0.0);
        a.orig_qty = qty;
        assert_eq!(classify(Ok(&a)), OrderResolution::Timeout);
    }
    let mut a = ack("NEW", 0.0);
    a.client_order_id = "another-intent".into();
    assert_eq!(classify(Ok(&a)), OrderResolution::Timeout);
    for e in [
        "HTTP 500",
        "timeout",
        "-2013",
        "contains -2013 in unrelated text",
    ] {
        assert_eq!(classify(Err(e)), OrderResolution::Timeout);
    }
}

#[test]
fn get_query_requires_wire_fields_even_when_serde_has_defaults() {
    let good: serde_json::Value = serde_json::from_str(r#"{"symbol":"AUDITUSDT","clientOrderId":"intent-1","orderId":123,"side":"BUY","status":"CANCELED","origQty":"1","executedQty":"0"}"#).unwrap();
    assert!(parse_query_order_response(&good.to_string(), "AUDITUSDT", "intent-1").is_ok());
    for field in [
        "symbol",
        "clientOrderId",
        "orderId",
        "side",
        "status",
        "origQty",
        "executedQty",
    ] {
        let mut bad = good.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(
            parse_query_order_response(&bad.to_string(), "AUDITUSDT", "intent-1").is_err(),
            "{field}"
        );
    }
    assert!(parse_query_order_response(&good.to_string(), "AUDITUSDT", "other").is_err());
}

#[test]
fn maker_active_response_does_not_authorize_a_market_replacement() {
    for (status, qty) in [("NEW", 0.0), ("PARTIALLY_FILLED", 0.4)] {
        let e = terminal_maker_executed_quantity("AUDITUSDT", "intent-1", &ack(status, qty))
            .unwrap_err();
        assert!(e.starts_with("MAKER_CHASE_UNVERIFIED"));
    }
    for (status, qty) in [("CANCELED", 0.4), ("FILLED", 1.0), ("EXPIRED", 0.0)] {
        assert_eq!(
            terminal_maker_executed_quantity("AUDITUSDT", "intent-1", &ack(status, qty)).unwrap(),
            qty
        );
    }
    assert!(
        terminal_maker_executed_quantity("OTHERUSDT", "intent-1", &ack("CANCELED", 0.0)).is_err()
    );
}
