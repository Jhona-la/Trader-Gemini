use execution_engine::{
    execution_evidence::{classify_close_ack, CloseSettlement},
    order_types::OrderAck,
};

fn ack(status: &str, orig_qty: f64, executed_qty: f64) -> OrderAck {
    OrderAck {
        order_id: 7,
        symbol: "BTCUSDT".into(),
        status: status.into(),
        client_order_id: "close-contract".into(),
        orig_qty,
        executed_qty,
        avg_price: 60_000.0,
        side: "SELL".into(),
        order_type: "MARKET".into(),
        ..Default::default()
    }
}

#[test]
fn partial_fill_is_residual_and_preserves_the_remaining_quantity() {
    let settlement = classify_close_ack(&ack("PARTIALLY_FILLED", 1.0, 0.4));
    match settlement {
        CloseSettlement::Residual {
            report,
            remaining_qty,
        } => {
            assert_eq!(report.executed_qty, 0.4);
            assert_eq!(remaining_qty, 0.6);
            assert!(!report.is_filled);
        }
        _ => panic!("partial fill must be residual"),
    }
}

#[test]
fn filled_with_zero_and_empty_ack_are_unknown() {
    for candidate in [ack("FILLED", 1.0, 0.0), OrderAck::default()] {
        assert!(matches!(
            classify_close_ack(&candidate),
            CloseSettlement::Unknown(_)
        ));
    }
}

#[test]
fn nonfinite_and_inconsistent_close_evidence_is_unknown() {
    for candidate in [
        ack("FILLED", f64::NAN, 1.0),
        ack("FILLED", 1.0, f64::INFINITY),
        ack("FILLED", 1.0, 1.1),
        ack("FILLED", 1.0, 0.4),
        ack("NEW", 1.0, 0.4),
    ] {
        assert!(matches!(
            classify_close_ack(&candidate),
            CloseSettlement::Unknown(_)
        ));
    }
}

#[test]
fn close_requires_exchange_identity() {
    for candidate in [
        OrderAck {
            order_id: 0,
            ..ack("FILLED", 1.0, 1.0)
        },
        OrderAck {
            client_order_id: String::new(),
            ..ack("FILLED", 1.0, 1.0)
        },
        OrderAck {
            symbol: String::new(),
            ..ack("FILLED", 1.0, 1.0)
        },
        OrderAck {
            side: "HOLD".into(),
            ..ack("FILLED", 1.0, 1.0)
        },
    ] {
        assert!(matches!(
            classify_close_ack(&candidate),
            CloseSettlement::Unknown(_)
        ));
    }
}

#[test]
fn terminal_partial_fill_is_residual_not_closed() {
    match classify_close_ack(&ack("EXPIRED", 1.0, 0.4)) {
        CloseSettlement::Residual { remaining_qty, .. } => {
            assert!((remaining_qty - 0.6).abs() < 1e-12)
        }
        _ => panic!("terminal partial fill must be residual"),
    }
}

#[test]
fn coherent_full_fill_closes_with_its_real_report() {
    let settlement = classify_close_ack(&ack("FILLED", 1.0, 1.0));
    match settlement {
        CloseSettlement::Closed(report) => {
            assert_eq!(report.client_order_id, "close-contract");
            assert_eq!(report.symbol, "BTCUSDT");
            assert_eq!(report.executed_qty, 1.0);
            assert!(report.is_filled);
        }
        _ => panic!("coherent full fill must close"),
    }
}
