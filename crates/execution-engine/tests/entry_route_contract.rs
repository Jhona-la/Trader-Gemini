//! Entry paths actually consumed by god_engine. Offline paper-only fixtures.
use execution_engine::executor::{ExecutionProvider, OrderExecutor};
use execution_engine::{
    binance_api::validate_leverage_confirmation,
    entry_dispatch::{
        dispatch_entry, new_entry_client_id, EntryRequest, EntryRoute, EntryTransport,
    },
};
use std::sync::Mutex;

fn paper_executor() -> OrderExecutor {
    let mut ex = OrderExecutor::new(String::new(), "offline-test".into(), true);
    ex.set_paper_trading(true);
    ex
}

#[tokio::test]
async fn raw_entry_rejects_nonrepresentable_rounded_quantity() {
    let ex = paper_executor();
    assert!(ex
        .execute_raw_qty_with_client_id("AUDITUSDT", true, 0.4, f64::from_bits(1), "audit_raw")
        .await
        .is_err());
}

#[tokio::test]
async fn paper_does_not_invent_native_usdm_iceberg_support() {
    let ex = paper_executor();
    let result = ex
        .execute_iceberg_limit(
            "AUDITUSDT",
            true,
            2.0,
            0.2,
            100.0,
            0.01,
            0.1,
            "audit_iceberg",
        )
        .await;
    assert!(
        result.is_err(),
        "USD-M adapter must not advertise unsupported native iceberg: {result:?}"
    );
}

#[derive(Debug, PartialEq)]
enum Event {
    Configure(String, u32),
    Submit {
        id: String,
        symbol: String,
        quantity: f64,
        route: EntryRoute,
    },
}

struct MockTransport {
    native_iceberg: bool,
    confirmation_error: bool,
    submission_error: bool,
    events: Mutex<Vec<Event>>,
}

impl MockTransport {
    fn new() -> Self {
        Self {
            native_iceberg: true,
            confirmation_error: false,
            submission_error: false,
            events: Mutex::new(Vec::new()),
        }
    }
}

impl EntryTransport for MockTransport {
    fn supports_native_iceberg(&self) -> bool {
        self.native_iceberg
    }
    async fn configure_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {
        self.events
            .lock()
            .unwrap()
            .push(Event::Configure(symbol.into(), leverage));
        if self.confirmation_error {
            Err("synthetic confirmation failure".into())
        } else {
            Ok(())
        }
    }
    async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {
        self.events.lock().unwrap().push(Event::Submit {
            id: request.client_order_id.into(),
            symbol: request.symbol.into(),
            quantity: request.quantity,
            route: request.route,
        });
        if self.submission_error {
            Err("AMBIGUOUS synthetic submission timeout".into())
        } else {
            Ok(())
        }
    }
}

fn request(route: EntryRoute) -> EntryRequest<'static> {
    EntryRequest {
        symbol: "AUDITUSDT",
        is_long: true,
        quantity: 2.0,
        step_size: 0.01,
        tick_size: 0.1,
        leverage: 1,
        client_order_id: "same_intention",
        route,
    }
}

#[tokio::test]
async fn leverage_failure_prevents_every_entry_route() {
    for route in [
        EntryRoute::Market,
        EntryRoute::Maker { price: 100.0 },
        EntryRoute::Iceberg {
            price: 100.0,
            visible_quantity: 0.2,
        },
    ] {
        let mut mock = MockTransport::new();
        mock.confirmation_error = true;
        let error = dispatch_entry(&mock, &request(route)).await.unwrap_err();
        assert!(error.starts_with("ENTRY_LEVERAGE_UNCONFIRMED"));
        assert_eq!(
            *mock.events.lock().unwrap(),
            vec![Event::Configure("AUDITUSDT".into(), 1)]
        );
    }
}

#[tokio::test]
async fn every_route_confirms_one_x_and_preserves_named_units_and_identity() {
    for route in [
        EntryRoute::Market,
        EntryRoute::Maker { price: 100.0 },
        EntryRoute::Iceberg {
            price: 100.0,
            visible_quantity: 0.2,
        },
    ] {
        let mock = MockTransport::new();
        dispatch_entry(&mock, &request(route)).await.unwrap();
        assert_eq!(
            *mock.events.lock().unwrap(),
            vec![
                Event::Configure("AUDITUSDT".into(), 1),
                Event::Submit {
                    id: "same_intention".into(),
                    symbol: "AUDITUSDT".into(),
                    quantity: 2.0,
                    route,
                }
            ]
        );
    }
}

#[tokio::test]
async fn unsupported_iceberg_has_no_configuration_or_submission_effects() {
    let mut mock = MockTransport::new();
    mock.native_iceberg = false;
    let err = dispatch_entry(
        &mock,
        &request(EntryRoute::Iceberg {
            price: 100.0,
            visible_quantity: 0.2,
        }),
    )
    .await
    .unwrap_err();
    assert!(err.starts_with("UNSUPPORTED_NATIVE_ICEBERG"));
    assert!(mock.events.lock().unwrap().is_empty());
}

#[tokio::test]
async fn invalid_request_cannot_mutate_leverage_first() {
    let mock = MockTransport::new();
    let baseline = request(EntryRoute::Market);
    for leverage in [0, 126, u32::MAX] {
        assert!(dispatch_entry(
            &mock,
            &EntryRequest {
                leverage,
                ..baseline
            }
        )
        .await
        .is_err());
    }
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(dispatch_entry(
            &mock,
            &EntryRequest {
                quantity: bad,
                ..baseline
            }
        )
        .await
        .is_err());
        assert!(dispatch_entry(
            &mock,
            &EntryRequest {
                step_size: bad,
                ..baseline
            }
        )
        .await
        .is_err());
        assert!(dispatch_entry(
            &mock,
            &EntryRequest {
                route: EntryRoute::Maker { price: bad },
                ..baseline
            }
        )
        .await
        .is_err());
        assert!(dispatch_entry(
            &mock,
            &EntryRequest {
                route: EntryRoute::Maker { price: 100.0 },
                tick_size: bad,
                ..baseline
            }
        )
        .await
        .is_err());
    }
    assert!(dispatch_entry(
        &mock,
        &EntryRequest {
            step_size: f64::from_bits(1),
            ..baseline
        }
    )
    .await
    .is_err());
    for id in ["", "bad&id", "é", "1234567890123456789012345678901234567"] {
        assert!(dispatch_entry(
            &mock,
            &EntryRequest {
                client_order_id: id,
                ..baseline
            }
        )
        .await
        .is_err());
    }
    assert!(dispatch_entry(
        &mock,
        &EntryRequest {
            symbol: "",
            ..baseline
        }
    )
    .await
    .is_err());
    for visible_quantity in [0.0, -0.1, 2.1, f64::NAN, f64::INFINITY] {
        assert!(dispatch_entry(
            &mock,
            &request(EntryRoute::Iceberg {
                price: 100.0,
                visible_quantity
            })
        )
        .await
        .is_err());
    }
    assert!(mock.events.lock().unwrap().is_empty());
}

#[tokio::test]
async fn market_does_not_use_limit_tick_and_submission_unknown_stays_unknown() {
    let mut mock = MockTransport::new();
    mock.submission_error = true;
    let req = EntryRequest {
        tick_size: f64::NAN,
        ..request(EntryRoute::Market)
    };
    assert_eq!(
        dispatch_entry(&mock, &req).await.unwrap_err(),
        "AMBIGUOUS synthetic submission timeout"
    );
    assert_eq!(mock.events.lock().unwrap().len(), 2);
}

#[test]
fn leverage_ack_requires_matching_identity_and_exact_value() {
    assert!(validate_leverage_confirmation(
        r#"{"symbol":"BTCUSDT","leverage":1,"maxNotionalValue":"100000"}"#,
        "BTCUSDT",
        1
    )
    .is_ok());
    for body in [
        "{}",
        "null",
        "invalid",
        r#"{"symbol":"BTCUSDT"}"#,
        r#"{"symbol":"ETHUSDT","leverage":1}"#,
        r#"{"symbol":"BTCUSDT","leverage":4}"#,
        r#"{"symbol":"BTCUSDT","leverage":1.5}"#,
        r#"{"symbol":"BTCUSDT","leverage":"1"}"#,
        r#"{"symbol":"BTCUSDT","leverage":-1}"#,
        r#"{"symbol":"BTCUSDT","leverage":1,"leverage":1}"#,
    ] {
        assert!(
            validate_leverage_confirmation(body, "BTCUSDT", 1).is_err(),
            "{body}"
        );
    }
}

#[test]
fn entry_ids_are_unique_bounded_and_keep_direction() {
    let mut ids = std::collections::HashSet::new();
    for i in 0..1000 {
        let is_long = i % 2 == 0;
        let id = new_entry_client_id(is_long);
        assert_eq!(id.len(), 35);
        assert!(id.starts_with(if is_long { "cL_" } else { "cS_" }));
        assert!(id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_'));
        assert!(ids.insert(id));
    }
}

#[tokio::test]
async fn real_adapter_paper_path_obeys_the_dispatch_contract_without_network() {
    let ex = paper_executor();
    assert!(dispatch_entry(&ex, &request(EntryRoute::Market))
        .await
        .is_ok());
    for leverage in [0, 126] {
        assert!(ex.set_leverage("AUDITUSDT", leverage).await.is_err());
    }
    assert!(ex.set_leverage("", 1).await.is_err());
    assert!(ex.set_leverage("AUDITUSDT", 1).await.is_ok());
}
