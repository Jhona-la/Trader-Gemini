//! OPEN limitations, not positive safety assertions.
use execution_engine::{
    order_registry::{OrderRegistry, OrderStatus},
    order_types::OrderAck,
    reconciliation::{reconcile, PositionRiskEntry},
};
#[test]
fn open_debt_local_timeout_fabricates_exchange_expiry() {
    // TRIAJE B (GLM 107) — DRENADO: un timeout LOCAL ya no fabrica una
    // expiración DEL EXCHANGE. Estado Unknown (fail-closed): ausencia de
    // evidencia ≠ expiración. La evidencia real tardía aterriza encima
    // (merge rank 0) y await_resolution sigue esperando → resolve_via_rest.
    let r = OrderRegistry::new();
    r.register_intent("audit", "XIXUSDT", "BUY", "LONG", "LIMIT", 1.0, 1);
    r.cleanup_stale_orders(100, 1000);
    assert_eq!(
        r.get_status("audit"),
        Some(OrderStatus::Unknown),
        "timeout local = desconocido, no expiración del exchange"
    );
    // El ack REAL tardío ya no es absorbido por el terminal fabricado.
    r.apply_ack(
        &OrderAck {
            client_order_id: "audit".into(),
            symbol: "XIXUSDT".into(),
            executed_qty: 1.0,
            avg_price: 100.0,
            cum_quote: 100.0,
            status: "PARTIALLY_FILLED".into(),
            update_time: 1100,
            ..Default::default()
        },
        1100,
    );
    assert_eq!(
        r.get_status("audit"),
        Some(OrderStatus::PartiallyFilled),
        "la evidencia del wire aterriza sobre el timeout"
    );
}
#[test]
fn open_debt_equal_quantity_old_price_still_replaces_new_price() {
    let r = OrderRegistry::new();
    let ack = |price, time| OrderAck {
        client_order_id: "audit".into(),
        symbol: "XIXUSDT".into(),
        executed_qty: 1.0,
        avg_price: price,
        cum_quote: price,
        status: "PARTIALLY_FILLED".into(),
        update_time: time,
        ..Default::default()
    };
    r.apply_ack(&ack(100.0, 200), 200);
    r.apply_ack(&ack(90.0, 100), 300);
    assert_eq!(r.get("audit").unwrap().avg_price, 90.0);
}
#[test]
fn open_debt_same_timestamp_hedge_adoptions_collide_in_registry() {
    let r = OrderRegistry::new();
    let row = |qty, side: &str| PositionRiskEntry {
        symbol: "XIXUSDT".into(),
        position_amt: qty,
        position_side: side.into(),
        entry_price: 100.0,
        update_time: 5,
        ..Default::default()
    };
    let report = reconcile(&[row(1.0, "LONG"), row(-1.0, "SHORT")], &r);
    assert_eq!(report.apply_to_registry(&r, 0.0005), 2);
    assert_eq!(r.stats().total, 1);
}
