//! OPEN limitations, not positive safety assertions.
use execution_engine::{
    order_registry::{OrderRegistry, OrderStatus},
    order_types::OrderAck,
    reconciliation::{reconcile, PositionRiskEntry},
};
#[test]
fn open_debt_local_timeout_fabricates_exchange_expiry() {
    let r = OrderRegistry::new();
    r.register_intent("audit", "XIXUSDT", "BUY", "LONG", "LIMIT", 1.0, 1);
    r.cleanup_stale_orders(100, 1000);
    assert_eq!(r.get_status("audit"), Some(OrderStatus::Expired));
    assert_eq!(r.prune_terminated(2000), 1);
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
