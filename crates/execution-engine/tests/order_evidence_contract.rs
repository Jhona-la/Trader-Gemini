use execution_engine::{
    order_registry::{OrderRegistry, OrderStatus, TradeUpdate},
    order_types::OrderAck,
};
fn ack(qty: f64, price: f64, status: &str) -> OrderAck {
    OrderAck {
        client_order_id: "XIX-order".into(),
        symbol: "XIXUSDT".into(),
        side: "BUY".into(),
        orig_qty: 2.0,
        executed_qty: qty,
        avg_price: price,
        cum_quote: qty * price,
        status: status.into(),
        ..Default::default()
    }
}
fn update(qty: f64, price: f64, status: OrderStatus) -> TradeUpdate {
    TradeUpdate {
        client_order_id: "XIX-order".into(),
        symbol: "XIXUSDT".into(),
        side: "BUY".into(),
        position_side: "LONG".into(),
        order_type: "LIMIT".into(),
        execution_type: "TRADE".into(),
        order_id: 1,
        status,
        orig_qty: 2.0,
        cumulative_filled_qty: qty,
        last_filled_qty: 0.5,
        last_filled_price: price,
        avg_price: price,
        commission: 0.01,
        commission_asset: "USDT".into(),
        trade_id: 2,
        trade_time_ms: 1,
    }
}
#[test]
fn stale_rest_preserves_quantity_price_notional_tuple() {
    let r = OrderRegistry::new();
    r.apply_ack(&ack(1.0, 100.0, "PARTIALLY_FILLED"), 1);
    r.apply_ack(&ack(0.5, 90.0, "PARTIALLY_FILLED"), 2);
    let o = r.get("XIX-order").unwrap();
    assert_eq!(o.executed_qty, 1.0);
    assert_eq!(o.avg_price, 100.0);
    assert_eq!(o.cum_quote, 100.0);
}
#[test]
fn stale_ws_preserves_cumulative_price_but_keeps_fill_fee_evidence() {
    let r = OrderRegistry::new();
    r.apply_ack(&ack(1.0, 100.0, "PARTIALLY_FILLED"), 1);
    r.apply_trade_update(&update(0.5, 90.0, OrderStatus::PartiallyFilled), 2);
    let o = r.get("XIX-order").unwrap();
    assert_eq!(o.avg_price, 100.0);
    assert_eq!(o.executed_qty, 1.0);
    assert_eq!(o.fees_by_trade[&2], 0.01);
}
#[test]
fn filled_is_not_overwritten_by_late_terminal_rest_status() {
    let r = OrderRegistry::new();
    r.apply_ack(&ack(2.0, 100.0, "FILLED"), 1);
    r.apply_ack(&ack(2.0, 100.0, "CANCELED"), 2);
    assert_eq!(r.get_status("XIX-order"), Some(OrderStatus::Filled));
}
#[test]
fn filled_is_not_overwritten_by_late_terminal_ws_status() {
    let r = OrderRegistry::new();
    r.apply_ack(&ack(2.0, 100.0, "FILLED"), 1);
    r.apply_trade_update(&update(2.0, 100.0, OrderStatus::Expired), 2);
    assert_eq!(r.get_status("XIX-order"), Some(OrderStatus::Filled));
}
#[test]
fn delayed_fill_can_complete_a_previously_canceled_order() {
    let r = OrderRegistry::new();
    r.apply_ack(&ack(1.0, 100.0, "CANCELED"), 1);
    r.apply_trade_update(&update(2.0, 101.0, OrderStatus::Filled), 2);
    assert_eq!(r.get_status("XIX-order"), Some(OrderStatus::Filled));
}
#[test]
fn expired_in_match_is_terminal_expiry() {
    assert_eq!(OrderStatus::parse("EXPIRED_IN_MATCH"), OrderStatus::Expired);
}
