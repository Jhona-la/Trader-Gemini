//! Contract test for F1-027, F1-028, and F1-029 (Ola Ω156):
//!
//! - F1-027: Sequence inversion in `flatten_all_positions` (Protocol X-009). Positions are closed
//!   first; brackets are cancelled only upon confirmed fill; fails closed on unclosed positions.
//! - F1-028: Typed reduce-only execution with `&newOrderRespType=RESULT`, pre-dispatch `register_intent`,
//!   and validation of `executed_qty > 0.0`.
//! - F1-029: WS confirmation polling expanded to 200 ms (20 × 10 ms), and fresh `clientOrderId`
//!   issued for REST fallback to prevent duplicate `-4116` collisions.

use execution_engine::order_types::OrderAck;

fn executor_source() -> &'static str {
    include_str!("../src/executor.rs")
}

#[test]
fn test_f1_027_flatten_causal_sequence_contract() {
    let src = executor_source();

    let flatten_fn = src
        .split("pub async fn flatten_all_positions(&self)")
        .nth(1)
        .expect("flatten_all_positions must exist in OrderExecutor");

    let fn_body = &flatten_fn[..flatten_fn.find("/// F3.5").expect("end of flatten_all_positions")];

    // F1-027 Protocol X-009: Must close open positions first
    let close_pos_idx = fn_body.find("for p in &open_positions").expect("loop over open positions");
    let purge_sym_idx = fn_body.find("self.cancel_symbol_orders_and_brackets").expect("cancel_symbol_orders_and_brackets");

    assert!(
        close_pos_idx < purge_sym_idx,
        "Causal sequence violation: positions must be closed BEFORE brackets are purged"
    );

    // Protection brackets must only be cancelled if position was confirmed closed
    assert!(
        fn_body.contains("if position_closed {")
            && fn_body.contains("cancelled += self.cancel_symbol_orders_and_brackets(&p.symbol, &api_secret).await;"),
        "Brackets must only be purged upon confirmed position close"
    );

    // Fail-closed guarantee: if any position cannot be closed, return Err(FLATTEN_INCOMPLETE)
    assert!(
        fn_body.contains("FLATTEN_INCOMPLETE:"),
        "Fail-closed contract: must return Err(FLATTEN_INCOMPLETE) if unclosed positions remain"
    );
}

#[test]
fn test_f1_028_typed_reduce_only_contract() {
    let src = executor_source();

    let reduce_fn = src
        .split("async fn execute_reduce_only_market(")
        .nth(2)
        .expect("execute_reduce_only_market implementation must exist");

    let fn_body = &reduce_fn[..reduce_fn.find("/// FASE 8: Native Exchange Trailing Stop Market").expect("end of execute_reduce_only_market")];

    // F1-028: Must request RESULT to inspect executedQty
    assert!(
        fn_body.contains("&newOrderRespType=RESULT"),
        "execute_reduce_only_market must request RESULT"
    );

    // F1-028: Must register intent prior to dispatch
    let intent_idx = fn_body.find("self.order_registry.register_intent").expect("register_intent call");
    let dispatch_idx = fn_body.find("self.client.execute_order_payload_typed").expect("execute_order_payload_typed call");
    assert!(
        intent_idx < dispatch_idx,
        "register_intent must occur strictly before execute_order_payload_typed dispatch"
    );

    // F1-028: Must strictly validate executed_qty > 0.0 or terminal filled
    assert!(
        fn_body.contains("ack.executed_qty.is_finite() && ack.executed_qty > 0.0"),
        "Must check ack.executed_qty > 0.0"
    );
    assert!(
        fn_body.contains("CLOSE_UNFILLED:"),
        "Must return CLOSE_UNFILLED error on unfilled reduce order"
    );
}

#[test]
fn test_f1_029_ws_timeout_and_fresh_coid_contract() {
    let src = executor_source();

    let flatten_fn = src
        .split("pub async fn flatten_all_positions(&self)")
        .nth(1)
        .expect("flatten_all_positions must exist");

    let fn_body = &flatten_fn[..flatten_fn.find("/// F3.5").expect("end of flatten_all_positions")];

    // F1-029: 20 iterations of 10 ms = 200 ms confirmation window
    assert!(
        fn_body.contains("for _ in 0..20 {") && fn_body.contains("sleep(std::time::Duration::from_millis(10))"),
        "WS confirmation loop must be 20 × 10ms = 200ms"
    );

    // F1-029: REST fallback generates a distinct fresh clientOrderId
    assert!(
        fn_body.contains("let coid_rest = uuid::Uuid::now_v7().simple().to_string();"),
        "REST fallback must generate fresh coid_rest to prevent duplicate -4116"
    );
}

#[test]
fn test_order_ack_fill_semantics() {
    let empty_ack = OrderAck {
        executed_qty: 0.0,
        status: "EXPIRED".to_string(),
        ..Default::default()
    };
    assert!(!empty_ack.is_filled());
    assert!(!empty_ack.is_active());

    let filled_ack = OrderAck {
        executed_qty: 1.5,
        status: "FILLED".to_string(),
        ..Default::default()
    };
    assert!(filled_ack.is_filled());
    assert!(filled_ack.executed_qty > 0.0);
}
