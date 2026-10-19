//! Contract test for F1-027, F1-028, and F1-029 (Ola Ω156):
//!
//! - F1-027: Sequence inversion in `flatten_all_positions` (Protocol X-009). Positions are closed
//!   first; brackets are cancelled only upon confirmed fill; fails closed on unclosed positions.
//! - F1-028: Typed reduce-only execution with `&newOrderRespType=RESULT`, pre-dispatch `register_intent`,
//!   and validation of `executed_qty > 0.0`.
//! - F1-029: WS confirmation polling expanded to 200 ms (20 × 10 ms), and fresh `clientOrderId`
//!   issued for REST fallback to prevent duplicate `-4116` collisions.
//!
//! Sol (2026-10-19): estos contratos auditan el FUENTE de `executor.rs`. La versión
//! original delimitaba las funciones por anclas de comentarios (`/// F3.5`,
//! `/// FASE 8: ...`), que se rompieron cuando Ø157 (F1-031) extrajo la lógica
//! tipada a `execute_reduce_only_market_internal` y dejó los wrares vacíos:
//! `replay-contracts` quedó rojo en main con la lógica F1-027/F1-028 intacta.
//! Ahora la extracción es por BALANCEO DE LLAVAS (saltando strings y comentarios),
//! inmune a la mudanza de anclas textuales entre refactors.

use execution_engine::order_types::OrderAck;

fn executor_source() -> &'static str {
    include_str!("../src/executor.rs")
}

/// Extrae el cuerpo completo (firma incluidа) de la primera función cuya firma
/// contenga `signature`, por balanceo de llaves con salto de strings literales
/// y comentarios de línea. No depende de comentarios/anclas textuales.
fn fn_body_first(src: &str, signature: &str) -> String {
    let start = src
        .find(signature)
        .unwrap_or_else(|| panic!("signature not found in executor.rs: {signature}"));
    extract_balanced(src, start, signature)
}

/// Igual que `fn_body_first` pero desde la ÚLTIMA ocurrencia (para firmas que
/// aparecen primero en un trait y después en la implementación real).
fn fn_body_last(src: &str, signature: &str) -> String {
    let start = src
        .rfind(signature)
        .unwrap_or_else(|| panic!("signature not found in executor.rs: {signature}"));
    extract_balanced(src, start, signature)
}

fn extract_balanced(src: &str, start: usize, signature: &str) -> String {
    let bytes = src.as_bytes();
    let mut i = start;
    let mut depth: i32 = 0;
    let mut begun = false;
    let mut in_str = false;
    let mut in_line_comment = false;
    let mut prev: u8 = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_line_comment {
            if c == b'\n' {
                in_line_comment = false;
            }
        } else if in_str {
            if c == b'"' && prev != b'\\' {
                in_str = false;
            }
        } else if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            in_line_comment = true;
            i += 2;
            prev = c;
            continue;
        } else if c == b'"' {
            in_str = true;
        } else if c == b'{' {
            depth += 1;
            begun = true;
        } else if c == b'}' {
            depth -= 1;
            if begun && depth == 0 {
                return src[start..=i].to_string();
            }
        }
        prev = c;
        i += 1;
    }
    panic!("unbalanced braces after signature in executor.rs: {signature}");
}

#[test]
fn test_f1_027_flatten_causal_sequence_contract() {
    let src = executor_source();

    let fn_body = fn_body_first(src, "pub async fn flatten_all_positions(&self)");

    // F1-027 Protocol X-009: posiciones primero, purga de brackets después.
    let close_pos_idx = fn_body
        .find("for p in &open_positions")
        .expect("loop over open positions");
    let purge_sym_idx = fn_body
        .find("self.cancel_symbol_orders_and_brackets")
        .expect("cancel_symbol_orders_and_brackets");

    assert!(
        close_pos_idx < purge_sym_idx,
        "Causal sequence violation: positions must be closed BEFORE brackets are purged"
    );

    // La purga debe estar condicionada al cierre confirmado. Ø157 (F1-030)
    // refinó la llamada a tupla (n_cancelados, purgado) con `mark_dirty()`:
    // la semántica fail-safe se conserva y fortalece.
    let position_closed_idx = fn_body
        .find("if position_closed {")
        .expect("confirmed-close guard must exist");
    assert!(
        position_closed_idx < purge_sym_idx,
        "Brackets must only be purged upon confirmed position close (guard precedes purge)"
    );
    assert!(
        fn_body.contains("let (sym_canc, purged) = self.cancel_symbol_orders_and_brackets(&p.symbol, &api_secret).await;")
            && fn_body.contains("cancelled += sym_canc;"),
        "Purge must only run inside the confirmed-close branch (Ø157 tuple form)"
    );
    assert!(
        fn_body.contains("unpurged_symbols.push") && fn_body.contains("protection_health::mark_dirty()"),
        "Incomplete purge must be flagged (protection_health dirty) — fail-safe de Ø157"
    );

    // Fail-closed guarantee: if any position cannot be closed, return Err(FLATTEN_INCOMPLETE)
    assert!(
        fn_body.contains("FLATTEN_INCOMPLETE:"),
        "Fail-closed contract: must return Err(FLATTEN_INCOMPLETE) if unclosed positions remain"
    );
    assert!(
        fn_body.contains("unclosed_symbols.push(p.symbol.clone());"),
        "Unclosed positions must be tracked for the FLATTEN_INCOMPLETE error"
    );
}

#[test]
fn test_f1_028_typed_reduce_only_contract() {
    let src = executor_source();

    // F1-031 (Ø157): la lógica tipada vive en `execute_reduce_only_market_internal`;
    // el wrapper público `execute_reduce_only_market` delega en ella.
    let fn_body = fn_body_first(src, "async fn execute_reduce_only_market_internal(");
    let wrapper = fn_body_last(src, "async fn execute_reduce_only_market(");

    assert!(
        wrapper.contains("execute_reduce_only_market_internal("),
        "public execute_reduce_only_market must delegate to the typed internal path"
    );

    // F1-028: Must request RESULT to inspect executedQty
    assert!(
        fn_body.contains("&newOrderRespType=RESULT"),
        "execute_reduce_only_market must request RESULT"
    );

    // F1-028: Must register intent prior to dispatch
    let intent_idx = fn_body
        .find("self.order_registry.register_intent")
        .expect("register_intent call");
    let dispatch_idx = fn_body
        .find("self.client.execute_order_payload_typed")
        .expect("execute_order_payload_typed call");
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

    let fn_body = fn_body_first(src, "pub async fn flatten_all_positions(&self)");

    // F1-029: 20 iterations of 10 ms = 200 ms confirmation window
    assert!(
        fn_body.contains("for _ in 0..20 {") && fn_body.contains("from_millis(10)"),
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
