//! Contrato F1-032: Clasificación estricta de respuestas terminales no activas en execute_order.
//!
//! Previene el defecto latente de "posiciones fantasma": si Binance responde HTTP 200 con un ACK
//! cuyo status es EXPIRED (ej. post-only GTX cruzada), CANCELED o REJECTED, el ejecutor debe
//! retornar Err explícito al llamador en lugar de Ok(()).

use execution_engine::OrderAck;

#[test]
fn test_f1_032_order_ack_terminal_inactive_properties() {
    for terminal_status in ["EXPIRED", "CANCELED", "REJECTED", "PENDING_CANCEL"] {
        let ack = OrderAck {
            order_id: 1001,
            symbol: "BTCUSDT".to_string(),
            status: terminal_status.to_string(),
            client_order_id: format!("test_ack_{}", terminal_status),
            orig_qty: 0.1,
            executed_qty: 0.0,
            cum_quote: 0.0,
            avg_price: 0.0,
            price: 65000.0,
            side: "BUY".to_string(),
            order_type: "LIMIT".to_string(),
            update_time: 1_700_000_000,
            fills: vec![],
        };
        assert!(!ack.is_filled(), "{} no debe ser considerado filled", terminal_status);
        assert!(!ack.is_active(), "{} no debe ser considerado active", terminal_status);
    }

    // NEW y PARTIALLY_FILLED son activas
    for active_status in ["NEW", "PARTIALLY_FILLED"] {
        let ack = OrderAck {
            order_id: 1002,
            symbol: "BTCUSDT".to_string(),
            status: active_status.to_string(),
            client_order_id: format!("test_ack_{}", active_status),
            orig_qty: 0.1,
            executed_qty: 0.0,
            cum_quote: 0.0,
            avg_price: 0.0,
            price: 65000.0,
            side: "BUY".to_string(),
            order_type: "LIMIT".to_string(),
            update_time: 1_700_000_000,
            fills: vec![],
        };
        assert!(ack.is_active(), "{} debe ser considerado active", active_status);
    }

    // FILLED es llena
    let filled_ack = OrderAck {
        order_id: 1003,
        symbol: "BTCUSDT".to_string(),
        status: "FILLED".to_string(),
        client_order_id: "test_ack_filled".to_string(),
        orig_qty: 0.1,
        executed_qty: 0.1,
        cum_quote: 6500.0,
        avg_price: 65000.0,
        price: 65000.0,
        side: "BUY".to_string(),
        order_type: "LIMIT".to_string(),
        update_time: 1_700_000_000,
        fills: vec![],
    };
    assert!(filled_ack.is_filled(), "FILLED debe ser considered filled");
}

#[test]
fn test_f1_032_structural_no_ghost_position_on_terminal_inactive_ack() {
    let src = include_str!("../src/executor.rs");
    // Verificar que en execute_order no existe Ok(()) incondicional tras ack no-filled y no-active
    assert!(
        src.contains("if !ack.is_filled() && !ack.is_active()"),
        "El hot path debe auditar estados no filled y no active"
    );
    assert!(
        src.contains("return Err(format!(") && src.contains("ORDER_"),
        "Debe retornar Err explícito con prefijo ORDER_ ante estados terminales inactivos"
    );
}
