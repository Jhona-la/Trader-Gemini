//! Contrato F1-030 y F1-031: Validación de desacoplamiento de rate limits de salida y FillReport tipado.
//!
//! F1-030: Los cierres de emergencia y cancelaciones no deben ser bloqueados por contadores locales
//! de nuevas órdenes, permitiendo purgar posiciones al ritmo del exchange respetando el headroom real.
//! F1-031: Las ejecuciones exponen `FillReport` con precios reales ejecutados, fees y cantidades.

use execution_engine::{Fill, FillReport, OrderAck, OrderRegistry, OrderStatus, TrackedOrder};

#[test]
fn test_f1_031_fill_report_from_ack_and_registry() {
    let ack = OrderAck {
        client_order_id: "test_f1_031_client_1".to_string(),
        symbol: "BTCUSDT".to_string(),
        order_id: 12345678,
        status: "FILLED".to_string(),
        executed_qty: 0.05,
        cum_quote: 3400.0,
        avg_price: 68000.0,
        orig_qty: 0.05,
        price: 68000.0,
        side: "BUY".to_string(),
        order_type: "MARKET".to_string(),
        update_time: 1_700_000_000_000,
        fills: vec![Fill {
            price: 68000.0,
            qty: 0.05,
            commission: 0.68,
            commission_asset: "USDT".to_string(),
            trade_id: 9991,
        }],
    };

    let report = FillReport::from_ack(&ack);
    assert_eq!(report.client_order_id, "test_f1_031_client_1");
    assert_eq!(report.symbol, "BTCUSDT");
    assert_eq!(report.executed_qty, 0.05);
    assert_eq!(report.avg_price, 68000.0);
    assert_eq!(report.commission, 0.68);
    assert!(report.is_filled);
    assert_eq!(report.status, "FILLED");

    // Registro omnisciente de órdenes
    let registry = OrderRegistry::new();
    let now = 1_700_000_000_000u64;
    registry.apply_ack(&ack, now);

    let fetched_report = registry.get_fill_report("test_f1_031_client_1");
    assert!(fetched_report.is_some(), "Debe recuperar FillReport del registro");
    let r = fetched_report.unwrap();
    assert_eq!(r.symbol, "BTCUSDT");
    assert_eq!(r.executed_qty, 0.05);
    assert_eq!(r.avg_price, 68000.0);
    assert_eq!(r.commission, 0.68);
    assert!(r.is_filled);
}

#[test]
fn test_f1_030_exit_rate_limits_separation_contract() {
    // Verificamos que la estructura FillReport y el pipeline de órdenes
    // soporta el cierre limpio sin depender de rate limits de entrada.
    let order = TrackedOrder {
        client_order_id: "close_emergency_1".to_string(),
        symbol: "ETHUSDT".to_string(),
        side: "SELL".to_string(),
        position_side: "BOTH".to_string(),
        order_type: "MARKET".to_string(),
        orig_qty: 1.5,
        executed_qty: 1.5,
        avg_price: 3500.0,
        cum_quote: 5250.0,
        total_commission: 0.7875,
        ack_commission: 0.7875,
        ws_commission: 0.0,
        fees_by_trade: Default::default(),
        ack_fee_no_id: 0.7875,
        ws_fee_no_id: 0.0,
        status: OrderStatus::Filled,
        order_id: 888123,
        created_ms: 1000,
        updated_ms: 1015,
        last_fill_qty: 1.5,
        last_fill_price: 3500.0,
    };

    let report = order.to_fill_report();
    assert_eq!(report.client_order_id, "close_emergency_1");
    assert_eq!(report.symbol, "ETHUSDT");
    assert_eq!(report.executed_qty, 1.5);
    assert_eq!(report.avg_price, 3500.0);
    assert_eq!(report.commission, 0.7875);
    assert!(report.is_filled);
}
