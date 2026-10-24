//! Contratos de Certificación Formal para Ola Ω172:
//! - F1-065: Cola lock-free ArrayQueue zero-alloc en trade_accounting y fast-path has_pending_bracket_closes.

use execution_engine::trade_accounting::{
    drain_bracket_closes, drain_bracket_closes_into, has_pending_bracket_closes,
    record_bracket_close, BracketClose,
};

#[test]
fn f1_065_lockfree_bracket_accounting_queue_contract() {
    // Vaciar cola previa si hubiera algo
    let _ = drain_bracket_closes();
    assert!(!has_pending_bracket_closes(), "La cola debe arrancar vacía");

    // Drenar en vacío devuelve vector vacío sin asignar
    let empty_drained = drain_bracket_closes();
    assert!(empty_drained.is_empty());

    // Grabar un cierre de bracket
    record_bracket_close(BracketClose {
        ts_ms: 1000,
        symbol: "BTCUSDT".to_string(),
        was_long: true,
        qty: 0.1,
        entry_price: 65000.0,
        exit_price: 65500.0,
        stop_price: 64500.0,
        pnl_gross: 50.0,
        fees: 2.0,
        trigger: "TP",
        slippage_bps: 1.5,
    });

    assert!(has_pending_bracket_closes(), "has_pending_bracket_closes debe ser true tras push");

    // Drenar con drain_bracket_closes_into sobre buffer reutilizable
    let mut reused_buffer = Vec::new();
    drain_bracket_closes_into(&mut reused_buffer);
    assert_eq!(reused_buffer.len(), 1);
    assert_eq!(reused_buffer[0].symbol, "BTCUSDT");
    assert_eq!(reused_buffer[0].pnl_gross, 50.0);
    assert_eq!(reused_buffer[0].trigger, "TP");

    assert!(!has_pending_bracket_closes(), "La cola debe quedar vacía tras drain");
}
