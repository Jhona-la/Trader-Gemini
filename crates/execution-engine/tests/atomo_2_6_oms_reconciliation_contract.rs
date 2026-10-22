//! Contrato E2E del Reconciliador OMS en Vivo — Átomo 2.6 (Plan Maestro 2026-10-22 / Ola Ω217)
//!
//! QUÉ: Certifica que el reconciliador OMS:
//!      1. Adopta posiciones remotas huérfanas en el exchange (GET /fapi/v2/positionRisk) hacia
//!         OrderRegistry y GlobalArena con imputación conservadora de comisión taker.
//!      2. Detecta y purga órdenes locales sospechosas (fantasmas) cuya posición remota está plana.
//!      3. Identifica órdenes resting remotas no registradas localmente (GET /fapi/v1/openOrders).
//! POR QUÉ: Evita drift entre el host y Binance, blindando una cuenta micro ($13 USD) de
//!          órdenes duplicadas, posiciones no protegidas y bloqueos de margen fantasma.
//! CÓMO: Prueba unitaria reproducible determinista sin dependencias de red externa.

use execution_engine::order_registry::{OrderRegistry, OrderStatus};
use execution_engine::order_types::OrderAck;
use execution_engine::reconciliation::{
    reconcile_arena, reconcile_with_orders, PositionRiskEntry,
};
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::atomic::Ordering;

#[test]
fn test_atomo_2_6_orphan_position_adopted_with_imputed_taker_fee() {
    let registry = OrderRegistry::new();

    // 1. Exchange reporta posición abierta de 0.05 BTC a 65,000 USDT con apalancamiento 10x
    let remote_positions = vec![PositionRiskEntry {
        symbol: "BTCUSDT".to_string(),
        position_amt: 0.05,
        entry_price: 65000.0,
        unrealized_pnl: 12.50,
        liquidation_price: 58500.0,
        leverage: 10.0,
        position_side: "BOTH".to_string(),
        update_time: 1729600000000,
    }];

    let report = reconcile_with_orders(&remote_positions, &[], &registry);
    assert_eq!(report.open_positions.len(), 1);
    assert_eq!(report.suspicious_active_orders.len(), 0);

    // 2. Adopción al OrderRegistry con comisión taker del 0.05% (0.0005)
    let taker_fee = 0.0005;
    let adopted_count = report.apply_to_registry(&registry, taker_fee);
    assert_eq!(adopted_count, 1, "Debe adoptar exactamente 1 posición huérfana");

    // Verificar en OrderRegistry
    let expected_client_id = "adopted_BTCUSDT_1729600000000";
    let adopted_order = registry.get(expected_client_id).expect("Debe existir la orden adoptada");
    assert_eq!(adopted_order.symbol, "BTCUSDT");
    assert_eq!(adopted_order.executed_qty, 0.05);
    assert_eq!(adopted_order.avg_price, 65000.0);
    assert_eq!(adopted_order.status, OrderStatus::Filled);
    let expected_commission = 0.05 * 65000.0 * taker_fee; // 1.625 USDT
    assert!((adopted_order.total_commission - expected_commission).abs() < 1e-6);

    // 3. Adopción a GlobalArena
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    symbols::update_dynamic_universe(vec!["BTCUSDT".to_string()]);
    let arena = GlobalArena::build_in_own_stack(100.0);

    let arena_adoptions = reconcile_arena(&remote_positions, &arena, 1729600001000);
    assert_eq!(arena_adoptions, 1, "GlobalArena debe adoptar la posición en ranura libre");

    let btc_coin = &arena.coins[0];
    assert!(btc_coin.positions.slot_2.is_open(), "Debe abrir la ranura preferida slot_2");
    assert_eq!(btc_coin.positions.slot_2.entry_price.load(Ordering::Relaxed), 65000.0);
    assert_eq!(btc_coin.positions.slot_2.quantity.load(Ordering::Relaxed), 0.05);
    assert!(btc_coin.positions.slot_2.is_long.load(Ordering::Relaxed));
}

#[test]
fn test_atomo_2_6_ghost_active_order_purged_when_remote_position_is_flat() {
    let registry = OrderRegistry::new();

    // 1. Registrar una orden local activa que quedó pendiente (NEW)
    registry.register_intent("god_ghost_123", "ETHUSDT", "BUY", "BOTH", "LIMIT", 1.0, 1000);

    assert_eq!(registry.active_orders().len(), 1, "Debe haber 1 orden activa local");

    // 2. Exchange reporta posición plana para ETHUSDT
    let remote_positions = vec![PositionRiskEntry {
        symbol: "ETHUSDT".to_string(),
        position_amt: 0.0,
        entry_price: 0.0,
        unrealized_pnl: 0.0,
        liquidation_price: 0.0,
        leverage: 10.0,
        position_side: "BOTH".to_string(),
        update_time: 2000,
    }];

    let report = reconcile_with_orders(&remote_positions, &[], &registry);
    assert_eq!(report.suspicious_active_orders.len(), 1, "Debe marcar la orden local como sospechosa");

    // 3. Purgar órdenes sospechosas de forma fail-closed
    let purged = report.purge_suspicious_active_orders(&registry, 2050);
    assert_eq!(purged, 1, "Debe purgar exactamente la orden fantasma");

    // Verificar que ya no hay órdenes activas locales
    assert_eq!(
        registry.active_orders().len(),
        0,
        "El registro no debe mantener órdenes activas fantasma cuando la posición remota es plana"
    );
    assert_eq!(registry.get_status("god_ghost_123"), Some(OrderStatus::Canceled));
}

#[test]
fn test_atomo_2_6_orphan_remote_resting_order_detected() {
    let registry = OrderRegistry::new();

    // Exchange reporta posición plana y una orden resting huérfana en Binance
    let remote_positions = vec![PositionRiskEntry {
        symbol: "SOLUSDT".to_string(),
        position_amt: 0.0,
        entry_price: 0.0,
        unrealized_pnl: 0.0,
        liquidation_price: 0.0,
        leverage: 10.0,
        position_side: "BOTH".to_string(),
        update_time: 2000,
    }];

    let remote_orders = vec![OrderAck {
        order_id: 99887766,
        symbol: "SOLUSDT".to_string(),
        status: "NEW".to_string(),
        client_order_id: "binance_orphan_order_99".to_string(),
        orig_qty: 2.0,
        executed_qty: 0.0,
        cum_quote: 0.0,
        avg_price: 0.0,
        price: 145.0,
        side: "BUY".to_string(),
        order_type: "LIMIT".to_string(),
        update_time: 2000,
        fills: Vec::new(),
    }];

    let report = reconcile_with_orders(&remote_positions, &remote_orders, &registry);
    assert_eq!(report.orphan_remote_orders.len(), 1);
    assert_eq!(report.orphan_remote_orders[0].client_order_id, "binance_orphan_order_99");
}
