//! Contrato Formal de Viabilidad Financiera y Selección Adversa (Fase 3 Átomo 3.1) — Ola Ω220.
//!
//! # QUÉ
//! Contrato formal que certifica la eliminación total del bypass heurístico de confianza
//! (`net_confidence >= 0.65`) y la resolución dinámica de comisiones maker y costo de funding
//! desde `arena.config.live_fee_schedule` y `arena.coins[coin_id].expected_funding_cost`.
//!
//! # POR QUÉ
//! En el router previo, una orden con edge bruto insuficiente para pagar 3 veces la fricción
//! se ejecutaba silenciosamente si `net_confidence >= 0.65`. Este bypass heurístico causaba
//! fugas de capital constantes en mercados adversos, violando el Axioma de Viabilidad Financiera:
//!   E[ΔW_bruto] >= 3.0 * (fees + funding + half_spread + adverse_selection)
//!
//! # PARA QUÉ
//! Garantizar que el sistema opere en modo fail-closed absoluto frente a fricciones microestructurales,
//! independientemente del nivel de convicción del modelo, cerrando la sangría de comisiones hacia exchanges.
//!
//! # DÓNDE
//! `crates/execution-engine/tests/adverse_selection_viability_contract.rs`.
//!
//! # QUIÉN
//! Consejo Colegiado de 10 Roles / Quant Sr. Lead.

use arc_swap::ArcSwap;
use execution_engine::executor::OrderExecutor;
use execution_engine::router::QuantumOrderRouter;
use quantum_arena::fee_schedule::{FeeSchedule, SymbolFees};
use quantum_arena::{symbols, GlobalArena};
use signal_engine::orchestrator::TensorDecision;
use signal_engine::SignalType;
use std::sync::Arc;

#[tokio::test]
async fn test_contract_high_confidence_order_strictly_rejected_if_edge_insufficient() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    symbols::update_dynamic_universe(vec!["BTCUSDT".to_string()]);

    let mut executor = OrderExecutor::new(
        "dummy_key".to_string(),
        "dummy_secret".to_string(),
        true,
    );
    executor.set_paper_trading(true);
    executor.arena.store(Some(arena));

    let router = QuantumOrderRouter::new(Arc::new(ArcSwap::from_pointee(executor)));

    // Decisión con altísima confianza (0.95, superior al umbral anterior de bypass 0.65)
    // pero con volatilidad esperada ínfima (0.0001 = 1 bp) frente a spread de 4 bps.
    // El edge bruto no puede pagar 3x fricciones.
    let high_conf_weak_ev_decision = TensorDecision {
        signal: SignalType::Long,
        net_confidence: 0.95,
        expected_volatility: 0.0001, // 1 bp
        expected_lifetime_ms: 10_000,
    };

    let result = router
        .route_order(
            "BTCUSDT",
            &high_conf_weak_ev_decision,
            50000.0,
            0.001,
            0.001,
            0.1,
            20,     // 20 ms (latencia normal, no pánico)
            0.0004, // 4 bps spread
            "ORD_HIGH_CONF_INSUFFICIENT_EDGE_01",
        )
        .await;

    // En el código anterior este caso PASABA a mercado porque net_confidence >= 0.65.
    // Con el blindaje de Fase 3 Átomo 3.1, DEBE ser rechazado fail-closed.
    assert!(
        result.is_err(),
        "El Axioma de Viabilidad DEBE rechazar órdenes con edge insuficiente incluso con 95% de confianza"
    );
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("REJ_INSUFFICIENT_VIABILITY"),
        "El error debe especificar REJ_INSUFFICIENT_VIABILITY, obtenido: {err_msg}"
    );
}

#[tokio::test]
async fn test_contract_router_resolves_live_maker_fee_from_published_schedule() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    symbols::update_dynamic_universe(vec!["BTCUSDT".to_string()]);

    // Publicamos un FeeSchedule explícito donde BTCUSDT tiene un maker fee de 15 bps (0.0015)
    // en lugar del default de 2 bps (0.0002).
    let schedule = FeeSchedule::new(
        1,
        vec![SymbolFees {
            symbol: "BTCUSDT".to_string(),
            maker: 0.0015, // 15 bps maker
            taker: 0.0030,
        }],
    )
    .expect("FeeSchedule válido");

    arena.config.live_fee_schedule.store(Some(Arc::new(schedule)));

    let mut executor = OrderExecutor::new(
        "dummy_key".to_string(),
        "dummy_secret".to_string(),
        true,
    );
    executor.set_paper_trading(true);
    executor.arena.store(Some(arena));

    let router = QuantumOrderRouter::new(Arc::new(ArcSwap::from_pointee(executor)));

    // Decisión con volatilidad 15 bps (0.0015) y confianza 0.60.
    // Con maker fee default de 2 bps, podría ser viable. Con maker fee real de 15 bps,
    // la fricción total supera los 20 bps, por lo que EV < 3x fricción.
    let decision = TensorDecision {
        signal: SignalType::Long,
        net_confidence: 0.60,
        expected_volatility: 0.0015,
        expected_lifetime_ms: 10_000,
    };

    let result = router
        .route_order(
            "BTCUSDT",
            &decision,
            50000.0,
            0.001,
            0.001,
            0.1,
            20,
            0.0002, // 2 bps spread
            "ORD_FEE_SCHEDULE_TEST_01",
        )
        .await;

    assert!(
        result.is_err(),
        "Con maker fee real de 15 bps la orden no debe ser viable"
    );
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("REJ_INSUFFICIENT_VIABILITY"));
}

#[tokio::test]
async fn test_contract_router_resolves_dynamic_funding_cost() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    symbols::update_dynamic_universe(vec!["BTCUSDT".to_string()]);

    let coin_id = symbols::get_coin_id("BTCUSDT").expect("BTCUSDT indexado");
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // Publicamos un funding rate adverso muy alto para largos (15 bps por ciclo de 8h)
    // que empuja las fricciones totales por encima de un tercio del edge esperado (28 bps vs 3x(15+2+1) bps = 54 bps)
    arena.coins[coin_id].publish_funding_next(0.0015, now_ms);

    let mut executor = OrderExecutor::new(
        "dummy_key".to_string(),
        "dummy_secret".to_string(),
        true,
    );
    executor.set_paper_trading(true);
    executor.arena.store(Some(arena));

    let router = QuantumOrderRouter::new(Arc::new(ArcSwap::from_pointee(executor)));

    // Decisión a horizonte de 8 horas (ciclo completo de funding)
    let decision = TensorDecision {
        signal: SignalType::Long,
        net_confidence: 0.70,
        expected_volatility: 0.0020,
        expected_lifetime_ms: 8 * 3600 * 1000, // 8h
    };

    let result = router
        .route_order(
            "BTCUSDT",
            &decision,
            50000.0,
            0.001,
            0.001,
            0.1,
            20,
            0.0002,
            "ORD_FUNDING_CARRY_TEST_01",
        )
        .await;

    // El costo de carry se suma a las fricciones y se rechaza si no hay edge suficiente
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("REJ_INSUFFICIENT_VIABILITY"));
}

#[tokio::test]
async fn test_contract_router_allows_genuinely_viable_orders() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    symbols::update_dynamic_universe(vec!["BTCUSDT".to_string()]);

    let mut executor = OrderExecutor::new(
        "dummy_key".to_string(),
        "dummy_secret".to_string(),
        true,
    );
    executor.set_paper_trading(true);
    executor.arena.store(Some(arena));

    let router = QuantumOrderRouter::new(Arc::new(ArcSwap::from_pointee(executor)));

    // Oportunidad genuina: Volatilidad esperada de 200 bps (0.02) y confianza 80%
    // Edge bruto ~320 bps, fricciones ~5 bps -> viability ratio > 50x >> 3.0x
    let fertile_decision = TensorDecision {
        signal: SignalType::Long,
        net_confidence: 0.80,
        expected_volatility: 0.02,
        expected_lifetime_ms: 10_000,
    };

    let result = router
        .route_order(
            "BTCUSDT",
            &fertile_decision,
            50000.0,
            0.001,
            0.001,
            0.1,
            20,
            0.0001, // 1 bp spread
            "ORD_HIGH_VIABILITY_FERTILE_01",
        )
        .await;

    assert!(
        result.is_ok(),
        "Una orden con edge genuino superior a 3x fricciones DEBE rutear exitosamente"
    );
}
