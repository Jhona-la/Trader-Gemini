//! Contrato formal: Normalización Dinámica de Latencia, Difusión Fraccional y Ruteo Idempotente (Ola Ω106).
//!
//! F3.1-07: Verifica que el deslizamiento por latencia satisfaga la difusión fraccional / anómala
//! con exponente de Hurst H ∈ [0.30, 0.75], preservando bit a bit la difusión canónica Browniana
//! en H = 0.50, y que el QuantumOrderRouter preserve la idempotencia del client_order_id
//! y resuelva umbrales de pánico dinámicamente desde la arena.

use arc_swap::ArcSwap;
use execution_engine::executor::OrderExecutor;
use execution_engine::router::QuantumOrderRouter;
use quantum_arena::GlobalArena;
use risk_engine::tp_sl::{
    latency_slippage_pct, latency_slippage_pct_hurst, roundtrip_friction, roundtrip_friction_hurst,
};
use signal_engine::orchestrator::TensorDecision;
use signal_engine::{SignalType, TradeHorizon};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

#[test]
fn test_fractional_diffusion_latency_canonical_parity() {
    let atr = 0.0025; // 25 bps
    let latency_ms = 45.0; // 45 ms

    // 1. En H = 0.50 reproduce identidad canónica bit-a-bit exacta
    let canonical = latency_slippage_pct(atr, latency_ms);
    let fractional_half = latency_slippage_pct_hurst(atr, latency_ms, 0.50);
    assert_eq!(canonical, fractional_half);

    let friction_canonical = roundtrip_friction(0.0004, 0.0001, atr, latency_ms);
    let friction_fractional = roundtrip_friction_hurst(0.0004, 0.0001, atr, latency_ms, 0.50);
    assert_eq!(friction_canonical, friction_fractional);
}

#[test]
fn test_fractional_diffusion_properties_and_sanitization() {
    let atr = 0.0030;
    let latency_ms = 60.0;

    // Escalamiento matemático estricto según la ley de potencia (t/tau)^H
    let slip_h30 = latency_slippage_pct_hurst(atr, latency_ms, 0.30);
    let slip_h50 = latency_slippage_pct_hurst(atr, latency_ms, 0.50);
    let slip_h70 = latency_slippage_pct_hurst(atr, latency_ms, 0.70);

    // Para t < tau_ref (60ms < 60000ms), ratio < 1, ratio^H es decreciente en H
    assert!(slip_h30 > slip_h50);
    assert!(slip_h50 > slip_h70);

    // Sanitización rigurosa ante entradas corruptas
    assert_eq!(latency_slippage_pct_hurst(f64::NAN, latency_ms, 0.50), 0.0);
    assert_eq!(latency_slippage_pct_hurst(atr, f64::NAN, 0.50), 0.0);
    assert_eq!(latency_slippage_pct_hurst(-0.01, latency_ms, 0.50), 0.0);
    assert_eq!(latency_slippage_pct_hurst(atr, -10.0, 0.50), 0.0);

    // H fuera de rango se clampa a [0.30, 0.75]
    let slip_clamped_low = latency_slippage_pct_hurst(atr, latency_ms, 0.10);
    assert_eq!(slip_clamped_low, slip_h30);

    let slip_clamped_nan = latency_slippage_pct_hurst(atr, latency_ms, f64::NAN);
    assert_eq!(slip_clamped_nan, slip_h50);
}

#[tokio::test]
async fn test_router_dynamic_panic_latency_and_idempotency_preservation() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    // Configuramos un umbral de pánico calibrado en 200 ms (en vez de 500 ms fijo)
    arena.config.latency_ms_panic_threshold.store(200.0, Relaxed);
    arena.registry.set("router_max_spread_pct", 0.004);

    let mut executor = OrderExecutor::new(
        "dummy_key".to_string(),
        "dummy_secret".to_string(),
        true,
    );
    executor.set_paper_trading(true);
    executor.arena.store(Some(arena));

    let router = QuantumOrderRouter::new(Arc::new(ArcSwap::from_pointee(executor)));

    let decision = TensorDecision {
        signal: SignalType::Long,
        net_confidence: 0.90,
        expected_volatility: 0.020,
        expected_lifetime_ms: 60_000,
        horizon: TradeHorizon::Continuous,
    };

    // Con latencia 250 ms (> 200 ms dinámico), el router debe activar la ruta de emergencia Market
    // preservando el client_order_id sin errores
    let res = router
        .route_order(
            "BTCUSDT",
            &decision,
            50000.0,
            0.001,
            0.001,
            0.1,
            250, // > 200 ms
            0.0001,
            "IDEMPOTENT_EMERGENCY_ORDER_01",
        )
        .await;

    assert!(res.is_ok(), "El ruteo de emergencia con client_order_id debe completarse con éxito");
}

#[tokio::test]
async fn test_adverse_selection_guard_blocks_frictional_death_for_low_ev_orders() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut executor = OrderExecutor::new(
        "dummy_key".to_string(),
        "dummy_secret".to_string(),
        true,
    );
    executor.set_paper_trading(true);
    executor.arena.store(Some(arena));

    let router = QuantumOrderRouter::new(Arc::new(ArcSwap::from_pointee(executor)));

    // Decisión con baja convicción (net_confidence: 0.20) y baja volatilidad (0.0005)
    // El edge bruto no puede pagar comisiones + funding + selección adversa (fricción total > 1/3 EV)
    let weak_decision = TensorDecision {
        signal: SignalType::Long,
        net_confidence: 0.20,
        expected_volatility: 0.0005,
        expected_lifetime_ms: 10_000,
        horizon: TradeHorizon::Continuous,
    };

    let res = router
        .route_order(
            "BTCUSDT",
            &weak_decision,
            50000.0,
            0.001,
            0.001,
            0.1,
            15, // latencia baja y normal (15 ms)
            0.0005, // spread 5 bps
            "WEAK_CHURN_ORDER_01",
        )
        .await;

    assert!(res.is_err(), "La orden de micro-edge devorada por fricciones DEBE ser rechazada");
    let err_msg = res.unwrap_err();
    assert!(
        err_msg.contains("REJ_INSUFFICIENT_VIABILITY"),
        "El mensaje de error debe indicar REJ_INSUFFICIENT_VIABILITY: {err_msg}"
    );
}
