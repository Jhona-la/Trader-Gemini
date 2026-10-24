//! Contrato F1-067 (Ola Ω173): Telemetría causal de liquidaciones conectada a Binance.
//!
//! Verifica que:
//! 1. Sin observaciones de liquidación, `liquidation_severity_at(0, as_of_ms).ok().flatten().unwrap_or(0.0)`
//!    produce 0.0 de forma determinista y sin fallos.
//! 2. Al recibir eventos causales de liquidación de Binance (!forceOrder@arr), la severidad
//!    se actualiza exactamente y decae de forma continua conforme a la física del tensor.
//! 3. La consulta de telemetría es totalmente segura: nunca produce NaN ni valores negativos.

use god_engine_core::{GodEngineCore, liquidation_feed::LiquidationObservation};
use quantum_arena::{GlobalArena, symbol_registry};
use std::sync::{Arc, Mutex};

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn create_test_core() -> (Arc<GlobalArena>, GodEngineCore) {
    quantum_arena::symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "ETHUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("BTCUSDT"),
        symbol_registry::get_official_binance_spec("ETHUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    (arena, core)
}

fn make_liquidation(symbol: &str, time: u64, notional: f64) -> LiquidationObservation {
    LiquidationObservation {
        symbol: symbol.into(),
        event_time_ms: time,
        trade_time_ms: time.saturating_sub(1),
        is_buy: false,
        reported_filled_notional: notional,
    }
}

#[test]
fn test_f1_067_telemetry_feed_causal_binance_liquidation_contract() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_arena, mut core) = create_test_core();

    // 1. Estado inicial sin liquidaciones: telemetría honesta en 0.0
    let initial_severity = core.liquidation_severity_at(0, 1000).ok().flatten().unwrap_or(0.0);
    assert_eq!(initial_severity, 0.0, "La severidad inicial debe ser 0.0");

    // 2. Ingesta de observación causal de Binance (!forceOrder@arr)
    let obs = make_liquidation("BTCUSDT", 1000, 1_000_000.0);
    let update_res = core.observe_liquidation(obs);
    assert!(update_res.is_ok(), "Debe aceptar la observación de Binance");

    // 3. Severidad inmediatamente posterior al evento
    let at_event = core.liquidation_severity_at(0, 1000).ok().flatten().unwrap_or(0.0);
    assert_eq!(at_event, 1.0, "Severidad en t_evento debe ser 1.0 para notional saturado");

    // 4. Decaimiento físico exponencial continuo a t = 11000 (half-life = 10000 ms)
    let at_half_life = core.liquidation_severity_at(0, 11000).ok().flatten().unwrap_or(0.0);
    assert!((at_half_life - 0.5).abs() < 1e-4, "Severidad tras half-life debe ser ~0.5, fue {}", at_half_life);

    // 5. Inmunidad matemática: finitud garantizada en todo instante
    for dt in [0, 100, 5000, 10000, 60000, 3600000] {
        let val = core.liquidation_severity_at(0, 1000 + dt).ok().flatten().unwrap_or(0.0);
        assert!(val.is_finite() && val >= 0.0 && val <= 1.0, "Valor fuera de dominio: {}", val);
    }
}
