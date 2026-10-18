//! Contrato F1-116: Suavizado continuo de LH-04 con Wilson LCB n >= 1.0 sin escalón artificial n=10.
//!
//! Verifica que:
//! 1. Con pocas operaciones (ej. n=6 < 10) donde hay evidencia positiva y estadísticamente válida,
//!    LH-04 no se apaga artificialmente por un umbral duro n=10, sino que la cota de Wilson LCB
//!    amortigua suave y conservadoramente la fracción de riesgo f > 0.
//! 2. La transición de f con respecto a n es monótona y continua (sin saltos discretos).
//! 3. El tope de margen se evalúa directamente por el axioma de ruina (clamp_ruin <= 0.25)
//!    sin ramas muertas.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn fixture() -> (std::sync::Arc<GlobalArena>, SignalIntent) {
    std::env::set_var("TG_GENOME_ENV", "backtest");
    update_registry(vec![SymbolSpec {
        symbol: "BTCUSDT".into(),
        step_size: 0.001,
        tick_size: 0.01,
        min_qty: 0.001,
        min_notional: 5.0,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }]);
    let arena = GlobalArena::build_in_own_stack(1000.0);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    (arena, intent)
}

#[test]
fn test_f1_116_lh04_activates_for_low_n_with_conservative_wilson() {
    let (arena, intent) = fixture();
    // 6 operaciones (n < 10, antes bloqueado por el escalón duro): WR = 0.75, PF = 8.0 (pago b ≈ 2.67)
    arena.coins[0].metrics.win_rate.store(0.75, Relaxed);
    arena.coins[0].metrics.trade_count.store(6, Relaxed);
    arena.coins[0].metrics.profit_factor.store(8.0, Relaxed);

    let mut engine = RiskEngine::new(1000.0);
    let _ = engine.evaluate_quantum_order(0, &intent, &arena);

    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert!(
        f.is_finite() && f > 0.0,
        "Con n=6 < 10 y evidencia estadistica positiva, f debe ser > 0, obtenido: {f}"
    );
    assert!(
        f <= 0.25,
        "f debe respetar el axioma de ruina (<= 0.25), obtenido: {f}"
    );

    let p_lcb = arena.registry.get_for_coin_or(0, "lh04_p_lcb", f64::NAN);
    assert!(
        p_lcb < 0.75,
        "Wilson LCB con muestra pequeña debe ser fuertemente conservador: {p_lcb} < 0.75"
    );
}

#[test]
fn test_f1_116_wilson_lcb_monotonicity_with_sample_size() {
    let (arena, intent) = fixture();
    let mut engine = RiskEngine::new(1000.0);

    // Mismo WR (0.70) y PF (2.5), pero variando n: 5 vs 20 vs 100
    arena.coins[0].metrics.win_rate.store(0.70, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.5, Relaxed);

    arena.coins[0].metrics.trade_count.store(5, Relaxed);
    let _ = engine.evaluate_quantum_order(0, &intent, &arena);
    let p_lcb_5 = arena.registry.get_for_coin_or(0, "lh04_p_lcb", f64::NAN);

    arena.coins[0].metrics.trade_count.store(20, Relaxed);
    let _ = engine.evaluate_quantum_order(0, &intent, &arena);
    let p_lcb_20 = arena.registry.get_for_coin_or(0, "lh04_p_lcb", f64::NAN);

    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    let _ = engine.evaluate_quantum_order(0, &intent, &arena);
    let p_lcb_100 = arena.registry.get_for_coin_or(0, "lh04_p_lcb", f64::NAN);

    assert!(
        p_lcb_5 < p_lcb_20 && p_lcb_20 < p_lcb_100,
        "Wilson LCB debe crecer suavemente con mayor evidencia: {} < {} < {}",
        p_lcb_5, p_lcb_20, p_lcb_100
    );
}
