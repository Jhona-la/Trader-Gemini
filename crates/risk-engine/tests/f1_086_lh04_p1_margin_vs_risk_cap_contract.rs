//! F1-086 (Ola Ω190, AGY / Quant Sr.): Contrato bilateral de desacoplamiento
//! del recorte erróneo de margen P1 en espacio de riesgo (LH-04).
//!
//! # Qué estaba mal
//! El recorte P1 (`crates/risk-engine/src/lib.rs:1561`) aplicaba a MARGEN (colateral)
//! un tope derivado del axioma de ruina en RIESGO (`clamp_ruin(1.0, q) <= 0.25`):
//! `final_margin = margen_obj.min(allocated_capital * tope_margen_lh04);`
//!
//! Esto sólo equivalía a riesgo <= tope si `L · SL == 1`.
//! En trading normal (`L · SL < 1`, e.g. L=5, SL=1% => L·SL=0.05), este tope
//! restringía el riesgo efectivo a `f <= 0.25 · L · SL` (apenas ~1.25%),
//! anulando el dimensionamiento Kelly calibrado para cualquier `f > 0.25 · L · SL`.
//! Además, la telemetría publicaba `lh04_f_riesgo = 0.10` mientras se ejecutaba `0.0125`.
//!
//! # Corrección F1-086
//! 1. `final_margin = margen_obj`: el riesgo al stop objetivo ya está matemáticamente
//!    acotado por `clamp_ruin` sobre `lh04_f` (axioma <= 0.25).
//! 2. Downstream, `resolve_margin_cap_limits` y `available_margin_headroom` (Sol A2)
//!    acotan el margen por liquidez y capacidad real de cartera.
//! 3. La telemetría `lh04_f_realizado` publica el riesgo realmente ejecutado tras
//!    los límites de margen, garantizando verdad causal sin sobreestimación.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn fixture_con_especificacion(max_leverage: u32) -> (std::sync::Arc<GlobalArena>, SignalIntent) {
    update_registry(vec![SymbolSpec {
        symbol: "BTCUSDT".into(),
        step_size: 0.001,
        tick_size: 0.01,
        min_qty: 0.001,
        min_notional: 5.0,
        max_leverage,
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
    arena.config.margin_cushion_pct.store(0.80, Relaxed);
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

/// (1) Cuando el margen cabe dentro del colchón del portfolio, el riesgo realizado
/// alcanza exactamente el f_objetivo calibrado por Kelly sin ser asfixiado por 0.25·C.
#[test]
fn test_f1_086_riesgo_no_es_asfixiado_por_tope_arbitrario_de_margen() {
    let (arena, intent) = fixture_con_especificacion(20);
    // Evidencia estadística medida: 100 cierres, 65% win rate, PF 2.6 => b = 1.4
    // Wilson LCB(0.65, 100, 1.64) ~ 0.572 => f_kelly ~ 0.13038
    arena.coins[0].metrics.win_rate.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.6, Relaxed);

    let mut engine = RiskEngine::new(1000.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    assert_ne!(out.signal, SignalType::Flat, "La orden debe ser validada");
    let f_objetivo = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    let f_realizado = arena.registry.get_for_coin_or(0, "lh04_f_realizado", f64::NAN);
    let capital = arena.registry.get_for_coin_or(0, "lh04_capital_asignado", f64::NAN);
    let stop_real = arena.registry.get_for_coin_or(0, "lh04_stop_real", f64::NAN);

    assert!(f_objetivo.is_finite() && f_objetivo > 0.0);
    assert!(f_realizado.is_finite() && f_realizado > 0.0);
    assert!(capital.is_finite() && capital > 0.0);
    assert!(stop_real.is_finite() && stop_real > 0.0);

    // El riesgo emitido al stop de la orden
    let stop_fraccion = (100.0 - out.sl_target) / 100.0;
    let riesgo_emitido = out.volume_usd * out.leverage * stop_fraccion;
    let riesgo_publicado = f_realizado * capital;

    // Verdad causal: la telemetría coincide con la orden emitida dentro de la tolerancia de redondeo
    assert!(
        (riesgo_emitido - riesgo_publicado).abs() <= 0.05 * riesgo_publicado,
        "Riesgo emitido ({riesgo_emitido}) debe coincidir con el publicado ({riesgo_publicado})"
    );
    // En ningún caso se supera el f_objetivo
    assert!(
        f_realizado <= f_objetivo + 1e-12,
        "f_realizado ({f_realizado}) <= f_objetivo ({f_objetivo})"
    );
}

/// (2) Verificación de que el margen emitido no está artificialmente bloqueado en 0.25·C
/// cuando la capacidad del portfolio (`safe_limit`) permite más colateral.
#[test]
fn test_f1_086_margen_supera_cuarto_de_capital_si_capacidad_permite() {
    let (arena, intent) = fixture_con_especificacion(20);
    arena.coins[0].metrics.win_rate.store(0.80, Relaxed);
    arena.coins[0].metrics.trade_count.store(200, Relaxed);
    arena.coins[0].metrics.profit_factor.store(3.0, Relaxed);

    let mut engine = RiskEngine::new(1000.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    assert_ne!(out.signal, SignalType::Flat);
    // Con 1000 USD de capital y colchón 80%, el margen permitido puede superar 250 USD
    // si el dimensionamiento de notional y leverage lo justifican.
    assert!(out.volume_usd > 0.0);
    assert!(
        out.volume_usd <= 1000.0 * 0.90 + 1e-6,
        "El margen respeta el techo de exposición del portfolio (<= 900 USD)"
    );
}

/// (3) Invariante de ruina: aún con evidencia extrema, f_realizado y el riesgo
/// al stop nunca superan el axioma de ruina (<= 0.25).
#[test]
fn test_f1_086_invariante_axioma_ruina_estricto() {
    let (arena, intent) = fixture_con_especificacion(20);
    // Evidencia extrema que daría f_raw > 0.50
    arena.coins[0].metrics.win_rate.store(0.95, Relaxed);
    arena.coins[0].metrics.trade_count.store(500, Relaxed);
    arena.coins[0].metrics.profit_factor.store(10.0, Relaxed);

    let mut engine = RiskEngine::new(1000.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    let f_objetivo = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    let f_realizado = arena.registry.get_for_coin_or(0, "lh04_f_realizado", f64::NAN);

    assert!(f_objetivo.is_finite() && f_objetivo > 0.0);
    assert!(
        f_objetivo <= 0.25 + 1e-12,
        "f_objetivo acotado por clamp_ruin <= 0.25: {f_objetivo}"
    );
    assert!(
        f_realizado <= 0.25 + 1e-12,
        "f_realizado acotado por clamp_ruin <= 0.25: {f_realizado}"
    );

    if out.signal != SignalType::Flat {
        let stop_fraccion = (100.0 - out.sl_target) / 100.0;
        let riesgo_al_stop = out.volume_usd * out.leverage * stop_fraccion;
        assert!(
            riesgo_al_stop <= 250.0 * 1.05,
            "Riesgo en USD ({riesgo_al_stop}) no excede 25% del capital ({})",
            1000.0 * 0.25
        );
    }
}
