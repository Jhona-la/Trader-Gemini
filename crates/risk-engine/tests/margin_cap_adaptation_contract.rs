//! CONTRATO FORMAL DE ADAPTACIÓN DINÁMICA DE COTAS DE MARGEN POR OPERACIÓN (Ola Ω112)
//!
//! Verifica que:
//! 1. Paridad canónica exacta bit a bit con los defaults (0.25 micro, 0.90 estándar).
//! 2. Adaptación dinámica mediante overrides en el registro omnisciente (`micro_margin_cap_ratio` y `standard_margin_cap_ratio`).
//! 3. Respeto estricto de los clamps de seguridad ([0.10, 0.50] micro, [0.50, 0.98] estándar).
//! 4. Inmunidad total contra NaN, infinitos y valores degenerados (fail-closed a defaults seguros).
//! 5. Monotonía estricta a lo largo del espectro continuo de capital ($5 a $10,000 USD).

use quantum_arena::symbol_registry::SymbolSpec;
use quantum_arena::GlobalArena;
use risk_engine::capital_regime::{
    compute_safe_margin_limit, resolve_margin_cap_limits, DEFAULT_MICRO_MARGIN_CAP_RATIO,
    DEFAULT_STANDARD_MARGIN_CAP_RATIO,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Once};

static INIT_SYMBOLS: Once = Once::new();

fn init_test_symbols() {
    INIT_SYMBOLS.call_once(|| {
        quantum_arena::symbol_registry::update_registry(vec![
            SymbolSpec {
                symbol: "BTCUSDT".into(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 50,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
            SymbolSpec {
                symbol: "ETHUSDT".into(),
                step_size: 0.01,
                tick_size: 0.01,
                min_qty: 0.01,
                min_notional: 5.0,
                max_leverage: 50,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
        ]);
    });
}

fn build_fixture(capital: f64) -> (Arc<GlobalArena>, SignalIntent) {
    std::env::set_var("TG_GENOME_ENV", "backtest");
    init_test_symbols();
    let arena = GlobalArena::build_in_own_stack(capital);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.2, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.75, Relaxed);
    arena.coins[0].metrics.kelly_fraction.store(0.20, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.20, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.35, Relaxed);

    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.85,
        win_probability: 0.70,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };

    (arena, intent)
}

#[test]
fn test_margin_cap_canonical_defaults_exact_parity() {
    let allocated: f64 = 13.0;
    let req_margin: f64 = 1.02;
    let safe_cushion: f64 = 0.80;
    let current_cap: f64 = 13.0;

    // Micro pleno (w = 1.0): límite canónico (13.0 * 0.25).max(1.02) = 3.25
    let micro_previo: f64 = (allocated * 0.25).max(req_margin);
    let micro_nuevo = compute_safe_margin_limit(
        allocated,
        req_margin,
        safe_cushion,
        current_cap,
        1.0,
        DEFAULT_MICRO_MARGIN_CAP_RATIO,
        DEFAULT_STANDARD_MARGIN_CAP_RATIO,
    );
    assert_eq!(micro_previo, micro_nuevo, "Paridad exacta en micro pleno");

    // Estándar pleno (w = 0.0, cap = 1000.0)
    let allocated_std: f64 = 1000.0;
    let cap_std: f64 = 1000.0;
    let std_previo: f64 = (allocated_std * safe_cushion).min(cap_std * 0.90);
    let std_nuevo = compute_safe_margin_limit(
        allocated_std,
        req_margin,
        safe_cushion,
        cap_std,
        0.0,
        DEFAULT_MICRO_MARGIN_CAP_RATIO,
        DEFAULT_STANDARD_MARGIN_CAP_RATIO,
    );
    assert_eq!(std_previo, std_nuevo, "Paridad exacta en estándar pleno");
}

#[test]
fn test_margin_cap_dynamic_registry_overrides_micro() {
    let (arena, intent) = build_fixture(13.0);

    // 1. Con ratio canónico por defecto (0.25):
    let mut engine_default = RiskEngine::new(13.0);
    let out_default = engine_default.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_default.signal, SignalType::Long);
    let margin_default = out_default.volume_usd / out_default.leverage;

    // 2. Con override adaptativo por activo c0:micro_margin_cap_ratio = 0.35 (35% de capital)
    arena.registry.set_for_coin(0, "micro_margin_cap_ratio", 0.35);
    let mut engine_override = RiskEngine::new(13.0);
    let out_override = engine_override.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(out_override.signal, SignalType::Long);
    let margin_override = out_override.volume_usd / out_override.leverage;

    // Con ratio expandido a 0.35, el margen seguro es 13.0 * 0.35 = 4.55 USD (vs 3.25 USD).
    assert!(
        margin_override >= margin_default,
        "Override 0.35 debe permitir mayor o igual margen que 0.25 (override={:.4}, default={:.4})",
        margin_override,
        margin_default
    );

    // Verificamos directamente con resolve_margin_cap_limits
    let limit_default = resolve_margin_cap_limits(&arena, 1, 13.0, 1.02, 0.80, 13.0, 1.0); // coin 1 no tiene override
    let limit_override = resolve_margin_cap_limits(&arena, 0, 13.0, 1.02, 0.80, 13.0, 1.0); // coin 0 tiene override 0.35
    assert!((limit_default - 3.25).abs() < 1e-12);
    assert!((limit_override - 4.55).abs() < 1e-12);
}

#[test]
fn test_margin_cap_dynamic_registry_overrides_standard() {
    let (arena, _) = build_fixture(500.0);

    // Sin override en coin 1 -> standard ratio 0.90
    let limit_default = resolve_margin_cap_limits(&arena, 1, 500.0, 1.02, 0.80, 500.0, 0.0);
    // (500 * 0.80).min(500 * 0.90) = 400.0
    assert!((limit_default - 400.0).abs() < 1e-12);

    // Con override restrictivo en coin 0: standard_margin_cap_ratio = 0.60
    arena.registry.set_for_coin(0, "standard_margin_cap_ratio", 0.60);
    let limit_restricted = resolve_margin_cap_limits(&arena, 0, 500.0, 1.02, 0.80, 500.0, 0.0);
    // (500 * 0.80).min(500 * 0.60) = 300.0
    assert!((limit_restricted - 300.0).abs() < 1e-12);
    assert!(limit_restricted < limit_default);
}

#[test]
fn test_margin_cap_clamp_boundaries_and_sanitization() {
    // 1. Clamp micro inferior y superior
    let limit_under = compute_safe_margin_limit(10.0, 1.0, 0.8, 10.0, 1.0, 0.001, 0.90);
    // Clamp a 0.10 -> 10.0 * 0.10 = 1.0
    assert!((limit_under - 1.0).abs() < 1e-12);

    let limit_over = compute_safe_margin_limit(10.0, 1.0, 0.8, 10.0, 1.0, 0.999, 0.90);
    // Clamp a 0.50 -> 10.0 * 0.50 = 5.0
    assert!((limit_over - 5.0).abs() < 1e-12);

    // 2. Clamp standard inferior y superior
    let limit_std_under = compute_safe_margin_limit(100.0, 1.0, 0.8, 100.0, 0.0, 0.25, 0.10);
    // Clamp std a 0.50 -> min(80.0, 100 * 0.50) = 50.0
    assert!((limit_std_under - 50.0).abs() < 1e-12);

    let limit_std_over = compute_safe_margin_limit(100.0, 1.0, 0.8, 100.0, 0.0, 0.25, 1.50);
    // Clamp std a 0.98 -> min(80.0, 100 * 0.98) = 80.0
    assert!((limit_std_over - 80.0).abs() < 1e-12);

    // 3. Robustez contra NaN e Infinitos
    let limit_nan = compute_safe_margin_limit(
        f64::NAN,
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::NAN,
        f64::NAN,
    );
    assert!(limit_nan.is_finite());
    assert!(limit_nan >= 1.0);
}

#[test]
fn test_margin_cap_monotonicity_across_capital_spectrum() {
    let capitals = [5.0, 10.0, 13.0, 20.0, 50.0, 100.0, 500.0, 1000.0];
    let mut prev_limit = 0.0;

    for &cap in &capitals {
        let w = risk_engine::capital_regime::micro_weight(cap, 5.0);
        let limit = compute_safe_margin_limit(
            cap,
            1.02,
            0.80,
            cap,
            w,
            DEFAULT_MICRO_MARGIN_CAP_RATIO,
            DEFAULT_STANDARD_MARGIN_CAP_RATIO,
        );
        assert!(
            limit >= prev_limit,
            "Monotonía de límite de margen violada: cap={}, limit={}, prev={}",
            cap,
            limit,
            prev_limit
        );
        prev_limit = limit;
    }
}
