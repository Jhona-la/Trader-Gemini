//! Contrato de Certificación — F3.4-08 (Ola Ω110): Modulación Adaptativa Continua de Exploración Kelly
//!
//! Verifica:
//! 1. Paridad canónica bit a bit en arranque en frío (trades_n == 0, current == peak, sin overrides).
//! 2. Rampa continua de Profit Factor (pf ∈ [0.5, 1.0]) que atenúa suavemente y apaga la exploración en pf <= 0.50.
//! 3. Protección por drawdown de cartera que contrae proporcionalmente el apetito de exploración.
//! 4. Overrides por activo (c{id}:kelly_exploration_*) y globales desde el registro omnisciente lock-free.
//! 5. Bounding adaptativo cuando PF > 1.0 con soporte de overrides en registro.

use quantum_arena::GlobalArena;
use risk_engine::{
    compute_kelly_exploration_fraction, resolve_kelly_cold, DEFAULT_KELLY_EXPLORATION_MAX,
    DEFAULT_KELLY_EXPLORATION_RATIO, DRAWDOWN_EXPLORATION_FLOOR,
};

#[test]
fn test_cold_start_canonical_parity_bit_exact() {
    let clamp_mins: [f64; 6] = [0.005, 0.01, 0.04, 0.10, 0.20, 0.40];
    for &cmin in &clamp_mins {
        let canon = (cmin * 0.25f64).clamp(0.0, 0.05);
        let computed = compute_kelly_exploration_fraction(
            cmin,
            1.0, // pf neutral desconocido
            0,   // trades_n == 0
            13.0,
            13.0,
            DEFAULT_KELLY_EXPLORATION_RATIO,
            DEFAULT_KELLY_EXPLORATION_MAX,
        );
        assert_eq!(
            computed, canon,
            "Fallo de paridad canónica bit a bit para clamp_min={cmin}"
        );
    }
}

#[test]
fn test_pf_ramp_continuous_decay_on_losing_asset() {
    let cmin = 0.10;
    let trades_n = 10;
    let peak = 50.0;
    let current = 50.0;

    // PF = 1.0 -> pf_ramp = 1.0
    let f_100 = compute_kelly_exploration_fraction(
        cmin, 1.0, trades_n, peak, current, 0.25, 0.05,
    );
    assert_eq!(f_100, 0.025);

    // PF = 0.75 -> pf_ramp = 0.50
    let f_075 = compute_kelly_exploration_fraction(
        cmin, 0.75, trades_n, peak, current, 0.25, 0.05,
    );
    assert_eq!(f_075, 0.0125);

    // PF = 0.50 -> pf_ramp = 0.0
    let f_050 = compute_kelly_exploration_fraction(
        cmin, 0.50, trades_n, peak, current, 0.25, 0.05,
    );
    assert_eq!(f_050, 0.0);

    // PF = 0.20 -> pf_ramp = 0.0 (apagado para activos con pérdidas severas)
    let f_020 = compute_kelly_exploration_fraction(
        cmin, 0.20, trades_n, peak, current, 0.25, 0.05,
    );
    assert_eq!(f_020, 0.0);

    assert!(f_100 > f_075);
    assert!(f_075 > f_050);
}

#[test]
fn test_portfolio_drawdown_attenuates_exploration() {
    let cmin = 0.10;
    let trades_n = 0;
    let peak = 100.0;

    // Sin drawdown (current = 100.0) -> dd_factor = 1.0
    let f_no_dd = compute_kelly_exploration_fraction(
        cmin, 1.0, trades_n, peak, 100.0, 0.25, 0.05,
    );
    assert_eq!(f_no_dd, 0.025);

    // Drawdown moderado de 20% (current = 80.0) -> dd_factor = 0.80
    let f_dd_20 = compute_kelly_exploration_fraction(
        cmin, 1.0, trades_n, peak, 80.0, 0.25, 0.05,
    );
    assert!((f_dd_20 - 0.025 * 0.80).abs() < 1e-12);

    // Drawdown severo de 90% (current = 10.0) -> dd_factor clamped en floor 0.20
    let f_dd_90 = compute_kelly_exploration_fraction(
        cmin, 1.0, trades_n, peak, 10.0, 0.25, 0.05,
    );
    assert!((f_dd_90 - 0.025 * DRAWDOWN_EXPLORATION_FLOOR).abs() < 1e-12);
}

#[test]
fn test_omniscient_registry_per_coin_and_global_overrides() {
    let arena = GlobalArena::new(100.0);
    let cmin = 0.10;
    let trades_n = 0;

    // Baseline sin overrides: gamma = 0.25, max_cap = 0.05
    let f_base = resolve_kelly_cold(
        &arena, 0, 1.0, cmin, 0.20, 100.0, 100.0, trades_n,
    );
    assert_eq!(f_base, 0.025);

    // Override global de ratio a 0.50
    arena.registry.set("kelly_exploration_ratio", 0.50);
    let f_global = resolve_kelly_cold(
        &arena, 0, 1.0, cmin, 0.20, 100.0, 100.0, trades_n,
    );
    assert_eq!(f_global, 0.05); // 0.10 * 0.50 = 0.05

    // Override por moneda para coin 1 con ratio 0.80 y techo expandido 0.15
    arena.registry.set("c1:kelly_exploration_ratio", 0.80);
    arena.registry.set("c1:kelly_exploration_max", 0.15);

    let f_coin1 = resolve_kelly_cold(
        &arena, 1, 1.0, cmin, 0.20, 100.0, 100.0, trades_n,
    );
    assert!((f_coin1 - 0.08).abs() < 1e-12, "0.10 * 0.80 = 0.08");

    // Coin 0 conserva el global
    let f_coin0 = resolve_kelly_cold(
        &arena, 0, 1.0, cmin, 0.20, 100.0, 100.0, trades_n,
    );
    assert!((f_coin0 - 0.05).abs() < 1e-12, "0.10 * 0.50 = 0.05");
}

#[test]
fn test_bootstrap_cold_when_pf_greater_than_one() {
    let arena = GlobalArena::new(100.0);
    let cmin = 0.01;
    let trades_n = 0;

    // Con PF > 1.0: usa kelly_cold_raw clamped a [0.05, 0.35]
    let f_high_raw = resolve_kelly_cold(
        &arena, 0, 1.5, cmin, 0.50, 100.0, 100.0, trades_n,
    );
    assert_eq!(f_high_raw, 0.35);

    let f_low_raw = resolve_kelly_cold(
        &arena, 0, 1.5, cmin, 0.02, 100.0, 100.0, trades_n,
    );
    assert_eq!(f_low_raw, 0.05);

    // Override de límites de bootstrap en registro
    arena.registry.set("kelly_bootstrap_cold_max", 0.45);
    let f_overridden = resolve_kelly_cold(
        &arena, 0, 1.5, cmin, 0.50, 100.0, 100.0, trades_n,
    );
    assert_eq!(f_overridden, 0.45);
}
