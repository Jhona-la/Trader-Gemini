//! Contrato formal: Calibración Secuencial Adaptativa de Sonda y SPRT de Wald (Ola Ω107).
//!
//! F3.4-05: Verifica que la fase de sonda resuelva su techo dinámicamente desde el
//! registro omnisciente (por moneda o global) preservando el valor canónico de 5 cierres
//! por defecto, y que la prueba secuencial de razón de verosimilitudes (SPRT de Wald)
//! permita el egreso anticipado de la sonda ante evidencia estadística concluyente de ventaja.

use quantum_arena::GlobalArena;
use risk_engine::sonda::{
    admite_sonda, en_fase_de_sonda, hay_sonda_abierta, resolve_max_cierres_sonda,
    sprt_bernoulli, sprt_evalua_salida_sonda, SprtDecision, CIERRES_DE_SONDA,
};
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn test_sprt_bernoulli_analytical_properties() {
    let p0 = 0.50;
    let p1 = 0.70;
    let alpha = 0.05;
    let beta = 0.10;

    // 1. n = 0 o entradas degeneradas => ContinuarMuestreo
    assert_eq!(
        sprt_bernoulli(0, 0, p0, p1, alpha, beta),
        SprtDecision::ContinuarMuestreo
    );
    assert_eq!(
        sprt_bernoulli(5, 6, p0, p1, alpha, beta), // k > n
        SprtDecision::ContinuarMuestreo
    );

    // 2. Muestreo intermedio indecidible
    // n = 4, k = 2 (50% aciertos, cerca del nulo pero sin potencia para rechazar)
    assert_eq!(
        sprt_bernoulli(4, 2, p0, p1, alpha, beta),
        SprtDecision::ContinuarMuestreo
    );

    // 3. Ventaja concluyente (H1 aceptada):
    // n = 15, k = 14 (93.3% aciertos): fuerte evidencia de ventaja
    assert_eq!(
        sprt_bernoulli(15, 14, p0, p1, alpha, beta),
        SprtDecision::VentajaDemostrada
    );

    // 4. Sin ventaja (H0 aceptada):
    // n = 15, k = 3 (20% aciertos): rechaza la hipótesis de edge alternativo
    assert_eq!(
        sprt_bernoulli(15, 3, p0, p1, alpha, beta),
        SprtDecision::SinVentaja
    );
}

#[test]
fn test_resolve_max_cierres_sonda_hierarchy_and_sanitization() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // 1. Por defecto sin claves => CIERRES_DE_SONDA (5)
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), CIERRES_DE_SONDA);
    assert_eq!(CIERRES_DE_SONDA, 5);

    // 2. Override global en registro
    arena.registry.set("sonda_max_cierres", 8.0);
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), 8);
    assert_eq!(resolve_max_cierres_sonda(&arena, 1), 8);

    // 3. Override específico por moneda prevalece sobre el global
    arena.registry.set("c0:sonda_max_cierres", 3.0);
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), 3);
    assert_eq!(resolve_max_cierres_sonda(&arena, 1), 8); // Moneda 1 sigue global

    // 4. Sanitización contra valores degenerados o no finitos
    arena.registry.set("c0:sonda_max_cierres", -10.0);
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), CIERRES_DE_SONDA);

    arena.registry.set("c0:sonda_max_cierres", f64::NAN);
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), CIERRES_DE_SONDA);

    arena.registry.set("c0:sonda_max_cierres", 100.0); // Clampa a 50 máximo
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), 50);
}

#[test]
fn test_en_fase_de_sonda_canonical_and_dynamic() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    // Caso A: Moneda sin trades (n=0) => en fase de sonda
    coin.metrics.trade_count.store(0, Relaxed);
    assert!(en_fase_de_sonda(&arena, 0));

    // Caso B: Moneda con 4 trades (< 5 canónico, sin SPRT trigger) => en fase de sonda
    coin.metrics.trade_count.store(4, Relaxed);
    coin.metrics.win_rate.store(0.50, Relaxed);
    assert!(en_fase_de_sonda(&arena, 0));

    // Caso C: Moneda con 5 trades (>= 5 canónico) => fuera de fase de sonda
    coin.metrics.trade_count.store(5, Relaxed);
    assert!(!en_fase_de_sonda(&arena, 0));

    // Caso D: Con override dinámico a 3 cierres: en n=3 ya está fuera
    arena.registry.set("c0:sonda_max_cierres", 3.0);
    coin.metrics.trade_count.store(3, Relaxed);
    assert!(!en_fase_de_sonda(&arena, 0));

    // En n=2 con override a 3 cierres sigue en sonda
    coin.metrics.trade_count.store(2, Relaxed);
    coin.metrics.win_rate.store(0.50, Relaxed);
    assert!(en_fase_de_sonda(&arena, 0));
}

#[test]
fn test_sprt_early_exit_from_probe_phase() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    // Calibramos SPRT en la arena con alta sensibilidad para micro-scalping
    arena.registry.set("sonda_sprt_enabled", 1.0);
    arena.registry.set("sonda_sprt_p0", 0.50);
    arena.registry.set("sonda_sprt_p1", 0.85);
    arena.registry.set("sonda_sprt_alpha", 0.10);
    arena.registry.set("sonda_sprt_beta", 0.20);
    // bound_b = ln((1 - 0.20) / 0.10) = ln(8) ~= 2.079

    // En n=4 cierres con 100% win rate (k=4):
    // llr = 4 * ln(0.85 / 0.50) = 4 * 0.5306 = 2.122 >= 2.079 => VentajaDemostrada!
    coin.metrics.trade_count.store(4, Relaxed);
    coin.metrics.win_rate.store(1.0, Relaxed);

    assert!(sprt_evalua_salida_sonda(&arena, 0));
    // Aunque n=4 < 5 cierres, egresa de la fase de sonda por prueba de hipótesis concluyente
    assert!(!en_fase_de_sonda(&arena, 0));

    // Si SPRT se desactiva explícitamente en el registro, la moneda debe respetar el conteo rígido
    arena.registry.set("sonda_sprt_enabled", 0.0);
    assert!(!sprt_evalua_salida_sonda(&arena, 0));
    assert!(en_fase_de_sonda(&arena, 0)); // Vuelve a estar en sonda porque n=4 < 5
}

#[test]
fn test_portfolio_probe_management_and_admission() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // Moneda 0 sin historial (sonda)
    arena.coins[0].metrics.trade_count.store(0, Relaxed);
    // Moneda 1 sin historial (sonda)
    arena.coins[1].metrics.trade_count.store(0, Relaxed);

    // 1. Sin posiciones abiertas: admite sonda
    assert!(!hay_sonda_abierta(&arena));
    assert!(admite_sonda(&arena, 0));
    assert!(admite_sonda(&arena, 1));

    // 2. Abrimos posición en moneda 0 (que está en sonda)
    arena.coins[0].positions.slots()[0].open(true, 100.0, 0.01, 0.05, 1_000, 102.0, 99.0);
    assert!(hay_sonda_abierta(&arena));

    // Moneda 0 ya tiene una sonda abierta en cartera => no admite otra
    assert!(!admite_sonda(&arena, 0));
    // Moneda 1 tampoco puede abrir sonda simultánea (regla D4: una sola sonda en cartera)
    assert!(!admite_sonda(&arena, 1));

    // 3. Moneda con historial (n=10 trades) está fuera de sonda => admite orden independientemente
    arena.coins[2].metrics.trade_count.store(10, Relaxed);
    assert!(!en_fase_de_sonda(&arena, 2));
    assert!(admite_sonda(&arena, 2));
}

#[test]
fn test_sprt_runtime_upper_boundary_preserves_decisions_without_panicking() {
    // WR=1 is an algebraic endpoint fixture, not measured Bernoulli evidence
    // or a validation of the wrapper's reconstructed count/Wald error rates.
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    arena.registry.set("sonda_sprt_enabled", 1.0);
    arena.registry.set("sonda_sprt_p0", 0.90);
    arena.registry.set("sonda_sprt_p1", 0.70);
    coin.metrics.win_rate.store(1.0, Relaxed);

    // Two wins are insufficient at p0=0.90, p1=0.95: LLR ~=0.108 < ln(18).
    coin.metrics.trade_count.store(2, Relaxed);
    assert!(!sprt_evalua_salida_sonda(&arena, 0));
    assert!(en_fase_de_sonda(&arena, 0));
    assert_eq!(coin.metrics.trade_count.load(Relaxed), 2);

    // 100 wins cross the same SPRT boundary: LLR ~=5.407 > ln(18).
    // Call the SPRT API directly: the separate closure quota is already met.
    coin.metrics.trade_count.store(100, Relaxed);
    assert_eq!(
        sprt_bernoulli(100, 100, 0.90, 0.95, 0.05, 0.10),
        SprtDecision::VentajaDemostrada
    );
    assert!(sprt_evalua_salida_sonda(&arena, 0));
    assert_eq!(coin.metrics.trade_count.load(Relaxed), 100);
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), CIERRES_DE_SONDA);
}

#[test]
fn test_sprt_runtime_neighbors_of_upper_probability_bounds_preserve_decisions() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    arena.registry.set("sonda_sprt_enabled", 1.0);
    coin.metrics.win_rate.store(1.0, Relaxed);
    let p0_edge = 0.90_f64.to_bits();
    let p1_edge = 0.95_f64.to_bits();

    for p0_raw in [
        f64::from_bits(p0_edge - 1),
        0.90,
        f64::from_bits(p0_edge + 1),
    ] {
        for p1_raw in [
            f64::from_bits(p1_edge - 1),
            0.95,
            f64::from_bits(p1_edge + 1),
        ] {
            arena.registry.set("sonda_sprt_p0", p0_raw);
            arena.registry.set("sonda_sprt_p1", p1_raw);
            coin.metrics.trade_count.store(2, Relaxed);
            assert!(!sprt_evalua_salida_sonda(&arena, 0));
            assert!(en_fase_de_sonda(&arena, 0));
            coin.metrics.trade_count.store(100, Relaxed);
            assert!(sprt_evalua_salida_sonda(&arena, 0));
            assert_eq!(coin.metrics.trade_count.load(Relaxed), 100);
        }
    }
}

#[test]
fn test_sprt_nonfinite_registry_parameters_preserve_default_fallback_and_quota() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    arena.registry.set("sonda_sprt_enabled", 1.0);
    arena.registry.set("sonda_max_cierres", 50.0);
    let knobs = [
        ("sonda_sprt_p0", 0.50),
        ("sonda_sprt_p1", 0.70),
        ("sonda_sprt_alpha", 0.05),
        ("sonda_sprt_beta", 0.10),
    ];
    for (key, value) in knobs {
        arena.registry.set(key, value);
    }
    // Algebraic WR=1 fixture; the registry getter resolves invalid knobs to
    // their defaults before the SPRT primitive receives them.
    coin.metrics.win_rate.store(1.0, Relaxed);
    for n in [2, 5, 8, 9, 49] {
        coin.metrics.trade_count.store(n, Relaxed);
        let default_exit = sprt_evalua_salida_sonda(&arena, 0);
        let default_phase = en_fase_de_sonda(&arena, 0);
        assert_eq!(
            default_exit,
            sprt_bernoulli(n, n, 0.50, 0.70, 0.05, 0.10) == SprtDecision::VentajaDemostrada
        );
        assert_eq!(default_phase, !default_exit);
        for (key, valid) in knobs {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                arena.registry.set(key, invalid);
                assert_eq!(
                    sprt_evalua_salida_sonda(&arena, 0),
                    default_exit,
                    "n={n}, {key}={invalid}"
                );
                assert_eq!(
                    en_fase_de_sonda(&arena, 0),
                    default_phase,
                    "n={n}, {key}={invalid}"
                );
                assert_eq!(coin.metrics.trade_count.load(Relaxed), n);
                assert_eq!(resolve_max_cierres_sonda(&arena, 0), 50);

                // Getter fallback does not change the independent quota exit.
                coin.metrics.trade_count.store(50, Relaxed);
                assert!(!en_fase_de_sonda(&arena, 0), "{key}={invalid}");
                coin.metrics.trade_count.store(n, Relaxed);
                arena.registry.set(key, valid);
                assert_eq!(sprt_evalua_salida_sonda(&arena, 0), default_exit);
            }
        }
    }
}

#[test]
fn test_sprt_bernoulli_nonfinite_parameters_continue_sampling() {
    let canonical = [0.50, 0.70, 0.05, 0.10];
    assert_eq!(
        sprt_bernoulli(
            100,
            100,
            canonical[0],
            canonical[1],
            canonical[2],
            canonical[3],
        ),
        SprtDecision::VentajaDemostrada
    );
    for index in 0..canonical.len() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut params = canonical;
            params[index] = invalid;
            assert_eq!(
                sprt_bernoulli(100, 100, params[0], params[1], params[2], params[3]),
                SprtDecision::ContinuarMuestreo,
                "parameter {index}={invalid}"
            );
        }
    }
}
