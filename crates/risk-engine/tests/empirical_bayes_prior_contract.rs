//! Contrato formal: Inyección Dinámica de Prior Bayesiano y Empirical Bayes Pooling (Ola Ω108).
//!
//! F3.4-06: Verifica que el prior de contracción jerárquica de win rate se resuelva
//! dinámicamente desde el registro omnisciente (por activo o global) o a partir de la
//! tasa de acierto empírica agregada del portafolio (Empirical Bayes), preservando
//! bit a bit la paridad canónica en frío (0.55 y masa 10.0).

use quantum_arena::GlobalArena;
use risk_engine::evidence::{
    resolve_bayes_prior, win_rate_hierarchical_lcb, DEFAULT_BAYES_PRIOR_MASS,
    DEFAULT_BAYES_PRIOR_WIN_RATE,
};
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn test_bayes_prior_canonical_cold_start_parity() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // En arranque en frío sin historial ni llaves en el registro:
    let (p_prior, k_prior) = resolve_bayes_prior(&arena, 0);
    assert_eq!(p_prior, DEFAULT_BAYES_PRIOR_WIN_RATE);
    assert_eq!(p_prior, 0.55);
    assert_eq!(k_prior, DEFAULT_BAYES_PRIOR_MASS);
    assert_eq!(k_prior, 10.0);

    // La cota LCB calculada es exactamente la canónica
    let lcb_canonico = win_rate_hierarchical_lcb(0.0, 1.0, 0.55, 10.0);
    let lcb_dinamico = win_rate_hierarchical_lcb(0.0, 1.0, p_prior, k_prior);
    assert_eq!(lcb_canonico, lcb_dinamico);
}

#[test]
fn test_bayes_prior_registry_hierarchy_and_scoping() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // 1. Override global en registro
    arena.registry.set("bayes_prior_win_rate", 0.65);
    arena.registry.set("bayes_prior_mass", 15.0);

    let (p0, k0) = resolve_bayes_prior(&arena, 0);
    assert_eq!(p0, 0.65);
    assert_eq!(k0, 15.0);

    let (p1, k1) = resolve_bayes_prior(&arena, 1);
    assert_eq!(p1, 0.65);
    assert_eq!(k1, 15.0);

    // 2. Override específico por moneda toma precedencia
    arena.registry.set("c0:bayes_prior_win_rate", 0.72);
    arena.registry.set("c0:bayes_prior_mass", 25.0);

    let (p0_scoped, k0_scoped) = resolve_bayes_prior(&arena, 0);
    assert_eq!(p0_scoped, 0.72);
    assert_eq!(k0_scoped, 25.0);

    // Moneda 1 preserva la configuración global
    let (p1_scoped, k1_scoped) = resolve_bayes_prior(&arena, 1);
    assert_eq!(p1_scoped, 0.65);
    assert_eq!(k1_scoped, 15.0);
}

#[test]
fn test_bayes_prior_empirical_bayes_portfolio_pooling() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // Sin llaves en registro, pero otros activos en el portafolio han acumulado evidencia
    // Moneda 1: 20 operaciones, 70% win rate (14 ganancias)
    arena.coins[1].metrics.trade_count.store(20, Relaxed);
    arena.coins[1].metrics.win_rate.store(0.70, Relaxed);

    // Moneda 2: 20 operaciones, 80% win rate (16 ganancias)
    arena.coins[2].metrics.trade_count.store(20, Relaxed);
    arena.coins[2].metrics.win_rate.store(0.80, Relaxed);

    // Moneda 0 es nueva (0 operaciones). Evaluamos su prior empírico a partir del portafolio:
    // Total operaciones del pool: 20 + 20 = 40 >= 10
    // Total ganancias del pool: 14 + 16 = 30
    // Win rate medio ponderado del pool: 30 / 40 = 0.75
    let (p_pool, k_pool) = resolve_bayes_prior(&arena, 0);
    assert!((p_pool - 0.75).abs() < 1e-9, "p_pool esperado 0.75, obtenido {p_pool}");
    assert_eq!(k_pool, DEFAULT_BAYES_PRIOR_MASS);

    // Moneda 0 con 1 pérdida temprana (w=0.0, n=1):
    // Con prior empírico del 75%, el LCB es significativamente mayor que con el 55% rígido
    let lcb_con_pool = win_rate_hierarchical_lcb(0.0, 1.0, p_pool, k_pool);
    let lcb_con_estatico = win_rate_hierarchical_lcb(0.0, 1.0, 0.55, 10.0);
    assert!(
        lcb_con_pool > lcb_con_estatico,
        "El prior del portafolio debe proteger al activo joven ({lcb_con_pool} > {lcb_con_estatico})"
    );
}

#[test]
fn test_bayes_prior_sanitization_against_anomalies() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // Valores aberrantes en registro
    arena.registry.set("c0:bayes_prior_win_rate", -0.5);
    arena.registry.set("c0:bayes_prior_mass", f64::NAN);

    let (p, k) = resolve_bayes_prior(&arena, 0);
    assert_eq!(p, DEFAULT_BAYES_PRIOR_WIN_RATE);
    assert_eq!(k, DEFAULT_BAYES_PRIOR_MASS);

    arena.registry.set("c0:bayes_prior_win_rate", 1.5); // > 0.95
    arena.registry.set("c0:bayes_prior_mass", 0.1); // < 0.5

    let (p_hi, k_lo) = resolve_bayes_prior(&arena, 0);
    assert_eq!(p_hi, DEFAULT_BAYES_PRIOR_WIN_RATE);
    assert_eq!(k_lo, DEFAULT_BAYES_PRIOR_MASS);
}
