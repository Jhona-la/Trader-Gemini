//! Contrato Formal de Certificación F1-090 (Ola Ω193)
//!
//! Veredicto: Egreso precoz por SPRT de Wald alcanzable en el horizonte de sonda (n_max = 5),
//! inversión analítica de la regularización Bayesiana de win_rate para conteo de aciertos k sin sesgo,
//! fast-path O(1) para n < 2, y resolución zero-alloc de `resolve_max_cierres_sonda` vía `get_for_coin_or`.

use quantum_arena::genome::SuperGenotype;
use quantum_arena::GlobalArena;
use risk_engine::sonda::{
    admite_sonda, admite_sonda_con_fase, en_fase_de_sonda, hay_sonda_abierta,
    resolve_max_cierres_sonda, sprt_evalua_salida_sonda, CIERRES_DE_SONDA,
};
use std::sync::atomic::Ordering::Relaxed;

#[test]
fn test_f1_090_sprt_early_exit_reachable_with_production_defaults() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    // Precondición: NINGUNA clave configurada en el registro (pure defaults de producción).
    // Con n=4 cierres y 4 aciertos sucesivos simulados con la actualización Bayesiana viva.
    // LG-34: el win rate nace en el prior vivo (GlobalArena::build), no en 0,55.
    let mut wr = coin.metrics.win_rate.load(Relaxed);
    assert_eq!(wr, SuperGenotype::WORST_TOLERATED_WR);
    for n_prev in 0..4 {
        wr = SuperGenotype::posterior_win_rate(wr, n_prev as f64, true);
    }
    // wr ≈ 0.7061, pero el número real de aciertos k es exactamente 4
    coin.metrics.trade_count.store(4, Relaxed);
    coin.metrics.win_rate.store(wr, Relaxed);

    // Con defaults (p0=0.50, p1=0.85, alpha=0.10, beta=0.20):
    // B = ln((1-0.20)/0.10) = ln(8) ≈ 2.0794
    // 4 aciertos: LLR = 4 * ln(0.85/0.50) = 4 * 0.5306 = 2.1225 >= 2.0794
    // VentajaDemostrada debe ser TRUE y egresar precozmente de la sonda.
    assert!(
        sprt_evalua_salida_sonda(&arena, 0),
        "SPRT debe demostrar ventaja anticipada en 4 aciertos consecutivos con defaults de producción"
    );
    assert!(
        !en_fase_de_sonda(&arena, 0),
        "La moneda debe egresar de la fase de sonda al demostrar ventaja por SPRT"
    );

    // En cambio, con 3 aciertos y 1 pérdida:
    let mut wr_with_loss = SuperGenotype::WORST_TOLERATED_WR;
    wr_with_loss = SuperGenotype::posterior_win_rate(wr_with_loss, 0.0, true);
    wr_with_loss = SuperGenotype::posterior_win_rate(wr_with_loss, 1.0, true);
    wr_with_loss = SuperGenotype::posterior_win_rate(wr_with_loss, 2.0, true);
    wr_with_loss = SuperGenotype::posterior_win_rate(wr_with_loss, 3.0, false);
    coin.metrics.win_rate.store(wr_with_loss, Relaxed);

    assert!(
        !sprt_evalua_salida_sonda(&arena, 0),
        "3 aciertos y 1 pérdida no deben alcanzar la frontera de ventaja demostrada"
    );
    assert!(
        en_fase_de_sonda(&arena, 0),
        "La moneda debe permanecer en fase de sonda si no demuestra ventaja anticipada"
    );
}

#[test]
fn test_f1_090_sprt_fast_path_sub_2_trades() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    // n = 0 trades: fast path inmediato false
    coin.metrics.trade_count.store(0, Relaxed);
    coin.metrics.win_rate.store(1.0, Relaxed);
    assert!(!sprt_evalua_salida_sonda(&arena, 0));
    assert!(en_fase_de_sonda(&arena, 0));

    // n = 1 trade: fast path inmediato false
    coin.metrics.trade_count.store(1, Relaxed);
    coin.metrics.win_rate.store(1.0, Relaxed);
    assert!(!sprt_evalua_salida_sonda(&arena, 0));
    assert!(en_fase_de_sonda(&arena, 0));
}

#[test]
fn test_f1_090_resolve_max_cierres_zero_alloc_and_override() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // 1. Default canónico
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), CIERRES_DE_SONDA);
    assert_eq!(resolve_max_cierres_sonda(&arena, 1), 5);

    // 2. Override global
    arena.registry.set("sonda_max_cierres", 7.0);
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), 7);
    assert_eq!(resolve_max_cierres_sonda(&arena, 1), 7);

    // 3. Override por moneda (específico tiene precedencia sobre global)
    arena.registry.set("c0:sonda_max_cierres", 3.0);
    assert_eq!(resolve_max_cierres_sonda(&arena, 0), 3);
    assert_eq!(resolve_max_cierres_sonda(&arena, 1), 7);

    // 4. Clamping defensivo
    arena.registry.set("c1:sonda_max_cierres", 100.0);
    assert_eq!(resolve_max_cierres_sonda(&arena, 1), 50);

    arena.registry.set("c2:sonda_max_cierres", 0.0);
    assert_eq!(resolve_max_cierres_sonda(&arena, 2), 5); // fallback a default
}

#[test]
fn test_f1_090_admite_sonda_con_fase_equivalence() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];

    // Caso A: Moneda en fase de sonda, sin sondas abiertas en la cartera
    coin.metrics.trade_count.store(0, Relaxed);
    let es_sonda = en_fase_de_sonda(&arena, 0);
    assert!(es_sonda);
    assert_eq!(
        admite_sonda(&arena, 0),
        admite_sonda_con_fase(&arena, es_sonda)
    );
    assert!(admite_sonda_con_fase(&arena, es_sonda));

    // Caso B: Moneda en fase de sonda, con OTRA sonda abierta en la cartera
    // Abrir posición en coin 1 que está en sonda
    arena.coins[1].metrics.trade_count.store(1, Relaxed);
    arena.coins[1].positions.slots()[0].open(true, 100.0, 0.01, 0.05, 1_000, 102.0, 99.0);
    assert!(hay_sonda_abierta(&arena));

    assert_eq!(
        admite_sonda(&arena, 0),
        admite_sonda_con_fase(&arena, es_sonda)
    );
    assert!(!admite_sonda_con_fase(&arena, es_sonda));

    // Caso C: Moneda fuera de la fase de sonda (egreso por cierres)
    coin.metrics.trade_count.store(5, Relaxed);
    let es_sonda_c = en_fase_de_sonda(&arena, 0);
    assert!(!es_sonda_c);
    // Aunque haya sonda abierta en coin 1, coin 0 ya no es sonda => admite
    assert_eq!(
        admite_sonda(&arena, 0),
        admite_sonda_con_fase(&arena, es_sonda_c)
    );
    assert!(admite_sonda_con_fase(&arena, es_sonda_c));
}
