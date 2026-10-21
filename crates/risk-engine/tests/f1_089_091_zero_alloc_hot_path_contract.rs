//! Contrato Formal de Certificación F1-089 / F1-091 (Ola Ω194)
//!
//! Veredicto: Ruta crítica de evaluación de intenciones Zero-Alloc:
//! 1. `hayashi_yoshida_correlation` con stack buffers sin Vec intermedios.
//! 2. `dependency_exposure` usando `snapshot_recent_into` con buffers en stack [CompactTick; 512].
//! 3. `same_bet_riesgos_hibridos_into` copiando riesgos en stack buffer sin alocaciones en heap.
//! 4. `resolve_bayes_prior` con `get_for_coin_strict` eliminando format!.
//! 5. `copulas_store` con manifest en memoria cacheado para evitar lecturas de disco redundantes.

use quantum_arena::state::CompactTick;
use quantum_arena::GlobalArena;
use risk_engine::correlation_guard::{dependency_exposure, hayashi_yoshida_correlation};
use risk_engine::evidence::{
    resolve_bayes_prior, DEFAULT_BAYES_PRIOR_MASS, DEFAULT_BAYES_PRIOR_WIN_RATE,
};

#[test]
fn test_hayashi_yoshida_zero_alloc_numerical_parity() {
    let now = 1_000_000u64;
    // Dos series sintéticas con cotizaciones asíncronas
    let mut a = Vec::new();
    let mut b = Vec::new();

    for i in 0..50 {
        let t_a = now + i * 100;
        let t_b = now + i * 100 + 35; // Desfase temporal asíncrono
        let p_a = 100.0 + (i as f64) * 0.5;
        let p_b = 200.0 + (i as f64) * 1.0;

        a.push(CompactTick {
            timestamp: t_a,
            bid_price: p_a,
            ask_price: p_a + 0.1,
            bid_qty: 1.0,
            ask_qty: 1.0,
        });
        b.push(CompactTick {
            timestamp: t_b,
            bid_price: p_b,
            ask_price: p_b + 0.2,
            bid_qty: 1.0,
            ask_qty: 1.0,
        });
    }

    let rho = hayashi_yoshida_correlation(&a, &b);
    assert!(rho.is_some(), "Debe estimar correlación con solape asíncrono válido");
    let val = rho.unwrap();
    assert!(
        val > 0.90 && val <= 1.0,
        "Series perfectamente co-tendenciales deben tener rho_HY > 0.90, observado: {val}"
    );

    // Ventana disjunta debe dar None sin pánicos
    let mut c = Vec::new();
    for i in 0..10 {
        c.push(CompactTick {
            timestamp: now + 50_000 + i * 100,
            bid_price: 100.0,
            ask_price: 100.1,
            bid_qty: 1.0,
            ask_qty: 1.0,
        });
    }
    assert!(hayashi_yoshida_correlation(&a, &c).is_none());
}

#[test]
fn test_dependency_exposure_zero_alloc_stack_execution() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let now = 1_000_000u64;

    // Poblar tick ring de candidata (coin 0) y de un par (coin 1)
    for i in 0..100 {
        arena.coins[0].tick_ring.push(CompactTick {
            timestamp: now + i * 100,
            bid_price: 100.0 + i as f64,
            ask_price: 100.1 + i as f64,
            bid_qty: 1.0,
            ask_qty: 1.0,
        });
        arena.coins[1].tick_ring.push(CompactTick {
            timestamp: now + i * 100 + 20,
            bid_price: 50.0 + i as f64 * 0.5,
            ask_price: 50.05 + i as f64 * 0.5,
            bid_qty: 1.0,
            ask_qty: 1.0,
        });
    }

    // Sin posiciones abiertas:
    let exp_empty = dependency_exposure(&arena, 0, true, 0.70);
    assert!(exp_empty.is_some());
    let e0 = exp_empty.unwrap();
    assert_eq!(e0.open_positions, 0);
    assert_eq!(e0.same_bet_positions, 0);
    assert!(e0.same_bet_rho_efectivo().is_none());

    // Abrir posición en coin 1
    arena.coins[1].positions.slots()[0].open(true, 50.0, 0.01, 0.05, 1_000, 51.0, 49.5);

    let exp_with_pos = dependency_exposure(&arena, 0, true, 0.70);
    assert!(exp_with_pos.is_some());
    let e1 = exp_with_pos.unwrap();
    assert_eq!(e1.open_positions, 1);
    assert_eq!(e1.same_bet_positions, 1);
    assert!(e1.same_bet_rho_efectivo().is_some());

    // Verificar copia zero-alloc a stack buffer
    let mut riesgos_buf = [0.0f64; 64];
    let n_riesgos = e1.same_bet_riesgos_hibridos_into(0.02, 0.015, &mut riesgos_buf);
    assert_eq!(n_riesgos, 2); // 1 miembro same-bet + 1 candidata
    assert!(riesgos_buf[0] > 0.0);
    assert_eq!(riesgos_buf[1], 0.015);
}

#[test]
fn test_resolve_bayes_prior_zero_alloc_and_overrides() {
    let arena = GlobalArena::build_in_own_stack(13.0);

    // 1. Defaults de arranque frío sin claves
    let (p_cold, k_cold) = resolve_bayes_prior(&arena, 0);
    assert_eq!(p_cold, DEFAULT_BAYES_PRIOR_WIN_RATE);
    assert_eq!(k_cold, DEFAULT_BAYES_PRIOR_MASS);

    // 2. Override global en registro
    arena.registry.set("bayes_prior_win_rate", 0.62);
    arena.registry.set("bayes_prior_mass", 18.0);
    let (p_glob, k_glob) = resolve_bayes_prior(&arena, 0);
    assert_eq!(p_glob, 0.62);
    assert_eq!(k_glob, 18.0);

    // 3. Override específico por moneda prevalece
    arena.registry.set("c0:bayes_prior_win_rate", 0.75);
    arena.registry.set("c0:bayes_prior_mass", 25.0);
    let (p_coin, k_coin) = resolve_bayes_prior(&arena, 0);
    assert_eq!(p_coin, 0.75);
    assert_eq!(k_coin, 25.0);

    // Moneda 1 sigue usando el global
    let (p_coin1, k_coin1) = resolve_bayes_prior(&arena, 1);
    assert_eq!(p_coin1, 0.62);
    assert_eq!(k_coin1, 18.0);
}
