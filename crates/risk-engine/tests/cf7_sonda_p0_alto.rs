//! C-F7 · CL-147: la salida SPRT de la sonda no entra en pánico con p0 alto.
//!
//! Teoría. `sprt_evalua_salida_sonda` lee p0 del registro y lo recorta a
//! [0,10; 0,90], y después recorta p1 a [p0 + 0,05; 0,95]. Con p0 = 0,90,
//! 0,90 + 0,05 vale 0,9500000000000001 en f64: el suelo supera al techo y
//! `f64::clamp` entra en pánico (con `panic = "abort"`, el motor vivo muere).
//! Hoy ningún código de producción escribe la clave; basta con que alguien la
//! ajuste para que el primer cierre de una moneda en sonda tumbe el proceso.
//!
//! Fixture. Una moneda con 4 cierres y acierto 1,0, y la clave
//! `sonda_sprt_p0` en 0,90 y por encima (el recorte la deja en 0,90).

use quantum_arena::GlobalArena;
use risk_engine::sonda::sprt_evalua_salida_sonda;
use std::sync::atomic::Ordering::Relaxed;

/// Esperado: un veredicto, no un pánico, con p0 = 0,90 y con p0 > 0,90.
/// Antes: `f64::clamp(0,9500000000000001, 0,95)`.
#[test]
fn cl147_p0_en_el_techo_no_tumba_la_sonda() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    coin.metrics.trade_count.store(4, Relaxed);
    coin.metrics.win_rate.store(1.0, Relaxed);
    arena.registry.set("sonda_sprt_enabled", 1.0);
    for p0 in [0.90, 0.95, 0.99] {
        arena.registry.set("sonda_sprt_p0", p0);
        let sale = sprt_evalua_salida_sonda(&arena, 0);
        println!("observado: p0 = {p0} ⇒ sale de la sonda: {sale}");
    }
}

/// Falso positivo: con p0 por debajo del techo el veredicto no cambia.
/// Mismo caso que `test_sprt_early_exit_from_probe_phase`: p0 = 0,50,
/// p1 = 0,85, α = 0,10, β = 0,20 y 4 aciertos de 4 ⇒ ventaja demostrada.
#[test]
fn cl147_con_p0_bajo_el_veredicto_no_cambia() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    coin.metrics.trade_count.store(4, Relaxed);
    coin.metrics.win_rate.store(1.0, Relaxed);
    arena.registry.set("sonda_sprt_enabled", 1.0);
    arena.registry.set("sonda_sprt_p0", 0.50);
    arena.registry.set("sonda_sprt_p1", 0.85);
    arena.registry.set("sonda_sprt_alpha", 0.10);
    arena.registry.set("sonda_sprt_beta", 0.20);
    assert!(sprt_evalua_salida_sonda(&arena, 0));
}
