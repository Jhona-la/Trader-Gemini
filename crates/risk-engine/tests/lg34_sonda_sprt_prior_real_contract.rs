//! LG-34 (Línea G, ola 10; hallazgo LG10-04, confirmado por un escéptico): el
//! SPRT de la sonda (F1-090) reconstruye el número de aciertos invirtiendo el
//! posterior del win rate con un prior de 0,55, pero el win rate de cada
//! moneda nace en `SuperGenotype::WORST_TOLERATED_WR` = 0,40 (GlobalArena::build,
//! D-680) y se actualiza con `posterior_win_rate` (k = 1,96² pseudo-operaciones).
//! Con 4 aciertos de 4 la inversión daba k = 3 y la salida precoz al 4.º cierre,
//! que F1-090 promete, era inalcanzable. El contrato F1-090 partía de 0,55, un
//! valor que no existe en vivo.

use quantum_arena::genome::SuperGenotype;
use quantum_arena::GlobalArena;
use risk_engine::sonda::sprt_evalua_salida_sonda;
use std::sync::atomic::Ordering::Relaxed;

/// Reproduce la contabilidad viva del núcleo: el arena nace con el prior real y
/// cada cierre aplica `posterior_win_rate` (god-engine-core, cierre del núcleo).
fn moneda_con(resultados: &[bool]) -> std::sync::Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let coin = &arena.coins[0];
    assert_eq!(
        coin.metrics.win_rate.load(Relaxed),
        SuperGenotype::WORST_TOLERATED_WR,
        "precondición: el win rate nace en el prior real"
    );
    for &gana in resultados {
        let n_prev = coin.metrics.trade_count.fetch_add(1, Relaxed) as f64;
        let wr = coin.metrics.win_rate.load(Relaxed);
        coin.metrics
            .win_rate
            .store(SuperGenotype::posterior_win_rate(wr, n_prev, gana), Relaxed);
    }
    arena
}

#[test]
fn lg34_cuatro_aciertos_de_cuatro_salen_de_la_sonda() {
    let arena = moneda_con(&[true, true, true, true]);
    assert!(
        sprt_evalua_salida_sonda(&arena, 0),
        "4/4 con defaults (p0 0,50, p1 0,85, α 0,10, β 0,20) cruza ln 8: debe salir"
    );
}

#[test]
fn lg34_tres_de_cuatro_no_salen() {
    for orden in [
        [true, true, true, false],
        [false, true, true, true],
        [true, false, true, true],
    ] {
        let arena = moneda_con(&orden);
        assert!(!sprt_evalua_salida_sonda(&arena, 0), "3/4 ({orden:?}) no cruza la frontera");
    }
}
