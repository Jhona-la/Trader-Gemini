//! C-F3 / F3.4-04 (Línea C): el replay (T-1, examen WF del demonio) no
//! alimenta los muros L2 que el veto de muro lee en vivo.
//!
//! Teoría que se prueba: paridad vivo/replay de las entradas de un veto. El
//! host vivo escribe en cada evento de profundidad la cantidad del mejor
//! nivel como muro (`arena.update_l2_depth(coin_id, dbq, daq)`, en
//! src/bin/god_engine.rs) y `puertas_del_continuo` veta con
//! `calibration::muro_en_contra(l2_bid_wall, l2_ask_wall, gen, lado)`. Si el
//! replay que decide las promociones no escribe los muros, en_contra = 0, el
//! veto nunca dispara y el gen 57 es inerte para la evolución: el genoma se
//! selecciona sin el veto con el que luego opera.
//!
//! Solución conocida: 300 ticks de libro fijos (bid 100, ask 100,01,
//! bid_qty 1, ask_qty 99). Hacen falta más de warmup_ticks + 10 = 210 (el
//! replay sale antes con ≤ 10 ticks útiles). El observador mira el estado
//! ANTES de cada evento, así que desde el segundo evento los muros deben
//! valer exactamente (1, 99), como en vivo. Igualdad exacta: es una copia.
//!
//! Hoy ningún camino de backtest-engine llama a `update_l2_depth` (sus únicos
//! escritores son el host, audit_forensic_backtest, train_forest y
//! ws_client): los muros nacen en 0 y siguen en 0.
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use backtest_engine::booktick_replay::{
    run_booktick_replay_with_observer, ReplayConfig, ReplayTick,
};
use quantum_arena::genome::SuperGenotype;
use std::sync::atomic::Ordering::Relaxed;

// Cerrado en Ola Ω93 (F3.4-04): booktick_replay escribe los muros L2 vía core.arena.update_l2_depth.
#[test]
fn cf3_replay_escribe_los_muros() {
    let ticks: Vec<ReplayTick> = (0..300u64)
        .map(|i| ReplayTick {
            ts_ms: 1_700_000_000_000 + i * 1_000,
            bid: 100.0,
            ask: 100.01,
            bid_qty: 1.0,
            ask_qty: 99.0,
        })
        .collect();
    let mut muros = Vec::new();
    let _ = run_booktick_replay_with_observer(
        &ticks,
        &SuperGenotype::default(),
        None,
        &ReplayConfig::default(),
        |i, core| {
            if i > 0 {
                let c = &core.arena.coins[0];
                muros.push((c.l2_bid_wall.load(Relaxed), c.l2_ask_wall.load(Relaxed)));
            }
        },
    );
    assert!(!muros.is_empty(), "el replay no procesó eventos");
    println!(
        "observado: {} eventos; primer estado {:?}; último {:?}",
        muros.len(),
        muros.first(),
        muros.last()
    );
    // El host vivo escribe el nivel 0 en cada evento de depth.
    assert!(
        muros.iter().all(|&(b, a)| b == 1.0 && a == 99.0),
        "el replay no alimenta los muros: primer estado observado {:?}",
        muros.first()
    );
}
