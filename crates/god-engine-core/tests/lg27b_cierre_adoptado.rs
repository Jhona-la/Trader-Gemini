//! LG-27b (Línea G; LG27-STAT-01 de la revisión de LG-27): el detector de
//! pérdida de ventaja (LG-22) sólo aprende de cierres cuya geometría se fijó
//! AL ABRIR.
//!
//! Por qué. El e-CUSUM de LG-22 apuesta con λ = φ/(r_max − μ), y su garantía
//! exige que r_max sea conocida al abrir (λ predecible). Una posición adoptada
//! del exchange tras un reinicio llega con SL y TP 0; al cerrar, el núcleo
//! rehace (tp, sl) con el ATR del cierre, y la R del cierre deja de estar en
//! la escala de su orden. Mismo criterio que CL-130 (el freno duro).
//!
//! Fixture. Arnés de CL-130: cuenta de 100 USD, largo a 100. Control: con SL
//! propio en 99 se cierra en el stop y alimenta el detector. Adoptada: con
//! SL y TP 0, el precio baja hasta que el núcleo la cierra; el detector no
//! cambia.

use god_engine_core::{outcome_context::OutcomeContext, GodEngineCore};
use quantum_arena::{position::PositionHorizon, symbol_registry, GlobalArena};
use std::sync::{atomic::Ordering::Relaxed, Arc};

const CAPITAL: f64 = 100.0;
const ENTRADA: f64 = 100.0;
const SL: f64 = 99.0;
const TP: f64 = 102.25;

fn nucleo() -> (Arc<GlobalArena>, GodEngineCore) {
    quantum_arena::symbols::update_dynamic_universe(vec!["LGQAUSDT".into(), "LGQBUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("LGQAUSDT"),
        symbol_registry::get_official_binance_spec("LGQBUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(CAPITAL);
    let mut core = GodEngineCore::new_with_outcome_context(
        arena.clone(),
        OutcomeContext::IsolatedSimulation,
    );
    core.swing_nn = None;
    core.scalp_forest = None;
    (arena, core)
}

/// Abre un largo con (`tp`, `sl`) y baja el precio hasta que el núcleo lo
/// cierra. Devuelve el instante siguiente.
fn abrir_y_cerrar(arena: &GlobalArena, core: &mut GodEngineCore, tp: f64, sl: f64, t0: u64) -> u64 {
    let pos = &arena.coins[0].positions.slot_2;
    assert!(pos.open_with_horizon(true, ENTRADA, 1.0, ENTRADA / 10.0, t0, tp, sl, PositionHorizon::Continuous));
    arena.used_margin.store(ENTRADA / 10.0, Relaxed);
    let mut t = t0;
    for paso in 1..=200 {
        t += 1_000;
        let precio = ENTRADA - 0.25 * paso as f64;
        let (_, cierre, _) =
            core.process_tick_dual(0, precio, precio + 0.01, 5.0, 5.0, t, &[0.0; 54], true, false);
        if cierre.is_some() {
            assert!(!pos.is_open());
            return t + 1_000;
        }
    }
    panic!("el núcleo no cerró la posición (tp {tp}, sl {sl}) bajando el precio a 50");
}

#[test]
fn lg27b_solo_los_cierres_con_sl_propio_alimentan_el_detector() {
    let (arena, mut core) = nucleo();
    assert_eq!(core.lg22_detectores[0].cierres_totales, 0);

    // Control: SL propio ⇒ el cierre llega al detector.
    let t = abrir_y_cerrar(&arena, &mut core, TP, SL, 1_000);
    assert_eq!(core.lg22_detectores[0].cierres_totales, 1, "el control alimenta el detector");

    // Adoptada (SL y TP 0, como la deja la reconciliación) ⇒ no llega.
    abrir_y_cerrar(&arena, &mut core, 0.0, 0.0, t + 60_000);
    assert_eq!(core.lg22_detectores[0].cierres_totales, 1, "el cierre adoptado no cuenta");
}
