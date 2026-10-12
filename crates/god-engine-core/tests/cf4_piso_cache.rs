//! CL-88 y CL-89 (atr-2, atr-3; revisión de los PR #43–#55): el piso de
//! resolución del ATR de una moneda es su tick.
//!
//! - CL-89: el núcleo cachea el tick por moneda y lo vuelve a leer sólo
//!   cuando se republica el registro de specs o el universo. La caché no
//!   puede quedarse con un tick viejo: si cambia el spec del símbolo del
//!   slot, o el símbolo que ocupa el slot, el piso cambia en el evento
//!   siguiente.
//! - CL-88: un núcleo cuyo slot no es el símbolo vivo (el examen del demonio)
//!   fija su propio tick, que manda sobre el del registro.
//!
//! Universo y registro son globales: la prueba vive en su propio proceso.

use god_engine_core::GodEngineCore;
use quantum_arena::symbol_registry::{update_registry, SymbolSpec};
use quantum_arena::{symbols, GlobalArena};

fn spec(simbolo: &str, tick: f64) -> SymbolSpec {
    SymbolSpec {
        symbol: simbolo.into(),
        step_size: 0.001,
        tick_size: tick,
        min_qty: 0.001,
        min_notional: 5.0,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0005,
        is_shadow: false,
    }
}

/// Un evento de libro en la moneda 0 y el piso de resolución que deja.
fn piso_tras_evento(core: &mut GodEngineCore, ts: u64) -> f64 {
    let mid = 100.0;
    let _ = core.process_event(
        0, false, false, true, mid, 0.0, mid - 0.05, mid + 0.05, 1.0, 1.0, 0.0, 0.0, ts, false,
        &[0.0; 54], false,
    );
    core.feature_engines[0].piso_resolucion
}

/// Verdad: el piso de la moneda 0 es el tick del símbolo que ocupa su slot,
/// también tras republicar su spec o reordenar el universo, y el tick propio
/// manda mientras está fijado.
#[test]
fn cl89_el_piso_sigue_al_registro_y_al_universo() {
    symbols::update_dynamic_universe(vec!["AAAUSDT".into(), "BBBUSDT".into()]);
    update_registry(vec![spec("AAAUSDT", 0.01), spec("BBBUSDT", 0.5)]);
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena);
    core.swing_nn = None;
    core.scalp_forest = None;
    let mut ts = 1_000u64;
    let mut paso = |core: &mut GodEngineCore| {
        ts += 1_000;
        piso_tras_evento(core, ts)
    };

    assert_eq!(paso(&mut core), 0.01, "AAAUSDT en el slot 0");
    assert_eq!(paso(&mut core), 0.01, "sin cambios, el mismo tick");

    update_registry(vec![spec("AAAUSDT", 0.02)]);
    assert_eq!(paso(&mut core), 0.02, "el spec de AAAUSDT cambia de tick");

    symbols::update_dynamic_universe(vec!["BBBUSDT".into(), "AAAUSDT".into()]);
    assert_eq!(paso(&mut core), 0.5, "BBBUSDT pasa al slot 0");

    core.fijar_tick_propio(0, 0.003);
    assert_eq!(paso(&mut core), 0.003, "el tick propio manda");
    core.fijar_tick_propio(0, f64::NAN);
    assert_eq!(paso(&mut core), 0.5, "sin tick propio vuelve el del registro");

    symbols::update_dynamic_universe(vec!["BBBUSDT".into()]);
    update_registry(vec![spec("BBBUSDT", 0.25)]);
    assert_eq!(paso(&mut core), 0.25, "dos publicaciones seguidas");
}
