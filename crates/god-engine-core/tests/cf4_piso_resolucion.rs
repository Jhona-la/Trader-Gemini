//! CL-69 (C-F4): el ATR de las decisiones nunca baja de un tick del símbolo.
//!
//! Un ATR por debajo de la resolución del precio no es medible: el mid sólo
//! se mueve de tick en tick. Hasta ahora el único piso era un literal en
//! puntos básicos (10 pb del precio en `v_t`), que no sabe del tick: con una
//! moneda a 1 USD y tick de 0,01 (100 pb) el ATR quedaba en 10 pb, una
//! décima de tick.
//!
//! Verdad: tras dos horas de mercado plano en una moneda a 1 USD con tick
//! 0,01, `get_atr_pct()` ≥ tick/precio = 0,01 y `current_atr` ≥ 0,01 USD.
//! Control: sin spec no hay piso declarado y el ATR sigue siendo
//! `v_t/precio`.

use god_engine_core::GodEngineCore;
use quantum_arena::symbol_registry::{self, SymbolSpec};
use quantum_arena::{symbols, GlobalArena};
use std::sync::atomic::Ordering;
use std::sync::Mutex;

/// El tramo cambia el universo y el registro de símbolos globales.
static ENTORNO_GLOBAL: Mutex<()> = Mutex::new(());

fn mercado_plano(sym: &str, con_spec: bool) -> (f64, f64, f64) {
    symbols::update_dynamic_universe(vec![sym.into()]);
    let specs = if con_spec {
        vec![SymbolSpec {
            symbol: sym.into(),
            step_size: 1.0,
            tick_size: 0.01,
            min_qty: 1.0,
            min_notional: 5.0,
            max_leverage: 20,
            maker_fee: 0.0002,
            taker_fee: 0.0005,
            is_shadow: false,
        }]
    } else {
        Vec::new()
    };
    symbol_registry::update_registry(specs);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    // Dos horas de libro quieto (bid 0,99 / ask 1,01) a un evento por segundo.
    for i in 0..7_200u64 {
        let _ = core.process_tick_dual(
            0,
            0.99,
            1.01,
            1_000.0,
            1_000.0,
            10_000 + 1_000 * i,
            &[0.0; 54],
            true,
            false,
        );
    }
    let fe = &core.feature_engines[0];
    let atr_pct = fe.get_atr_pct();
    let atr_arena = arena.coins[0].current_atr.load(Ordering::Relaxed);
    (atr_pct, atr_arena, fe.v_t / fe.last_price)
}

#[test]
fn cl69_mercado_plano_el_atr_no_baja_de_un_tick() {
    let _g = ENTORNO_GLOBAL.lock().unwrap_or_else(|e| e.into_inner());
    let (atr_pct, atr_arena, vt_pct) = mercado_plano("PISOUSDT", true);
    assert!(
        atr_pct >= 0.01 - 1e-12 && atr_pct.is_finite(),
        "ATR {atr_pct} por debajo de un tick (0,01 del precio); v_t/p = {vt_pct}"
    );
    assert!(
        atr_arena >= 0.01 - 1e-12,
        "current_atr {atr_arena} USD por debajo de un tick de 0,01 USD"
    );
}

#[test]
fn cl69_sin_spec_no_hay_piso_declarado() {
    let _g = ENTORNO_GLOBAL.lock().unwrap_or_else(|e| e.into_inner());
    let (atr_pct, _, vt_pct) = mercado_plano("SINSPECUSDT", false);
    assert_eq!(
        atr_pct.to_bits(),
        vt_pct.to_bits(),
        "sin spec el ATR es v_t/p"
    );
}
