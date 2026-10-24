//! CL-66 (C-F4, LC-F3-02): el núcleo no publica `atr_1s`, `atr_5s` ni
//! `atr_1m`, y el coaxial no vota con escalas fabricadas.
//!
//! Antes del arreglo las tres claves eran el mismo `v_t` del motor con
//! multiplicadores fijos y suelos absolutos en USD:
//! `atr_1s = max(v_t, 0,0001)`, `atr_5s = max(v_t, 0,0005)` y
//! `atr_1m = max(7,746·v_t, 0,0020)`. Con precios normales el coaxial daba
//! `comp_1s = max(1 − √5, 0) = 0` y votaba 0. Con una moneda a 0,05 USD
//! (`v_t` = 10 pb · 0,05 = 5e-5 USD) mordían los suelos:
//! `n1 = 1e-4`, `n5 = 5e-4/√5 = 2,236e-4` y `n1m = 0,002/√60 = 2,582e-4`.
//! Eso da `comp_1s = 0,553`, `comp_5s = 0,134` y `squeeze = tanh(0,296) = 0,288`.
//! El voto, 0,288·tanh(dirección), dependía sólo del nivel de precio y no de
//! ninguna compresión medida.
//!
//! Verdad tras el arreglo: ninguna de las tres claves existe en ningún ámbito
//! (global, moneda o símbolo) y el coaxial se abstiene (voto 0 exacto) aunque
//! la dirección del flujo sea máxima.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use signal_engine::coaxial_breakout::CoaxialBreakoutEngine;
use std::sync::Mutex;
use strategy_core::QuantumStrategy;

/// El tramo cambia el universo y el registro de símbolos globales.
static ENTORNO_GLOBAL: Mutex<()> = Mutex::new(());

const CLAVES: [&str; 3] = ["atr_1s", "atr_5s", "atr_1m"];

#[test]
fn cl66_moneda_de_cinco_centavos_sin_atr_por_escala_ni_voto_fabricado() {
    let _g = ENTORNO_GLOBAL.lock().unwrap_or_else(|e| e.into_inner());
    let sym = "DOGEUSDT";
    symbols::update_dynamic_universe(vec![sym.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(sym)]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    // 120 eventos de depth separados 1 s alrededor de 0,05 USD (dos velas
    // internas de 60 s cerradas: el ATR ya pasó por su EMA).
    for i in 0..120u64 {
        let bid = 0.05 + 0.00001 * (i % 7) as f64;
        let ask = bid + 0.00001;
        let _ = core.process_tick_dual(
            0,
            bid,
            ask,
            50_000.0,
            50_000.0,
            10_000 + 1_000 * i,
            &[0.0; 54],
            true,
            false,
        );
    }
    let reg = &arena.registry;
    // Control: el núcleo sí publicó en este ámbito (si no, la ausencia de
    // las claves no diría nada).
    assert!(
        reg.get_for_coin_or(0, "mid_price", f64::NAN).is_finite(),
        "el núcleo no publicó mid_price: el tramo no ejercitó la publicación"
    );
    for clave in CLAVES {
        assert!(
            reg.get_scoped_parameter(Some(sym), Some(0), clave, "cl66")
                .is_none()
                && reg
                    .get_scoped_parameter(None, Some(0), clave, "cl66")
                    .is_none()
                && reg
                    .get_scoped_parameter(Some(sym), None, clave, "cl66")
                    .is_none()
                && reg
                    .get_scoped_parameter(None, None, clave, "cl66")
                    .is_none(),
            "el núcleo sigue publicando {clave}"
        );
    }
    // Dirección del flujo máxima: con escalas fabricadas el voto sería
    // 0,288·tanh(1) ≈ 0,219.
    reg.set_scoped(sym, "order_flow_direction", 1.0);
    let mut coax = CoaxialBreakoutEngine::new();
    coax.init(reg.clone()).expect("init");
    let voto = coax.evaluate_for_coin(0, sym);
    assert_eq!(
        voto, 0.0,
        "el coaxial vota {voto} sin ningún ATR por escala medido"
    );
}
