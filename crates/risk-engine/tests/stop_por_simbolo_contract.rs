//! LC-F1-03 (CL-62): el gen del stop de BTC (`sl_atr_mult_btc`) va con la
//! IDENTIDAD de BTCUSDT, no con el slot 0.
//!
//! El gate de riesgo y la sonda de banda operable del núcleo elegían el gen
//! con `coin_id == 0`. Nada garantiza que el slot 0 sea BTCUSDT: el forense y
//! el entrenador ponen ahí el símbolo que miden, y en vivo lo es sólo porque
//! el bootloader lista BTC primero. ADR-0011: el slot es una posición; la
//! identidad es el símbolo.
//! Estado sintético local; sin exchange ni genoma activo.
use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    symbols::update_dynamic_universe,
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

const UNIVERSO: [&str; 3] = ["IDSLAUSDT", "IDSLBUSDT", "BTCUSDT"];

fn arena() -> Arc<GlobalArena> {
    static REGISTRY: Once = Once::new();
    REGISTRY.call_once(|| {
        update_dynamic_universe(UNIVERSO.iter().map(|s| s.to_string()).collect());
        update_registry(
            UNIVERSO
                .iter()
                .map(|s| SymbolSpec {
                    symbol: s.to_string(),
                    step_size: 0.001,
                    tick_size: 0.01,
                    min_qty: 0.001,
                    min_notional: 5.0,
                    max_leverage: 20,
                    maker_fee: 0.0002,
                    taker_fee: 0.0004,
                    is_shadow: false,
                })
                .collect(),
        )
    });
    let a = GlobalArena::build_in_own_stack(1_000.0);
    for c in 0..UNIVERSO.len() {
        a.coins[c].current_price.store(100.0, Relaxed);
        a.coins[c].current_atr.store(0.2, Relaxed);
        a.coins[c].hurst_exponent.store(0.5, Relaxed);
    }
    a.config.global_leverage.store(20.0, Relaxed);
    a.config.global_max_drawdown.store(0.2, Relaxed);
    a.config.kelly_clamp_min.store(0.01, Relaxed);
    a.config.kelly_clamp_max.store(0.25, Relaxed);
    a.config.latency_penalty_ms.store(0.0, Relaxed);
    a.config.live_taker_fee.store(0.0004, Relaxed);
    a.config.base_slippage_floor.store(0.00001, Relaxed);
    a.config.sl_atr_mult_btc.store(3.0, Relaxed);
    a.config.sl_atr_multiplier.store(1.5, Relaxed);
    a.config.tp_rr_ratio_btc.store(3.0, Relaxed);
    a
}

/// Distancia relativa del stop de un largo a τ = 1 h en el slot `coin_id`.
fn stop_relativo(coin_id: usize) -> f64 {
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 3_600_000,
        ..Default::default()
    };
    let orden = RiskEngine::new(1_000.0).evaluate_quantum_order(coin_id, &intent, &arena());
    assert_eq!(
        orden.signal,
        SignalType::Long,
        "slot {coin_id}: {}",
        risk_engine::reject_report()
    );
    (100.0 - orden.sl_target) / 100.0
}

#[test]
fn cl62_el_stop_no_depende_del_slot() {
    let a = stop_relativo(0);
    let b = stop_relativo(1);
    assert!(a > 0.0 && b > 0.0, "stops {a} y {b}");
    assert_eq!(a, b, "dos alts idénticas llevan el mismo stop en el slot 0 y en el 1");
}

#[test]
fn cl62_btc_lleva_su_gen_en_cualquier_slot() {
    let alt = stop_relativo(1);
    let btc = stop_relativo(2);
    assert!(
        btc > alt,
        "BTCUSDT en el slot 2 usa sl_atr_mult_btc (3,0 > 1,5): stop {btc} frente a {alt}"
    );
}

#[test]
fn cl62_identidad_del_slot_sin_copiar_el_nombre() {
    let _ = arena();
    assert!(quantum_arena::symbol_registry::slot_es_simbolo(2, "BTCUSDT"));
    assert!(quantum_arena::symbol_registry::slot_es_simbolo(2, "btcusdt"));
    assert!(!quantum_arena::symbol_registry::slot_es_simbolo(0, "BTCUSDT"));
    assert!(!quantum_arena::symbol_registry::slot_es_simbolo(99, "BTCUSDT"));
}

#[test]
fn cl62_gate_y_sonda_del_nucleo_eligen_el_gen_igual() {
    // El gate y la sonda de banda operable del núcleo deben leer el MISMO
    // multiplicador; si uno vuelve a elegir por slot, la sonda aprueba τ que
    // el gate rechaza (o al revés).
    for (nombre, fuente) in [
        ("risk-engine/src/lib.rs", include_str!("../src/lib.rs")),
        ("god-engine-core/src/lib.rs", include_str!("../../god-engine-core/src/lib.rs")),
    ] {
        let produccion = fuente.split("#[cfg(test)]").next().unwrap();
        assert!(
            produccion.contains("sl_atr_multiplier_for(coin_id)"),
            "{nombre} no usa GlobalArena::sl_atr_multiplier_for"
        );
        assert!(
            !produccion.contains("sl_atr_mult_btc"),
            "{nombre} lee sl_atr_mult_btc por su cuenta"
        );
    }
}
