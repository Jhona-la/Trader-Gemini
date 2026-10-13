//! C-F3 · F3.7-04 (Línea C): el filtro de reversión conformal no lee la z de
//! otra moneda.
//!
//! Mismo defecto que LC-F1-07 en el tercer lector de `vecm_zscore`: con
//! símbolo, si la moneda no tiene su z, el lector caía a la clave global, la
//! de la última moneda que escribió (`set_reg` del núcleo escribe global,
//! moneda y símbolo). Verdad: la moneda 1 sin z propia no vota; la moneda 0,
//! con z 2,5 y tendencia en contra (−1), sí (control).

use omniscient_registry::OmniscientRegistry;
use signal_engine::conformal_reversion_filter::ConformalReversionFilterEngine;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

fn set_reg(r: &OmniscientRegistry, coin: usize, sym: &str, k: &str, v: f64) {
    r.set(k, v);
    r.set_for_coin(coin, k, v);
    r.set_scoped(sym, k, v);
}

#[test]
fn cl80_conformal_moneda_sin_z_propia_no_hereda_la_global() {
    let r = Arc::new(OmniscientRegistry::new());
    set_reg(&r, 0, "BTCUSDT", "vecm_zscore", 2.5);
    set_reg(&r, 0, "BTCUSDT", "ema_trend_swing_z", -1.0);
    let mut cf = ConformalReversionFilterEngine::new();
    assert!(cf.init(r).is_ok());
    let (v0, v1) = (cf.evaluate_for_coin(0, "BTCUSDT"), cf.evaluate_for_coin(1, "ETHUSDT"));
    println!("observado: conformal moneda 0 = {v0:.4}, moneda 1 = {v1:.4}");
    assert!(v0 < 0.0, "control: la moneda 0 vota reversión ({v0})");
    assert_eq!(v1, 0.0, "la moneda 1 vota con la z de la moneda 0: {v1}");
}
