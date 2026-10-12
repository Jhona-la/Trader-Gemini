//! C-F3 · F3.7-04 / LC-F1-07 (Línea C): una moneda sin física propia no
//! vota con la z de otra moneda.
//!
//! Teoría que se prueba: el núcleo escribe cada dato por moneda en tres
//! claves (`set_reg`: global, por moneda y por símbolo). Un lector por
//! moneda que cae a la clave global cuando faltan las suyas lee el dato de
//! la ÚLTIMA moneda que escribió, no «desconocido». StatArb compara además
//! la t½ de esa otra moneda con SU τ* y vota reversión con la z ajena.
//! La verdad: con símbolo, una clave ausente para la moneda es ausencia
//! (abstención o el respaldo de la propia moneda); sin símbolo
//! (`evaluate()`), la lectura global se conserva.
//!
//! Solución conocida (LC-F1-07): moneda 0 (BTCUSDT) madura con
//! statarb_ou_zscore 2,5, t½ 60 s y τ* 60 s; moneda 1 (ETHUSDT) sólo con su
//! τ* y su vecm_zscore 0. StatArb(30; 1,5) da −tanh(1/1,5) ≈ −0,583 para la
//! moneda 0 y debe dar 0 para la 1.

use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::{stat_arb::StatArbEngine, vecm_arbitrage::JohansenVecmEngine, QuantumStrategy};

/// Escritura como la de `set_reg` del núcleo: global, moneda y símbolo.
fn set_reg(r: &OmniscientRegistry, coin: usize, sym: &str, k: &str, v: f64) {
    r.set(k, v);
    r.set_for_coin(coin, k, v);
    r.set_scoped(sym, k, v);
}

/// Sólo las claves propias de la moneda (sin tocar la global).
fn set_propia(r: &OmniscientRegistry, coin: usize, sym: &str, k: &str, v: f64) {
    r.set_for_coin(coin, k, v);
    r.set_scoped(sym, k, v);
}

#[test]
fn cl80_statarb_moneda_sin_fisica_propia_no_hereda_z_global() {
    let r = Arc::new(OmniscientRegistry::new());
    set_reg(&r, 0, "BTCUSDT", "statarb_ou_zscore", 2.5);
    set_reg(&r, 0, "BTCUSDT", "statarb_half_life_ms", 60_000.0);
    set_reg(&r, 0, "BTCUSDT", "dominant_tau_ms", 60_000.0);
    set_propia(&r, 1, "ETHUSDT", "dominant_tau_ms", 60_000.0);
    set_propia(&r, 1, "ETHUSDT", "vecm_zscore", 0.0);
    let mut sa = StatArbEngine::new(30, 1.5);
    assert!(sa.init(r).is_ok());
    let (v0, v1) = (sa.evaluate_for_coin(0, "BTCUSDT"), sa.evaluate_for_coin(1, "ETHUSDT"));
    println!("observado: StatArb moneda 0 = {v0:.4}, moneda 1 = {v1:.4}");
    assert!((v0 + (1.0f64 / 1.5).tanh()).abs() < 1e-12, "control: moneda 0 = {v0}");
    assert_eq!(v1, 0.0, "la moneda 1 vota con la z de la moneda 0: {v1}");
}

#[test]
fn cl80_vecm_moneda_sin_z_propia_no_hereda_la_global() {
    let r = Arc::new(OmniscientRegistry::new());
    set_reg(&r, 0, "BTCUSDT", "vecm_zscore", 2.5);
    let mut ve = JohansenVecmEngine::default();
    assert!(ve.init(r).is_ok());
    let (v0, v1) = (ve.evaluate_for_coin(0, "BTCUSDT"), ve.evaluate_for_coin(1, "ETHUSDT"));
    println!("observado: VECM moneda 0 = {v0:.4}, moneda 1 = {v1:.4}");
    assert!(v0 < 0.0, "control: la moneda 0 vota reversión ({v0})");
    assert_eq!(v1, 0.0, "la moneda 1 vota con la z de la moneda 0: {v1}");
}

/// Sin símbolo (`evaluate()`), la lectura global se conserva: es la
/// interfaz de una sola moneda (pruebas y llamadores sin universo).
#[test]
fn cl80_sin_simbolo_se_conserva_la_lectura_global() {
    let r = Arc::new(OmniscientRegistry::new());
    r.set("vecm_zscore", 2.5);
    let mut ve = JohansenVecmEngine::default();
    let mut sa = StatArbEngine::new(30, 1.5);
    assert!(ve.init(r.clone()).is_ok());
    assert!(sa.init(r).is_ok());
    assert!(ve.evaluate() < 0.0);
    assert!(sa.evaluate() < 0.0);
}
