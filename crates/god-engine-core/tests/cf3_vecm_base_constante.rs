//! C-F3 · F3.7-01 (Línea C): una base futuro-spot constante es equilibrio,
//! no dislocación.
//!
//! Teoría que se prueba: un z de reversión mide la desviación de la base
//! b_t = F_t − S_t respecto de SU media, en unidades de SU dispersión:
//! z = (b_t − E[b])/σ_b. Con una prima de financiación constante (contango
//! de 30 pb) más ruido simétrico, E[z] = 0: no hay nada que revierta, y un
//! voto de reversión no puede tener signo sistemático. El núcleo publica
//! `vecm_zscore` = (F − S)/(F·max(ATR%, 5 pb)), recortado a ±3: el NIVEL de
//! la base dividido por la volatilidad del FUTURO, sin restar la media de la
//! base. Una prima de 30 pb con un ATR% de ~12 pb da z ≈ 2,4 en todos los
//! eventos y VECM vota reversión (corto) en todos.
//!
//! Generador: spot en paseo aleatorio gaussiano de 1 pb por segundo (σ de
//! 60 s ≈ 7,7 pb ⇒ ATR% ≈ √(8/π)·7,7 ≈ 12 pb, una volatilidad de futuro
//! realista), futuro = spot·1,003·(1 + 1e-5·w) con w uniforme en [−1, 1]
//! (prima constante de 30 pb con ruido de ±0,1 pb, independiente del
//! spot). 1 800 ticks de 1 s por `process_tick_dual` con entradas
//! permitidas (el escritor sólo corre en eventos con entradas).
//!
//! Corrección del borrador: pedía |z| < 0,5 en el último evento y un futuro
//! alterno ±1e-5. Con ese alterno los retornos de 60 s son exactamente 0, la
//! σ̂60 de CL-71 no arranca y el ATR% se queda en el prior de 50 pb (z = 0,6
//! sólo por el prior). Y un z bien construido sobre un ruido i.i.d. tiene
//! varianza 1, así que |z| < 0,5 en un instante tampoco es su verdad. La
//! verdad es de sesgo: la media de z y la del voto VECM en la segunda media
//! hora deben ser ≈ 0.
//!
//! Convención (fase C-F3): un test que hoy falla por la causa de un hallazgo
//! lleva `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use strategy_core::{vecm_arbitrage::JohansenVecmEngine, QuantumStrategy};

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Verdad: en la segunda media hora, |media de vecm_zscore| < 0,5 y
/// |media del voto VECM| < 0,1. Con un z centrado en la media de la base y
/// varianza 1, el error típico de la media de 900 lecturas es ≤ 1/30 ≈ 0,033
/// aun sin independencia perfecta: 0,5 son 15 errores típicos; el voto es
/// una función impar de z, acotada en [−1, 1], y su media cae igual a 0.
#[test]
#[ignore = "OPEN F3.7-01 (Línea C, escritor en lib.rs y lectores de strategy-core): vecm_zscore es (F − S)/(F·max(ATR%, 5 pb)), el nivel de la base entre el ATR del futuro, sin restar su media; una prima constante se publica como dislocación y VECM vota reversión en todos los eventos"]
fn cf3_vecm_base_constante_no_vota_reversion() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    let mut vecm = JohansenVecmEngine::default();
    assert!(vecm.init(arena.registry.clone()).is_ok());
    let mut rng = Lcg(0xCF3_7001);
    let mut ln_s = 50_000f64.ln();
    let mut ts = 1_000u64;
    let (mut suma_z, mut suma_voto, mut n) = (0.0f64, 0.0f64, 0usize);
    for k in 0..1_800u64 {
        ln_s += 1e-4 * rng.n();
        let spot = ln_s.exp();
        arena.update_spot_data(0, spot - 0.1, spot + 0.1, 1.0, 1.0);
        let fut = spot * 1.003 * (1.0 + 1e-5 * (2.0 * rng.u() - 1.0));
        let _ = core.process_tick_dual(
            0,
            fut - 0.5,
            fut + 0.5,
            1.0,
            1.0,
            ts,
            &[0.0; 54],
            true,
            true,
        );
        ts += 1_000;
        if k >= 900 {
            let z = arena
                .registry
                .get_scoped_strict("BTCUSDT", "vecm_zscore")
                .expect("vecm_zscore publicado por el núcleo");
            suma_z += z;
            suma_voto += vecm.evaluate_for_coin(0, "BTCUSDT");
            n += 1;
        }
    }
    let (media_z, media_voto) = (suma_z / n as f64, suma_voto / n as f64);
    let atr = arena
        .registry
        .get_scoped_value_or("BTCUSDT", "atr_pct", f64::NAN);
    println!(
        "observado: media de vecm_zscore = {media_z:.4}, media del voto VECM = {media_voto:.4}, atr_pct final = {atr:e} ({n} lecturas)"
    );
    assert!(
        media_z.abs() < 0.5,
        "base constante (equilibrio) publicada con z medio = {media_z:.4}"
    );
    assert!(
        media_voto.abs() < 0.1,
        "VECM vota reversión sobre una prima constante: voto medio {media_voto:.4}"
    );
}
