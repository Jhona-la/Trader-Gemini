//! C-F2.2 (Línea C): momentos de feature-engine frente a soluciones conocidas.
//!
//! Teoría que se prueba:
//! - `WelfordOnline` estima la media y la varianza de una muestra iid. En
//!   modo acumulativo (< 2000 observaciones) es el algoritmo exacto de
//!   Welford; después pasa a una EWMA con α = 2/2001 (n efectivo ≈ 2000),
//!   cuyo sesgo relativo en la varianza es ≈ −α/2 ≈ −5·10⁻⁴.
//! - `Ewma::update_elapsed` integra dm/dt = (x − m)/τ con la ganancia
//!   −expm1(−dt/τ). Ante un escalón 0 → 1 sostenido durante T la solución
//!   cerrada es 1 − e^(−T/τ), sea cual sea la partición de T en pasos dt.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.
//!
//! La prueba de la correlación media con la cesta (F2.2-11) se descartó: el
//! módulo `correlation.rs` se retira por falta de consumidor y la
//! cancelación del factor (1 − α)² es algebraica (correlation.rs:88-107);
//! se registra en la ficha de retirada, no como test que nace para borrarse.

use feature_engine::{Ewma, WelfordOnline};

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn new(semilla: u64) -> Self {
        Lcg(semilla ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    /// Normal estándar por Box-Muller.
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Verdad: N(3, 2²), media 3 y varianza 4.
/// Modo acumulativo (n = 1999): sd(media) = 2/√1999 = 0,045 y
/// sd(var) = 4·√(2/1998) = 0,127; tolerancias 0,2 y 0,55 (≥ 4,3 sd).
/// Modo exponencial (n efectivo ≈ 2000, sesgo ≈ −5·10⁻⁴): sd(media) ≈ 0,045
/// y sd(var) ≤ 0,18 (cota del escéptico); tolerancias 0,25 y 0,75 (≥ 4,2 sd).
#[test]
fn cf2_welford_media_y_varianza_conocidas() {
    let mut g = Lcg::new(1);
    let mut w = WelfordOnline::new();
    for _ in 0..1999 {
        w.update(3.0 + 2.0 * g.normal());
    }
    assert!(!w.is_decay, "con 1999 muestras sigue en modo acumulativo");
    assert!(
        (w.mean() - 3.0).abs() < 0.2 && (w.variance() - 4.0).abs() < 0.55,
        "acumulativo: media {} (verdad 3), varianza {} (verdad 4)",
        w.mean(),
        w.variance()
    );
    for _ in 0..100_000 {
        w.update(3.0 + 2.0 * g.normal());
    }
    assert!(w.is_decay, "tras 2000 muestras pasa a modo exponencial");
    assert!(
        (w.mean() - 3.0).abs() < 0.25 && (w.variance() - 4.0).abs() < 0.75,
        "exponencial: media {} (verdad 3), varianza {} (verdad 4)",
        w.mean(),
        w.variance()
    );
}

/// Verdad: 1 − e^(−100/37) tras un escalón 0 → 1 sostenido 100 ms con
/// τ = 37 ms, repartido en pasos dt aleatorios ≤ 5 ms. La ganancia
/// −expm1(−dt/τ) compone exactamente; con ≈ 40 pasos el error de redondeo
/// acumulado es ≈ 1e-14, de ahí la tolerancia de 1e-12.
#[test]
fn cf2_ewma_tiempo_fisico_respuesta_al_escalon() {
    let tau = 37.0;
    let mut a = Ewma::new(0.5);
    a.update_elapsed(0.0, 0.0, tau).unwrap();
    let mut g = Lcg::new(9);
    let mut t = 0.0f64;
    let mut pasos = 0usize;
    while t < 100.0 {
        let dt = (5.0 * g.u()).min(100.0 - t);
        a.update_elapsed(1.0, dt, tau).unwrap();
        t += dt;
        pasos += 1;
    }
    let esperado = 1.0 - (-100.0f64 / tau).exp();
    assert!(
        (a.get() - esperado).abs() < 1e-12,
        "{} frente a {esperado} tras {pasos} pasos",
        a.get()
    );
}
