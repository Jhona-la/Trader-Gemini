//! C-F2.6 (Línea C): el «Lyapunov» de QuantumTensorStore frente a la
//! solución conocida del mapa logístico.
//!
//! Teoría que se prueba: el exponente de Lyapunov de un mapa x ↦ f(x) es
//! λ = lim (1/n)·Σ ln|f'(x_k)|. En el mapa logístico f(x) = r·x(1 − x):
//! - con r = 2,9 el punto fijo x* = 1 − 1/2,9 es atractor y λ = ln|f'(x*)| =
//!   ln 0,9 < 0 (contracción);
//! - con r = 4, λ = ln 2 (caos), y λ es invariante ante cualquier cambio de
//!   coordenadas suave, en particular ante el reescalado x ↦ a·x.
//! `calculate_lyapunov_chaos` es RMS(tf0 − tf1) entre dos vectores de
//! features: no puede ser negativo y escala con la unidad de los datos.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.
//! Son citas de falsación para la ficha de retirada de F2.6-05: el módulo no
//! tiene consumidor y estas pruebas se van con él.
//!
//! Las dos pruebas de SimdNeuralNet (igualdad bit a bit con una referencia
//! escalar y gradiente de la entropía cruzada) se descartaron: la primera es
//! tautológica (la referencia replica el mismo orden de operaciones) y las
//! dos certifican un módulo sin consumidor que se retira por el grep
//! (F2.6-06); no aportan evidencia a ninguna decisión viva.

use feature_engine::{QuantumTensorStore, NUM_FEATURES};

/// Órbita del mapa logístico tras `quemado` iteraciones: n valores.
fn orbita_logistica(r: f64, x0: f64, quemado: usize, n: usize) -> Vec<f64> {
    let mut x = x0;
    for _ in 0..quemado {
        x = r * x * (1.0 - x);
    }
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(x);
        x = r * x * (1.0 - x);
    }
    v
}

/// Escribe x_0..x_63 en tf0 y x_1..x_64 en tf1 (pares consecutivos de la
/// órbita, multiplicados por `escala`) y devuelve el «Lyapunov» del código.
fn caos(xs: &[f64], escala: f64) -> f64 {
    let store = QuantumTensorStore::new(1);
    for i in 0..NUM_FEATURES {
        store.write_feature(0, 0, i, escala * xs[i]);
        store.write_feature(0, 1, i, escala * xs[i + 1]);
    }
    store.calculate_lyapunov_chaos(0)
}

/// Verdad: r = 2,9, x* = 1 − 1/2,9 ≈ 0,6552 y λ = ln|r(1 − 2x*)| = ln 0,9 =
/// −0,1054 (exacto). Se usa el transitorio desde x_0 = 0,665 sin quemado, con
/// desviaciones d_n ≈ 0,01·(−0,9)^n: d_64 ≈ 1,2e-5, todavía resoluble frente al
/// redondeo de x* (revisión cruzada, CL-65: con r = 2,5 y d_n = 0,01·2^{−n} la
/// cola de la serie ya era x* exacto hacia n ≈ 47 y ningún estimador honesto
/// podía dar ln 0,5 con ella). Control del generador: la media de
/// ln|d_(n+1)/d_n| en los 64 pasos da −0,10518 (el término no lineal r·d_n
/// alterna de signo y casi se cancela). Tolerancia 0,03 en los dos, que deja
/// λ separado de 0 por más de 2,5 tolerancias.
/// Hoy el código da una RMS ≥ 0: no puede ser negativo.
#[test]
#[ignore = "OPEN F2.6-05 (Línea C): calculate_lyapunov_chaos es RMS(tf0 − tf1) ≥ 0 y no puede dar el exponente negativo de un punto fijo estable"]
fn cf2_6_lyapunov_de_un_punto_fijo_estable_es_negativo() {
    let r = 2.9f64;
    let x_fijo = 1.0 - 1.0 / r;
    let verdad = 0.9f64.ln();
    let xs = orbita_logistica(r, 0.665, 0, NUM_FEATURES + 1);
    let referencia = (0..NUM_FEATURES)
        .map(|n| ((xs[n + 1] - x_fijo) / (xs[n] - x_fijo)).abs().ln())
        .sum::<f64>()
        / NUM_FEATURES as f64;
    assert!(
        (referencia - verdad).abs() < 0.03,
        "control del generador: media de ln|d(n+1)/d(n)| = {referencia}"
    );
    let l = caos(&xs, 1.0);
    assert!(
        (l - verdad).abs() < 0.03,
        "λ̂ = {l}; λ verdadero = ln 0,9 = {verdad} (referencia de la serie {referencia})"
    );
}

/// Verdad: r = 4, λ = ln 2 en cualquier unidad de x (el exponente es
/// invariante ante x ↦ a·x). Se exige λ̂(1000·x) = λ̂(x) con tolerancia 1e-9,
/// la del redondeo. Hoy la función es una RMS lineal en la escala: bajo la
/// densidad arcoseno la RMS de x − f(x) es 0,5, así que la diferencia ronda
/// 999·0,5 ≈ 500.
#[test]
#[ignore = "OPEN F2.6-05 (Línea C): el «Lyapunov» es una RMS lineal en la unidad de los datos; multiplicar la serie por 1000 lo multiplica por 1000"]
fn cf2_6_lyapunov_no_depende_de_la_unidad_de_medida() {
    let xs = orbita_logistica(4.0, 0.123_456_7, 1_000, NUM_FEATURES + 1);
    let (l1, l1000) = (caos(&xs, 1.0), caos(&xs, 1_000.0));
    assert!(
        (l1000 - l1).abs() < 1e-9,
        "λ̂(x) = {l1}, λ̂(1000·x) = {l1000}: depende de la unidad (λ = ln 2 = {})",
        2.0f64.ln()
    );
}
