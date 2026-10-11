//! C-F3.5 (Línea C), hallazgo F3.5-04: el sesgo spot no tiene espejo cuando
//! la base del bosque no es ½; el mismo diferencial empuja (1 − b)/b veces
//! más hacia el corto.
//!
//! Teoría que se prueba:
//! - `compose_ml_prob(p, s) = clamp(p + s; 0; 1)` suma el sesgo spot en
//!   unidades absolutas de probabilidad.
//! - Los dos consumidores que miden la opinión contra la base del modelo
//!   dividen por semiintervalos distintos, (1 − b) por encima y b por
//!   debajo: `nn_score` (CL-13, `StatefulEngine::update_ml_prediction`) y las
//!   puertas (`normalized_directional_divergence`, K-06).
//! - Con la opinión en su base: f(+s) = s/(1 − b) y f(−s) = −s/b. Con
//!   b = 0,181 y s = 0,01: +0,0122 frente a −0,0552.
//!
//! Propiedad: el sesgo spot es IMPAR en la escala que leen `nn_score` y las
//! puertas: f(b, +s) + f(b, −s) = 0. Los escépticos de F3.5-04 precisan que
//! un arreglo «en log-odds» (logit p + κ·s) NO pasa esta prueba (con
//! b = 0,181 y κs = 0,3: +0,0595 frente a −0,2227); sí pasa sumar κ·s
//! DESPUÉS de la divergencia con el mismo peso a cada lado. Prueba y arreglo
//! eligen la misma escala. Misma raíz que F3.1-07, F3.3-04 y F3.8-01.
//!
//! Bases: las de los bosques versionados (0,181 BNB, 0,215 ATOM) y 0,30.
//! Tolerancia 1e-12: una suma y una división en f64 (error ≲ 1e-16); el
//! defecto más pequeño de la rejilla (b = 0,30, s = 0,01) mide 1,9·10⁻².
//!
//! Convención de la fase C-F3 (docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un
//! test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlo:
//! `cargo test ... -- --include-ignored`.

use god_engine_core::calibration::{compose_ml_prob, normalized_directional_divergence};
use god_engine_core::stateful_engine::StatefulEngine;

/// Verdad: nn_score(b, +s) + nn_score(b, −s) = 0 y lo mismo para la
/// divergencia de las puertas, en toda la rejilla b × s.
#[test]
#[ignore = "OPEN F3.5-04 (Línea C): el sesgo spot se suma en p y nn_score y las puertas lo dividen por (1-b) arriba y por b abajo; con b = 0,181 y s = 0,01, nn_score da +0,0122 frente a -0,0552"]
fn cf3_sesgo_spot_es_impar_con_base_honesta() {
    let mut fe = StatefulEngine::new();
    let mut fallos = Vec::new();
    for &b in &[0.181_f64, 0.215, 0.30] {
        for &s in &[0.01_f64, 0.03, 0.05] {
            let up = fe.update_ml_prediction(compose_ml_prob(b, s), b);
            let dn = fe.update_ml_prediction(compose_ml_prob(b, -s), b);
            let du = normalized_directional_divergence(compose_ml_prob(b, s), b);
            let dd = normalized_directional_divergence(compose_ml_prob(b, -s), b);
            println!(
                "observado: b = {b:.3}, s = {s:.2}: nn_score {up:+.4} / {dn:+.4} (suma {:+.4}); \
                 puertas {du:+.4} / {dd:+.4} (suma {:+.4})",
                up + dn,
                du + dd
            );
            if (up + dn).abs() >= 1e-12 {
                fallos.push(format!("nn_score b={b} s={s}: {up} vs {dn}"));
            }
            if (du + dd).abs() >= 1e-12 {
                fallos.push(format!("puertas b={b} s={s}: {du} vs {dd}"));
            }
        }
    }
    assert!(
        fallos.is_empty(),
        "el sesgo spot no es impar con base honesta: {fallos:#?}"
    );
}
