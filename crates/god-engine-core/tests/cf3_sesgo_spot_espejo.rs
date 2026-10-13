//! C-F3.1 (Línea C), hallazgo F3.1-07: un sesgo spot «simétrico» mueve
//! `nn_score` 2,3 veces más hacia abajo que hacia arriba cuando la base del
//! bosque es 0,30.
//!
//! Teoría que se prueba:
//! - El sesgo de microestructura del núcleo es impar en la base
//!   spot-futuros: `spot_bias = clamp(basis_pb / 10; ±0,05)` (lib.rs, bloque
//!   «Spot vs Futures»). Una base de +b pb y otra de −b pb dan ±s.
//! - `compose_ml_prob` suma s en UNIDADES DE PROBABILIDAD: p = b + s.
//! - `nn_score` (CL-13, `StatefulEngine::update_ml_prediction`) mide la
//!   opinión contra la base del modelo normalizando cada lado por su propio
//!   recorrido: d/(1 − b) por encima de la base y d/b por debajo.
//! - Compuestos: nn(+s) = s/(1 − b) y nn(−s) = −s/b, cuya suma es
//!   s·(1/(1 − b) − 1/b). Sólo es nula con b = ½. Con b = 0,30 y s = 0,05:
//!   +0,0714 frente a −0,1667 (suma −0,0952); con b = 0,18: +0,0610 frente a
//!   −0,2778 (suma −0,2168).
//!
//! Propiedad: un dato de microestructura simétrico no debe dar al compuesto
//! un empuje distinto según el lado. Es la misma raíz que F3.3-04 (puertas),
//! F3.5-04 (nn_score y puertas a la vez) y F3.8-01 (núcleo completo): el
//! sesgo se suma en p y después se mide en la escala por semiintervalos.
//! Las cuatro pruebas comparten una definición de espejo: imparidad en la
//! escala que lee el consumidor (escéptico de F3.5-04: sumar el sesgo en
//! log-odds NO pasa esta prueba; sí pasa sumarlo DESPUÉS de la divergencia
//! con el mismo peso a cada lado).
//!
//! Tolerancia 1e-9: las operaciones son una suma y una división en f64 con
//! error relativo ≲ 1e-15; el defecto mide ≥ 9,5·10⁻² con b = 0,30.
//!
//! Convención de la fase C-F3 (docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un
//! test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlo:
//! `cargo test ... -- --include-ignored`.

use god_engine_core::calibration::compose_ml_prob;
use god_engine_core::stateful_engine::StatefulEngine;

/// Verdad: nn(b, +0,05) + nn(b, −0,05) = 0 para cada base. Con b = 0,50 se
/// cumple hoy (control); con b = 0,30 y 0,18 no.
#[test]
#[ignore = "OPEN F3.1-07 (Codex: compose_ml_prob, con el autor de CL-13; punto de aplicación en lib.rs, Línea C): el sesgo spot se suma en p antes del mapeo por semiintervalos de CL-13, así que +s da s/(1-b) y -s da -s/b"]
fn cf3_un_sesgo_spot_simetrico_mueve_nn_score_simetricamente() {
    let mut e = StatefulEngine::new();
    let mut filas = Vec::new();
    for &b in &[0.50, 0.30, 0.18] {
        let arriba = e.update_ml_prediction(compose_ml_prob(b, 0.05), b);
        let abajo = e.update_ml_prediction(compose_ml_prob(b, -0.05), b);
        println!(
            "observado: base {b:.2}: nn(+0,05) = {arriba:+.4}, nn(-0,05) = {abajo:+.4}, suma {:+.4}",
            arriba + abajo
        );
        filas.push((b, arriba, abajo));
    }
    // Control: con b = ½ el mapeo es simétrico y la suma es exactamente 0.
    let (_, a50, d50) = filas[0];
    assert!(
        (a50 + d50).abs() < 1e-9,
        "control b = 0,50: el fixture no reproduce el mapeo de CL-13 ({a50} y {d50})"
    );
    for (b, arriba, abajo) in filas {
        assert!(
            (arriba + abajo).abs() < 1e-9,
            "base {b}: sesgo +0,05 da {arriba:.4} y -0,05 da {abajo:.4}"
        );
    }
}
