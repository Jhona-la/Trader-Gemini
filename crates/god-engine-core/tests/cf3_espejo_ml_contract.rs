//! C-F3.3 (Línea C), hallazgo F3.3-04: el sesgo spot rompe el espejo de la
//! puerta ML de `puertas_del_continuo` (F-009 / K-06).
//!
//! Teoría que se prueba:
//! - La puerta pondera la confianza de cada intención con la divergencia
//!   direccional `d = normalized_directional_divergence(ml_prob, base)`
//!   (lib.rs, `puertas_del_continuo`, paso 3): d = (p − b)/(1 − b) si p ≥ b y
//!   (p − b)/b si p < b. El largo usa d y el corto −d; la confianza se
//!   multiplica por (1 + d_lado/2) a favor y por (1 + d_lado/2) ∈ [0,2; 1]
//!   en contra, con veto si d_lado < −0,80.
//! - `ml_prob = compose_ml_prob(ensamble, spot_bias)` suma el sesgo spot en
//!   unidades de probabilidad.
//! - Con la opinión del modelo en su base (sin información) y |s| igual a
//!   cada lado: d(+s) = s/(1 − b) y d(−s) = −s/b. El corto espejo recibe
//!   (1 − b)/b veces el impulso del largo: 4,56 con b = 0,18 (BNB) y 3,65
//!   con b = 0,215 (ATOM).
//!
//! Propiedad (condición NECESARIA del espejo, no suficiente: el escéptico
//! de F3.3-04 recuerda que el bosque sirve P(TP del largo), sin espejo propio,
//! ver F3.8-02): invertir el signo del sesgo spot invierte la divergencia que
//! ve la puerta, d(b, +s) + d(b, −s) = 0. Es la misma raíz que F3.1-07,
//! F3.5-04 y F3.8-01.
//!
//! Tolerancia 1e-9: dos restas y una división en f64; el defecto más pequeño
//! de la rejilla (b = 0,30, s = 0,01) mide 1,9·10⁻².
//!
//! Convención de la fase C-F3 (docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un
//! test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlo:
//! `cargo test ... -- --include-ignored`.

use god_engine_core::calibration::{compose_ml_prob, normalized_directional_divergence as ndd};

/// Verdad: ndd(compose(b, s), b) + ndd(compose(b, −s), b) = 0 para las bases
/// de los bosques servidos (0,18 BNB, 0,215 ATOM) y 0,30.
#[test]
#[ignore = "OPEN F3.3-04 (Línea C): compose_ml_prob suma el sesgo spot en p y la puerta lo normaliza por (1-b) arriba y por b abajo; el corto espejo recibe (1-b)/b veces el impulso del largo"]
fn cf3_sesgo_spot_es_espejo_en_la_puerta_ml() {
    let mut fallos = Vec::new();
    for b in [0.18_f64, 0.215, 0.30] {
        for s in [0.01_f64, 0.03, 0.05] {
            let largo = ndd(compose_ml_prob(b, s), b);
            let corto = ndd(compose_ml_prob(b, -s), b);
            println!(
                "observado: b = {b:.3}, s = {s:.2}: d(+s) = {largo:+.4} (x{:.4} al largo), \
                 d(-s) = {corto:+.4} (x{:.4} al corto), suma {:+.4}",
                1.0 + 0.5 * largo,
                1.0 - 0.5 * corto,
                largo + corto
            );
            if (largo + corto).abs() >= 1e-9 {
                fallos.push((b, s, largo, corto));
            }
        }
    }
    assert!(
        fallos.is_empty(),
        "el sesgo spot no es espejo en la puerta ML; (b, s, d(+s), d(-s)): {fallos:?}"
    );
}
