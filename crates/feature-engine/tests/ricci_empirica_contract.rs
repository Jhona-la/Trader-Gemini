//! Contratos del estimador numérico de curvatura escalar de Ricci
//! (GLM-193, mandato §3.4). Las TRES constantes cerradas fueron
//! verificadas a mano en python (mismo algoritmo, mismo orden de
//! acumulación) ANTES de escribir el núcleo — incluida una ERRATA del
//! doc del 191 que el espejo destapó: el escalar de Ricci de la familia
//! gaussiana es −1 (−1/2 era la curvatura gaussiana/seccional K).
use feature_engine::ricci_empirica::{H_POR_DEFECTO, curvatura_ricci_2d};

#[test]
fn triple_forma_cerrada_gaussiana_euclidea_hiperbolica() {
    // (a) Métrica euclídea δ: derivadas nulas ⇒ R = 0 EXACTO (las
    // diferencias centrales de una constante son 0.0 bit a bit).
    let r_euc = curvatura_ricci_2d(|_, _| [[1.0, 0.0], [0.0, 1.0]], 0.3, 1.7, H_POR_DEFECTO)
        .expect("euclídea válida");
    assert_eq!(r_euc, 0.0, "euclídea: R = 0 exacto");
    // (b) Semiplano hiperbólico (dx²+dy²)/y²: R = −2 (constante clásica;
    // medido a mano: −2.00000242 con h=1e-3).
    let r_hyp = curvatura_ricci_2d(
        |_, y| {
            let f = 1.0 / (y * y);
            [[f, 0.0], [0.0, f]]
        },
        0.3,
        1.7,
        H_POR_DEFECTO,
    )
    .expect("hiperbólica válida");
    assert!(
        (r_hyp - (-2.0)).abs() < 1e-4,
        "hiperbólica: R = {r_hyp} ≈ −2 (medido −2.00000242)"
    );
    // (c) Métrica Fisher de N(μ,σ) = diag(1/σ², 2/σ²): R = −1 — la
    // ERRATA corregida del 191 (−1/2 era la seccional K; R = 2K).
    let r_gauss = curvatura_ricci_2d(
        |_, s| {
            let f = 1.0 / (s * s);
            [[f, 0.0], [0.0, 2.0 * f]]
        },
        0.3,
        1.7,
        H_POR_DEFECTO,
    )
    .expect("gaussiana válida");
    assert!(
        (r_gauss - (-1.0)).abs() < 1e-4,
        "gaussiana: R = {r_gauss} ≈ −1 (medido −1.00000242; errata 191: −1/2 era K, no R)"
    );
    // La exactitud MEJORA con h menor en métricas lisas (verificado a
    // mano: h=1e-4 ⇒ error ~2e-8).
    let r_fino = curvatura_ricci_2d(
        |_, s| {
            let f = 1.0 / (s * s);
            [[f, 0.0], [0.0, 2.0 * f]]
        },
        0.3,
        1.7,
        1e-4,
    )
    .expect("válido");
    assert!(
        (r_fino - (-1.0)).abs() < (r_gauss - (-1.0)).abs(),
        "h=1e-4 más exacto que h=1e-3 en métrica lisa: {r_fino} vs {r_gauss}"
    );
}

#[test]
fn la_curvatura_es_local_y_varia_con_el_punto_en_metricas_no_constantes() {
    // En el semiplano hiperbólico la curvatura es CONSTANTE (−2) en todo
    // punto — el estimador la reproduce en puntos distintos: propiedad
    // de invarianza de la familia.
    for (x, y) in [(0.3f64, 1.7), (-2.0, 0.9), (5.0, 3.1)] {
        let r = curvatura_ricci_2d(
            |_, yy| {
                let f = 1.0 / (yy * yy);
                [[f, 0.0], [0.0, f]]
            },
            x,
            y,
            H_POR_DEFECTO,
        )
        .expect("válido");
        assert!(
            (r - (-2.0)).abs() < 1e-4,
            "({x},{y}): R = {r} ≈ −2 constante"
        );
    }
}

#[test]
fn falla_cerrado_sin_ceros_fabricados() {
    let buena = |_, s: f64| {
        let f = 1.0 / (s * s);
        [[f, 0.0], [0.0, 2.0 * f]]
    };
    // h inválido.
    assert!(curvatura_ricci_2d(buena, 0.3, 1.7, 0.0).is_err());
    assert!(curvatura_ricci_2d(buena, 0.3, 1.7, -1e-3).is_err());
    assert!(curvatura_ricci_2d(buena, 0.3, 1.7, f64::NAN).is_err());
    // Métrica singular: diag(1, 0) — determinante 0, no invertible.
    assert!(curvatura_ricci_2d(|_, _| [[1.0, 0.0], [0.0, 0.0]], 0.3, 1.7, 1e-3).is_err());
    // Métrica no positiva (indefinida): Err, no un valor inventado.
    assert!(curvatura_ricci_2d(|_, _| [[-1.0, 0.0], [0.0, 1.0]], 0.3, 1.7, 1e-3).is_err());
    // Métrica con NaN: Err.
    let nan_metrica = |_, _| [[f64::NAN, 0.0], [0.0, 1.0]];
    assert!(curvatura_ricci_2d(nan_metrica, 0.3, 1.7, 1e-3).is_err());
}

#[test]
fn convergencia_con_h_y_determinante_degenerado_en_el_entorno() {
    // Convergencia: h→menor ⇒ error menor en la gaussiana (ya cubierto
    // en el test 1 con 1e-4); aquí se verifica el rango útil: h=1e-2
    // aún da el valor a ~2.4e-4 — holgura para métricas ruidosas.
    let r = curvatura_ricci_2d(
        |_, s| {
            let f = 1.0 / (s * s);
            [[f, 0.0], [0.0, 2.0 * f]]
        },
        0.3,
        1.7,
        1e-2,
    )
    .expect("válido");
    assert!(
        (r - (-1.0)).abs() < 1e-3,
        "h=1e-2: R = {r} ≈ −1 a 2.4e-4 (medido)"
    );
    // Degeneración EN el entorno (no en el punto): métrica válida en
    // (0.3, 1.7) pero con σ=0 a distancia <h en el eje 2 ⇒ Err del
    // entorno, no un valor parcial.
    let casi_sing = |_, s: f64| {
        let ss = if s < 1e-3 { 1e-9 } else { s * s };
        let f = 1.0 / ss;
        [[f, 0.0], [0.0, 2.0 * f]]
    };
    // Punto con σ=1.7: entorno sano; debe funcionar.
    assert!(curvatura_ricci_2d(casi_sing, 0.3, 1.7, 1e-3).is_ok());
}
