//! C-F2.3b (Línea C): las firmas de nivel 2 de `path_signatures.rs` frente a
//! formas cerradas.
//!
//! Teoría que se prueba:
//! - Firma geométrica de un camino colineal: para p(t) = 100·e^{μt} el camino
//!   (log P, t/T) es una recta de (0, 0) a (a, 1) con a = μT, y su firma es
//!   S¹ = a, S² = 1, S^{ij} = X^i X^j/2 con cualquier subdivisión, también con
//!   timestamps irregulares.
//! - Área de Lévy de un lazo cerrado: ½(S¹² − S²¹) es el área orientada
//!   (fórmula de Green); para el polígono regular de n lados inscrito en el
//!   círculo unidad vale (n/2)·sin(2π/n), y S¹ = S² = 0.
//!
//! El archivo también probaba `ShannonEntropyEngine`, el gemelo sin
//! consumidor de la entropía del núcleo; se retiró con él (RET-46) y su
//! forma cerrada se portó al estimador vivo en
//! crates/god-engine-core/tests/cf2_vpin_entropia_solucion_conocida.rs.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use feature_engine::path_signatures::{firma_nivel2_incrementos, firma_ventana_logprecio};

/// Verdad: firma de la recta de (0, 0) a (a, 1), a = μT = 0,01. Tolerancia
/// 1e-12: 9 pasos con un error de ln_1p de ≈ 2e-16 cada uno.
#[test]
fn cf2_firma_exponencial_irregular_forma_cerrada() {
    let mu = 2.5e-6; // log-retorno por ms
    let ts: Vec<u64> = vec![0, 7, 19, 20, 150, 151, 900, 1_000, 2_345, 4_000];
    let p: Vec<f64> = ts.iter().map(|&t| 100.0 * (mu * t as f64).exp()).collect();
    let s = firma_ventana_logprecio(&p, &ts).expect("ventana válida");
    let a = mu * 4_000.0;
    let tol = 1e-12;
    assert!((s.level1[0] - a).abs() < tol, "S1 {}", s.level1[0]);
    assert!((s.level1[1] - 1.0).abs() < tol, "S2 {}", s.level1[1]);
    assert!((s.level2[0][0] - a * a / 2.0).abs() < tol, "S11 {}", s.level2[0][0]);
    assert!((s.level2[0][1] - a / 2.0).abs() < tol, "S12 {}", s.level2[0][1]);
    assert!((s.level2[1][0] - a / 2.0).abs() < tol, "S21 {}", s.level2[1][0]);
    assert!((s.level2[1][1] - 0.5).abs() < tol, "S22 {}", s.level2[1][1]);
}

/// Verdad: área orientada (n/2)·sin(2π/n) para n = 3, 6 y 64, y S¹ = S² = 0.
/// Tolerancia 1e-12: la fórmula de Green es exacta para lados rectos y el
/// redondeo es de orden 1e-15.
#[test]
fn cf2_firma_area_de_levy_poligono_regular() {
    for n in [3usize, 6, 64] {
        let v: Vec<(f64, f64)> = (0..=n)
            .map(|k| {
                let th = std::f64::consts::TAU * k as f64 / n as f64;
                (th.cos(), th.sin())
            })
            .collect();
        let inc: Vec<(f64, f64)> = v
            .windows(2)
            .map(|w| (w[1].0 - w[0].0, w[1].1 - w[0].1))
            .collect();
        let s = firma_nivel2_incrementos(&inc);
        let area = 0.5 * (s.level2[0][1] - s.level2[1][0]);
        let esperado = 0.5 * n as f64 * (std::f64::consts::TAU / n as f64).sin();
        assert!((area - esperado).abs() < 1e-12, "n={n}: {area} frente a {esperado}");
        assert!(
            s.level1[0].abs() < 1e-12 && s.level1[1].abs() < 1e-12,
            "n={n}: lazo no cerrado ({}, {})",
            s.level1[0],
            s.level1[1]
        );
    }
}
