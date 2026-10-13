//! C-F2.3b (Línea C): `KalmanFilter1D` frente al modelo lineal gaussiano que
//! lo define.
//!
//! Teoría que se prueba: para el paseo aleatorio x_t = x_{t−1} + w_t,
//! w ~ N(0, q), observado como y_t = x_t + v_t, v ~ N(0, r), el filtro de
//! Kalman escalar es el estimador óptimo. Con q = 1 y r = 4:
//! - la covarianza converge a la solución de la Riccati estacionaria,
//!   P⁻ = (q + √(q² + 4qr))/2 = (1 + √17)/2 ≈ 2,5616 y P⁺ = P⁻r/(P⁻ + r)
//!   ≈ 1,5616;
//! - el error cuadrático medio del estimado vale P⁺;
//! - las innovaciones ν = y − x̂⁻ son blancas con varianza S = P⁻ + r;
//! - el intervalo x̂ ± 1,96·√P⁺ cubre el estado verdadero el 95 % de las
//!   veces.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use feature_engine::kalman::KalmanFilter1D;

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }
    /// Normal estándar por Box-Muller.
    fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Riccati estacionaria del paseo aleatorio observado: (P⁻, P⁺).
fn riccati(q: f64, r: f64) -> (f64, f64) {
    let p_pred = (q + (q * q + 4.0 * q * r).sqrt()) / 2.0;
    (p_pred, p_pred * r / (p_pred + r))
}

/// Simula `n` pasos tras `quemado` con el filtro inicializado en P⁺ y
/// devuelve (errores x̂ − x, innovaciones ν) del tramo medido.
fn simular(q: f64, r: f64, n: usize, quemado: usize, semilla: u64) -> (Vec<f64>, Vec<f64>) {
    let (_, p_post) = riccati(q, r);
    let mut rng = Lcg(semilla);
    let mut kf = KalmanFilter1D::new(0.0, p_post, q, r);
    let mut x = 0.0f64;
    let mut errores = Vec::with_capacity(n);
    let mut nus = Vec::with_capacity(n);
    for t in 0..(n + quemado) {
        x += q.sqrt() * rng.n();
        let y = x + r.sqrt() * rng.n();
        let nu = y - kf.x;
        let est = kf.update(y);
        if t >= quemado {
            errores.push(est - x);
            nus.push(nu);
        }
    }
    (errores, nus)
}

/// Verdad: P⁺ = 1,5616 (Riccati). Determinista: (1 − K)² ≈ 0,37 por paso,
/// así que tras 200 pasos el transitorio desde P₀ = 10 es < 1e-80 y la
/// tolerancia 1e-9 sólo cubre el redondeo.
#[test]
fn cf2_kalman_riccati_estacionaria() {
    let (_, p_post) = riccati(1.0, 4.0);
    assert!((p_post - 1.5616).abs() < 1e-4, "P+ cerrada {p_post}");
    let mut kf = KalmanFilter1D::new(0.0, 10.0, 1.0, 4.0);
    for _ in 0..200 {
        kf.update(0.0);
    }
    assert!((kf.p - p_post).abs() < 1e-9, "P {} frente a {}", kf.p, p_post);
}

/// Verdades: ECM = P⁺, E[ν²/S] = 1 y autocorrelación de orden 1 de ν = 0.
/// Tolerancias con N = 20 000:
/// - ECM ± 0,10: el error es un AR(1) con φ = r/S = 0,6096, el factor de
///   autocorrelación de e² es (1 + φ²)/(1 − φ²) = 2,183 y la SE es
///   1,5616·√(2·2,183/20 000) = 0,023 (4,3 SE);
/// - ν²/S ± 0,04: SE = √(2/N) = 0,01 (4 SE);
/// - ρ₁ ± 0,028: SE = 1/√N = 0,0071 (4 SE).
#[test]
fn cf2_kalman_recupera_modelo_lineal_gaussiano() {
    let (q, r) = (1.0f64, 4.0f64);
    let (p_pred, p_post) = riccati(q, r);
    let s = p_pred + r;
    let n = 20_000usize;
    let (errores, nus) = simular(q, r, n, 500, 0x4A13_A11);
    let mse = errores.iter().map(|e| e * e).sum::<f64>() / n as f64;
    let z2 = nus.iter().map(|v| v * v / s).sum::<f64>() / n as f64;
    let m = nus.iter().sum::<f64>() / n as f64;
    let var = nus.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
    let rho1 = nus
        .windows(2)
        .map(|w| (w[0] - m) * (w[1] - m))
        .sum::<f64>()
        / (n as f64 * var);
    eprintln!("[cf2 kalman] ECM {mse:.4} (P+ {p_post:.4}), nu^2/S {z2:.4}, rho1 {rho1:.4}");
    assert!((mse - p_post).abs() < 0.10, "ECM {mse:.4} frente a P+ {p_post:.4}");
    assert!((z2 - 1.0).abs() < 0.04, "media de nu^2/S {z2:.4}");
    assert!(rho1.abs() < 0.028, "autocorrelación de innovaciones {rho1:.4}");
}

/// Verdad: cobertura 0,95 de x̂ ± 1,96·√P⁺ (puerta de cobertura del plan,
/// añadida por el escéptico). Tolerancia 0,01: con N = 20 000 la SE iid es
/// √(0,95·0,05/N) = 0,0015; el indicador de |e| ≤ c es una función par de
/// un AR(1) gaussiano, su autocorrelación a lag k es ≤ φ^{2k} y el factor de
/// varianza queda acotado por (1 + φ²)/(1 − φ²) = 2,183, así que la SE es
/// ≤ 0,0023 y la tolerancia son ≥ 4,4 SE.
#[test]
fn cf2_kalman_cobertura_95_estacionaria() {
    let (q, r) = (1.0f64, 4.0f64);
    let (_, p_post) = riccati(q, r);
    let n = 20_000usize;
    let (errores, _) = simular(q, r, n, 500, 0x4A13_A12);
    let c = 1.959_963_984_540_054 * p_post.sqrt();
    let dentro = errores.iter().filter(|e| e.abs() <= c).count();
    let cobertura = dentro as f64 / n as f64;
    eprintln!("[cf2 kalman] cobertura {cobertura:.4}");
    assert!((cobertura - 0.95).abs() < 0.01, "cobertura {cobertura:.4} frente a 0,95");
}
