//! Cópula t bivariada por par — medición de dependencia de COLA (LXXI).
//!
//! Portón ADR-0006: la medición precede al cableado (doctrina Fisher/D-754,
//! como transfer_entropy en XLVIII·E). El veto same-bet usa hoy ρ̄
//! (Hayashi-Yoshida×signo + curl_share², XLVI·D/E); una cópula t aporta lo
//! que un ρ lineal no ve: dependencia de cola
//! λ = P(Y en su cola | X en la suya) — la cantidad que gobierna los
//! stop-out simultáneos de un grupo misma-apuesta. Si ν̂ resulta grande o
//! λ̂≈0 en los horizontes de trading, el tratamiento gaussiano/EWMA basta y
//! la decoración se cierra antes de nacer.
//!
//! Nada de este módulo consume el veto; es instrumento de medición. Las
//! convenciones: τ-a de Kendall exacto O(n²) (los empates cuentan en el
//! denominador y en ninguno de los numeradores); ρ por inversión elíptica
//! ρ = sin(πτ/2); ν por máxima verosimilitud sobre grid con ρ fijo;
//! λ implícita de la fórmula cerrada de la t-cópula.

use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────
// 1. Kendall τ-a exacto
// ─────────────────────────────────────────────────────────────────

/// τ-a de Kendall O(n²) exacto. `None` si n<2, longitudes difieren, algún
/// valor no es finito, o una serie es constante (sin variación: medición
/// no realizada, no independencia).
pub fn tau_kendall(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    if !x.iter().chain(y.iter()).all(|v| v.is_finite()) {
        return None;
    }
    let mut con = 0i64;
    let mut dis = 0i64;
    let mut x_varia = false;
    let mut y_varia = false;
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            if dx != 0.0 {
                x_varia = true;
            }
            if dy != 0.0 {
                y_varia = true;
            }
            let prod = dx * dy;
            if prod > 0.0 {
                con += 1;
            } else if prod < 0.0 {
                dis += 1;
            }
            // prod == 0 (empate en x o en y): τ-a no lo cuenta en ningún
            // numerador pero SÍ queda en el denominador n(n−1)/2.
        }
    }
    if !x_varia || !y_varia {
        return None;
    }
    let den = (x.len() * (x.len() - 1) / 2) as f64;
    Some((con - dis) as f64 / den)
}

/// Inversión elíptica (válida para cópulas t y gaussiana):
/// ρ = sin(πτ/2).
pub fn rho_desde_tau(tau: f64) -> f64 {
    (PI * tau / 2.0).sin()
}

// ─────────────────────────────────────────────────────────────────
// 2. t-Student univariada (sin deps: Lanczos + fracción continua)
// ─────────────────────────────────────────────────────────────────

/// log Γ(x) — aproximación de Lanczos (g=7, n=9).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // reflexión: Γ(x)Γ(1−x) = π/sin(πx)
        (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Beta incompleta regularizada I_x(a,b) — fracción continua de
/// Numerical Recipes §6.4 con expansión simétrica.
fn beta_incompleta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let frontal = (a * x.ln() + b * (1.0 - x).ln() + lbeta).exp();
    // fracción continua (modificada Lentz)
    if x < (a + 1.0) / (a + b + 2.0) {
        frontal * beta_cf(a, b, x) / a
    } else {
        // simetría I_x(a,b) = 1 − I_{1−x}(b,a)
        1.0 - frontal_alt(a, b, x) / b
    }
}

fn frontal_alt(a: f64, b: f64, x: f64) -> f64 {
    let lbeta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    (b * (1.0 - x).ln() + a * x.ln() + lbeta).exp() * beta_cf(b, a, 1.0 - x)
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        // incluso
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // impar
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// CDF de la t de Student con ν grados de libertad.
pub fn t_cdf(x: f64, nu: f64) -> f64 {
    debug_assert!(nu > 0.0);
    let nu = nu.max(1e-9);
    let beta = 0.5 * nu;
    let z = nu / (nu + x * x);
    let incompleta = beta_incompleta(beta, 0.5, z);
    if x > 0.0 {
        1.0 - 0.5 * incompleta
    } else {
        0.5 * incompleta
    }
}

/// log-PDF de la t de Student.
pub fn t_log_pdf(x: f64, nu: f64) -> f64 {
    let nu = nu.max(1e-9);
    ln_gamma((nu + 1.0) / 2.0) - ln_gamma(nu / 2.0)
        - 0.5 * (nu * PI).ln()
        - ((nu + 1.0) / 2.0) * (1.0 + x * x / nu).ln()
}

/// Cuantil de la t de Student por bisección sobre la CDF.
/// p fuera de (0,1) o ν≤0 ⇒ None.
pub fn t_cuantil(p: f64, nu: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) || nu <= 0.0 || !p.is_finite() {
        return None;
    }
    if p == 0.5 {
        return Some(0.0);
    }
    // bisección simétrica: T es creciente; rango ±1e3 cubre ν≥1
    let (mut lo, mut hi) = (-1e3f64, 1e3f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, nu) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

// ─────────────────────────────────────────────────────────────────
// 3. Cópula t bivariada
// ─────────────────────────────────────────────────────────────────

/// log-densidad de la cópula t en scores (u,v)∈(0,1) con parámetros
/// (ρ, ν). `None` si algún score sale de (0,1), |ρ|≥1 o ν≤0.
pub fn log_copula_t(u: f64, v: f64, rho: f64, nu: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return None;
    }
    if rho.abs() >= 1.0 || nu <= 0.0 {
        return None;
    }
    let x = t_cuantil(u, nu)?;
    let y = t_cuantil(v, nu)?;
    let det = 1.0 - rho * rho;
    let cuad = (x * x - 2.0 * rho * x * y + y * y) / (nu * det);
    // densidad bivariada t₂
    let log_f2 = ln_gamma((nu + 2.0) / 2.0) - ln_gamma(nu / 2.0)
        - (nu * PI).ln()
        - 0.5 * det.ln()
        - ((nu + 2.0) / 2.0) * (1.0 + cuad).ln();
    Some(log_f2 - t_log_pdf(x, nu) - t_log_pdf(y, nu))
}

/// Grid de ν para el MLE: de colas muy pesadas (2) a gaussiana efectiva
/// (100). Coarse a propósito: el veredicto del portón es cualitativo.
pub const GRID_NU: [f64; 13] = [
    2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0,
];

/// MLE de ν sobre `GRID_NU` con ρ fijo (por inversión de τ). Devuelve
/// (ν̂, log-verosimilitud). `None` si algún score es inválido o n==0.
pub fn nu_mle(u: &[f64], v: &[f64], rho: f64) -> Option<(f64, f64)> {
    if u.len() != v.len() || u.is_empty() || rho.abs() >= 1.0 {
        return None;
    }
    let mut mejor: Option<(f64, f64)> = None;
    for &nu in GRID_NU.iter() {
        let mut ll = 0.0f64;
        let mut valido = true;
        for (&ui, &vi) in u.iter().zip(v.iter()) {
            match log_copula_t(ui, vi, rho, nu) {
                Some(l) => ll += l,
                None => {
                    valido = false;
                    break;
                }
            }
        }
        if !valido {
            continue;
        }
        if mejor.is_none_or(|(_, m)| ll > m) {
            mejor = Some((nu, ll));
        }
    }
    mejor
}

/// Dependencia de cola implícita de la cópula t (simétrica U/L):
/// λ = 2·T_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))).
pub fn lambda_cola_t(rho: f64, nu: f64) -> f64 {
    if rho.abs() >= 1.0 {
        return if rho >= 1.0 { 1.0 } else { 0.0 };
    }
    let nu1 = nu + 1.0;
    let arg = ((nu1 * (1.0 - rho) / (1.0 + rho)).sqrt()).min(1e3);
    2.0 * t_cdf(-arg, nu1)
}

/// λ empírica no paramétrica por cuantiles de muestra: devuelve
/// (λ_U, λ_L) al nivel q (p.ej. 0.95: cola superior = 1−q).
/// `None` si no hay suficientes observaciones en la cola (<5).
pub fn lambda_empirica(x: &[f64], y: &[f64], q: f64) -> Option<(f64, f64)> {
    if x.len() != y.len() || x.len() < 20 || !(0.5..1.0).contains(&q) {
        return None;
    }
    fn cmp_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    }
    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(cmp_f64);
    ys.sort_by(cmp_f64);
    let idx = |v: &[f64], p: f64| -> usize {
        let k = (p * v.len() as f64) as usize;
        k.min(v.len() - 1)
    };
    let qx_hi = xs[idx(&xs, q)];
    let qy_hi = ys[idx(&ys, q)];
    let qx_lo = xs[idx(&xs, 1.0 - q)];
    let qy_lo = ys[idx(&ys, 1.0 - q)];
    // λ_U: P(Y > qy_hi | X > qx_hi); λ_L: P(Y < qy_lo | X < qx_lo)
    let (mut nx_up, mut nxy_up, mut nx_lo, mut nxy_lo) = (0usize, 0usize, 0usize, 0usize);
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if xi > qx_hi {
            nx_up += 1;
            if yi > qy_hi {
                nxy_up += 1;
            }
        }
        if xi < qx_lo {
            nx_lo += 1;
            if yi < qy_lo {
                nxy_lo += 1;
            }
        }
    }
    if nx_up < 5 || nx_lo < 5 {
        return None;
    }
    Some((
        nxy_up as f64 / nx_up as f64,
        nxy_lo as f64 / nx_lo as f64,
    ))
}

// ─────────────────────────────────────────────────────────────────
// Contratos (corren siempre: matemática con dientes)
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cerca(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn tau_comonotonica_y_antimonotonica() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let z: Vec<f64> = x.iter().map(|v| -v).collect();
        assert!((tau_kendall(&x, &y).unwrap() - 1.0).abs() < 1e-12);
        assert!((tau_kendall(&x, &z).unwrap() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn tau_rechaza_degenerados() {
        let x = vec![1.0, 2.0, 3.0];
        assert!(tau_kendall(&x, &[1.0]).is_none()); // longitudes
        assert!(tau_kendall(&[1.0], &[1.0]).is_none()); // n<2
        assert!(tau_kendall(&x, &[f64::NAN, 1.0, 2.0]).is_none()); // no finito
        assert!(tau_kendall(&x, &[5.0, 5.0, 5.0]).is_none()); // constante
    }

    #[test]
    fn inversion_eliptica_ancla_conocidas() {
        assert!(cerca(rho_desde_tau(0.0), 0.0, 1e-15));
        assert!(cerca(rho_desde_tau(1.0), 1.0, 1e-15));
        // τ=1/3 ⇒ ρ=sin(π/6)=0.5
        assert!(cerca(rho_desde_tau(1.0 / 3.0), 0.5, 1e-12));
    }

    #[test]
    fn t_cdf_simetria_y_cero() {
        for nu in [1.0, 2.0, 5.0, 30.0] {
            assert!(cerca(t_cdf(0.0, nu), 0.5, 1e-12));
            for x in [-2.5, -0.7, 0.3, 1.9] {
                let a = t_cdf(x, nu);
                let b = 1.0 - t_cdf(-x, nu);
                assert!(cerca(a, b, 1e-12), "nu={nu} x={x}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn t_cuantil_valores_tabla() {
        // valores de tabla clásicos, tolerancia de bisección
        assert!(cerca(t_cuantil(0.975, 1.0).unwrap(), 12.7062, 1e-3));
        assert!(cerca(t_cuantil(0.975, 30.0).unwrap(), 2.0423, 1e-4));
        assert!(cerca(t_cuantil(0.975, 100.0).unwrap(), 1.9840, 1e-4));
        assert!(cerca(t_cuantil(0.5, 7.0).unwrap(), 0.0, 1e-9));
        assert!(t_cuantil(1.5, 5.0).is_none());
        assert!(t_cuantil(0.5, -1.0).is_none());
    }

    #[test]
    fn lambda_cola_decae_con_nu_y_valores_ancla() {
        // ρ=0.5, ν=2: λ = 2·T₃(−1) ≈ 2·0.2063... valor de tabla t₃
        let lambda_v2 = lambda_cola_t(0.5, 2.0);
        assert!(cerca(lambda_v2, 0.3928, 5e-3), "λ(0.5, ν=2)={lambda_v2}");
        // monotonía en ν: colas más pesadas ⇒ más λ
        let lambda_v30 = lambda_cola_t(0.5, 30.0);
        assert!(lambda_v2 > lambda_v30);
        // gaussiana efectiva: λ→0
        assert!(lambda_cola_t(0.5, 1e4) < 0.02);
        // ρ negativo: la cópula t también tiene colas (λ>0 pequeño)
        assert!(lambda_cola_t(-0.5, 4.0) > 0.0);
        // perfecta: λ=1; independencia con ν finita: λ>0 (firma de la t)
        assert!(cerca(lambda_cola_t(1.0, 5.0), 1.0, 1e-12));
    }

    #[test]
    fn copula_t_converge_a_gaussiana_cuando_nu_grande() {
        // densidad cópula gaussiana: c = (1−ρ²)^−1/2 · exp((2ρxy−x²−y²)ρ'²/(2(1−ρ²)))
        // con (x,y) cuantiles normales de (u,v)
        let (u, v, rho): (f64, f64, f64) = (0.15, 0.85, 0.6);
        let x = inversa_normal(u);
        let y = inversa_normal(v);
        let det = 1.0 - rho * rho;
        // c = f₂/(φ(x)φ(y)): el +(x²+y²)/2 es la división por las marginales
        let log_c_gauss = -0.5 * det.ln() - (x * x - 2.0 * rho * x * y + y * y)
            / (2.0 * det)
            + 0.5 * (x * x + y * y);
        let log_c_t = log_copula_t(u, v, rho, 1e5).unwrap();
        assert!(
            cerca(log_c_t, log_c_gauss, 5e-3),
            "t(ν=1e5)={log_c_t} vs gauss={log_c_gauss}"
        );
    }

    /// Aproximación de Acklam para Φ⁻¹ (sólo para el contrato de límite).
    fn inversa_normal(p: f64) -> f64 {
        // racional de Peter Acklam, suficiente a 1e-9
        const A: [f64; 6] = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        const B: [f64; 5] = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        const C: [f64; 6] = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        const D: [f64; 4] = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];
        let p_low = 0.02425;
        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        } else if p <= 1.0 - p_low {
            let q = p - 0.5;
            let r = q * q;
            (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5])
                * q
                / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        }
    }

    /// xorshift64* determinista para el contrato del MLE.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn normal(&mut self) -> f64 {
            // Box-Muller con caché
            let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-12;
            let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }

    #[test]
    fn nu_mle_recupera_nu_sintetica() {
        // muestra t-bivariada con ρ=0.5, ν=4: z gaussiana escalada por
        // √(ν/W) con W~χ²_ν. El MLE con ρ por τ debe devolver ν̂ en el
        // grid cerca de 4 (tolerancia generosa: el grid es coarse y τ≠ρ).
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let (rho_t, nu_t, n) = (0.5f64, 4.0f64, 4000usize);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let z1 = rng.normal();
            let z2 = rng.normal();
            let yz = rho_t * z1 + (1.0 - rho_t * rho_t).sqrt() * z2;
            // W ~ χ²_ν = Gamma(ν/2, 2): suma de exponenciales es tosca para
            // ν=4: Gamma(2,2) = −2·ln(u1·u2)
            let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-12;
            let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-12;
            let w = -2.0 * (u1 * u2).ln(); // χ²₄
            let esc = (nu_t / w).sqrt();
            x.push(z1 * esc);
            y.push(yz * esc);
        }
        // τ sobre la muestra ⇒ ρ̂ por inversión
        let tau = tau_kendall(&x, &y).expect("muestra válida");
        let rho_hat = rho_desde_tau(tau);
        assert!((rho_hat - 0.5).abs() < 0.06, "ρ̂={rho_hat}");
        // scores por rangos empíricos
        let (u, v) = scores_por_rango(&x, &y);
        let (nu_hat, _ll) = nu_mle(&u, &v, rho_hat).expect("MLE válido");
        assert!(
            (3.0..=8.0).contains(&nu_hat),
            "ν̂={nu_hat} fuera de la banda esperada para ν=4"
        );
    }

    fn scores_por_rango(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let ranks = |v: &[f64]| -> Vec<f64> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
            let mut r = vec![0usize; v.len()];
            for (pos, &i) in idx.iter().enumerate() {
                r[i] = pos + 1;
            }
            r.iter()
                .map(|&p| p as f64 / (v.len() + 1) as f64)
                .collect()
        };
        (ranks(x), ranks(y))
    }

    #[test]
    fn lambda_empirica_en_independencia_y_comonotonia() {
        let mut rng = Rng(12345);
        let n = 2000;
        // independiente: λ empírica ≈ (1−q)/(1−q)=q_independiente... en
        // realidad P(Y>qy|X>qx) ≈ 1−q ≈ 0.05 con q=0.95
        let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let (lu, ll) = lambda_empirica(&x, &y, 0.95).unwrap();
        assert!(lu < 0.12, "λU indep={lu}");
        assert!(ll < 0.12, "λL indep={ll}");
        // comonótona: λ=1
        let z: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let (lu, ll) = lambda_empirica(&z, &z, 0.95).unwrap();
        assert!(cerca(lu, 1.0, 1e-9));
        assert!(cerca(ll, 1.0, 1e-9));
        // rechazos
        assert!(lambda_empirica(&x[..10], &y[..10], 0.95).is_none());
        assert!(lambda_empirica(&x, &y, 1.5).is_none());
    }
}
