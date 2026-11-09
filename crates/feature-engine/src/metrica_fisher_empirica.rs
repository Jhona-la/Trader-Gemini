//! Métrica Fisher EMPÍRICA de la familia q-gaussiana con β_q — GLM-221
//! (Fase C1 del plan maestro; diseño del ciclo 195, corregido aquí).
//!
//! La familia (Tsallis) con normalización de varianza:
//!   p(z | μ, σ, q) = f(u) / (C_q·σ),  f(u) = [1 − (1−q)·β_q·u²]^{1/(1−q)},  u = (z−μ)/σ
//! con **β_q = 1/(5−3q)** tiene varianza EXACTAMENTE σ² para todo q del
//! rango abierto (1, 5/3) — el techo 5/3 es la divergencia de la varianza —
//! y límite q→1⁺ = N(μ, σ²) exacto (ciclo 195; la familia SIN β convergía a
//! exp(−z²), no exp(−z²/2): advertencia del 194). C_q no depende de (μ, σ);
//! el 1/σ sí, y entra en el score de σ.
//!
//! Para q > 1 la familia es una t de Student con ν = (3−q)/(q−1) grados de
//! libertad reescalada a varianza σ². Es de localización-escala, así que su
//! métrica de Fisher es g(μ, σ) = G/σ², con G = E[s̃·s̃ᵀ] constante bajo la
//! densidad estándar f/C_q (s̃ = score con σ = 1): G = diag(a, b),
//! a = ν(ν+1)/((ν+3)(ν−2)), b = 2ν/(ν+3). Para G = [[a, c], [c, b]], la
//! curvatura escalar de Ricci de g es R = −2a/(ab − c²), que no depende de
//! (μ, σ); con c = 0, **R = −2/b = −(ν+3)/ν = −1 − 3(q−1)/(3−q)**: q→1⁺ ⇒
//! R→−1 (continuidad con el semiplano hiperbólico de Fisher-Rao, 191/193);
//! q = 1.5 ⇒ R = −2; q→5/3 ⇒ R→−2.5. Las colas pesadas CURVAN la geometría,
//! no la aplanan.
//!
//! ERRATA del 194/195 (corregida aquí). La versión anterior tenía dos
//! defectos. El score de σ omitía el término −1/σ de la normalización, de
//! modo que E_p[s_σ] = 1/σ ≠ 0 y el producto externo no era la métrica de
//! Fisher. Y el Ricci derivaba la media empírica con las muestras FIJAS, que
//! fuera del punto de muestreo es E_{θ0}[s_θ·s_θᵀ], no g(θ); éste era el
//! error dominante (con sólo el score corregido, ese método da R ≈ +2 en el
//! límite gaussiano). El Rust anterior daba R ≈ +0.001 en q = 1.001 y
//! R ≈ +0.12 en q = 1.5, curvatura positiva, y su propio contrato fallaba en
//! CI. Las cifras |R| ≈ 0.87 / 0.48 y la conclusión «las colas aplanan»
//! (R ≈ −0.40 en q = 1.5) venían del espejo Python del ciclo 194/195, no de
//! este código. Ahora la única cantidad empírica es G, estimada con las
//! muestras en su propio punto, y el campo g(θ) = G/σ² sale exacto de la
//! estructura de localización-escala.
//!
//! Fail-closed SIEMPRE: q fuera de (1, 5/3), σ ≤ 0, muestra fuera del
//! soporte (sólo posible con q < 1), muestras insuficientes o métrica
//! degenerada devuelven `None` (métrica) o `Err` (Ricci) — nunca un cero o
//! un número fabricado.

/// Rango abierto de q: 1 < q < 5/3 (en 5/3 la varianza de la q-gaussiana
/// diverge; en 1 no es q-gaussiana sino la gaussiana límite, cubierta por
/// `fisher_rao`).
pub const Q_MIN: f64 = 1.0 + 1e-9;
pub const Q_MAX: f64 = 5.0 / 3.0;

/// Muestras mínimas para estimar G = E[s̃·s̃ᵀ]. Con 512 el error estándar
/// de R ronda 0.12–0.17 (q = 1.5 y límite gaussiano): todavía separa
/// q→1 (R = −1) de q = 1.5 (R = −2) por ≈ 5 errores.
pub const M_MIN_MUESTRAS: usize = 512;

/// β_q = 1/(5−3q): la normalización que da varianza σ² exacta a toda la
/// familia. `None` si q ∉ (1, 5/3) (abierto) o la resta degenera.
pub fn beta_q(q: f64) -> Option<f64> {
    if !(Q_MIN..Q_MAX).contains(&q) || !q.is_finite() {
        return None;
    }
    let denominador = 5.0 - 3.0 * q;
    if denominador <= 0.0 {
        return None;
    }
    Some(1.0 / denominador)
}

/// Score (∇ log p) de la q-gaussiana en (μ, σ) para una observación z:
///   s_μ = 2βu / (σ·base)
///   s_σ = (2βu²/base − 1) / σ,   base = 1 − (1−q)βu²
/// El −1/σ es la derivada de −ln σ (la normalización de p), y hace que
/// E_p[s] = 0. Con q > 1, cuando |u|→∞, s_μ→0 y s_σ→(2/(q−1) − 1)/σ.
/// `None` = fuera de soporte (base ≤ 0), σ ≤ 0, q inválido o no finito:
/// el llamador decide, aquí no se fabrica nada.
pub fn score_q_gaussiana(q: f64, mu: f64, sigma: f64, z: f64) -> Option<[f64; 2]> {
    if !sigma.is_finite() || sigma <= 0.0 || !mu.is_finite() || !z.is_finite() {
        return None;
    }
    let beta = beta_q(q)?;
    let u = (z - mu) / sigma;
    let base = 1.0 - (1.0 - q) * beta * u * u;
    if !(base > 0.0) {
        return None;
    }
    let s_mu = 2.0 * beta * u / (sigma * base);
    let s_sigma = (2.0 * beta * u * u / base - 1.0) / sigma;
    if !(s_mu.is_finite() && s_sigma.is_finite()) {
        return None;
    }
    Some([s_mu, s_sigma])
}

/// Métrica de Fisher empírica 2D (θ = (μ, σ)) por producto externo de
/// scores: g ≈ (1/N)·Σ s(z_i|θ)·s(z_i|θ)ᵀ. Las muestras `z` deben venir de
/// la propia p(·|μ, σ, q): sólo así la media estima E_p[s·sᵀ] (identidad de
/// regularidad). `None` sólo por guardas fail-closed: menos de
/// [`M_MIN_MUESTRAS`], algún score no finito (σ ≤ 0, q inválido, z no
/// finito) o matriz degenerada (det ≤ 0). Con q < 1 (fuera del rango) la
/// familia tiene soporte compacto y la guarda `base ≤ 0` del score además
/// excluye muestras fuera de él.
pub fn metrica_fisher_empirica(
    q: f64,
    mu: f64,
    sigma: f64,
    muestras: &[f64],
) -> Option<[[f64; 2]; 2]> {
    if muestras.len() < M_MIN_MUESTRAS {
        return None;
    }
    let mut g = [[0.0f64; 2]; 2];
    for &z in muestras {
        let s = score_q_gaussiana(q, mu, sigma, z)?;
        g[0][0] += s[0] * s[0];
        g[0][1] += s[0] * s[1];
        g[1][0] += s[0] * s[1];
        g[1][1] += s[1] * s[1];
    }
    let n = muestras.len() as f64;
    for fila in &mut g {
        for celda in fila.iter_mut() {
            *celda /= n;
        }
    }
    let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    if !det.is_finite() || det <= 0.0 {
        return None;
    }
    Some(g)
}

/// Curvatura escalar de Ricci de la métrica Fisher empírica de la familia,
/// con el estimador numérico genérico del ciclo 193 (derivadas centrales →
/// Christoffels → Riemann → contracción). `muestras` son de
/// p(·|mu, sigma, q). Se estima G = σ²·g(mu, sigma) una sola vez y el
/// estimador recibe el campo exacto θ ↦ G/θ_σ² (la familia es de
/// localización-escala): no se deriva la media empírica con las muestras
/// fijas, que fuera del punto de muestreo no es la métrica. Como R no
/// depende de (μ, σ), el estimador se evalúa en el punto estándar (0, 1):
/// su paso h es absoluto y con σ muy pequeña o muy grande daría un
/// resultado sin sentido. `Err` = q fuera de rango, menos de
/// [`M_MIN_MUESTRAS`], algún score no finito o métrica degenerada.
pub fn ricci_q_gaussiana(
    q: f64,
    mu: f64,
    sigma: f64,
    muestras: &[f64],
) -> Result<f64, &'static str> {
    if beta_q(q).is_none() {
        return Err("q fuera del rango (1, 5/3)");
    }
    let g = metrica_fisher_empirica(q, mu, sigma, muestras)
        .ok_or("metrica indefinida en el punto (soporte/muestras/degenerada)")?;
    let s2 = sigma * sigma;
    let g_estandar = [[g[0][0] * s2, g[0][1] * s2], [g[1][0] * s2, g[1][1] * s2]];
    let campo = |_t1: f64, t2: f64| -> [[f64; 2]; 2] {
        if !t2.is_finite() || t2 <= 0.0 {
            return [[f64::NAN; 2]; 2];
        }
        let inv = 1.0 / (t2 * t2);
        [
            [g_estandar[0][0] * inv, g_estandar[0][1] * inv],
            [g_estandar[1][0] * inv, g_estandar[1][1] * inv],
        ]
    };
    crate::ricci_empirica::curvatura_ricci_2d(campo, 0.0, 1.0, crate::ricci_empirica::H_POR_DEFECTO)
}
