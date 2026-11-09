//! Métrica Fisher EMPÍRICA de la familia q-gaussiana con β_q — GLM-221
//! (Fase C1 del plan maestro; diseño cerrado en el ciclo 195).
//!
//! La familia (Tsallis) con normalización de varianza:
//!   p(z | μ, σ, q) ∝ [1 − (1−q)·β_q·u²]^{1/(1−q)},  u = (z−μ)/σ
//! con **β_q = 1/(5−3q)** tiene varianza EXACTAMENTE σ² para todo q del
//! rango abierto (1, 5/3) — el techo 5/3 es la divergencia de la varianza —
//! y límite q→1⁺ = N(μ, σ²) exacto (ciclo 195; la familia SIN β convergía a
//! exp(−z²), no exp(−z²/2): advertencia del 194).
//!
//! La métrica de Fisher se estima por producto externo de scores:
//!   g_ij(θ) = E_z[s_i(z|θ)·s_j(z|θ)],  s = ∇_θ log p
//! sobre un conjunto fijo de muestras (identidad de regularidad). La
//! curvatura escalar de Ricci de esa métrica 2D (θ = (μ, σ)) reutiliza el
//! estimador numérico genérico del ciclo 193 (`ricci_empirica`), verificado
//! contra tres constantes cerradas (euclídea 0, hiperbólica −2, gaussiana
//! −1). Números de contrato (medidos en el 194/195): q→1⁺ ⇒ R→−1
//! (continuidad con el semiplano hiperbólico de Fisher-Rao); q = 1.5 ⇒
//! R ≈ −0.40 — las colas pesadas APLANAN la geometría de la información.
//!
//! Fail-closed SIEMPRE: q fuera de (1, 5/3), σ ≤ 0, muestra fuera del
//! soporte, muestras insuficientes o métrica degenerada devuelven `None`
//! (métrica) o `Err` (Ricci) — nunca un cero o un número fabricado.

/// Rango abierto de q: 1 < q < 5/3 (en 5/3 la varianza de la q-gaussiana
/// diverge; en 1 no es q-gaussiana sino la gaussiana límite, cubierta por
/// `fisher_rao`).
pub const Q_MIN: f64 = 1.0 + 1e-9;
pub const Q_MAX: f64 = 5.0 / 3.0;

/// Muestras mínimas para estimar E[s_i·s_j] con la estabilidad que el
/// estimador de Ricci del 193 exige (sus derivadas centrales amplifican el
/// ruido de una media corta).
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
///   s_μ     = 2βu / (σ·(1−(1−q)βu²))
///   s_σ     = 2βu² / (σ·(1−(1−q)βu²))
/// `None` = fuera de soporte (1−(1−q)βu² ≤ 0), σ ≤ 0, q inválido o
/// no-finito: el llamador decide, aquí no se fabrica nada.
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
    let s_sigma = 2.0 * beta * u * u / (sigma * base);
    if !(s_mu.is_finite() && s_sigma.is_finite()) {
        return None;
    }
    Some([s_mu, s_sigma])
}

/// Métrica de Fisher empírica 2D (θ = (μ, σ)) por producto externo de
/// scores sobre las MISMAS muestras `z` (identidad de regularidad: la
/// esperanza bajo p de s·sᵀ es la métrica; con muestras propias de p es su
/// estimador consistente). Para q > 1 el soporte es ℝ (ley de potencias) y
/// ningún score diverge: `None` sólo por guardas fail-closed — menos de
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

/// Curvatura escalar de Ricci de la métrica Fisher empírica de la familia
/// en el punto (mu, sigma), reutilizando el estimador numérico genérico del
/// ciclo 193 (derivadas centrales → Christoffels → Riemann → contracción).
/// Las muestras se mantienen FIJAS para todas las evaluaciones del entorno
/// (diferenciar la MISMA función empírica, no re-muestrar). `Err` =
/// propagación fail-closed del estimador (métrica degenerada en el punto o
/// su entorno; típicamente un h que empuja μ fuera del soporte de las
/// muestras).
pub fn ricci_q_gaussiana(
    q: f64,
    mu: f64,
    sigma: f64,
    muestras: &[f64],
) -> Result<f64, &'static str> {
    if beta_q(q).is_none() {
        return Err("q fuera del rango (1, 5/3)");
    }
    // Falla temprana en el punto central (mensaje propio, no el genérico).
    metrica_fisher_empirica(q, mu, sigma, muestras)
        .ok_or("metrica indefinida en el punto (soporte/muestras/degenerada)")?;
    let metrica = |t1: f64, t2: f64| -> [[f64; 2]; 2] {
        metrica_fisher_empirica(q, t1, t2, muestras)
            .unwrap_or([[f64::NAN; 2]; 2])
    };
    crate::ricci_empirica::curvatura_ricci_2d(metrica, mu, sigma, crate::ricci_empirica::H_POR_DEFECTO)
}
