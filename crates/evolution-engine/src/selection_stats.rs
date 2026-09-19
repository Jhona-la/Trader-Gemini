//! QO-M1 — ESTADÍSTICA DE SELECCIÓN: Deflated Sharpe Ratio (DSR) y
//! Probabilistic Sharpe Ratio (PSR), Bailey & López de Prado (2014).
//!
//! # Por qué existe
//! La auditoría matemática halló que TODAS las puertas de promoción
//! (online_daemon, ShadowForest, walkforward) seleccionan sobre Sharpe/WR/PF
//! crudos SIN control de multiplicidad: con 2000 candidatos × 26 monedas ×
//! 32 escalas, el mejor por pura suerte supera cualquier umbral fijo. El
//! DSR corrige el Sharpe por (a) el número de pruebas realizadas y (b) la
//! curtosis de los retornos: sólo un edge que SOBREVIVE la corrección es
//! estadísticamente real.
//!
//! # Matemática
//! PSR(SR*) = Φ( (SR − SR*) · √(n−1) / √(1 − γ₃·SR + (γ₄−1)/4 · SR²) )
//!   donde γ₃ = skewness, γ₄ = kurtosis (no exceso), Φ = CDF normal.
//! DSR = PSR(SR*) con SR* = el Sharpe esperado del MEJOR de N pruebas bajo
//!   H₀ (todas sin edge): SR* ≈ √(ln N) · (4·ln N − γ) / (4·ln N − 2γ)
//!   con γ = Euler-Mascheroni.
//!
//! Regla de decisión: DSR > 0.95 ⇒ el edge es real al 95% DESPUÉS de
//! corregir por N pruebas.

/// Momentos de una serie de retornos (media, desviación, skewness, kurtosis).
/// Kurtosis NO excesiva (normal = 3).
#[derive(Debug, Clone, Copy)]
pub struct ReturnMoments {
    pub mean: f64,
    pub sd: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub n: usize,
}

pub fn compute_moments(returns: &[f64]) -> Option<ReturnMoments> {
    let clean: Vec<f64> = returns.iter().copied().filter(|r| r.is_finite()).collect();
    if clean.len() < 20 {
        return None;
    }
    let n = clean.len() as f64;
    let mean = clean.iter().sum::<f64>() / n;
    let var = clean.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();
    if sd <= 1e-12 {
        return None;
    }
    let m3 = clean.iter().map(|r| ((r - mean) / sd).powi(3)).sum::<f64>() / n;
    let m4 = clean.iter().map(|r| ((r - mean) / sd).powi(4)).sum::<f64>() / n;
    Some(ReturnMoments {
        mean,
        sd,
        skewness: m3,
        kurtosis: m4,
        n: clean.len(),
    })
}

/// Sharpe del período (media/σ). El llamador anualiza si quiere.
#[inline]
pub fn sharpe(m: &ReturnMoments) -> f64 {
    if m.sd <= 1e-12 {
        return 0.0;
    }
    m.mean / m.sd
}

/// PSR — Probabilistic Sharpe Ratio (Bailey & LdP 2012, eq. 4).
#[inline]
pub fn psr(m: &ReturnMoments, sr: f64, sr_benchmark: f64) -> f64 {
    let n = m.n as f64;
    let denom_sq = 1.0 - m.skewness * sr + (m.kurtosis - 1.0) / 4.0 * sr * sr;
    if denom_sq <= 1e-12 {
        return 0.5;
    }
    let z = (sr - sr_benchmark) * (n - 1.0).sqrt() / denom_sq.sqrt();
    normal_cdf(z)
}

/// DSR — Deflated Sharpe Ratio (Bailey & LdP 2014).
#[inline]
pub fn dsr(m: &ReturnMoments, n_trials: usize) -> f64 {
    let sr = sharpe(m);
    let n = n_trials.max(2) as f64;
    let ln_n = n.ln();
    let gamma = 0.577_215_664_901_532_9; // Euler-Mascheroni
    let sr_star = ln_n.sqrt() * (4.0 * ln_n - gamma) / (4.0 * ln_n - 2.0 * gamma);
    let sr_benchmark = sr_star / (m.n as f64).sqrt();
    psr(m, sr, sr_benchmark)
}

/// CDF normal estándar — Abramowitz & Stegun 7.1.26 (|ε| < 7.5e-8).
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * ax);
    let poly = t * (0.319_381_530
        + t * (-0.356_563_782
            + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let cdf_pos = 1.0 - poly * (-0.5 * ax * ax).exp() / (2.0 * std::f64::consts::PI).sqrt();
    0.5 * (1.0 + sign * (2.0 * cdf_pos - 1.0))
}

/// Umbral DSR estándar de la industria.
pub const DSR_THRESHOLD: f64 = 0.95;

#[derive(Debug, Clone, Copy)]
pub struct SelectionVerdict {
    pub dsr: f64,
    pub passes: bool,
    pub n_trials: usize,
    pub note: &'static str,
}

pub fn edge_survives_multiplicity(returns: &[f64], n_trials: usize) -> SelectionVerdict {
    match compute_moments(returns) {
        Some(m) => {
            let d = dsr(&m, n_trials);
            SelectionVerdict {
                dsr: d,
                passes: d >= DSR_THRESHOLD,
                n_trials,
                note: if d >= DSR_THRESHOLD {
                    "edge REAL al 95% tras corrección por multiplicidad"
                } else {
                    "ruido: el mejor de N pruebas sin edge supera este Sharpe por azar"
                },
            }
        }
        None => SelectionVerdict {
            dsr: 0.0,
            passes: false,
            n_trials,
            note: "muestra insuficiente (<20 retornos finitos)",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psr_edge_real_supera_umbral() {
        let returns: Vec<f64> = (0..500)
            .map(|i| 0.002 + ((i % 17) as f64 - 8.0) * 0.0001)
            .collect();
        let m = compute_moments(&returns).unwrap();
        let sr = sharpe(&m);
        let p = psr(&m, sr, 0.0);
        assert!(p > 0.95, "edge real debe PSR>0.95, got {:.4}", p);
    }

    #[test]
    fn psr_ruido_puro_no_supera() {
        let returns: Vec<f64> = (0..500)
            .map(|i| ((i * 7919) % 1000) as f64 / 500_000.0 - 0.001)
            .collect();
        let m = compute_moments(&returns).unwrap();
        let sr = sharpe(&m);
        let p = psr(&m, sr, 0.0);
        assert!(p < 0.8, "ruido puro no debe PSR>0.8, got {:.4}", p);
    }

    #[test]
    fn dsr_mas_pruebas_menor_dsr() {
        let returns: Vec<f64> = (0..300)
            .map(|i| 0.001 + ((i % 23) as f64 - 11.0) * 0.00005)
            .collect();
        let m = compute_moments(&returns).unwrap();
        let dsr_low = dsr(&m, 10);
        let dsr_high = dsr(&m, 5000);
        assert!(dsr_high <= dsr_low, "{:.4} vs {:.4}", dsr_high, dsr_low);
    }

    #[test]
    fn edge_survives_rechaza_muestra_corta() {
        let v = edge_survives_multiplicity(&[0.01; 10], 100);
        assert!(!v.passes);
    }

    #[test]
    fn normal_cdf_correcta() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-4);
    }
}
