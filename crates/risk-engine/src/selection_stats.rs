//! QO-M1 / S5 — ESTADÍSTICA DE SELECCIÓN Y CONTROL DE MULTIPLICIDAD:
//! Deflated Sharpe Ratio (DSR) y Probabilistic Sharpe Ratio (PSR), Bailey & López de Prado (2014).
//!
//! # Por qué existe
//! La auditoría matemática halló que las compuertas de optimización y promoción
//! seleccionan sobre Sharpe/WR/PF crudos SIN control de multiplicidad: con N candidatos,
//! el mejor por puro azar supera cualquier umbral fijo. El DSR corrige el Sharpe por:
//! (a) el número de pruebas realizadas N y (b) la asimetría y curtosis de los retornos.
//! Sólo un edge que SOBREVIVE la corrección es estadísticamente real.
//!
//! # Matemática
//! PSR(SR*) = Φ( (SR − SR*) · √(n−1) / √(1 − γ₃·SR + (γ₄−1)/4 · SR²) )
//!   donde γ₃ = skewness, γ₄ = kurtosis (no exceso), Φ = CDF normal.
//! DSR = PSR(SR*) con SR* = el Sharpe esperado del MEJOR de N pruebas bajo H₀ (Gumbel, eq. 5):
//!
//! E[max SR] ≈ σ_SR · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

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
    let se = sharpe_std_error(m, sr);
    if se <= 1e-12 {
        return 0.5;
    }
    let z = (sr - sr_benchmark) / se;
    normal_cdf(z)
}

/// Error estándar asintótico del estimador de Sharpe bajo retornos no-normales
/// (Mertens 2002, Lo 2002, Bailey & López de Prado 2012 eq. 4, 2014 eq. 7).
///
/// ```text
/// σ_SR = √[ (1 − γ₃·SR + ((γ₄ − 1)/4)·SR²) / (n − 1) ]
/// ```
///
/// Para distribuciones leptocúrticas (γ₄ > 3) como retornos cripto, σ_SR es
/// estrictamente mayor que el i.i.d. gaussiano 1/√(n−1), elevando el listón
/// de deflación E[max SR] para evitar falsos positivos por colas pesadas.
#[inline]
pub fn sharpe_std_error(m: &ReturnMoments, sr: f64) -> f64 {
    let n = m.n as f64;
    if n < 2.0 {
        return 0.0;
    }
    // R4-B3/B6 + R5-B3: γ₄ pisado al gaussiano (3.0) y γ₃ desconfiado en
    // la región degenerada. El discriminante de Mertens 1−γ₃·SR+((γ₄−1)/4)SR²
    // cruza ≤0 cuando γ₃² > γ₄−1 (ej γ₃=2, SR=2 → −1): ahí el fallback
    // 1/√(n−1) era ANTI-conservador y la cota al discriminante dejaría σ→0
    // en el vértice (peor aún). Resolución: si la asimetría medida vuelve
    // IMAGINARIA la varianza asintótica, se desconfía del término de skew
    // (γ₃→0) y se conserva el de curtosis — denom ≥ 1+((g4−1)/4)SR² ≥ 1:
    // leptocúrtico honesto, jamás sub-gaussiano, justo donde la muestra
    // es débil. La región legítima (denom > 0) usa la fórmula intacta.
    let g4 = m.kurtosis.max(3.0);
    let denom_raw = 1.0 - m.skewness * sr + (g4 - 1.0) / 4.0 * sr * sr;
    let denom_sq = if denom_raw > 0.0 {
        denom_raw
    } else {
        1.0 + (g4 - 1.0) / 4.0 * sr * sr
    };
    (denom_sq / (n - 1.0)).sqrt()
}

/// Constante de Euler–Mascheroni (γ), la que aparece en la aproximación de
/// Gumbel al máximo de N variables normales.
pub const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// E[max SR] de N pruebas independientes bajo H₀ — Bailey & López de Prado (2014), ec. 5:
///
/// ```text
/// E[max SR] ≈ σ_SR · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
/// ```
#[inline]
pub fn expected_max_sharpe(n_trials: usize, sr_sigma: f64) -> f64 {
    let n = (n_trials.max(2)) as f64;
    let z1 = inverse_normal_cdf(1.0 - 1.0 / n);
    let z2 = inverse_normal_cdf(1.0 - 1.0 / (n * std::f64::consts::E));
    let e_max = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2;
    if e_max.is_finite() && sr_sigma.is_finite() {
        sr_sigma * e_max
    } else {
        0.0
    }
}

/// DSR — Deflated Sharpe Ratio (Bailey & LdP 2014, eq. 7 y 8).
/// H1-4: Utiliza el error estándar asintótico no-normal `sharpe_std_error` que incorpora
/// el sesgo γ₃ y la curtosis γ₄ pesada de la muestra, garantizando que el listón
/// de deflación E[max SR] sea riguroso y conservador frente a ruido leptocúrtico.
#[inline]
pub fn dsr(m: &ReturnMoments, n_trials: usize) -> f64 {
    let sr = sharpe(m);
    let n_obs = m.n as f64;
    if n_obs < 2.0 {
        return 0.0;
    }
    let sr_sigma = sharpe_std_error(m, sr);
    let sr_benchmark = expected_max_sharpe(n_trials, sr_sigma);
    psr(m, sr, sr_benchmark)
}

/// Φ⁻¹ — función cuantil de la normal estándar (Peter Acklam, error relativo < 1,15e-9).
#[inline]
pub fn inverse_normal_cdf(p: f64) -> f64 {
    if !p.is_finite() || p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
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
    fn inverse_normal_cdf_valores_conocidos() {
        let casos = [
            (0.5, 0.0),
            (0.95, 1.644_853_626_951_472),
            (0.975, 1.959_963_984_540_054),
            (0.99, 2.326_347_874_040_841),
            (0.999, 3.090_232_306_167_813),
            (0.025, -1.959_963_984_540_054),
            (0.001, -3.090_232_306_167_813),
        ];
        for (p, esperado) in casos {
            let z = inverse_normal_cdf(p);
            assert!(
                (z - esperado).abs() < 1e-6,
                "Φ⁻¹({p}) = {z}, esperado {esperado}"
            );
        }
    }

    /// D-741 — el umbral de deflación con N = 2 000 DEBE ser mayor que el que
    /// producía la fórmula inventada. Falla con el código viejo: allí el
    /// benchmark era 2,8114·σ_SR (y además dividido por √n en vez de √(n−1)).
    #[test]
    fn d741_umbral_sube_respecto_de_la_formula_vieja() {
        let n_trials = 2_000usize;
        let ln_n = (n_trials as f64).ln();
        let viejo = ln_n.sqrt() * (4.0 * ln_n - EULER_MASCHERONI)
            / (4.0 * ln_n - 2.0 * EULER_MASCHERONI);
        let nuevo = expected_max_sharpe(n_trials, 1.0);
        assert!(
            (viejo - 2.811_4).abs() < 1e-3,
            "la fórmula vieja valía 2,8114·σ_SR, medido {viejo:.4}"
        );
        assert!(
            (nuevo - 3.447_3).abs() < 1e-3,
            "E[max SR] correcto ≈ 3,4473·σ_SR, medido {nuevo:.4}"
        );
        assert!(
            nuevo > viejo * 1.2,
            "el umbral correcto supera al viejo en >20%: {nuevo:.4} vs {viejo:.4}"
        );
    }

    /// D-741 — consecuencia operativa: una serie que el DSR viejo aprobaba
    /// con 2 000 pruebas ya no pasa, porque el listón subió un 23 %.
    #[test]
    fn d741_el_dsr_es_mas_exigente_que_el_viejo() {
        let returns: Vec<f64> = (0..400)
            .map(|i| 0.0009 + ((i % 31) as f64 - 15.0) * 0.0004)
            .collect();
        let m = compute_moments(&returns).unwrap();
        let n_obs = m.n as f64;
        let ln_n = 2000f64.ln();
        let sr_star_viejo = ln_n.sqrt() * (4.0 * ln_n - EULER_MASCHERONI)
            / (4.0 * ln_n - 2.0 * EULER_MASCHERONI);
        let dsr_viejo = psr(&m, sharpe(&m), sr_star_viejo / n_obs.sqrt());
        let dsr_nuevo = dsr(&m, 2_000);
        assert!(
            dsr_nuevo < dsr_viejo,
            "el DSR corregido debe ser MÁS exigente: nuevo {dsr_nuevo:.4} vs viejo {dsr_viejo:.4}"
        );
    }

    #[test]
    fn expected_max_sharpe_crece_con_las_pruebas() {
        let a = expected_max_sharpe(10, 1.0);
        let b = expected_max_sharpe(2_000, 1.0);
        let c = expected_max_sharpe(200_000, 1.0);
        assert!(a < b && b < c, "{a:.4} < {b:.4} < {c:.4}");
    }

    #[test]
    fn normal_cdf_correcta() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-4);
    }

    #[test]
    fn omega15_h1_4_dsr_sharpe_std_error_leptocurtico() {
        // Serie con alta curtosis (outliers / colas pesadas cripto)
        let mut returns: Vec<f64> = vec![0.001; 100];
        returns[10] = 0.05;
        returns[20] = -0.05;
        returns[30] = 0.08;
        returns[40] = -0.07;

        let m = compute_moments(&returns).unwrap();
        assert!(m.kurtosis > 3.0, "Kurtosis debe ser leptocúrtica (>3), dio {}", m.kurtosis);

        let sr = sharpe(&m);
        let se_no_normal = sharpe_std_error(&m, sr);
        let se_iid_gauss = 1.0 / (m.n as f64 - 1.0).sqrt();

        // El error estándar no-normal debe ser estrictamente mayor que el gaussiano ingenuo
        assert!(
            se_no_normal > se_iid_gauss,
            "se_no_normal ({:.6}) debe ser mayor que se_iid_gauss ({:.6})",
            se_no_normal,
            se_iid_gauss
        );

        // En consecuencia, el benchmark E[max SR] es más exigente
        let bm_no_normal = expected_max_sharpe(100, se_no_normal);
        let bm_gauss = expected_max_sharpe(100, se_iid_gauss);
        assert!(bm_no_normal > bm_gauss, "El benchmark E[max SR] debe ser más estricto con colas pesadas");
    }

/// R5-B3 — el discriminante de Mertens puede volverse negativo (γ₃² > γ₄−1):
/// con γ₃=2, SR=2 y γ₄=3 la varianza asintótica "vale" −1. La resolución
/// desconfía del término de skew y conserva el de curtosis: denom ≥ 1 —
/// JAMÁS sub-gaussiano, a diferencia del fallback 1/√(n−1) viejo.
#[test]
fn r5_b3_discriminante_degenerado_jamas_sub_gaussiano() {
    let m = ReturnMoments { n: 100, mean: 0.0, sd: 1.0, skewness: 2.0, kurtosis: 3.0 };
    let sr = 2.0_f64;
    let se = sharpe_std_error(&m, sr);
    // Con γ₃=0 y γ₄=3: denom = 1 + (2/4)·4 = 3 ⇒ se = √(3/99).
    let esperado = (3.0_f64 / 99.0).sqrt();
    assert!(
        (se - esperado).abs() < 1e-12,
        "región degenerada usa γ₃=0 con curtosis intacta: {se} vs {esperado}"
    );
    let gaussiano = 1.0_f64 / 99.0_f64.sqrt();
    assert!(
        se > gaussiano,
        "jamás sub-gaussiano donde la muestra es débil: {se} vs {gaussiano}"
    );
    // Región legítima intacta: γ₃=0.5, SR=1, γ₄=5 → denom = 1 − 0.5 + 1 = 1.5.
    let m_legit = ReturnMoments { n: 100, mean: 0.0, sd: 1.0, skewness: 0.5, kurtosis: 5.0 };
    let se_legit = sharpe_std_error(&m_legit, 1.0);
    assert!((se_legit - (1.5_f64 / 99.0).sqrt()).abs() < 1e-12);
}
}
