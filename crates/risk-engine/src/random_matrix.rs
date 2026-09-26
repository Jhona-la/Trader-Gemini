//! MATRICES ALEATORIAS — BORDE DE MARCHENKO-PASTUR (Ola XLI·C1).
//!
//! Contrato (protocolo del repo):
//! - Variable: matriz de correlación C (N×N, simétrica, diagonal 1) de los
//!   retornos de N activos del universo sobre T observaciones.
//! - Operador: el TEOREMA de Marchenko-Pastur dice que el espectro de
//!   C cuando los retornos son ruido iid (sin estructura) se concentra en
//!   [λ−, λ+] con λ± = σ²(1 ± √γ)², γ = T/N. Todo eigenvalor FUERA de ese
//!   borde es estructura REAL (modo de mercado, factores), no ruido.
//! - Unidades: adimensionales (eigenvalores de correlación).
//! - Condiciones de contorno: N < 2 o T ≤ N → None (no se afirma borde con
//!   γ ≥ 1: el borde inferior se degenera y la muestra es la población).
//! - Identificabilidad: `systemic_mode` separa "correlación medida" (modo
//!   fuera del borde) de "ruido que parece correlación" (dentro del borde).
//!   El RUIDO NO VETA: un par Pearson alto dentro de la banda de ruido no es
//!   evidencia de misma-apuesta.
//! - Coste: iteración de potencias O(N²·iters) con N = tamaño del grupo
//!   (típicamente ≤ 30) — despreciable frente a la adquisición de ticks.
//! - Falsación: (a) ruido gaussiano iid con γ conocido → λ_max ≤ borde MP
//!   (test); (b) un factor común inyectado → λ_max > borde (test).
//!
//! Esta es la familia matemática de las conjeturas de brecha espectral
//! (Yang-Mills) integrada como lo que aquí es: un TEOREMA con condiciones
//! exactas, sin decoración.

/// Borde superior de Marchenko-Pastur para una matriz de correlación N×N
/// construida con T observaciones (varianza de población 1 por definición de
/// correlación). γ = N/T: el espectro de ruido iid vive en
/// [(1−√γ)², (1+√γ)²]. None si T ≤ N (muestra insuficiente: γ ≥ 1 y el
/// borde inferior colapsa — no se afirma nada).
#[inline]
pub fn mp_upper_edge(n_assets: usize, t_observations: usize) -> Option<f64> {
    if n_assets < 2 || t_observations <= n_assets {
        return None;
    }
    let gamma = n_assets as f64 / t_observations as f64;
    Some((1.0 + gamma.sqrt()).powi(2))
}

/// Mayor eigenvalor por iteración de potencias con deflación de la traza
/// (la matriz de correlación tiene traza N: el modo trivial ~1 por activo
/// está garantizado; lo que importa es si EXCEDE el borde de ruido).
/// None en matriz vacía/degenerada o sin convergencia en 100 iteraciones.
pub fn largest_eigenvalue(corr: &[Vec<f64>]) -> Option<f64> {
    let n = corr.len();
    if n < 2 || corr.iter().any(|r| r.len() != n) {
        return None;
    }
    // vector inicial uniforme: no favorece ningún activo
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda_prev = 0.0;
    for _ in 0..100 {
        // y = C·v
        let mut y = vec![0.0; n];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, &cij) in corr[i].iter().enumerate() {
                acc += cij * v[j];
            }
            *yi = acc;
        }
        let norm: f64 = y.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm > 1e-12) || !norm.is_finite() {
            return None;
        }
        for (vi, yi) in v.iter_mut().zip(&y) {
            *vi = yi / norm;
        }
        // λ = vᵀCv (Rayleigh)
        let mut rayleigh = 0.0;
        for i in 0..n {
            for (j, &cij) in corr[i].iter().enumerate() {
                rayleigh += v[i] * cij * v[j];
            }
        }
        if (rayleigh - lambda_prev).abs() < 1e-10 * rayleigh.abs().max(1e-10) {
            return Some(rayleigh);
        }
        lambda_prev = rayleigh;
    }
    None
}

/// ¿Existe un modo SISTEMÁTICO (estructura real de correlación) en la matriz,
/// por encima de lo que el ruido explicaría? None sin muestra suficiente
/// (γ ≤ 1): en ese caso la política de veto NO cambia (no se afirma nada).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MppVerdict {
    /// λ_max ≤ borde MP: toda la correlación observada es compatible con
    /// ruido iid. Los pares altos NO cuentan como misma-apuesta medible.
    AllNoise,
    /// λ_max > borde: existe al menos un modo real (mercado/factor).
    SystematicMode { lambda_max: f64, mp_edge: f64 },
}

pub fn systematic_mode(corr: &[Vec<f64>], t_observations: usize) -> Option<MppVerdict> {
    let edge = mp_upper_edge(corr.len(), t_observations)?;
    let lambda = largest_eigenvalue(corr)?;
    Some(if lambda > edge {
        MppVerdict::SystematicMode {
            lambda_max: lambda,
            mp_edge: edge,
        }
    } else {
        MppVerdict::AllNoise
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Falsación (a): ruido gaussiano iid, γ = T/N = 4 → λ_max dentro del
    /// borde MP. El ruido NO es modo sistemático.
    #[test]
    fn xli_c1_ruido_iid_no_supera_el_borde_mp() {
        let n = 8usize;
        let t = 512usize; // γ = 64
        // retornos iid deterministas (LCG)
        let mut seed = 0xDEADBEEFCAFEBABEu64;
        let mut rets = vec![0.0f64; n * t];
        for r in rets.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
            *r = u;
        }
        let corr = correlation_matrix(&rets, n, t);
        let verdict = systematic_mode(&corr, t).expect("gamma>1");
        assert!(
            matches!(verdict, MppVerdict::AllNoise),
            "ruido iid debe quedar dentro del borde MP: {:?}",
            verdict
        );
    }

    /// Falsación (b): un FACTOR COMÚN inyectado → λ_max supera el borde.
    #[test]
    fn xli_c1_factor_comun_es_modo_sistematico() {
        let n = 8usize;
        let t = 512usize;
        let mut seed = 0x1234567890ABCDEFu64;
        let mut rets = vec![0.0f64; n * t];
        for k in 0..t {
            // factor común fuerte + idiosincrático débil
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let factor = (((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) * 0.05;
            for i in 0..n {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idio = (((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) * 0.01;
                rets[i * t + k] = factor + idio;
            }
        }
        let corr = correlation_matrix(&rets, n, t);
        let verdict = systematic_mode(&corr, t).expect("gamma>1");
        match verdict {
            MppVerdict::SystematicMode {
                lambda_max,
                mp_edge,
            } => {
                assert!(lambda_max > mp_edge);
                // Un factor en 8 activos debe explicar la mayor parte de la
                // traza: λ_max del orden de N·(share de varianza del factor).
                assert!(lambda_max > 4.0, "lambda_max={lambda_max} debia ser dominante");
            }
            other => panic!("factor comun debia ser sistematico: {other:?}"),
        }
    }

    #[test]
    fn xli_c1_muestra_insuficiente_no_afirma_nada() {
        assert_eq!(mp_upper_edge(4, 4), None);
        assert_eq!(mp_upper_edge(1, 100), None);
        assert!(systematic_mode(&[], 100).is_none());
    }

    /// Matriz de correlación de Pearson O(N·T) para los tests.
    fn correlation_matrix(rets: &[f64], n: usize, t: usize) -> Vec<Vec<f64>> {
        let mut means = vec![0.0; n];
        for i in 0..n {
            means[i] = rets[i * t..i * t + t].iter().sum::<f64>() / t as f64;
        }
        let mut corr = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            corr[i][i] = 1.0;
            for j in (i + 1)..n {
                let mut num = 0.0;
                let mut di = 0.0;
                let mut dj = 0.0;
                for k in 0..t {
                    let a = rets[i * t + k] - means[i];
                    let b = rets[j * t + k] - means[j];
                    num += a * b;
                    di += a * a;
                    dj += b * b;
                }
                let denom = (di * dj).sqrt().max(1e-18);
                let c = (num / denom).clamp(-1.0, 1.0);
                corr[i][j] = c;
                corr[j][i] = c;
            }
        }
        corr
    }
}
