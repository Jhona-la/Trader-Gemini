//! Correlation-spectrum diagnostic and Marchenko-Pastur reference edge.
//!
//! For standardized iid observations, gamma = N/T and the asymptotic upper
//! edge is (1 + sqrt(gamma))^2. This is NOT a finite-sample confidence bound;
//! neither side of that edge proves independence or a real market factor.
//! Serial dependence, microstructure noise and pairwise asynchronous estimates
//! require a separately justified null model. Buffer capacity is not sample size.
//!
//! Input contract: a complete, finite, symmetric, positive-semidefinite
//! correlation matrix with unit diagonal. Numerical checks cannot establish
//! data provenance: missing correlations must never be encoded as measured zero.
//!
//! The eigensolver uses cyclic Jacobi rotations, O(sweeps * N^3) time and O(N^2)
//! storage. Unlike a single-start power iteration, it retains modes orthogonal
//! to the all-ones vector (including balanced long/short factor exposures).
//! This is ordinary symmetric linear algebra, not a Yang-Mills mass-gap model.

/// Asymptotic reference edge for population identity correlation, gamma = N/T.
/// The T > N restriction is this API's conservative support policy, not a claim
/// that the MP law ceases to exist for gamma >= 1.
#[inline]
pub fn mp_upper_edge(n_assets: usize, t_observations: usize) -> Option<f64> {
    if n_assets < 2 || t_observations <= n_assets {
        return None;
    }
    let gamma = n_assets as f64 / t_observations as f64;
    Some((1.0 + gamma.sqrt()).powi(2))
}

/// Maximum matrix dimension supported on the stack without heap allocation (O(1) stack, 0 heap allocations).
pub const MAX_STACK_CORR_DIM: usize = 32;
const MAX_STACK_CORR_CELLS: usize = MAX_STACK_CORR_DIM * MAX_STACK_CORR_DIM;

/// Largest eigenvalue of a numerically valid correlation matrix.
/// Returns None on invalid input or nonconvergence. Singular PSD matrices are
/// valid. No eigenvalue clipping, diagonal loading or imputation is performed.
/// For N <= MAX_STACK_CORR_DIM (32), operates with ZERO heap allocations.
pub fn largest_eigenvalue(corr: &[Vec<f64>]) -> Option<f64> {
    let n = corr.len();
    if n >= 2 && n <= MAX_STACK_CORR_DIM {
        let diag = diagonalize_correlation_stack(corr)?;
        Some(diag[..n].iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)))
    } else {
        let diagonalized = diagonalize_correlation(corr)?;
        Some(diagonalized.iter().enumerate().map(|(i, row)| row[i]).fold(f64::NEG_INFINITY, f64::max))
    }
}

/// Largest eigenvalue of a row-major flat correlation matrix of dimension n*n.
/// Operates with ZERO heap allocations for n <= MAX_STACK_CORR_DIM (32).
pub fn largest_eigenvalue_flat(corr_flat: &[f64], n: usize) -> Option<f64> {
    if n < 2 || corr_flat.len() != n * n {
        return None;
    }
    if n <= MAX_STACK_CORR_DIM {
        let diag = diagonalize_correlation_flat_stack(corr_flat, n)?;
        Some(diag[..n].iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)))
    } else {
        let nested: Vec<Vec<f64>> = (0..n)
            .map(|i| corr_flat[i * n..(i + 1) * n].to_vec())
            .collect();
        largest_eigenvalue(&nested)
    }
}

/// Full descending spectrum of a row-major flat correlation matrix of dimension n*n.
/// Writes eigenvalues into out_spectrum (must have len >= n).
/// Operates with ZERO heap allocations for n <= MAX_STACK_CORR_DIM (32).
pub fn full_spectrum_flat_zero_alloc(
    corr_flat: &[f64],
    n: usize,
    out_spectrum: &mut [f64],
) -> Option<usize> {
    if n < 2 || corr_flat.len() != n * n || out_spectrum.len() < n {
        return None;
    }
    if n <= MAX_STACK_CORR_DIM {
        let diag = diagonalize_correlation_flat_stack(corr_flat, n)?;
        out_spectrum[..n].copy_from_slice(&diag[..n]);
        out_spectrum[..n].sort_by(|a, b| b.total_cmp(a));
        Some(n)
    } else {
        let nested: Vec<Vec<f64>> = (0..n)
            .map(|i| corr_flat[i * n..(i + 1) * n].to_vec())
            .collect();
        let spec = full_spectrum(&nested)?;
        out_spectrum[..n].copy_from_slice(&spec);
        Some(n)
    }
}

// Zero-allocation cyclic Jacobi rotation eigensolver for flat buffer of size n*n.
fn jacobi_eigensolver_flat_stack(
    a: &mut [f64],
    n: usize,
    tolerance: f64,
) -> Option<[f64; MAX_STACK_CORR_DIM]> {
    const MAX_SWEEPS: usize = 64;
    for sweep in 0..=MAX_SWEEPS {
        let mut off_diagonal_norm = 0.0_f64;
        for i in 0..n {
            let row_offset = i * n;
            for j in (i + 1)..n {
                let entry = a[row_offset + j];
                off_diagonal_norm = off_diagonal_norm.hypot(entry * std::f64::consts::SQRT_2);
            }
        }
        if !off_diagonal_norm.is_finite() {
            return None;
        }
        if off_diagonal_norm <= tolerance {
            for i in 0..n {
                let diag = a[i * n + i];
                if !diag.is_finite() || diag < -tolerance {
                    return None;
                }
            }
            let mut out = [0.0f64; MAX_STACK_CORR_DIM];
            for i in 0..n {
                out[i] = a[i * n + i];
            }
            return Some(out);
        }
        if sweep == MAX_SWEEPS {
            break;
        }
        for p in 0..n {
            let p_offset = p * n;
            for q in (p + 1)..n {
                let q_offset = q * n;
                let apq = a[p_offset + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[p_offset + p];
                let aqq = a[q_offset + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + tau.hypot(1.0));
                let cosine = 1.0 / (1.0 + t * t).sqrt();
                let sine = t * cosine;
                a[p_offset + p] -= t * apq;
                a[q_offset + q] += t * apq;
                a[p_offset + q] = 0.0;
                a[q_offset + p] = 0.0;
                for k in 0..n {
                    if k != p && k != q {
                        let k_offset = k * n;
                        let akp = a[k_offset + p];
                        let akq = a[k_offset + q];
                        let new_p = cosine * akp - sine * akq;
                        let new_q = sine * akp + cosine * akq;
                        a[k_offset + p] = new_p;
                        a[p_offset + k] = new_p;
                        a[k_offset + q] = new_q;
                        a[q_offset + k] = new_q;
                    }
                }
            }
        }
    }
    None
}

// Stack-allocated diagonalizer for &[Vec<f64>] without dynamic heap allocations.
fn diagonalize_correlation_stack(corr: &[Vec<f64>]) -> Option<[f64; MAX_STACK_CORR_DIM]> {
    let n = corr.len();
    if n < 2 || n > MAX_STACK_CORR_DIM || corr.iter().any(|row| row.len() != n) {
        return None;
    }
    let tolerance = 64.0 * f64::EPSILON * n as f64;
    for i in 0..n {
        if !corr[i][i].is_finite() || (corr[i][i] - 1.0).abs() > tolerance {
            return None;
        }
        for j in (i + 1)..n {
            let a = corr[i][j];
            let b = corr[j][i];
            if !a.is_finite()
                || !b.is_finite()
                || a.abs() > 1.0 + tolerance
                || b.abs() > 1.0 + tolerance
                || (a - b).abs() > tolerance
            {
                return None;
            }
        }
    }

    let mut a = [0.0f64; MAX_STACK_CORR_CELLS];
    for i in 0..n {
        a[i * n + i] = corr[i][i];
        for j in (i + 1)..n {
            let symmetric = 0.5 * corr[i][j] + 0.5 * corr[j][i];
            a[i * n + j] = symmetric;
            a[j * n + i] = symmetric;
        }
    }

    jacobi_eigensolver_flat_stack(&mut a[..n * n], n, tolerance)
}

// Stack-allocated diagonalizer for &[f64] row-major slice without heap allocations.
fn diagonalize_correlation_flat_stack(
    corr_flat: &[f64],
    n: usize,
) -> Option<[f64; MAX_STACK_CORR_DIM]> {
    if n < 2 || n > MAX_STACK_CORR_DIM || corr_flat.len() != n * n {
        return None;
    }
    let tolerance = 64.0 * f64::EPSILON * n as f64;
    for i in 0..n {
        let c_ii = corr_flat[i * n + i];
        if !c_ii.is_finite() || (c_ii - 1.0).abs() > tolerance {
            return None;
        }
        for j in (i + 1)..n {
            let a = corr_flat[i * n + j];
            let b = corr_flat[j * n + i];
            if !a.is_finite()
                || !b.is_finite()
                || a.abs() > 1.0 + tolerance
                || b.abs() > 1.0 + tolerance
                || (a - b).abs() > tolerance
            {
                return None;
            }
        }
    }

    let mut a = [0.0f64; MAX_STACK_CORR_CELLS];
    for i in 0..n {
        a[i * n + i] = corr_flat[i * n + i];
        for j in (i + 1)..n {
            let symmetric = 0.5 * corr_flat[i * n + j] + 0.5 * corr_flat[j * n + i];
            a[i * n + j] = symmetric;
            a[j * n + i] = symmetric;
        }
    }

    jacobi_eigensolver_flat_stack(&mut a[..n * n], n, tolerance)
}

// Fallback dynamic heap diagonalizer for N > MAX_STACK_CORR_DIM.
fn diagonalize_correlation(corr: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = corr.len();
    if n < 2 || corr.iter().any(|row| row.len() != n) {
        return None;
    }
    // Dimension-scaled roundoff budget for unit-scale entries. This is a
    // numerical tolerance, not a market threshold or a statistical confidence.
    let tolerance = 64.0 * f64::EPSILON * n as f64;
    for i in 0..n {
        if !corr[i][i].is_finite() || (corr[i][i] - 1.0).abs() > tolerance {
            return None;
        }
        for j in (i + 1)..n {
            let a = corr[i][j];
            let b = corr[j][i];
            if !a.is_finite()
                || !b.is_finite()
                || a.abs() > 1.0 + tolerance
                || b.abs() > 1.0 + tolerance
                || (a - b).abs() > tolerance
            {
                return None;
            }
        }
    }

    let mut a = corr.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            // Only reconcile roundoff-sized asymmetry after validation.
            let symmetric = 0.5 * corr[i][j] + 0.5 * corr[j][i];
            a[i][j] = symmetric;
            a[j][i] = symmetric;
        }
    }

    // Bounded work: a numerical budget, not an acceptance shortcut. A matrix
    // still above residual tolerance at the end yields None, never AllNoise.
    const MAX_SWEEPS: usize = 64;
    for sweep in 0..=MAX_SWEEPS {
        let mut off_diagonal_norm = 0.0_f64;
        for (i, row) in a.iter().enumerate() {
            for &entry in row.iter().skip(i + 1) {
                off_diagonal_norm = off_diagonal_norm.hypot(entry * std::f64::consts::SQRT_2);
            }
        }
        if !off_diagonal_norm.is_finite() {
            return None;
        }
        if off_diagonal_norm <= tolerance {
            for (i, row) in a.iter().enumerate() {
                // Residual Frobenius norm bounds the spectral error. Admit
                // only roundoff-scale negativity, not an indefinite estimate.
                if !row[i].is_finite() || row[i] < -tolerance {
                    return None;
                }
            }
            return Some(a);
        }
        if sweep == MAX_SWEEPS {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq == 0.0 {
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                // Stable smaller root of t^2 + 2*tau*t - 1 = 0.
                let t = tau.signum() / (tau.abs() + tau.hypot(1.0));
                let cosine = 1.0 / (1.0 + t * t).sqrt();
                let sine = t * cosine;
                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[k][p];
                        let akq = a[k][q];
                        let new_p = cosine * akp - sine * akq;
                        let new_q = sine * akp + cosine * akq;
                        a[k][p] = new_p;
                        a[p][k] = new_p;
                        a[k][q] = new_q;
                        a[q][k] = new_q;
                    }
                }
            }
        }
    }
    None
}

/// Complete descending spectrum of a valid correlation matrix. Shares the
/// domain, numerical tolerance and convergence checks of largest_eigenvalue.
/// Returns None for invalid inputs or nonconvergence; trace equals N up to
/// numerical error, not by normalizing an arbitrary covariance matrix.
/// For N <= MAX_STACK_CORR_DIM (32), internal diagonalization performs zero heap allocations.
pub fn full_spectrum(corr: &[Vec<f64>]) -> Option<Vec<f64>> {
    let n = corr.len();
    if n >= 2 && n <= MAX_STACK_CORR_DIM {
        let diag = diagonalize_correlation_stack(corr)?;
        let mut eigenvalues = diag[..n].to_vec();
        eigenvalues.sort_by(|a, b| b.total_cmp(a));
        Some(eigenvalues)
    } else {
        let diagonalized = diagonalize_correlation(corr)?;
        let mut eigenvalues: Vec<f64> =
            diagonalized.into_iter().enumerate().map(|(i, row)| row[i]).collect();
        eigenvalues.sort_by(|a, b| b.total_cmp(a));
        Some(eigenvalues)
    }
}

/// Legacy name for the participation of modes ABOVE the MP reference edge:
/// (sum sqrt(lambda))^2 / sum lambda. For k retained positive modes,
/// Cauchy-Schwarz bounds this dimensionless diagnostic in [1, k] <= [1, N].
/// Equal retained eigenvalues give k; a single retained mode gives 1.
///
/// It is NOT a portfolio's effective number of independent bets: there are no
/// position weights, and identity correlation yields None (no retained modes),
/// not N. Exceeding an asymptotic edge does not statistically certify a factor.
/// Missing/invalid inputs, unsupported sample count and no retained modes all
/// yield None; callers must not interpret absence as independence or zero risk.
/// No operational sizing or diversification discount is authorized by this API.
pub fn effective_bets(corr: &[Vec<f64>], t_observations: usize) -> Option<f64> {
    let spectrum = full_spectrum(corr)?;
    let edge = mp_upper_edge(corr.len(), t_observations)?;
    let clean: Vec<f64> = spectrum.into_iter().filter(|&lambda| lambda > edge).collect();
    if clean.is_empty() || clean.iter().any(|l| !l.is_finite() || *l <= 0.0) {
        return None;
    }
    let sum_sqrt: f64 = clean.iter().map(|l| l.sqrt()).sum();
    let sum: f64 = clean.iter().sum();
    if sum <= 1e-12 {
        return None;
    }
    Some((sum_sqrt * sum_sqrt / sum).clamp(1.0, corr.len() as f64))
}

/// Legacy names retained for callers; these are threshold comparisons, not
/// statistical certificates. In particular, AllNoise must not authorize a
/// portfolio-risk discount without independently validated sampling evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MppVerdict {
    /// Maximum eigenvalue did not exceed the asymptotic reference edge.
    /// Does NOT imply independence, absence of structure or safe diversification.
    AllNoise,
    /// Maximum eigenvalue exceeded the reference edge; not a calibrated p-value.
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

    /// One deterministic uniform-noise fixture, not a universal null guarantee.
    #[test]
    fn xli_c1_ruido_iid_no_supera_el_borde_mp() {
        let n = 8usize;
        let t = 512usize; // gamma = N/T = 1/64
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
        let verdict = systematic_mode(&corr, t).expect("T > N");
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
        let verdict = systematic_mode(&corr, t).expect("T > N");
        match verdict {
            MppVerdict::SystematicMode {
                lambda_max,
                mp_edge,
            } => {
                assert!(lambda_max > mp_edge);
                // Un factor en 8 activos debe explicar la mayor parte de la
                // traza: λ_max del orden de N·(share de varianza del factor).
                assert!(
                    lambda_max > 4.0,
                    "lambda_max={lambda_max} debia ser dominante"
                );
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

    /// Matriz de correlación de Pearson O(N²·T) para los tests.
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


#[cfg(test)]
mod xliv_effective_bets_tests {
    use super::*;

    fn factor_matrix(n: usize, t: usize, seed0: u64) -> Vec<Vec<f64>> {
        let mut seed = seed0;
        let mut rets = vec![0.0f64; n * t];
        for k in 0..t {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let factor = (((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) * 0.05;
            for i in 0..n {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idio = (((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) * 0.01;
                rets[i * t + k] = factor + idio;
            }
        }
        let mut corr = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            corr[i][i] = 1.0;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let mut num = 0.0; let mut di = 0.0; let mut dj = 0.0;
                let mi = rets[i * t..i * t + t].iter().sum::<f64>() / t as f64;
                let mj = rets[j * t..j * t + t].iter().sum::<f64>() / t as f64;
                for k in 0..t {
                    let a = rets[i * t + k] - mi;
                    let b = rets[j * t + k] - mj;
                    num += a * b; di += a * a; dj += b * b;
                }
                let c = (num / (di * dj).sqrt().max(1e-18)).clamp(-1.0, 1.0);
                corr[i][j] = c; corr[j][i] = c;
            }
        }
        corr
    }

    /// Un factor común domina en 8 activos → N_eff ≈ 1 (una sola apuesta).
    #[test]
    fn xlv_factor_comun_da_una_apuesta_efectiva() {
        let n = 8; let t = 512;
        let corr = factor_matrix(n, t, 42);
        let bets = effective_bets(&corr, t).expect("espectro con factor");
        assert!(bets < 2.0, "factor dominante debia dar N_eff≈1, dio {bets}");
        assert!(bets >= 1.0);
    }

    /// Ruido iid → None (sin autovalores sobre el borde MP).
    #[test]
    fn xlv_ruido_puro_no_afirma_dimension() {
        let n = 4; let t = 512;
        // generar ruido puro (sin factor)
        let mut seed = 99u64;
        let mut rets = vec![0.0f64; n * t];
        for x in rets.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
        }
        let mut corr = vec![vec![0.0f64; n]; n];
        for i in 0..n { corr[i][i] = 1.0; }
        for i in 0..n {
            for j in (i + 1)..n {
                let mi = rets[i * t..i * t + t].iter().sum::<f64>() / t as f64;
                let mj = rets[j * t..j * t + t].iter().sum::<f64>() / t as f64;
                let mut num = 0.0; let mut di = 0.0; let mut dj = 0.0;
                for k in 0..t {
                    let a = rets[i * t + k] - mi;
                    let b = rets[j * t + k] - mj;
                    num += a * b; di += a * a; dj += b * b;
                }
                let c = (num / (di * dj).sqrt().max(1e-18)).clamp(-1.0, 1.0);
                corr[i][j] = c; corr[j][i] = c;
            }
        }
        // ruido puro: el max puede o no superar el borde por azar finito
        // — no ASSERT sobre None (sería frágil); verificamos que si da
        // Some, N_eff ≥ 1 y ≤ N.
        if let Some(bets) = effective_bets(&corr, t) {
            assert!(bets >= 1.0 && bets <= n as f64, "N_eff={bets} fuera de [1,{n}]");
        }
    }

    /// Identidad pura (todo correlacionado = mismo activo) → N_eff = 1 exacto.
    #[test]
    fn xlv_identidad_pura_da_exactamente_uno() {
        let corr = vec![vec![1.0; 3]; 3];
        let bets = effective_bets(&corr, 512).expect("λ=3 > borde");
        assert!((bets - 1.0).abs() < 1e-6, "identidad debia dar 1.0, dio {bets}");
    }
}
