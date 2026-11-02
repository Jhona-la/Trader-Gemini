//! Matriz de correlación EWMA por pares + filtrado de Marchenko-Pastur
//! (mandato §12.1 del operador, GLM-182).
//!
//! «Al calcular matrices de covarianza y tensores de correlación entre el
//! Top 10 de futuros y el resto del mercado, el 80 % de los autovalores
//! suele ser ruido aleatorio de alta frecuencia. Filtra todos los
//! autovalores que caigan dentro de los límites teóricos de la
//! distribución de Marchenko-Pastur, conservando únicamente los
//! autovectores super-críticos (el movimiento real institucional).»
//!
//! Dos piezas ortogonales:
//! - [`filtrar_mp`]: función PURA que clasifica el espectro de una matriz
//!   de correlación contra la banda MP λ± = (1 ± √(N/T))². La razón de
//!   aspecto es N/T con T = muestras efectivas.
//! - [`MatrizCorrelacionEwma`]: estimador streaming O(N²) por corte
//!   sincronizado completo. El T efectivo del EWMA (ganancia α por corte)
//!   es (2−α)/α — con `Ewma::from_period(P)` da exactamente P, así que el
//!   período nominal ES el T efectivo de la banda (documentado abajo).
//!
//! Medir-antes-de-decidir (patrón de esta línea): sin consumidor cableado
//! aún; el wiring cross-activo (CoinArena / Top-10) es titularidad de AGY.

use crate::ewma::Ewma;

/// Informe del filtrado de Marchenko-Pastur sobre un espectro de
/// correlación. La banda es [λ−, λ+]; los autovalores se clasifican en
/// RUIDO (dentro), SEÑAL (super-críticos: > λ+, el movimiento
/// institucional) y SUB-CRÍTICOS (< λ−: estructura que el ruido iid no
/// produce, p. ej. anti-correlación fuerte).
#[derive(Debug, Clone, PartialEq)]
pub struct InformeMp {
    /// N de la matriz (activos).
    pub n: usize,
    /// Muestras efectivas usadas para la banda (T en N/T).
    pub t_efectivo: f64,
    /// λ− = (1 − √(N/T))², acotado a ≥ 0.
    pub lambda_menos: f64,
    /// λ+ = (1 + √(N/T))².
    pub lambda_mas: f64,
    /// Autovalores dentro de la banda: ruido de alta frecuencia.
    pub n_ruido: usize,
    /// Autovalores > λ+: modos super-críticos (señal institucional).
    pub n_senal: usize,
    /// Autovalores < λ−: sub-críticos (estructura no-ruido por debajo).
    pub n_subcritico: usize,
    /// Espectro completo ordenado descendente (longitud N).
    pub autovalores: Vec<f64>,
}

/// Clasifica el espectro de una matriz de correlación N×N (aplanada por
/// filas, diagonal 1) contra la banda de Marchenko-Pastur.
///
/// Errores (CL-15, ausencia honesta): dimensión inválida, longitud
/// incorrecta, asimetría > 1e-9, diagonal ≠ 1 (> 1e-6) o T efectivo no
/// finito < 2. Nunca inventa banda.
pub fn filtrar_mp(matriz: &[f64], n: usize, t_efectivo: f64) -> Result<InformeMp, &'static str> {
    if n < 2 {
        return Err("MP requiere N >= 2");
    }
    if matriz.len() != n * n {
        return Err("longitud de matriz != n*n");
    }
    if !t_efectivo.is_finite() || t_efectivo < 2.0 {
        return Err("T efectivo debe ser finito y >= 2");
    }
    for i in 0..n {
        if (matriz[i * n + i] - 1.0).abs() > 1e-6 {
            return Err("la diagonal debe ser 1 (matriz de correlación)");
        }
        for j in (i + 1)..n {
            let asim = (matriz[i * n + j] - matriz[j * n + i]).abs();
            if asim > 1e-9 {
                return Err("matriz no simétrica");
            }
        }
    }
    let mut autovalores = autovalores_jacobi_simetrico(matriz, n);
    autovalores.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));

    let ratio = n as f64 / t_efectivo;
    let raiz = ratio.sqrt();
    let lambda_menos = ((1.0 - raiz) * (1.0 - raiz)).max(0.0);
    let lambda_mas = (1.0 + raiz) * (1.0 + raiz);

    let (mut n_ruido, mut n_senal, mut n_subcritico) = (0usize, 0usize, 0usize);
    for &lambda in &autovalores {
        if !lambda.is_finite() {
            return Err("autovalor no finito");
        }
        if lambda > lambda_mas {
            n_senal += 1;
        } else if lambda < lambda_menos {
            n_subcritico += 1;
        } else {
            n_ruido += 1;
        }
    }
    Ok(InformeMp {
        n,
        t_efectivo,
        lambda_menos,
        lambda_mas,
        n_ruido,
        n_senal,
        n_subcritico,
        autovalores,
    })
}

/// Jacobi simétrico cíclico (rotaciones de Givens) sobre una matriz
/// aplanada por filas. Devuelve sólo los autovalores (el consumidor de
/// hoy es la clasificación espectral; los vectores se añaden cuando un
/// consumidor los pida). N ≤ 32 en el uso previsto: ~O(n³) despreciable.
fn autovalores_jacobi_simetrico(matriz: &[f64], n: usize) -> Vec<f64> {
    let mut a = matriz.to_vec();
    // Barridos cíclicos hasta convergencia del off-diagonal.
    for _ in 0..64 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-18 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                // Ángulo de rotación que anula a[p][q]: tan(2θ) = 2apq/(aqq−app).
                let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
                let (c, s) = (theta.cos(), theta.sin());
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[i * n + i]).collect()
}

/// Estimador streaming de la matriz de correlación N×N por EWMA por pares.
///
/// T efectivo del EWMA de ganancia α: (2−α)/α (muestras efectivas de la
/// varianza exponencialmente ponderada). Con `period` P (α = 2/(P+1)) da
/// exactamente P ⇒ la banda MP del análisis usa ese mismo P como T.
///
/// O(N²) por corte, sin asignaciones tras la construcción. La API exige
/// cortes completos sincronizados por el llamador; no infiere reloj.
pub struct MatrizCorrelacionEwma {
    n: usize,
    alpha: f64,
    mean: Vec<Ewma>,
    var: Vec<Ewma>,
    cov: Vec<Ewma>,
    last: Vec<f64>,
    rets: Vec<f64>,
    desv: Vec<f64>,
    corte: u64,
}

impl MatrizCorrelacionEwma {
    /// Constructor sin pánico (`RB-P1` zero-panic ratchet): sanea `n < 2` a `2`
    /// y `period` inválido a `20.0`. Para validación estricta usar [`Self::try_new`].
    pub fn new(n: usize, period: f64) -> Self {
        let n_safe = n.max(2);
        let p_safe = if period.is_finite() && period >= 1.0 {
            period
        } else {
            20.0
        };
        let alpha = 2.0 / (p_safe + 1.0);
        Self {
            n: n_safe,
            alpha,
            mean: (0..n_safe).map(|_| Ewma::from_period(p_safe)).collect(),
            var: (0..n_safe).map(|_| Ewma::from_period(p_safe)).collect(),
            cov: (0..n_safe * n_safe)
                .map(|_| Ewma::from_period(p_safe))
                .collect(),
            last: vec![0.0; n_safe],
            rets: vec![0.0; n_safe],
            desv: vec![0.0; n_safe],
            corte: 0,
        }
    }

    pub fn try_new(n: usize, period: f64) -> Result<Self, &'static str> {
        if n < 2 {
            return Err("la matriz requiere N >= 2");
        }
        let semilla = Ewma::try_from_period(period)?;
        let alpha = semilla.alpha;
        Ok(Self {
            n,
            alpha,
            mean: (0..n).map(|_| Ewma::from_period(period)).collect(),
            var: (0..n).map(|_| Ewma::from_period(period)).collect(),
            cov: (0..n * n).map(|_| Ewma::from_period(period)).collect(),
            last: vec![0.0; n],
            rets: vec![0.0; n],
            desv: vec![0.0; n],
            corte: 0,
        })
    }

    /// T efectivo del estimador (documentado en el doc del tipo).
    pub fn t_efectivo(&self) -> f64 {
        (2.0 - self.alpha) / self.alpha
    }

    /// Ingresa un corte sincronizado completo de precios. Err ⇒ corte
    /// inválido y estado intacto; Ok(None) ⇒ primer corte (sólo fija
    /// referencia de precios).
    pub fn try_update(&mut self, precios: &[f64]) -> Result<Option<()>, &'static str> {
        if precios.len() != self.n {
            return Err("corte incompleto");
        }
        if precios.iter().any(|p| !p.is_finite() || *p <= 0.0) {
            return Err("precios deben ser finitos y positivos");
        }
        if self.corte == 0 {
            self.last.copy_from_slice(precios);
            self.corte += 1;
            return Ok(None);
        }
        for i in 0..self.n {
            let ret = (precios[i] - self.last[i]) / self.last[i];
            if !ret.is_finite() {
                return Err("retorno no representable");
            }
            self.rets[i] = ret;
        }
        for i in 0..self.n {
            let mu = self.mean[i].update(self.rets[i]);
            if !mu.is_finite() {
                return Err("media no representable");
            }
            self.desv[i] = self.rets[i] - mu;
        }
        for i in 0..self.n {
            let d2 = self.desv[i] * self.desv[i];
            if !d2.is_finite() {
                return Err("varianza no representable");
            }
            let v = self.var[i].update(d2);
            if !v.is_finite() {
                return Err("varianza no representable");
            }
        }
        for i in 0..self.n {
            for j in i..self.n {
                let c = self.cov[i * self.n + j].update(self.desv[i] * self.desv[j]);
                if !c.is_finite() {
                    return Err("covarianza no representable");
                }
            }
        }
        self.last.copy_from_slice(precios);
        self.corte += 1;
        Ok(Some(()))
    }

    /// Analiza el espectro de la correlación actual contra la banda MP con
    /// el T efectivo del EWMA. Err mientras la varianza marginal aún no es
    /// positiva (warmup) — ausencia honesta.
    pub fn analizar(&self) -> Result<InformeMp, &'static str> {
        let mut matriz = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            let vi = self.var[i].get();
            if !vi.is_finite() || vi <= 0.0 {
                return Err("warmup: varianza marginal no positiva");
            }
            matriz[i * self.n + i] = 1.0;
            for j in (i + 1)..self.n {
                let vj = self.var[j].get();
                if !vj.is_finite() || vj <= 0.0 {
                    return Err("warmup: varianza marginal no positiva");
                }
                let c = self.cov[i * self.n + j].get();
                let r = (c / (vi * vj).sqrt()).clamp(-1.0, 1.0);
                matriz[i * self.n + j] = r;
                matriz[j * self.n + i] = r;
            }
        }
        filtrar_mp(&matriz, self.n, self.t_efectivo())
    }
}
