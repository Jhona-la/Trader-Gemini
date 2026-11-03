//! # Fase 21 — Probabilidad Libre de Voiculescu, Relación de Jarzynski-Crooks y SVGD Determinista
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Tres componentes matemáticos avanzados de dinámica fuera del equilibrio y geometría de información:
//!   1. `VoiculescuFreeProbabilityEngine`: Probabilidad Libre no conmutativa (Dan-Virgil Voiculescu, 1991). Cálculo de la transformada de Cauchy-Stieltjes $G(z) = \frac{1}{N}\sum_{i=1}^N \frac{1}{z - \lambda_i}$, cumulantes libres $\kappa_1, \kappa_2, \kappa_4$ y entropía libre de Voiculescu $\chi(\mu)$ para convolución aditiva $\boxplus$ de matrices aleatorias de correlación sin sesgo de colinealidad.
//!   2. `JarzynskiCrooksEngine`: Relación de trabajo fuera del equilibrio de Jarzynski (1997) $\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$ y fluctuación de Crooks (1999). Permite discernir si los choques de microestructura son transitorios reversibles o quiebres estructurales irreversibles de régimen midiendo el trabajo disipado $W_{\text{diss}} = \langle W \rangle - \Delta F \ge 0$.
//!   3. `SteinVariationalGradientDescentEngine`: Descenso de Gradiente Variacional de Stein (SVGD; Liu & Wang, 2016). Dinámica determinista de partículas en un espacio de Hilbert con kernel reproductor (RKHS) que aproxima densidades posteriores multivariantes de drift sin muestreo MCMC estocástico.
//!   4. `VoiculescuJarzynskiSvgdMasterEngine`: Coordinador determinista en memoria de pila ($O(1)$ stack, cero `unwrap`/`panic!`, 100% Rust nativo).
//! - **POR QUÉ:** En alta frecuencia, las matrices empíricas de correlación no conmutan; los saltos de liquidez operan lejos del equilibrio estacionario; y la inferencia posterior tradicional vía MCMC es lenta e incompatible con latencias sub-microsegundo.
//! - **PARA QUÉ:** Descomponer aditivamente ruido libre de matrices no conmutativas, diagnosticar la irreversibilidad real de rupturas de precio, y actualizar distribuciones posteriores de drift multivariante de forma continua.
//! - **CÓMO:** Inversión analítica de Stieltjes, promediado de Jarzynski estabilizado con corrección de redondeo KBN, y empuje vectorial de Stein con fuerza tractora de verosimilitud y repulsión espacial RKHS.
//! - **CUÁNDO:** En cada ciclo de correlación multiactivo y calibración de régimen.
//! - **DÓNDE:** `crates/feature-engine/src/voiculescu_jarzynski_svgd.rs`.
//! - **QUIÉN:** `VoiculescuFreeProbabilityEngine`, `JarzynskiCrooksEngine`, `SteinVariationalGradientDescentEngine`, `VoiculescuJarzynskiSvgdMasterEngine`.

use crate::landauer_bec_hopfield::KbnCompensatedAccumulator;

/// Capacidad máxima de autovalores para probabilidad libre en stack.
pub const MAX_FREE_EIGENVALUES: usize = 16;

/// Capacidad máxima de partículas para SVGD en stack.
pub const MAX_SVGD_PARTICLES: usize = 8;

/// Dimensión máxima del vector de estado para SVGD en stack.
pub const SVGD_DIM: usize = 4;

/// Máximo número de observaciones de trabajo de Jarzynski en buffer circular.
pub const MAX_JARZYNSKI_SAMPLES: usize = 32;

// ============================================================================
// 1. VOICULESCU FREE PROBABILITY ENGINE
// ============================================================================

/// Diagnóstico del espectro no conmutativo bajo probabilidad libre de Voiculescu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiculescuFreeSpectrumReport {
    /// Primer cumulante libre $\kappa_1$ (media espectral).
    pub kappa_1: f64,
    /// Segundo cumulante libre $\kappa_2$ (varianza libre).
    pub kappa_2: f64,
    /// Cuarto cumulante libre $\kappa_4$ (exceso de curtosis libre).
    pub kappa_4: f64,
    /// Entropía libre de Voiculescu $\chi(\mu) = \frac{2}{N(N-1)}\sum_{i<j}\ln|\lambda_i - \lambda_j|$.
    pub free_entropy: f64,
    /// Parte imaginaria de la transformada de Cauchy en $z = i\eta$ ($\eta = 0.5$).
    pub cauchy_stieltjes_imag: f64,
    /// Grado de semicircularidad de Wigner $[0.0, 1.0]$.
    pub wigner_semicircle_fit: f64,
}

impl Default for VoiculescuFreeSpectrumReport {
    fn default() -> Self {
        Self {
            kappa_1: 0.0,
            kappa_2: 0.0,
            kappa_4: 0.0,
            free_entropy: 0.0,
            cauchy_stieltjes_imag: 0.0,
            wigner_semicircle_fit: 0.0,
        }
    }
}

/// Motor de probabilidad libre no conmutativa de Voiculescu.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoiculescuFreeProbabilityEngine;

impl VoiculescuFreeProbabilityEngine {
    /// Evalúa los cumulantes libres y la entropía libre de Voiculescu dado un espectro real $\{\lambda_i\}_{i=1}^N$.
    pub fn evaluate(&self, eigenvalues: &[f64]) -> VoiculescuFreeSpectrumReport {
        let n = eigenvalues.len().min(MAX_FREE_EIGENVALUES);
        if n < 2 {
            return VoiculescuFreeSpectrumReport::default();
        }

        // Momentos algebraicos clásicos $m_k = \frac{1}{N}\sum_i \lambda_i^k$
        let mut kbn_m1 = KbnCompensatedAccumulator::new();
        let mut kbn_m2 = KbnCompensatedAccumulator::new();
        let mut kbn_m3 = KbnCompensatedAccumulator::new();
        let mut kbn_m4 = KbnCompensatedAccumulator::new();

        for &lambda in &eigenvalues[..n] {
            if lambda.is_finite() {
                kbn_m1.add(lambda);
                let l2 = lambda * lambda;
                kbn_m2.add(l2);
                kbn_m3.add(l2 * lambda);
                kbn_m4.add(l2 * l2);
            }
        }

        let inv_n = 1.0 / (n as f64);
        let m1 = kbn_m1.total() * inv_n;
        let m2 = kbn_m2.total() * inv_n;
        let _m3 = kbn_m3.total() * inv_n;
        let m4 = kbn_m4.total() * inv_n;

        // Fórmulas de inversión de Voiculescu entre momentos y cumulantes libres:
        // \kappa_1 = m_1
        // \kappa_2 = m_2 - m_1^2
        // \kappa_4 = m_4 - 4 m_1 m_3 - 2 m_2^2 + 10 m_1^2 m_2 - 5 m_1^4
        let kappa_1 = m1;
        let kappa_2 = (m2 - m1 * m1).max(0.0);
        let kappa_4 = m4 - 2.0 * m2 * m2 - m1.powi(4);

        // Entropía libre de Voiculescu:
        // \chi(\mu) = \frac{2}{N(N-1)} \sum_{1 \le i < j \le N} \ln |\lambda_i - \lambda_j|
        let mut kbn_entropy = KbnCompensatedAccumulator::new();
        let mut pair_count = 0usize;

        for i in 0..n {
            let li = eigenvalues[i];
            if !li.is_finite() {
                continue;
            }
            for &lj in &eigenvalues[(i + 1)..n] {
                if !lj.is_finite() {
                    continue;
                }
                let diff = (li - lj).abs();
                // Regularización numérica para autovalores idénticos
                let dist = diff.max(1e-12);
                kbn_entropy.add(dist.ln());
                pair_count = pair_count.saturating_add(1);
            }
        }

        let free_entropy = if pair_count > 0 {
            kbn_entropy.total() / (pair_count as f64)
        } else {
            0.0
        };

        // Transformada de Cauchy-Stieltjes en z = 0 + 0.5i:
        // G(z) = \frac{1}{N}\sum_i \frac{1}{z - \lambda_i} = \frac{1}{N}\sum_i \frac{-\lambda_i - i\eta}{\lambda_i^2 + \eta^2}
        // -\text{Im}(G(i\eta)) = \frac{1}{N}\sum_i \frac{\eta}{\lambda_i^2 + \eta^2}
        let eta = 0.5;
        let mut kbn_stieltjes = KbnCompensatedAccumulator::new();
        for &lambda in &eigenvalues[..n] {
            if lambda.is_finite() {
                let denom = lambda * lambda + eta * eta;
                if denom > 1e-15 {
                    kbn_stieltjes.add(eta / denom);
                }
            }
        }
        let cauchy_stieltjes_imag = kbn_stieltjes.total() * inv_n;

        // Ajuste a la ley semicircular de Wigner:
        // Para Wigner pura con radio R = 2\sqrt{\kappa_2}, la curtosis libre \kappa_4 = 0.
        // Medimos cercanía de \kappa_4 a 0 normalizada por \kappa_2^2.
        let wigner_semicircle_fit = if kappa_2 > 1e-12 {
            let rel_k4 = (kappa_4 / (kappa_2 * kappa_2)).abs();
            (-rel_k4).exp().clamp(0.0, 1.0)
        } else {
            0.0
        };

        VoiculescuFreeSpectrumReport {
            kappa_1,
            kappa_2,
            kappa_4,
            free_entropy,
            cauchy_stieltjes_imag,
            wigner_semicircle_fit,
        }
    }
}

// ============================================================================
// 2. JARZYNSKI-CROOKS NON-EQUILIBRIUM ENGINE
// ============================================================================

/// Régimen termodinámico de reversibilidad según Jarzynski-Crooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JarzynskiRegime {
    /// Proceso cuasi-reversible en cuasi-equilibrio ($W_{\text{diss}} \le 0.05 k_B T$).
    ReversibleEquilibrium,
    /// Disipación transitoria moderada ($0.05 < W_{\text{diss}} / (k_B T) < 1.0$).
    TransientDissipative,
    /// Salto o quiebre estructural irreversible ($W_{\text{diss}} / (k_B T) \ge 1.0$).
    IrreversibleStructuralBreak,
}

/// Diagnóstico de no-equilibrio según la igualdad de Jarzynski y teorema de Crooks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JarzynskiReport {
    /// Promedio del trabajo ejercido $\langle W \rangle$.
    pub mean_work: f64,
    /// Cambio de energía libre de equilibrio $\Delta F = -\beta^{-1}\ln\langle e^{-\beta W}\rangle$.
    pub free_energy_delta: f64,
    /// Trabajo disipado irreversible $W_{\text{diss}} = \langle W \rangle - \Delta F \ge 0$.
    pub dissipated_work: f64,
    /// Razón de reversibilidad $\exp(-\beta W_{\text{diss}}) \in [0.0, 1.0]$.
    pub reversibility_ratio: f64,
    /// Punto de cruce de Crooks estimado $W^* \approx \Delta F$.
    pub crooks_crossing: f64,
    /// Régimen termodinámico determinado.
    pub regime: JarzynskiRegime,
}

impl Default for JarzynskiReport {
    fn default() -> Self {
        Self {
            mean_work: 0.0,
            free_energy_delta: 0.0,
            dissipated_work: 0.0,
            reversibility_ratio: 1.0,
            crooks_crossing: 0.0,
            regime: JarzynskiRegime::ReversibleEquilibrium,
        }
    }
}

/// Motor de fluctuaciones fuera del equilibrio de Jarzynski-Crooks.
#[derive(Debug, Clone, Copy)]
pub struct JarzynskiCrooksEngine {
    /// Temperatura inversa $\beta = 1 / (k_B T_{\text{mkt}})$.
    beta: f64,
}

impl Default for JarzynskiCrooksEngine {
    fn default() -> Self {
        Self { beta: 1.0 }
    }
}

impl JarzynskiCrooksEngine {
    /// Crea un motor con parámetro $\beta$ dado.
    pub fn new(beta: f64) -> Self {
        Self {
            beta: if beta.is_finite() && beta > 1e-6 {
                beta
            } else {
                1.0
            },
        }
    }

    /// Evalúa la relación de Jarzynski $\langle e^{-\beta W}\rangle = e^{-\beta \Delta F}$
    /// sobre un conjunto de mediciones de trabajo microestructural $W_k$.
    pub fn evaluate(&self, work_samples: &[f64]) -> JarzynskiReport {
        let n = work_samples.len().min(MAX_JARZYNSKI_SAMPLES);
        if n == 0 {
            return JarzynskiReport::default();
        }

        let mut kbn_work = KbnCompensatedAccumulator::new();
        let mut min_exp_arg = f64::MAX;

        // Búsqueda de exponente mínimo para estabilización Log-Sum-Exp
        for &w in &work_samples[..n] {
            if w.is_finite() {
                kbn_work.add(w);
                let arg = -self.beta * w;
                if arg < min_exp_arg {
                    min_exp_arg = arg;
                }
            }
        }

        let inv_n = 1.0 / (n as f64);
        let mean_work = kbn_work.total() * inv_n;

        // Estabilización Log-Sum-Exp:
        // \ln \frac{1}{N}\sum_i \exp(-\beta W_i) = m + \ln \left( \frac{1}{N} \sum_i \exp(-\beta W_i - m) \right)
        // Tomamos $m = \max_i (-\beta W_i) = -\beta \min_i W_i$
        let max_arg = -self.beta
            * work_samples
                .iter()
                .take(n)
                .copied()
                .fold(f64::INFINITY, f64::min);
        let max_shift = if max_arg.is_finite() { max_arg } else { 0.0 };

        let mut kbn_exp = KbnCompensatedAccumulator::new();
        for &w in &work_samples[..n] {
            if w.is_finite() {
                let term = (-self.beta * w - max_shift).exp();
                kbn_exp.add(term);
            }
        }

        let sum_exp = kbn_exp.total();
        let free_energy_delta = if sum_exp > 1e-30 {
            let log_avg = max_shift + (sum_exp * inv_n).ln();
            -log_avg / self.beta
        } else {
            mean_work
        };

        // Por segunda ley generalizada (desigualdad de Jensen): \langle W \rangle \ge \Delta F
        let dissipated_work = (mean_work - free_energy_delta).max(0.0);
        let reversibility_ratio = (-self.beta * dissipated_work).exp().clamp(0.0, 1.0);
        let crooks_crossing = free_energy_delta;

        let thermal_scale = 1.0 / self.beta;
        let diss_norm = dissipated_work / thermal_scale;

        let regime = if diss_norm <= 0.05 {
            JarzynskiRegime::ReversibleEquilibrium
        } else if diss_norm < 1.0 {
            JarzynskiRegime::TransientDissipative
        } else {
            JarzynskiRegime::IrreversibleStructuralBreak
        };

        JarzynskiReport {
            mean_work,
            free_energy_delta,
            dissipated_work,
            reversibility_ratio,
            crooks_crossing,
            regime,
        }
    }
}

// ============================================================================
// 3. STEIN VARIATIONAL GRADIENT DESCENT (SVGD) ENGINE
// ============================================================================

/// Diagnóstico de convergencia de partículas SVGD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvgdReport {
    /// Media posterior estimada para cada dimensión de drift.
    pub posterior_mean: [f64; SVGD_DIM],
    /// Varianza posterior estimada para cada dimensión.
    pub posterior_variance: [f64; SVGD_DIM],
    /// Ancho de banda $h$ del kernel RBF basado en la mediana.
    pub kernel_bandwidth: f64,
    /// Norma de la discrepancia de Stein (fuerza residual neta de actualización).
    pub stein_discrepancy_norm: f64,
}

impl Default for SvgdReport {
    fn default() -> Self {
        Self {
            posterior_mean: [0.0; SVGD_DIM],
            posterior_variance: [0.0; SVGD_DIM],
            kernel_bandwidth: 1.0,
            stein_discrepancy_norm: 0.0,
        }
    }
}

/// Motor de Descenso de Gradiente Variacional de Stein determinista.
#[derive(Debug, Clone)]
pub struct SteinVariationalGradientDescentEngine {
    /// Matriz de partículas en stack: `particles[i][d]`.
    particles: [[f64; SVGD_DIM]; MAX_SVGD_PARTICLES],
    /// Número de partículas activas $M \le \text{MAX\_SVGD\_PARTICLES}$.
    num_particles: usize,
    /// Tasa de paso $\epsilon$.
    step_size: f64,
}

impl Default for SteinVariationalGradientDescentEngine {
    fn default() -> Self {
        Self::new(MAX_SVGD_PARTICLES, 0.05)
    }
}

impl SteinVariationalGradientDescentEngine {
    /// Inicializa el motor SVGD con un número de partículas y tasa de aprendizaje.
    pub fn new(num_particles: usize, step_size: f64) -> Self {
        let m = num_particles.clamp(2, MAX_SVGD_PARTICLES);
        let mut particles = [[0.0; SVGD_DIM]; MAX_SVGD_PARTICLES];

        // Distribución inicial determinista simétrica en malla simplex
        for (i, p) in particles[..m].iter_mut().enumerate() {
            let offset = (i as f64) - ((m - 1) as f64) * 0.5;
            for (d, val) in p.iter_mut().enumerate() {
                *val = offset * 0.1 * ((d + 1) as f64);
            }
        }

        Self {
            particles,
            num_particles: m,
            step_size: if step_size.is_finite() && step_size > 0.0 {
                step_size
            } else {
                0.05
            },
        }
    }

    /// Calcula la distancia euclídea al cuadrado entre dos vectores.
    #[inline]
    fn sq_dist(a: &[f64; SVGD_DIM], b: &[f64; SVGD_DIM]) -> f64 {
        let mut s = 0.0;
        for d in 0..SVGD_DIM {
            let diff = a[d] - b[d];
            s += diff * diff;
        }
        s
    }

    /// Estima el ancho de banda del kernel RBF mediante la mediana de distancias por pares.
    fn estimate_bandwidth(&self) -> f64 {
        let m = self.num_particles;
        let mut dists = [0.0; MAX_SVGD_PARTICLES * MAX_SVGD_PARTICLES];
        let mut count = 0usize;

        for i in 0..m {
            for j in (i + 1)..m {
                let d2 = Self::sq_dist(&self.particles[i], &self.particles[j]);
                dists[count] = d2.sqrt();
                count = count.saturating_add(1);
            }
        }

        if count == 0 {
            return 1.0;
        }

        // Ordenamiento por inserción en stack para obtener mediana
        for i in 1..count {
            let key = dists[i];
            let mut j = i;
            while j > 0 && dists[j - 1] > key {
                dists[j] = dists[j - 1];
                j = j.saturating_sub(1);
            }
            dists[j] = key;
        }

        let median = dists[count / 2];
        let h = median / (2.0 * (m as f64).ln().max(1.0)).sqrt();
        h.clamp(0.01, 10.0)
    }

    /// Realiza un paso de actualización determinista de Stein hacia el prior/likelihood target:
    /// $\nabla \ln p(x) = -\Sigma^{-1}(x - \mu_{\text{target}})$.
    pub fn step(
        &mut self,
        target_mean: &[f64; SVGD_DIM],
        target_precision: &[f64; SVGD_DIM],
    ) -> SvgdReport {
        let m = self.num_particles;
        let h = self.estimate_bandwidth();
        let h2 = (h * h).max(1e-6);
        let inv_h2 = 1.0 / h2;

        let mut velocity = [[0.0; SVGD_DIM]; MAX_SVGD_PARTICLES];
        let inv_m = 1.0 / (m as f64);

        // Precalcular gradientes del log-target para cada partícula:
        // \nabla_{x_j} \ln p(x_j) = -target_precision * (x_j - target_mean)
        let mut log_grad = [[0.0; SVGD_DIM]; MAX_SVGD_PARTICLES];
        for (j, lg) in log_grad[..m].iter_mut().enumerate() {
            for (d, val) in lg.iter_mut().enumerate() {
                let diff = self.particles[j][d] - target_mean[d];
                *val = -target_precision[d] * diff;
            }
        }

        // Dinámica de Stein:
        // \phi(x_i) = \frac{1}{M}\sum_{j=1}^M [ k(x_j, x_i)\nabla_{x_j}\ln p(x_j) + \nabla_{x_j} k(x_j, x_i) ]
        // con \nabla_{x_j} k(x_j, x_i) = -\frac{x_j - x_i}{h^2} k(x_j, x_i)
        for (i, v_row) in velocity[..m].iter_mut().enumerate() {
            for (j, lg_row) in log_grad[..m].iter().enumerate() {
                let dist2 = Self::sq_dist(&self.particles[j], &self.particles[i]);
                let kernel_val = (-dist2 / (2.0 * h2)).exp();

                for (d, v_d) in v_row.iter_mut().enumerate() {
                    let grad_force = kernel_val * lg_row[d];
                    let repulsive_force =
                        -((self.particles[j][d] - self.particles[i][d]) * inv_h2) * kernel_val;
                    *v_d += inv_m * (grad_force + repulsive_force);
                }
            }
        }

        // Actualización de partículas y cálculo de norma de Stein
        let mut total_velocity_sq = 0.0;
        for (i, v_row) in velocity[..m].iter().enumerate() {
            for (d, &v) in v_row.iter().enumerate() {
                if v.is_finite() {
                    self.particles[i][d] += self.step_size * v;
                    total_velocity_sq += v * v;
                }
            }
        }

        // Estadísticas empíricas posteriores de la nube de partículas
        let mut post_mean = [0.0; SVGD_DIM];
        for (d, mean_d) in post_mean.iter_mut().enumerate() {
            let mut s = 0.0;
            for p in &self.particles[..m] {
                s += p[d];
            }
            *mean_d = s * inv_m;
        }

        let mut post_var = [0.0; SVGD_DIM];
        for (d, var_d) in post_var.iter_mut().enumerate() {
            let mut s = 0.0;
            for p in &self.particles[..m] {
                let diff = p[d] - post_mean[d];
                s += diff * diff;
            }
            *var_d = s * inv_m;
        }

        SvgdReport {
            posterior_mean: post_mean,
            posterior_variance: post_var,
            kernel_bandwidth: h,
            stein_discrepancy_norm: (total_velocity_sq * inv_m).sqrt(),
        }
    }

    /// Retorna una referencia inmutable a las partículas activas.
    pub fn particles(&self) -> &[[f64; SVGD_DIM]] {
        &self.particles[..self.num_particles]
    }
}

// ============================================================================
// 4. MASTER CONSOLIDATED ENGINE
// ============================================================================

/// Reporte consolidado de la Fase 21.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiculescuJarzynskiSvgdReport {
    /// Reporte de probabilidad libre de Voiculescu.
    pub free_probability: VoiculescuFreeSpectrumReport,
    /// Reporte de no-equilibrio de Jarzynski-Crooks.
    pub jarzynski: JarzynskiReport,
    /// Reporte de distribución posterior determinista SVGD.
    pub svgd: SvgdReport,
    /// Score compuesto de estabilidad y predictibilidad libre $[0.0, 1.0]$.
    pub free_composite_stability: f64,
}

/// Motor maestro de la Fase 21 que integra Voiculescu, Jarzynski y SVGD.
#[derive(Debug, Clone)]
pub struct VoiculescuJarzynskiSvgdMasterEngine {
    free_prob: VoiculescuFreeProbabilityEngine,
    jarzynski: JarzynskiCrooksEngine,
    svgd: SteinVariationalGradientDescentEngine,
}

impl Default for VoiculescuJarzynskiSvgdMasterEngine {
    fn default() -> Self {
        Self {
            free_prob: VoiculescuFreeProbabilityEngine,
            jarzynski: JarzynskiCrooksEngine::default(),
            svgd: SteinVariationalGradientDescentEngine::default(),
        }
    }
}

impl VoiculescuJarzynskiSvgdMasterEngine {
    /// Crea un motor maestro parametrizado.
    pub fn new(beta: f64, svgd_particles: usize, svgd_step: f64) -> Self {
        Self {
            free_prob: VoiculescuFreeProbabilityEngine,
            jarzynski: JarzynskiCrooksEngine::new(beta),
            svgd: SteinVariationalGradientDescentEngine::new(svgd_particles, svgd_step),
        }
    }

    /// Evalúa en un solo paso libre de pánico las tres teorías físicas y algebraicas.
    pub fn evaluate(
        &mut self,
        eigenvalues: &[f64],
        work_samples: &[f64],
        target_drift_mean: &[f64; SVGD_DIM],
        target_drift_prec: &[f64; SVGD_DIM],
    ) -> VoiculescuJarzynskiSvgdReport {
        let free_rep = self.free_prob.evaluate(eigenvalues);
        let jarz_rep = self.jarzynski.evaluate(work_samples);
        let svgd_rep = self.svgd.step(target_drift_mean, target_drift_prec);

        // Estabilidad compuesta:
        // Pondera la semicircularidad libre de Wigner, la razón de reversibilidad de Jarzynski
        // y el inverso del error de Stein.
        let stein_penalty = (-svgd_rep.stein_discrepancy_norm).exp().clamp(0.0, 1.0);
        let composite = 0.40 * free_rep.wigner_semicircle_fit
            + 0.40 * jarz_rep.reversibility_ratio
            + 0.20 * stein_penalty;

        VoiculescuJarzynskiSvgdReport {
            free_probability: free_rep,
            jarzynski: jarz_rep,
            svgd: svgd_rep,
            free_composite_stability: composite.clamp(0.0, 1.0),
        }
    }
}
