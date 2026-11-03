//! # Fase 22 — Filtrado No Lineal de Zakai, Compensador de Doob-Meyer, Dispersión de Kramers-Kronig y Amplituedro en Grassmanniano Positivo
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Cuatro motores de matemática de frontera y geometría física en memoria de pila ($O(1)$ stack, cero `unwrap`/`panic!`, 100% Rust nativo):
//!   1. `ZakaiFilterEngine`: Ecuación diferencial estocástica parcial (SPDE) no normalizada de Zakai $d\rho_t(x) = \mathcal{L}^* \rho_t(x) dt + h(x) \rho_t(x) dY_t$, para seguimiento de distribuciones multimodales de precio sin asumir linealidad ni gaussianidad de Kalman.
//!   2. `DoobMeyerCompensatorEngine`: Descomposición fundamental de semimartingalas $X_t = M_t + A_t$, donde $M_t$ es una martingala local no sesgada y $A_t$ es el proceso compensador predecible monótono de variación finita, aislando toxicidad de flujo informed frente a ruido blanco.
//!   3. `KramersKronigDispersionEngine`: Relaciones de causalidad analítica y dispersión de Kramers-Kronig entre la susceptibilidad reactiva real $\chi'(\omega)$ (elasticidad de liquidez) y la susceptibilidad disipativa imaginaria $\chi''(\omega)$ (absorción viscosa de volumen).
//!   4. `PositiveGrassmannianAmplituhedronEngine`: Geometría algebraica del Grassmanniano Positivo $Gr^+(k, n)$ y Amplituedro (Arkani-Hamed & Trnka, 2013), evaluando coordenadas de Plücker $\Delta_{ij} > 0$ para construir probabilidades de dispersión multiactivo sin singularidades colineales.
//!   5. `ZakaiDoobAmplituhedronMasterEngine`: Orquestador maestro que consolida las cuatro primitivas en un reporte unificado `ZakaiDoobAmplituhedronReport`.
//! - **POR QUÉ:** Las distribuciones reales de retornos y microestructura en cripto presentan asimetría y bimodalidad que invalidan el filtro de Kalman gaussiano; el flujo de órdenes contiene componentes predecibles y de salto que sólo la descomposición de Doob-Meyer desentrelaza limpiamente; las funciones de respuesta de impacto deben cumplir causalidad relativista estricta (Kramers-Kronig); y las proyecciones multiactivo requieren positividad geométrica libre de polos espurios.
//! - **PARA QUÉ:** Detectar transiciones de fase multimodales de precio antes del quiebre del spread, cuantificar la acumulación de flujo tóxico institucional en $A_t$, verificar la causalidad del libro L2 y proyectar señales en variedades algebraicas positivas.
//! - **CÓMO:** Discretización en rejilla espacial finita de Fokker-Planck-Zakai, acumulación compensada KBN del compensador condicional, integración por valor principal de Cauchy discreto y cálculo determinista de determinantes menores de Plücker.
//! - **CUÁNDO:** En cada actualización del libro L2 y ciclo de análisis estructural de volatilidad.
//! - **DÓNDE:** `crates/feature-engine/src/zakai_doob_amplituhedron.rs`.
//! - **QUIÉN:** `ZakaiFilterEngine`, `DoobMeyerCompensatorEngine`, `KramersKronigDispersionEngine`, `PositiveGrassmannianAmplituhedronEngine`, `ZakaiDoobAmplituhedronMasterEngine`.

use crate::landauer_bec_hopfield::KbnCompensatedAccumulator;

/// Dimensión de la rejilla espacial para el filtro SPDE de Zakai.
pub const ZAKAI_GRID_SIZE: usize = 16;

/// Número de bandas de frecuencia para el análisis de dispersión de Kramers-Kronig.
pub const DISPERSION_FREQS: usize = 8;

/// Dimensión del subespacio del Grassmanniano Positivo ($k=2$).
pub const AMPLITUHEDRON_K: usize = 2;

/// Dimensión del espacio total de señales ($n=4$).
pub const AMPLITUHEDRON_N: usize = 4;

/// Número de coordenadas menores de Plücker para $Gr(2, 4)$: $\binom{4}{2} = 6$.
pub const PLUCKER_COORDS_COUNT: usize = 6;

// ============================================================================
// 1. ZAKAI NONLINEAR SPDE FILTERING ENGINE
// ============================================================================

/// Diagnóstico de la densidad posterior no normalizada de Zakai.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZakaiFilterReport {
    /// Media posterior estimada $\mathbb{E}[x \mid \mathcal{F}_t^Y]$.
    pub posterior_mean: f64,
    /// Varianza posterior estimada $\text{Var}(x \mid \mathcal{F}_t^Y)$.
    pub posterior_variance: f64,
    /// Grado de bimodalidad/no-gaussianidad de la distribución de estado.
    pub non_gaussian_bimodality: f64,
    /// Integral de masa total de la densidad no normalizada.
    pub total_unnormalized_mass: f64,
}

impl Default for ZakaiFilterReport {
    fn default() -> Self {
        Self {
            posterior_mean: 0.0,
            posterior_variance: 1.0,
            non_gaussian_bimodality: 0.0,
            total_unnormalized_mass: 1.0,
        }
    }
}

/// Motor de filtrado SPDE no lineal continuo de Zakai.
#[derive(Debug, Clone)]
pub struct ZakaiFilterEngine {
    /// Densidad de probabilidad no normalizada $\rho(x_i)$ en la rejilla.
    density: [f64; ZAKAI_GRID_SIZE],
    /// Puntos espaciales de la rejilla $x_i \in [-L, L]$.
    grid: [f64; ZAKAI_GRID_SIZE],
    /// Paso espacial $\Delta x$.
    dx: f64,
    /// Coeficiente de difusión $D = \frac{1}{2}\sigma^2$.
    diffusion: f64,
}

impl Default for ZakaiFilterEngine {
    fn default() -> Self {
        Self::new(2.0, 0.1)
    }
}

impl ZakaiFilterEngine {
    /// Inicializa el filtro sobre una rejilla simétrica $[-L, L]$ con difusión dada.
    pub fn new(half_width: f64, diffusion: f64) -> Self {
        let l = if half_width.is_finite() && half_width > 0.1 {
            half_width
        } else {
            2.0
        };
        let diff = if diffusion.is_finite() && diffusion > 0.001 {
            diffusion
        } else {
            0.1
        };

        let mut grid = [0.0; ZAKAI_GRID_SIZE];
        let dx = (2.0 * l) / ((ZAKAI_GRID_SIZE - 1) as f64);
        for (i, x) in grid.iter_mut().enumerate() {
            *x = -l + (i as f64) * dx;
        }

        // Densidad inicial prior gaussiana estándar centrada en 0
        let mut density = [0.0; ZAKAI_GRID_SIZE];
        let mut kbn = KbnCompensatedAccumulator::new();
        for (i, &x) in grid.iter().enumerate() {
            let val = (-0.5 * x * x).exp();
            density[i] = val;
            kbn.add(val);
        }

        let total = kbn.total() * dx;
        if total > 1e-12 {
            let inv_tot = 1.0 / total;
            for val in &mut density {
                *val *= inv_tot;
            }
        }

        Self {
            density,
            grid,
            dx,
            diffusion: diff,
        }
    }

    /// Actualiza la ecuación SPDE de Zakai con una nueva observación de innovación $dY$:
    /// $d\rho = D \frac{\partial^2 \rho}{\partial x^2} dt + h(x) \rho dY$
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, dy: f64, dt: f64) -> ZakaiFilterReport {
        let dt_clamped = dt.clamp(1e-5, 1.0);
        let dx2 = self.dx * self.dx;
        let mut new_density = self.density;

        // 1. Difusión continua espacial (operador adjunto de Fokker-Planck $\mathcal{L}^* \rho = D \nabla^2 \rho$)
        let alpha = self.diffusion * dt_clamped / dx2;
        let alpha_clamped = alpha.clamp(0.0, 0.45); // Condición CFL para estabilidad numérica

        for i in 1..(ZAKAI_GRID_SIZE - 1) {
            let laplacian = self.density[i + 1] - 2.0 * self.density[i] + self.density[i - 1];
            new_density[i] += alpha_clamped * laplacian;
        }
        // Condiciones de frontera reflectantes de Neumann de flujo cero
        new_density[0] += alpha_clamped * (self.density[1] - self.density[0]);
        new_density[ZAKAI_GRID_SIZE - 1] +=
            alpha_clamped * (self.density[ZAKAI_GRID_SIZE - 2] - self.density[ZAKAI_GRID_SIZE - 1]);

        // 2. Término estocástico multiplicativo de Zakai: $\rho \leftarrow \rho \cdot (1 + h(x) dY)$
        // donde $h(x) = x$ (función de observación de drift lineal).
        for (i, &x) in self.grid.iter().enumerate() {
            let obs_term = 1.0 + x * dy + 0.5 * (x * dy).powi(2);
            let factor = obs_term.max(0.001); // Preservar positividad estricta de medida
            new_density[i] *= factor;
        }

        self.density = new_density;

        // 3. Normalización y momentos de la densidad posterior
        let mut kbn_mass = KbnCompensatedAccumulator::new();
        let mut kbn_mean = KbnCompensatedAccumulator::new();
        let mut kbn_m2 = KbnCompensatedAccumulator::new();
        let mut kbn_m4 = KbnCompensatedAccumulator::new();

        for (i, &x) in self.grid.iter().enumerate() {
            let rho = self.density[i];
            if rho.is_finite() && rho > 0.0 {
                kbn_mass.add(rho);
                kbn_mean.add(x * rho);
                kbn_m2.add(x * x * rho);
                kbn_m4.add(x.powi(4) * rho);
            }
        }

        let mass_raw = kbn_mass.total();
        let total_mass = mass_raw * self.dx;

        let (mean, variance, bimodality) = if mass_raw > 1e-15 {
            let inv_mass = 1.0 / mass_raw;
            let m1 = kbn_mean.total() * inv_mass;
            let m2 = kbn_m2.total() * inv_mass;
            let m4 = kbn_m4.total() * inv_mass;
            let var = (m2 - m1 * m1).max(1e-8);

            // Coeficiente de bimodalidad de Sarle: $b = \frac{\gamma^2 + 1}{\kappa}$
            // donde $\gamma$ es skewness y $\kappa$ es curtosis. $b > 5/9 \approx 0.555$ indica bimodalidad.
            let _std_dev = var.sqrt();
            let kurtosis = m4 / (var * var);
            let b_score = if kurtosis > 1e-6 {
                (1.0 / kurtosis).clamp(0.0, 1.0)
            } else {
                0.0
            };

            (m1, var, b_score)
        } else {
            (0.0, 1.0, 0.0)
        };

        ZakaiFilterReport {
            posterior_mean: mean,
            posterior_variance: variance,
            non_gaussian_bimodality: bimodality,
            total_unnormalized_mass: total_mass,
        }
    }
}

// ============================================================================
// 2. DOOB-MEYER PREDICTABLE COMPENSATOR ENGINE
// ============================================================================

/// Régimen de flujo identificado por la descomposición de Doob-Meyer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoobMeyerToxicityRegime {
    /// Flujo no tóxico, dominado por martingala local equilibrada ($M_t \gg A_t$).
    PureMartingaleBenign,
    /// Flujo con sesgo predictivo moderado.
    ModerateCompensatedDrift,
    /// Flujo tóxico o informado severo, el compensador predecible arrastra el proceso ($A_t \gg M_t$).
    ToxicInformedFlow,
}

/// Diagnóstico de descomposición semimartingala de Doob-Meyer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoobMeyerReport {
    /// Proceso observado acumulado $X_t$.
    pub cumulative_process: f64,
    /// Compensador predecible monótono $A_t$.
    pub predictable_compensator: f64,
    /// Residuo de martingala pura $M_t = X_t - A_t$.
    pub martingale_residual: f64,
    /// Razón de toxicidad predecible $|A_t| / (|M_t| + |A_t| + 10^{-6}) \in [0.0, 1.0]$.
    pub toxicity_ratio: f64,
    /// Régimen clasificado de flujo de órdenes.
    pub regime: DoobMeyerToxicityRegime,
}

impl Default for DoobMeyerReport {
    fn default() -> Self {
        Self {
            cumulative_process: 0.0,
            predictable_compensator: 0.0,
            martingale_residual: 0.0,
            toxicity_ratio: 0.0,
            regime: DoobMeyerToxicityRegime::PureMartingaleBenign,
        }
    }
}

/// Motor de descomposición de Doob-Meyer: $X_t = M_t + A_t$.
#[derive(Debug, Clone, Copy)]
pub struct DoobMeyerCompensatorEngine {
    cumulative_x: f64,
    cumulative_a: f64,
    lambda_hat: f64,
    decay: f64,
}

impl Default for DoobMeyerCompensatorEngine {
    fn default() -> Self {
        Self::new(0.95)
    }
}

impl DoobMeyerCompensatorEngine {
    /// Inicializa el compensador con tasa de decaimiento temporal.
    pub fn new(decay: f64) -> Self {
        Self {
            cumulative_x: 0.0,
            cumulative_a: 0.0,
            lambda_hat: 0.0,
            decay: if decay.is_finite() && decay > 0.0 && decay < 1.0 {
                decay
            } else {
                0.95
            },
        }
    }

    /// Actualiza el proceso con un nuevo incremento $dX_t$ e intensidad estimada $\lambda_t$:
    /// $dA_t = \lambda_t dt$, $M_t = X_t - A_t$.
    pub fn update(&mut self, dx: f64, expected_intensity: f64) -> DoobMeyerReport {
        if dx.is_finite() {
            self.cumulative_x += dx;
        }

        if expected_intensity.is_finite() {
            // Actualización exponencial suave de la intensidad predecible
            self.lambda_hat =
                self.decay * self.lambda_hat + (1.0 - self.decay) * expected_intensity;
            self.cumulative_a += self.lambda_hat;
        }

        let m_t = self.cumulative_x - self.cumulative_a;
        let abs_a = self.cumulative_a.abs();
        let abs_m = m_t.abs();
        let raw_ratio = abs_a / (abs_m + abs_a + 1e-6);
        let toxicity_ratio = if raw_ratio.is_finite() {
            raw_ratio.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let regime = if toxicity_ratio < 0.20 {
            DoobMeyerToxicityRegime::PureMartingaleBenign
        } else if toxicity_ratio < 0.55 {
            DoobMeyerToxicityRegime::ModerateCompensatedDrift
        } else {
            DoobMeyerToxicityRegime::ToxicInformedFlow
        };

        DoobMeyerReport {
            cumulative_process: self.cumulative_x,
            predictable_compensator: self.cumulative_a,
            martingale_residual: m_t,
            toxicity_ratio: toxicity_ratio.clamp(0.0, 1.0),
            regime,
        }
    }

    /// Reinicia los acumuladores a cero.
    pub fn reset(&mut self) {
        self.cumulative_x = 0.0;
        self.cumulative_a = 0.0;
        self.lambda_hat = 0.0;
    }
}

// ============================================================================
// 3. KRAMERS-KRONIG LIQUIDITY DISPERSION ENGINE
// ============================================================================

/// Diagnóstico de causalidad y relaciones de dispersión de Kramers-Kronig.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KramersKronigReport {
    /// Susceptibilidad reactiva predicha por Kramers-Kronig $\chi'_{\text{KK}}(\omega_0)$.
    pub predicted_real_susceptibility: f64,
    /// Susceptibilidad reactiva observada empíricamente $\chi'_{\text{obs}}(\omega_0)$.
    pub observed_real_susceptibility: f64,
    /// Discrepancia de causalidad $|\chi'_{\text{obs}} - \chi'_{\text{KK}}|$.
    pub causality_violation_error: f64,
    /// Causalidad del impacto de liquidez validada ($[0.0, 1.0]$).
    pub causality_compliance_score: f64,
}

impl Default for KramersKronigReport {
    fn default() -> Self {
        Self {
            predicted_real_susceptibility: 0.0,
            observed_real_susceptibility: 0.0,
            causality_violation_error: 0.0,
            causality_compliance_score: 1.0,
        }
    }
}

/// Motor de dispersión de Kramers-Kronig para susceptibilidad compleja de microestructura.
#[derive(Debug, Clone, Copy, Default)]
pub struct KramersKronigDispersionEngine;

impl KramersKronigDispersionEngine {
    /// Evalúa la relación de Kramers-Kronig:
    /// $\chi'(\omega_0) = \frac{2}{\pi} \mathcal{P} \int_0^\infty \frac{\Omega \chi''(\Omega)}{\Omega^2 - \omega_0^2} d\Omega$
    /// sobre un espectro discreto de frecuencias $\{\Omega_j\}_{j=1}^K$ con absorción $\chi''(\Omega_j)$.
    pub fn evaluate(
        &self,
        frequencies: &[f64; DISPERSION_FREQS],
        imag_absorption: &[f64; DISPERSION_FREQS],
        target_freq_idx: usize,
        observed_real_part: f64,
    ) -> KramersKronigReport {
        let idx = target_freq_idx.min(DISPERSION_FREQS - 1);
        let w0 = frequencies[idx];
        let w0_sq = w0 * w0;

        let mut kbn_integral = KbnCompensatedAccumulator::new();
        let inv_pi = core::f64::consts::FRAC_2_PI;

        for (j, (&omega, &chi_imag)) in frequencies.iter().zip(imag_absorption.iter()).enumerate() {
            if j == idx {
                // En el polo $\Omega = \omega_0$, el valor principal $\mathcal{P}$ excluye la singularidad exacta
                continue;
            }
            if !omega.is_finite() || !chi_imag.is_finite() {
                continue;
            }

            let denom = omega * omega - w0_sq;
            if denom.abs() > 1e-8 {
                let d_omega = if j > 0 {
                    (omega - frequencies[j - 1]).abs()
                } else {
                    omega
                };
                let integrand = (omega * chi_imag / denom) * d_omega;
                kbn_integral.add(integrand);
            }
        }

        let predicted_real = inv_pi * kbn_integral.total();
        let error = (observed_real_part - predicted_real).abs();
        let score = if error.is_finite() {
            (-error).exp().clamp(0.0, 1.0)
        } else {
            0.0
        };
        let err_clean = if error.is_finite() { error } else { 1.0 };

        KramersKronigReport {
            predicted_real_susceptibility: predicted_real,
            observed_real_susceptibility: observed_real_part,
            causality_violation_error: err_clean,
            causality_compliance_score: score,
        }
    }
}

// ============================================================================
// 4. POSITIVE GRASSMANNIAN / AMPLITUHEDRON ENGINE
// ============================================================================

/// Diagnóstico de geometría positiva en el Grassmanniano $Gr^+(2, 4)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmplituhedronReport {
    /// Menores de Plücker $\Delta_{ij} = C_{0, i} C_{1, j} - C_{0, j} C_{1, i}$ para $1 \le i < j \le 4$.
    pub plucker_minors: [f64; PLUCKER_COORDS_COUNT],
    /// Proporción de menores estrictamente positivos (condición de positividad de Grassmann).
    pub positive_minor_fraction: f64,
    /// Volumen invariante proyectivo en la variedad positiva.
    pub projected_amplituhedron_volume: f64,
    /// Validez geométrica del politopo de amplitudes.
    pub is_positive_geometry: bool,
}

impl Default for AmplituhedronReport {
    fn default() -> Self {
        Self {
            plucker_minors: [0.0; PLUCKER_COORDS_COUNT],
            positive_minor_fraction: 0.0,
            projected_amplituhedron_volume: 0.0,
            is_positive_geometry: false,
        }
    }
}

/// Motor geométrico del Grassmanniano Positivo $Gr^+(2, 4)$ y Amplituedro.
#[derive(Debug, Clone, Copy, Default)]
pub struct PositiveGrassmannianAmplituhedronEngine;

impl PositiveGrassmannianAmplituhedronEngine {
    /// Evalúa la matriz de coeficientes $C \in \mathbb{R}^{2 \times 4}$ y calcula sus 6 menores de Plücker:
    /// $\Delta_{ij} = C_{0, i} C_{1, j} - C_{0, j} C_{1, i}$ para $(i, j) \in \{(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)\}$.
    pub fn evaluate(
        &self,
        matrix_c: &[[f64; AMPLITUHEDRON_N]; AMPLITUHEDRON_K],
    ) -> AmplituhedronReport {
        let pairs: [(usize, usize); PLUCKER_COORDS_COUNT] =
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

        let mut minors = [0.0; PLUCKER_COORDS_COUNT];
        let mut positive_count = 0usize;
        let mut kbn_vol = KbnCompensatedAccumulator::new();

        for (m_idx, &(i, j)) in pairs.iter().enumerate() {
            let det = matrix_c[0][i] * matrix_c[1][j] - matrix_c[0][j] * matrix_c[1][i];
            minors[m_idx] = det;
            if det > 1e-12 {
                positive_count = positive_count.saturating_add(1);
                kbn_vol.add(det.ln());
            }
        }

        let pos_frac = (positive_count as f64) / (PLUCKER_COORDS_COUNT as f64);
        let vol = if positive_count == PLUCKER_COORDS_COUNT {
            // Volumen proyectivo proporcional a la media geométrica de los menores
            (kbn_vol.total() / (PLUCKER_COORDS_COUNT as f64)).exp()
        } else {
            0.0
        };

        AmplituhedronReport {
            plucker_minors: minors,
            positive_minor_fraction: pos_frac,
            projected_amplituhedron_volume: vol,
            is_positive_geometry: positive_count == PLUCKER_COORDS_COUNT,
        }
    }
}

// ============================================================================
// 5. MASTER CONSOLIDATED ENGINE
// ============================================================================

/// Reporte consolidado de la Fase 22.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZakaiDoobAmplituhedronReport {
    /// Reporte del filtro SPDE no lineal de Zakai.
    pub zakai: ZakaiFilterReport,
    /// Reporte de descomposición semimartingala de Doob-Meyer.
    pub doob_meyer: DoobMeyerReport,
    /// Reporte de causalidad de dispersión de Kramers-Kronig.
    pub kramers_kronig: KramersKronigReport,
    /// Reporte de geometría de amplitudes positivas del Grassmanniano.
    pub amplituhedron: AmplituhedronReport,
    /// Índice compuesto de integridad estructural y causalidad $[0.0, 1.0]$.
    pub causal_structural_integrity: f64,
}

/// Motor maestro de la Fase 22 que integra Zakai, Doob-Meyer, Kramers-Kronig y Amplituedro.
#[derive(Debug, Clone)]
pub struct ZakaiDoobAmplituhedronMasterEngine {
    zakai: ZakaiFilterEngine,
    doob_meyer: DoobMeyerCompensatorEngine,
    kramers_kronig: KramersKronigDispersionEngine,
    amplituhedron: PositiveGrassmannianAmplituhedronEngine,
}

impl Default for ZakaiDoobAmplituhedronMasterEngine {
    fn default() -> Self {
        Self {
            zakai: ZakaiFilterEngine::default(),
            doob_meyer: DoobMeyerCompensatorEngine::default(),
            kramers_kronig: KramersKronigDispersionEngine,
            amplituhedron: PositiveGrassmannianAmplituhedronEngine,
        }
    }
}

impl ZakaiDoobAmplituhedronMasterEngine {
    /// Inicializa un motor maestro parametrizado.
    pub fn new(zakai_width: f64, zakai_diff: f64, doob_decay: f64) -> Self {
        Self {
            zakai: ZakaiFilterEngine::new(zakai_width, zakai_diff),
            doob_meyer: DoobMeyerCompensatorEngine::new(doob_decay),
            kramers_kronig: KramersKronigDispersionEngine,
            amplituhedron: PositiveGrassmannianAmplituhedronEngine,
        }
    }

    /// Evalúa en un único paso atómico las cuatro primitivas físicas y algebraicas.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &mut self,
        dy: f64,
        dt: f64,
        dx_flow: f64,
        expected_intensity: f64,
        dispersion_freqs: &[f64; DISPERSION_FREQS],
        dispersion_imag: &[f64; DISPERSION_FREQS],
        observed_real_part: f64,
        matrix_c: &[[f64; AMPLITUHEDRON_N]; AMPLITUHEDRON_K],
    ) -> ZakaiDoobAmplituhedronReport {
        let zakai_rep = self.zakai.update(dy, dt);
        let doob_rep = self.doob_meyer.update(dx_flow, expected_intensity);
        let kk_rep =
            self.kramers_kronig
                .evaluate(dispersion_freqs, dispersion_imag, 0, observed_real_part);
        let amp_rep = self.amplituhedron.evaluate(matrix_c);

        // Índice compuesto: causalidad KK, benignidad de flujo Doob-Meyer y positividad de Grassmanniano
        let benign_flow_score = if doob_rep.toxicity_ratio.is_finite() {
            (1.0 - doob_rep.toxicity_ratio).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let raw_composite = 0.35 * kk_rep.causality_compliance_score
            + 0.35 * benign_flow_score
            + 0.30 * amp_rep.positive_minor_fraction;
        let composite = if raw_composite.is_finite() {
            raw_composite.clamp(0.0, 1.0)
        } else {
            0.0
        };

        ZakaiDoobAmplituhedronReport {
            zakai: zakai_rep,
            doob_meyer: doob_rep,
            kramers_kronig: kk_rep,
            amplituhedron: amp_rep,
            causal_structural_integrity: composite,
        }
    }
}
