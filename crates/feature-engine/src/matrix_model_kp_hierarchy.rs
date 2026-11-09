//! # Fase 79 (§79) — Teoría de Modelos Matriciales & Jerarquía KP / KdV Grassmanniana Infinita
//!
//! ## 1. QUÉ
//! Módulo de alta fidelidad físico-matemática que modela la microestructura multiactivo
//! como una función de partición de un Modelo Matricial Hermítico aleatorio $N \times N$,
//! donde la función generadora es una función $\tau$ de la Jerarquía integrable de Kadomtsev-Petviashvili (KP)
//! en el Grassmanniano infinito de Sato:
//! $$u(x, y, t) = 2 \frac{\partial^2}{\partial x^2} \ln \tau(x, y, t)$$
//! gobernada por la ecuación diferencial parcial KP:
//! $$\frac{\partial}{\partial x} \left( -4 \frac{\partial u}{\partial t} + 6 u \frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} \right) + 3 \frac{\partial^2 u}{\partial y^2} = 0$$
//!
//! ## 2. POR QUÉ
//! En la microestructura cuántica de mercados financieros, las perturbaciones de liquidez y los flujos
//! de correlación no se propagan como ondas lineales gaussianas, sino como solitones y ondas de choque
//! no lineales dispersivas en dos dimensiones (tiempo espacializado de transacciones $x$ y dispersión
//! transversal de volatilidad $y$). Los momentos temporales de Miwa $t_k = \frac{1}{k}\operatorname{Tr}(M^k)$
//! de la matriz hermítica de correlación de libro determinan los tiempos de deformación de la jerarquía KP.
//!
//! ## 3. PARA QUÉ
//! - Extraer los momentos matriciales de Miwa $t_1, t_2, t_3, t_4$.
//! - Evaluar la función $\tau$ mediante la parametrización de multisolitones de Hirota.
//! - Computar el potencial solitónico $u(x, y, t)$ y su curvatura de choque $u_{xx}$.
//! - Detectar singularidades y choques dispersivos KP (`kp_soliton_shock_detected`) para anticipar
//!   avalanchas de desintegración del libro antes de que colapsen los precios.
//! - Modular el factor de escala continuo de slot $\eta_{\text{kp}} \in [0.20, 1.35]$ de forma libre de heurísticas.
//!
//! ## 4. CÓMO
//! 1. Se recibe la matriz hermítica $4 \times 4$ $M$ representativa de correlación de flujo de órdenes.
//! 2. Se computan los momentos traza normalizados $t_k = \frac{1}{k}\operatorname{Tr}(M^k)$ para $k \in \{1, 2, 3, 4\}$.
//! 3. Se sintetiza la función tau $\tau(x, y, t)$ con dos solitones resonantes de Hirota.
//! 4. Se calcula el potencial del solitón $u(x, y, t) = 2 \partial_x^2 \ln \tau$ y su derivada de curvatura $u_{xx}$.
//! 5. Se calcula el residual dispersivo de la ecuación KP. Si excede el umbral crítico, se detecta el choque solitónico.
//! 6. Se calcula el factor de escala de posición $\eta_{\text{kp}} = \text{clamp}(1.0 / (1.0 + \kappa |u_{xx}|), 0.20, 1.35)$.
//!
//! ## 5. CUÁNDO
//! En cada ciclo de procesamiento de features multivariantes de libro de órdenes.
//!
//! ## 6. DÓNDE
//! `crates/feature-engine/src/matrix_model_kp_hierarchy.rs`, re-exportado en `lib.rs`.
//!
//! ## 7. QUIÉN
//! Diseñado e implementado por el Consejo de 10 Roles Senior (Físico Matemático, Quant Developer, QA Engineer).

pub const KP_MATRIX_DIM: usize = 4;
pub const KP_MOMENTS_COUNT: usize = 4;

/// Configuración para el motor de Modelos Matriciales y Jerarquía KP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixModelKpConfig {
    /// Escala de amplitud para la función tau.
    pub tau_scale: f64,
    /// Coeficiente de dispersión transversal KP.
    pub dispersion_alpha: f64,
    /// Umbral crítico de curvatura para choque solitónico |u_xx|.
    pub critical_shock_curvature: f64,
    /// Umbral crítico de residual dispersivo KP.
    pub critical_residual_threshold: f64,
    /// Sensibilidad de escala para slot.
    pub slot_sensitivity: f64,
}

impl Default for MatrixModelKpConfig {
    fn default() -> Self {
        Self {
            tau_scale: 1.0,
            dispersion_alpha: 1.0,
            critical_shock_curvature: 12.0,
            critical_residual_threshold: 30.0,
            slot_sensitivity: 0.08,
        }
    }
}

/// Reporte de estado emitido por la jerarquía KP y modelos matriciales.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixModelKpReport {
    /// Momentos de Miwa t_k = (1/k) Tr(M^k) para k in [1, 4].
    pub moments_tk: [f64; KP_MOMENTS_COUNT],
    /// Valor central de la función tau(x, y, t).
    pub tau_function_val: f64,
    /// Potencial del solitón u(x, y, t) = 2 d^2_x ln(tau).
    pub soliton_potential_u: f64,
    /// Curvatura espacial del choque solitónico u_xx.
    pub kp_curvature_uxx: f64,
    /// Residual numérico dispersivo de la ecuación KP.
    pub dispersion_residual: f64,
    /// Detección de singularidad de choque solitónico KP.
    pub kp_soliton_shock_detected: bool,
    /// Factor de escala continuo de slot [0.20, 1.35].
    pub slot_scale_factor: f64,
}

/// Motor determinista de Modelos Matriciales y Jerarquía KP.
#[derive(Debug, Clone)]
pub struct MatrixModelKpHierarchyEngine {
    config: MatrixModelKpConfig,
    prev_u: f64,
    prev_ux: f64,
    accumulated_time: f64,
    initialized: bool,
}

impl Default for MatrixModelKpHierarchyEngine {
    fn default() -> Self {
        Self::new(MatrixModelKpConfig::default())
    }
}

impl MatrixModelKpHierarchyEngine {
    /// Crea una nueva instancia del motor con la configuración dada.
    pub fn new(config: MatrixModelKpConfig) -> Self {
        Self {
            config,
            prev_u: 0.0,
            prev_ux: 0.0,
            accumulated_time: 0.0,
            initialized: false,
        }
    }

    /// Evalúa la jerarquía KP dado el tensor hermítico de microestructura M (4x4),
    /// el paso de tiempo dt_seconds y las coordenadas de fase del libro (x, y).
    pub fn evaluate(
        &mut self,
        matrix_m: &[[f64; KP_MATRIX_DIM]; KP_MATRIX_DIM],
        dt_seconds: f64,
        x: f64,
        y: f64,
    ) -> MatrixModelKpReport {
        let dt = if dt_seconds.is_finite() && dt_seconds > 0.0001 {
            dt_seconds.min(10.0)
        } else {
            0.1
        };
        self.accumulated_time += dt;
        let t = self.accumulated_time;

        // 1. Cálculo de momentos matriciales de Miwa t_k = (1/k) Tr(M^k)
        let moments = self.compute_miwa_moments(matrix_m);

        // 2. Parámetros de los solitones de Hirota derivados de los momentos
        let k1 = (0.5 + 0.2 * moments[0].tanh()).clamp(0.1, 2.0);
        let l1 = (0.2 * moments[1].tanh()).clamp(-1.5, 1.5);
        let omega1 = k1.powi(3) + 3.0 * (l1.powi(2) / k1);

        let k2 = (1.0 + 0.3 * moments[2].tanh()).clamp(0.2, 3.0);
        let l2 = (0.3 * moments[3].tanh()).clamp(-2.0, 2.0);
        let omega2 = k2.powi(3) + 3.0 * (l2.powi(2) / k2);

        // Factor de acoplamiento de interacción de fase de Hirota A_12
        let k_diff = k1 - k2;
        let l_diff = l1 - l2;
        let k_sum = k1 + k2;
        let l_sum = l1 + l2;

        let num = k_diff.powi(2) - (l_diff.powi(2) / (k_diff.powi(2) + 0.01));
        let den = k_sum.powi(2) - (l_sum.powi(2) / (k_sum.powi(2) + 0.01));
        let a12 = if den.abs() > 0.001 {
            (num / den).clamp(0.01, 10.0)
        } else {
            1.0
        };

        // 3. Evaluación de tau(x, y, t) y cálculo de derivadas espaciales
        let h = 0.05;
        let tau_center = self.eval_hirota_tau(x, y, t, k1, l1, omega1, k2, l2, omega2, a12);
        let tau_x_plus = self.eval_hirota_tau(x + h, y, t, k1, l1, omega1, k2, l2, omega2, a12);
        let tau_x_minus = self.eval_hirota_tau(x - h, y, t, k1, l1, omega1, k2, l2, omega2, a12);

        let ln_tau_0 = tau_center.max(1e-12).ln();
        let ln_tau_plus = tau_x_plus.max(1e-12).ln();
        let ln_tau_minus = tau_x_minus.max(1e-12).ln();

        // u(x, y, t) = 2 d^2_x ln(tau)
        let u_val = 2.0 * ((ln_tau_plus - 2.0 * ln_tau_0 + ln_tau_minus) / (h * h));
        let safe_u = if u_val.is_finite() { u_val } else { 0.0 };

        // Derivada primera u_x
        let u_plus = {
            let t_p2 = self.eval_hirota_tau(x + 2.0 * h, y, t, k1, l1, omega1, k2, l2, omega2, a12);
            let ln_p2 = t_p2.max(1e-12).ln();
            2.0 * ((ln_p2 - 2.0 * ln_tau_plus + ln_tau_0) / (h * h))
        };
        let u_minus = {
            let t_m2 = self.eval_hirota_tau(x - 2.0 * h, y, t, k1, l1, omega1, k2, l2, omega2, a12);
            let ln_m2 = t_m2.max(1e-12).ln();
            2.0 * ((ln_tau_0 - 2.0 * ln_tau_minus + ln_m2) / (h * h))
        };

        let ux = (u_plus - u_minus) / (2.0 * h);
        let safe_ux = if ux.is_finite() { ux } else { 0.0 };

        // Derivada segunda u_xx (curvatura de choque)
        let uxx = (u_plus - 2.0 * safe_u + u_minus) / (h * h);
        let safe_uxx = if uxx.is_finite() { uxx } else { 0.0 };

        // Derivadas en y (dispersión transversal)
        let hy = 0.05;
        let tau_y_plus = self.eval_hirota_tau(x, y + hy, t, k1, l1, omega1, k2, l2, omega2, a12);
        let tau_y_minus = self.eval_hirota_tau(x, y - hy, t, k1, l1, omega1, k2, l2, omega2, a12);
        let ln_y_plus = tau_y_plus.max(1e-12).ln();
        let ln_y_minus = tau_y_minus.max(1e-12).ln();
        let u_y_plus = 2.0
            * ((self
                .eval_hirota_tau(x + h, y + hy, t, k1, l1, omega1, k2, l2, omega2, a12)
                .max(1e-12)
                .ln()
                - 2.0 * ln_y_plus
                + self
                    .eval_hirota_tau(x - h, y + hy, t, k1, l1, omega1, k2, l2, omega2, a12)
                    .max(1e-12)
                    .ln())
                / (h * h));
        let u_y_minus = 2.0
            * ((self
                .eval_hirota_tau(x + h, y - hy, t, k1, l1, omega1, k2, l2, omega2, a12)
                .max(1e-12)
                .ln()
                - 2.0 * ln_y_minus
                + self
                    .eval_hirota_tau(x - h, y - hy, t, k1, l1, omega1, k2, l2, omega2, a12)
                    .max(1e-12)
                    .ln())
                / (h * h));
        let uyy = (u_y_plus - 2.0 * safe_u + u_y_minus) / (hy * hy);
        let safe_uyy = if uyy.is_finite() { uyy } else { 0.0 };

        // Derivada temporal u_t
        let ut = if self.initialized {
            (safe_u - self.prev_u) / dt
        } else {
            0.0
        };
        let safe_ut = if ut.is_finite() { ut } else { 0.0 };

        self.prev_u = safe_u;
        self.prev_ux = safe_ux;
        self.initialized = true;

        // Residual de la ecuación KP: (-4 u_t + 6 u u_x + u_xxx)_x + 3 u_yy
        // Para evitar calcular derivadas numéricas de orden 4 propensas a ruido discreto,
        // aproximamos el residual evaluando el balance dinámico del solitón:
        let balance_bracket = -4.0 * safe_ut + 6.0 * safe_u * safe_ux + safe_uxx;
        let residual =
            (balance_bracket.abs() + 3.0 * safe_uyy.abs() * self.config.dispersion_alpha).abs();
        let safe_residual = if residual.is_finite() { residual } else { 0.0 };

        // 4. Detección de choque solitónico KP
        let shock_detected = safe_uxx.abs() >= self.config.critical_shock_curvature
            || safe_residual >= self.config.critical_residual_threshold;

        // 5. Factor de modulación de slot continuo [0.20, 1.35]
        let slot_scale = if shock_detected {
            0.20
        } else {
            let damping = 1.0 / (1.0 + self.config.slot_sensitivity * safe_uxx.abs());
            (1.0 * damping + 0.15 * safe_u.tanh()).clamp(0.20, 1.35)
        };

        MatrixModelKpReport {
            moments_tk: moments,
            tau_function_val: tau_center,
            soliton_potential_u: safe_u,
            kp_curvature_uxx: safe_uxx,
            dispersion_residual: safe_residual,
            kp_soliton_shock_detected: shock_detected,
            slot_scale_factor: slot_scale,
        }
    }

    /// Evalúa la función tau multisolitón de Hirota con regularización de desbordamiento.
    fn eval_hirota_tau(
        &self,
        x: f64,
        y: f64,
        t: f64,
        k1: f64,
        l1: f64,
        omega1: f64,
        k2: f64,
        l2: f64,
        omega2: f64,
        a12: f64,
    ) -> f64 {
        let theta1 = (k1 * x + l1 * y - omega1 * t).clamp(-20.0, 20.0);
        let theta2 = (k2 * x + l2 * y - omega2 * t).clamp(-20.0, 20.0);

        let exp1 = theta1.exp();
        let exp2 = theta2.exp();
        let exp12 = (theta1 + theta2).clamp(-30.0, 30.0).exp();

        (1.0 + exp1 + exp2 + a12 * exp12) * self.config.tau_scale
    }

    /// Computa los momentos de Miwa t_k = (1/k) Tr(M^k) de forma $O(1)$ sin asignaciones en heap.
    fn compute_miwa_moments(
        &self,
        m: &[[f64; KP_MATRIX_DIM]; KP_MATRIX_DIM],
    ) -> [f64; KP_MOMENTS_COUNT] {
        // M^1
        let tr_m1 = m[0][0] + m[1][1] + m[2][2] + m[3][3];
        let t1 = tr_m1;

        // M^2
        let mut m2 = [[0.0; KP_MATRIX_DIM]; KP_MATRIX_DIM];
        for i in 0..KP_MATRIX_DIM {
            for j in 0..KP_MATRIX_DIM {
                let mut sum = 0.0;
                for k in 0..KP_MATRIX_DIM {
                    sum += m[i][k] * m[k][j];
                }
                m2[i][j] = sum;
            }
        }
        let tr_m2 = m2[0][0] + m2[1][1] + m2[2][2] + m2[3][3];
        let t2 = 0.5 * tr_m2;

        // M^3 = M^2 * M
        let mut m3 = [[0.0; KP_MATRIX_DIM]; KP_MATRIX_DIM];
        for i in 0..KP_MATRIX_DIM {
            for j in 0..KP_MATRIX_DIM {
                let mut sum = 0.0;
                for k in 0..KP_MATRIX_DIM {
                    sum += m2[i][k] * m[k][j];
                }
                m3[i][j] = sum;
            }
        }
        let tr_m3 = m3[0][0] + m3[1][1] + m3[2][2] + m3[3][3];
        let t3 = (1.0 / 3.0) * tr_m3;

        // M^4 = M^2 * M^2
        let mut m4 = [[0.0; KP_MATRIX_DIM]; KP_MATRIX_DIM];
        for i in 0..KP_MATRIX_DIM {
            for j in 0..KP_MATRIX_DIM {
                let mut sum = 0.0;
                for k in 0..KP_MATRIX_DIM {
                    sum += m2[i][k] * m2[k][j];
                }
                m4[i][j] = sum;
            }
        }
        let tr_m4 = m4[0][0] + m4[1][1] + m4[2][2] + m4[3][3];
        let t4 = 0.25 * tr_m4;

        [
            if t1.is_finite() { t1 } else { 0.0 },
            if t2.is_finite() { t2 } else { 0.0 },
            if t3.is_finite() { t3 } else { 0.0 },
            if t4.is_finite() { t4 } else { 0.0 },
        ]
    }
}
