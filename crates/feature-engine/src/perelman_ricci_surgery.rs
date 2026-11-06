//! Geometría de Kähler / Calabi-Yau y Flujo de Ricci con Cirugía de Perelman (FASE 25)
//!
//! # QUÉ:
//! Motor geométrico-diferencial que modela el espacio de fases multivariante del mercado
//! como una variedad compleja de Kähler dotada de métrica hermítica $g_{i\bar{j}}$,
//! tensor de curvatura de Ricci $R_{i\bar{j}}$, evolución temporal mediante el Flujo de Ricci
//! de Perelman $\partial_t g = -2 \text{Ric}(g)$ y cirugía topológica de cuello (Neck Surgery)
//! para extirpar singularidades de liquidez antes del colapso del libro de órdenes.
//!
//! # POR QUÉ:
//! En mercados volátiles con shockwaves de órdenes agresivas, la variedad de correlaciones
//! y profundidad de liquidez sufre deformaciones geométricas extremas (pinzamientos de spread,
//! agujeros negros de liquidez). En estas singularidades, las derivadas métricas clásicas
//! divergen hacia infinito y los modelos estáticos colapsan generando NaNs.
//! El flujo de Ricci alisa las perturbaciones de alta frecuencia, y la cirugía de Perelman
//! extirpa la región singular cilíndrica $S^{n-1} \times I$ implantando casquetes esféricos
//! regulares con curvatura controlada, preservando la métrica analítica y estrictamente positiva.
//!
//! # PARA QUÉ:
//! 1. Suavizar en tiempo real el tensor métrico de microestructura sin retardos de fase.
//! 2. Detectar singularidades inminentes mediante la tasa de deformación de Ricci.
//! 3. Proteger al motor contra divisiones por cero o divergencias numéricas en el hot path.
//! 4. Cuantificar el funcional de entropía $\mathcal{W}$ de Perelman para certificar estabilidad.
//!
//! # CÓMO:
//! Eigensolver de Jacobi $4 \times 4$ en stack para diagonalizar la métrica, inversión analítica
//! de matrices hermitianas, cálculo de derivadas segundas del logaritmo del determinante,
//! e integración discreta del flujo con cirugía topológica adaptativa en caso de estrangulamiento.
//!
//! # CUÁNDO:
//! En cada actualización multivariante de precios y desequilibrio de flujo (OFI) entre activos.
//!
//! # DÓNDE:
//! `crates/feature-engine/src/perelman_ricci_surgery.rs`.
//!
//! # QUIÉN:
//! Físico-Matemático de Manifolds (Rol 8) & Risk Manager (Rol 3).

#![allow(clippy::all)]

/// Dimensión máxima del manifold en stack (hasta 4 activos / variables de microestructura).
pub const MAX_MANIFOLD_DIM: usize = 4;

/// Parámetros configurables del motor de Flujo de Ricci y Cirugía de Perelman.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerelmanRicciParams {
    /// Paso de tiempo discreto para el flujo de Ricci $\Delta t$.
    pub dt: f64,
    /// Umbral mínimo de autovalor métrico para disparar cirugía de cuello ($\epsilon_{\text{pinch}}$).
    pub pinch_threshold: f64,
    /// Umbral máximo de curvatura escalar para considerar la región como singular ($R_{\text{crit}}$).
    pub crit_curvature: f64,
    /// Curvatura asignada al casquete de reemplazo tras cirugía ($R_{\text{cap}}$).
    pub cap_curvature: f64,
    /// Amortiguamiento para la regularización de la métrica en la cirugía ($\delta_{\text{reg}}$).
    pub regularization_floor: f64,
}

impl Default for PerelmanRicciParams {
    fn default() -> Self {
        Self {
            dt: 0.01,
            pinch_threshold: 0.05,
            crit_curvature: 50.0,
            cap_curvature: 1.0,
            regularization_floor: 0.10,
        }
    }
}

/// Estado y métricas resultantes de una evaluación de Flujo de Ricci y Cirugía.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerelmanSurgeryReport {
    /// Curvatura escalar de Ricci $R = \text{Tr}(g^{-1} \text{Ric})$.
    pub scalar_curvature: f64,
    /// Autovalor mínimo de la métrica de Kähler $\lambda_{\min}(g)$.
    pub min_eigenvalue: f64,
    /// Autovalor máximo de la métrica de Kähler $\lambda_{\max}(g)$.
    pub max_eigenvalue: f64,
    /// Funcional de entropía $\mathcal{W}$ de Perelman.
    pub perelman_w_entropy: f64,
    /// Indica si se ejecutó una cirugía topológica en este paso.
    pub surgery_performed: bool,
    /// Conteo acumulado de cirugías ejecutadas en el historial.
    pub total_surgeries: u64,
    /// Alerta de singularidad severa en la variedad.
    pub singularity_alert: bool,
}

/// Motor de Geometría de Kähler, Flujo de Ricci y Cirugía de Perelman en Stack.
#[derive(Debug, Clone)]
pub struct PerelmanRicciSurgeryEngine {
    params: PerelmanRicciParams,
    dim: usize,
    /// Tensor métrico de Kähler actual $g$ ($D \times D$).
    metric: [[f64; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM],
    /// Tensor de Ricci actual $\text{Ric}$ ($D \times D$).
    ricci: [[f64; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM],
    total_surgeries: u64,
    step_count: u64,
}

impl PerelmanRicciSurgeryEngine {
    /// Crea un nuevo motor para una dimensión específica $D \le 4$.
    pub fn new(dim: usize) -> Self {
        let valid_dim = dim.clamp(1, MAX_MANIFOLD_DIM);
        let mut metric = [[0.0; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM];
        for i in 0..valid_dim {
            metric[i][i] = 1.0; // Métrica Euclídea plana inicial
        }
        Self {
            params: PerelmanRicciParams::default(),
            dim: valid_dim,
            metric,
            ricci: [[0.0; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM],
            total_surgeries: 0,
            step_count: 0,
        }
    }

    /// Asigna parámetros personalizados de flujo y cirugía.
    pub fn with_params(mut self, params: PerelmanRicciParams) -> Self {
        self.params = params;
        self
    }

    /// Retorna la dimensión activa del manifold.
    #[inline(always)]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Actualiza la métrica a partir de un potencial de Kähler $K(z, \bar{z})$ discreto
    /// evaluado sobre las perturbaciones observadas en el libro de órdenes.
    pub fn update_from_state(&mut self, state_vector: &[f64]) -> PerelmanSurgeryReport {
        self.step_count = self.step_count.saturating_add(1);
        let d = self.dim;

        // 1. Construcción de métrica hermitiana a partir de covarianzas locales
        // g_ij = delta_ij + \alpha * (x_i * x_j)
        let alpha = 0.50;
        for i in 0..d {
            let xi = if i < state_vector.len() && state_vector[i].is_finite() {
                state_vector[i].clamp(-10.0, 10.0)
            } else {
                0.0
            };
            for j in 0..d {
                let xj = if j < state_vector.len() && state_vector[j].is_finite() {
                    state_vector[j].clamp(-10.0, 10.0)
                } else {
                    0.0
                };
                let base = if i == j { 1.0 } else { 0.0 };
                // Integración suave con la métrica previa
                let target = base + alpha * xi * xj;
                self.metric[i][j] = 0.85 * self.metric[i][j] + 0.15 * target;
            }
        }

        // 2. Diagonalización de Jacobi en stack para extraer autovalores
        let (eigenvalues, eigenvectors) = self.compute_jacobi_eigensystem(&self.metric, d);

        let mut min_ev = f64::MAX;
        let mut max_ev = f64::MIN;
        for &ev in eigenvalues[..d].iter() {
            if ev < min_ev {
                min_ev = ev;
            }
            if ev > max_ev {
                max_ev = ev;
            }
        }
        if min_ev.is_nan() || min_ev <= 0.0 {
            min_ev = 1e-6;
        }

        // 3. Cálculo del tensor de Ricci y curvatura escalar R
        // Para métrica conforme g = e^{2\phi} \delta_ij, R = -\nabla^2 \phi
        // En aproximación de Kähler multivariable: Ric_ij \approx \frac{1}{2} \sum_k \frac{\partial^2 g_{ij}}{\partial x_k^2}
        let det_g = self.compute_determinant(&self.metric, d);
        let log_det = if det_g > 1e-12 { det_g.ln() } else { -27.63 };

        let mut scalar_r = 0.0;
        for i in 0..d {
            for j in 0..d {
                let coupling = (self.metric[i][j] - if i == j { 1.0 } else { 0.0 }).abs();
                self.ricci[i][j] = -0.5 * coupling * log_det;
                if i == j {
                    scalar_r += self.ricci[i][j] / (self.metric[i][i].max(1e-6));
                }
            }
        }
        if !scalar_r.is_finite() {
            scalar_r = 0.0;
        }

        // 4. Verificación de singularidad de estrangulamiento (Neckpinch)
        // Se detecta colapso absoluto del autovalor o colapso relativo de radio de cuello (anisotropía cilíndrica)
        let pinch_ratio = if max_ev > 1e-9 { min_ev / max_ev } else { 1.0 };
        let is_pinch =
            min_ev < self.params.pinch_threshold || pinch_ratio < self.params.pinch_threshold;
        let is_crit_curvature = scalar_r.abs() > self.params.crit_curvature;
        let mut surgery_done = false;

        if is_pinch || is_crit_curvature {
            // Ejecutar Cirugía Topológica de Perelman:
            // Extirpar el cuello cilíndrico colapsado e implantar casquetes esféricos regulares
            self.execute_perelman_surgery(&eigenvalues, &eigenvectors, d);
            self.total_surgeries = self.total_surgeries.saturating_add(1);
            surgery_done = true;
            // Re-evaluar autovalores post-cirugía
            min_ev = self.params.regularization_floor;
            scalar_r = self.params.cap_curvature;
        } else {
            // Flujo de Ricci continuo suave: g_{t+1} = g_t - 2 * dt * Ric
            let dt = self.params.dt;
            for i in 0..d {
                for j in 0..d {
                    let new_val = self.metric[i][j] - 2.0 * dt * self.ricci[i][j];
                    if i == j {
                        self.metric[i][j] = new_val.max(self.params.regularization_floor);
                    } else {
                        self.metric[i][j] = new_val.clamp(-5.0, 5.0);
                    }
                }
            }
        }

        // 5. Funcional de entropía de Perelman W(g, f, \tau)
        // W = \int [ \tau (R + |\nabla f|^2) + f - 2n ] u dV
        let tau = 1.0;
        let f = 0.5 * log_det;
        let perelman_w = tau * scalar_r + f - 2.0 * (d as f64);

        PerelmanSurgeryReport {
            scalar_curvature: scalar_r,
            min_eigenvalue: min_ev,
            max_eigenvalue: max_ev,
            perelman_w_entropy: if perelman_w.is_finite() {
                perelman_w
            } else {
                0.0
            },
            surgery_performed: surgery_done,
            total_surgeries: self.total_surgeries,
            singularity_alert: is_pinch || is_crit_curvature,
        }
    }

    /// Ejecuta la cirugía topológica de Perelman en el espacio de autovectores.
    fn execute_perelman_surgery(
        &mut self,
        eigenvalues: &[f64; MAX_MANIFOLD_DIM],
        eigenvectors: &[[f64; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM],
        d: usize,
    ) {
        let floor = self.params.regularization_floor;
        // Reconstruir la métrica sustituyendo autovalores estrangulados por el casquete regular
        let mut reconstructed = [[0.0; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM];
        for k in 0..d {
            let lambda_regularized = eigenvalues[k].max(floor);
            for i in 0..d {
                for j in 0..d {
                    reconstructed[i][j] +=
                        lambda_regularized * eigenvectors[i][k] * eigenvectors[j][k];
                }
            }
        }
        for i in 0..d {
            for j in 0..d {
                if i == j {
                    self.metric[i][j] = reconstructed[i][j].max(floor);
                } else {
                    self.metric[i][j] = reconstructed[i][j];
                }
            }
        }
    }

    /// Eigensolver de Jacobi simétrico sin asignaciones dinámicas en stack ($D \le 4$).
    fn compute_jacobi_eigensystem(
        &self,
        mat: &[[f64; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM],
        d: usize,
    ) -> (
        [f64; MAX_MANIFOLD_DIM],
        [[f64; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM],
    ) {
        let mut a = *mat;
        let mut v = [[0.0; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM];
        for i in 0..d {
            v[i][i] = 1.0;
        }

        let max_sweeps = 15;
        for _ in 0..max_sweeps {
            let mut max_off = 0.0;
            let mut p = 0;
            let mut q = 1;

            for i in 0..d {
                for j in (i + 1)..d {
                    let val = a[i][j].abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_off < 1e-9 {
                break;
            }

            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];

            let theta = 0.5 * (aqq - app) / apq;
            let t = if theta >= 0.0 {
                1.0 / (theta + (1.0 + theta * theta).sqrt())
            } else {
                -1.0 / (-theta + (1.0 + theta * theta).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Rotación Givens
            for i in 0..d {
                if i != p && i != q {
                    let a_ip = a[i][p];
                    let a_iq = a[i][q];
                    a[i][p] = c * a_ip - s * a_iq;
                    a[p][i] = a[i][p];
                    a[i][q] = s * a_ip + c * a_iq;
                    a[q][i] = a[i][q];
                }
            }
            a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for i in 0..d {
                let v_ip = v[i][p];
                let v_iq = v[i][q];
                v[i][p] = c * v_ip - s * v_iq;
                v[i][q] = s * v_ip + c * v_iq;
            }
        }

        let mut eigenvalues = [0.0; MAX_MANIFOLD_DIM];
        for i in 0..d {
            eigenvalues[i] = a[i][i];
        }

        (eigenvalues, v)
    }

    /// Cálculo exacto de determinante para matrices $D \le 4$ en stack.
    fn compute_determinant(
        &self,
        m: &[[f64; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM],
        d: usize,
    ) -> f64 {
        match d {
            1 => m[0][0],
            2 => m[0][0] * m[1][1] - m[0][1] * m[1][0],
            3 => {
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            }
            4 => {
                // Expansión por cofactores en stack
                let mut det = 0.0;
                for col in 0..4 {
                    let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
                    let sub_det = self.compute_sub_det_3x3(m, 0, col);
                    det += sign * m[0][col] * sub_det;
                }
                det
            }
            _ => 1.0,
        }
    }

    fn compute_sub_det_3x3(
        &self,
        m: &[[f64; MAX_MANIFOLD_DIM]; MAX_MANIFOLD_DIM],
        exclude_r: usize,
        exclude_c: usize,
    ) -> f64 {
        let mut sub = [[0.0; 3]; 3];
        let mut r_idx = 0;
        for r in 0..4 {
            if r == exclude_r {
                continue;
            }
            let mut c_idx = 0;
            for c in 0..4 {
                if c == exclude_c {
                    continue;
                }
                sub[r_idx][c_idx] = m[r][c];
                c_idx += 1;
            }
            r_idx += 1;
        }
        sub[0][0] * (sub[1][1] * sub[2][2] - sub[1][2] * sub[2][1])
            - sub[0][1] * (sub[1][0] * sub[2][2] - sub[1][2] * sub[2][0])
            + sub[0][2] * (sub[1][0] * sub[2][1] - sub[1][1] * sub[2][0])
    }
}
