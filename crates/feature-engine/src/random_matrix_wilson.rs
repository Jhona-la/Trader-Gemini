//! # Fase 9: Matrices Aleatorias & Wilson Renormalization Group (RG)
//!
//! ## Paradigma Matemático & Físico
//! En el universo multivariante continuo temporal-espectral $\mathcal{H} = L^2(\mathbb{R}^{d \times \tau}, \mu)$,
//! la estimación empírica de correlaciones entre activos e interacciones de microestructura sufre de:
//! 1. **Contaminación por Ruido Wishart:** Para $N$ activos observados en $T$ ventanas temporales,
//!    la matriz empírica $C = \frac{1}{T} X X^T$ tiene un espectro ensanchado por ruido puro de acuerdo
//!    a la distribución de Marchenko-Pastur:
//!    $$\rho(\lambda) = \frac{T}{2\pi \sigma^2 N \lambda} \sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}$$
//!    con bordes analíticos $\lambda_\pm = \sigma^2 (1 \pm \sqrt{N/T})^2$.
//!    Los autovalores dentro de $[\lambda_-, \lambda_+]$ son ruido puro no interpretable.
//!    El filtrado de Marchenko-Pastur aísla los autovalores $\lambda > \lambda_+$ (modo de mercado y factores
//!    estructurales) y contrae el bulk de ruido preservando estrictamente la traza: $\text{Tr}(C_{\text{clean}}) = N$.
//!
//! 2. **Grupo de Renormalización de Wilson (Coarse-Graining Momentum-Shell):**
//!    Las micro-fluctuaciones de libro y flujos de órdenes se integran recursivamente en $K=8$ escalas
//!    diádicas $\tau_k = 2^k \tau_0$.
//!    La función beta de renormalización $\beta_i = \frac{d g_i}{d \ln b}$ clasifica los operadores en:
//!    - **Relevantes ($\beta > 0$):** Flujos persistentes macro/meso institucionales (alpha duradero).
//!    - **Marginales ($\beta \approx 0$):** Invarianza de escala crítica (multifractalidad).
//!    - **Irrelevantes ($\beta < 0$):** Ruido de microestructura, rebote bid-ask o spoofing transitorio (veto).
//!
//! 3. **Estimador Continuo del Máximo Exponente de Lyapunov ($\lambda_{\max}$):**
//!    Rastrea la tasa orbital de divergencia exponencial en el espacio de fases de Hilbert:
//!    $\|\delta Z(t)\| \sim \|\delta Z(0)\| e^{\lambda_{\max} t}$.
//!    Cuando $\lambda_{\max} > 0$, la dinámica entra en régimen caótico determinista, modulando
//!    automáticamente el factor de compresión de horizonte $\gamma_{\text{horizon}} = (1 + \max(0, \lambda_{\max}))^{-1}$
//!    y ensanchando los intervalos de incertidumbre conformal para prevenir sobreconfianza en colas pesadas.
//!
//! Cero asignaciones en heap en la ruta crítica, matrices de tamaño acotado en stack,
//! resolución numérica O(N^3) vía rotaciones cíclicas de Jacobi para matrices simétricas,
//! y cero pánicos garantizados por diseño formal.

/// Número máximo de activos concurrentes en el universo multivariante denso.
pub const MAX_ASSETS: usize = 16;
/// Número máximo de escalas diádicas en el Grupo de Renormalización de Wilson ($2^0$ a $2^7$).
pub const MAX_DYADIC_SCALES: usize = 8;
/// Dimensión de inmersión en espacio de fases para el cálculo de Lyapunov ($m=4$).
pub const LYAPUNOV_EMBED_DIM: usize = 4;
/// Capacidad del buffer circular de trayectorias en espacio de fases.
pub const LYAPUNOV_BUFFER_SIZE: usize = 64;
/// Ventana temporal circular de retornos por activo.
pub const RETURNS_HISTORY_SIZE: usize = 256;

/// Límites analíticos del espectro de ruido de Marchenko-Pastur.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarchenkoPasturBounds {
    /// Límite inferior del bulk de ruido $\lambda_-$.
    pub lambda_minus: f64,
    /// Límite superior del bulk de ruido $\lambda_+$.
    pub lambda_plus: f64,
    /// Varianza del ruido estimada $\sigma^2$.
    pub noise_variance: f64,
    /// Ratio espectral $q = N / T$.
    pub spectral_ratio_q: f64,
}

impl Default for MarchenkoPasturBounds {
    fn default() -> Self {
        Self {
            lambda_minus: 0.0,
            lambda_plus: 4.0,
            noise_variance: 1.0,
            spectral_ratio_q: 1.0,
        }
    }
}

/// Resultado de la descomposición espectral y denoising de Marchenko-Pastur.
#[derive(Debug, Clone, Copy)]
pub struct DenoisedCorrelationResult {
    /// Matriz de correlación filtrada $N \times N$ limpia de ruido.
    pub cleaned_matrix: [[f64; MAX_ASSETS]; MAX_ASSETS],
    /// Autovalores originales ordenados descendentemente.
    pub raw_eigenvalues: [f64; MAX_ASSETS],
    /// Autovalores filtrados con conservación de traza.
    pub cleaned_eigenvalues: [f64; MAX_ASSETS],
    /// Cantidad de autovalores identificados como señal estructural ($\lambda > \lambda_+$).
    pub signal_modes_count: usize,
    /// Relación Señal-Ruido (SNR = energía en modos señal / traza total).
    pub signal_to_noise_ratio: f64,
    /// Bordes analíticos Marchenko-Pastur calculados.
    pub bounds: MarchenkoPasturBounds,
}

/// Clasificación de operadores en el flujo del Grupo de Renormalización de Wilson.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WilsonOperatorClass {
    /// Operador relevante: $\beta > 0.05$. El acoplamiento crece a escalas mayores (tendencia macro persistente).
    Relevant,
    /// Operador marginal: $|\beta| \le 0.05$. Invarianza de escala o criticidad multifractal.
    Marginal,
    /// Operador irrelevante: $\beta < -0.05$. El acoplamiento decae a escalas macro (ruido microestructural efímero).
    Irrelevant,
}

/// Estado de renormalización de Wilson en múltiples escalas diádicas para un activo.
#[derive(Debug, Clone, Copy)]
pub struct WilsonRGScaleReport {
    /// Acoplamientos de autocorrelación $g_1^{(k)}$ en cada escala $k \in [0, 7]$.
    pub g1_couplings: [f64; MAX_DYADIC_SCALES],
    /// Acoplamientos no lineales (exceso de curtosis) $g_2^{(k)}$ en cada escala.
    pub g2_couplings: [f64; MAX_DYADIC_SCALES],
    /// Funciones beta $\beta_1(k) = \frac{d g_1}{d \ln b}$ entre escalas adyacentes.
    pub beta1_flows: [f64; MAX_DYADIC_SCALES - 1],
    /// Beta efectiva promedio en escalas macro intermedias.
    pub effective_beta1: f64,
    /// Clasificación física del operador dominante.
    pub operator_class: WilsonOperatorClass,
}

impl Default for WilsonRGScaleReport {
    fn default() -> Self {
        Self {
            g1_couplings: [0.0; MAX_DYADIC_SCALES],
            g2_couplings: [0.0; MAX_DYADIC_SCALES],
            beta1_flows: [0.0; MAX_DYADIC_SCALES - 1],
            effective_beta1: 0.0,
            operator_class: WilsonOperatorClass::Marginal,
        }
    }
}

/// Diagnóstico de estabilidad no lineal y caos mediante exponente de Lyapunov.
#[derive(Debug, Clone, Copy)]
pub struct LyapunovDiagnostic {
    /// Máximo exponente de Lyapunov estimado $\lambda_{\max}$.
    pub lambda_max: f64,
    /// Factor de compresión dinámica de horizonte: $\gamma_{\text{horizon}} \in (0, 1]$.
    pub horizon_compression: f64,
    /// Indicador booleano de régimen caótico determinista ($\lambda_{\max} > 0.02$).
    pub is_chaotic: bool,
    /// Distancia orbital promedio en espacio de fases.
    pub phase_space_diameter: f64,
}

impl Default for LyapunovDiagnostic {
    fn default() -> Self {
        Self {
            lambda_max: 0.0,
            horizon_compression: 1.0,
            is_chaotic: false,
            phase_space_diameter: 0.0,
        }
    }
}

// ============================================================================
// EIGENSOLVER SIMÉTRICO DE JACOBI (CERO ASIGNACIONES EN HEAP, O(N^3))
// ============================================================================

/// Resuelve el problema de autovalores simétrico $A v = \lambda v$ mediante rotaciones cíclicas de Jacobi.
/// Garantiza convergencia incondicional, conservación estricta de ortogonalidad y cero pánicos.
pub fn jacobi_eigensolver_symmetric(
    matrix: &[[f64; MAX_ASSETS]; MAX_ASSETS],
    n: usize,
    eigenvalues: &mut [f64; MAX_ASSETS],
    eigenvectors: &mut [[f64; MAX_ASSETS]; MAX_ASSETS],
) -> bool {
    let dim = n.min(MAX_ASSETS);
    if dim == 0 {
        return false;
    }

    // Copia de trabajo de la matriz en stack
    let mut a = *matrix;

    // Inicializar matriz de autovectores como la identidad
    for i in 0..MAX_ASSETS {
        for j in 0..MAX_ASSETS {
            eigenvectors[i][j] = if i == j && i < dim { 1.0 } else { 0.0 };
        }
    }

    // Inicializar autovalores con la diagonal
    for i in 0..dim {
        eigenvalues[i] = a[i][i];
    }

    const MAX_SWEEPS: usize = 30;
    const EPSILON: f64 = 1e-12;

    for _sweep in 0..MAX_SWEEPS {
        // Calcular norma fuera de la diagonal
        let mut off_diag_sum = 0.0;
        for p in 0..dim {
            for q in (p + 1)..dim {
                off_diag_sum += a[p][q].abs();
            }
        }

        if off_diag_sum < EPSILON {
            break;
        }

        for p in 0..dim {
            for q in (p + 1)..dim {
                let apq = a[p][q];
                if apq.abs() < 1e-15 {
                    continue;
                }

                let app = a[p][p];
                let aqq = a[q][q];
                let diff = aqq - app;

                // Cálculo estable de t = tan(theta)
                let t = if diff.abs() < 1e-15 {
                    1.0 // theta = pi / 4
                } else {
                    let tau = diff / (2.0 * apq);
                    if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    }
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau_rot = s / (1.0 + c);

                // Actualizar elementos diagonales
                let delta_p = t * apq;
                a[p][p] -= delta_p;
                a[q][q] += delta_p;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                // Actualizar resto de elementos fuera de diagonal
                for r in 0..dim {
                    if r != p && r != q {
                        let arp = a[r][p];
                        let arq = a[r][q];
                        a[r][p] = arp - s * (arq + arp * tau_rot);
                        a[p][r] = a[r][p];
                        a[r][q] = arq + s * (arp - arq * tau_rot);
                        a[q][r] = a[r][q];
                    }
                }

                // Acumular rotación en la base ortonormal de autovectores
                for r in 0..dim {
                    let vrp = eigenvectors[r][p];
                    let vrq = eigenvectors[r][q];
                    eigenvectors[r][p] = vrp - s * (vrq + vrp * tau_rot);
                    eigenvectors[r][q] = vrq + s * (vrp - vrq * tau_rot);
                }
            }
        }
    }

    // Extraer autovalores diagonales finales
    for i in 0..dim {
        eigenvalues[i] = if a[i][i].is_finite() { a[i][i] } else { 0.0 };
    }

    // Ordenar autovalores y columnas de autovectores en orden descendente
    for i in 0..dim {
        let mut max_idx = i;
        for j in (i + 1)..dim {
            if eigenvalues[j] > eigenvalues[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            eigenvalues.swap(i, max_idx);
            for r in 0..dim {
                let temp = eigenvectors[r][i];
                eigenvectors[r][i] = eigenvectors[r][max_idx];
                eigenvectors[r][max_idx] = temp;
            }
        }
    }

    true
}

// ============================================================================
// MARCHENKO-PASTUR SPECTRAL DENOISING
// ============================================================================

/// Filtro espectral de Marchenko-Pastur para matrices de covarianza y correlación.
#[derive(Debug, Clone, Copy)]
pub struct MarchenkoPasturFilter;

impl MarchenkoPasturFilter {
    /// Calcula los bordes teóricos $[\lambda_-, \lambda_+]$ de la distribución de Marchenko-Pastur.
    #[inline]
    pub fn compute_bounds(n_assets: usize, t_samples: usize, noise_variance: f64) -> MarchenkoPasturBounds {
        let n = (n_assets.clamp(1, MAX_ASSETS)) as f64;
        let t = (t_samples.max(2)) as f64;
        let q = (n / t).clamp(0.001, 1.0);
        let sigma_sq = if noise_variance.is_finite() && noise_variance > 1e-8 {
            noise_variance
        } else {
            1.0
        };

        let sqrt_q = q.sqrt();
        let lambda_minus = sigma_sq * (1.0 - sqrt_q).powi(2);
        let lambda_plus = sigma_sq * (1.0 + sqrt_q).powi(2);

        MarchenkoPasturBounds {
            lambda_minus,
            lambda_plus,
            noise_variance: sigma_sq,
            spectral_ratio_q: q,
        }
    }

    /// Ejecuta el proceso de Denoising con preservación estricta de traza $\text{Tr}(C) = N$.
    pub fn denoise_correlation(
        corr_matrix: &[[f64; MAX_ASSETS]; MAX_ASSETS],
        n_assets: usize,
        t_samples: usize,
    ) -> DenoisedCorrelationResult {
        let dim = n_assets.clamp(1, MAX_ASSETS);
        let mut raw_eigs = [0.0; MAX_ASSETS];
        let mut evecs = [[0.0; MAX_ASSETS]; MAX_ASSETS];

        // 1. Descomposición espectral de la matriz empírica
        jacobi_eigensolver_symmetric(corr_matrix, dim, &mut raw_eigs, &mut evecs);

        // 2. Estimación de varianza del ruido a partir del bulk inferior
        // En una matriz de correlación normalizada, Tr(C) = dim.
        // Si no hay autovalores gigantes, sigma^2 ≈ 1 - lambda_max / dim.
        let lambda_max = raw_eigs[0].max(0.0);
        let estimated_noise_var = if dim > 1 && lambda_max < (dim as f64) {
            (1.0 - lambda_max / (dim as f64)).clamp(0.10, 1.0)
        } else {
            1.0
        };

        let bounds = Self::compute_bounds(dim, t_samples, estimated_noise_var);

        // 3. Identificar modos de señal vs modos de ruido
        let mut signal_count = 0usize;
        let mut noise_trace_sum = 0.0;
        let mut total_trace_sum = 0.0;
        let mut signal_trace_sum = 0.0;

        for i in 0..dim {
            let lambda = raw_eigs[i].max(0.0);
            total_trace_sum += lambda;
            if lambda > bounds.lambda_plus {
                signal_count += 1;
                signal_trace_sum += lambda;
            } else {
                noise_trace_sum += lambda;
            }
        }

        // 4. Shrinkage del bulk de ruido preservando la masa de traza
        let noise_count = dim.saturating_sub(signal_count);
        let avg_noise_lambda = if noise_count > 0 {
            (noise_trace_sum / (noise_count as f64)).max(1e-8)
        } else {
            1.0
        };

        let mut cleaned_eigs = [0.0; MAX_ASSETS];
        for i in 0..dim {
            if i < signal_count {
                cleaned_eigs[i] = raw_eigs[i].max(1e-8);
            } else {
                cleaned_eigs[i] = avg_noise_lambda;
            }
        }

        // 5. Reconstrucción de la matriz limpia: C_clean = V * Lambda_clean * V^T
        let mut cleaned = [[0.0; MAX_ASSETS]; MAX_ASSETS];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    sum += evecs[i][k] * cleaned_eigs[k] * evecs[j][k];
                }
                cleaned[i][j] = sum;
            }
        }

        // 6. Normalización para garantizar diagonal idéntica a 1.0 y simetría exacta
        let mut diag_inv_sqrt = [1.0; MAX_ASSETS];
        for i in 0..dim {
            let d = cleaned[i][i].max(1e-12);
            diag_inv_sqrt[i] = 1.0 / d.sqrt();
        }

        let mut final_matrix = [[0.0; MAX_ASSETS]; MAX_ASSETS];
        for i in 0..dim {
            for j in 0..dim {
                if i == j {
                    final_matrix[i][j] = 1.0;
                } else {
                    let val = cleaned[i][j] * diag_inv_sqrt[i] * diag_inv_sqrt[j];
                    final_matrix[i][j] = val.clamp(-1.0, 1.0);
                }
            }
        }

        let snr = if total_trace_sum > 1e-12 {
            (signal_trace_sum / total_trace_sum).clamp(0.0, 1.0)
        } else {
            0.0
        };

        DenoisedCorrelationResult {
            cleaned_matrix: final_matrix,
            raw_eigenvalues: raw_eigs,
            cleaned_eigenvalues: cleaned_eigs,
            signal_modes_count: signal_count,
            signal_to_noise_ratio: snr,
            bounds,
        }
    }
}

// ============================================================================
// GRUPO DE RENORMALIZACIÓN DE WILSON (MOMENTUM-SHELL COARSE-GRAINING)
// ============================================================================

/// Motor del Grupo de Renormalización de Wilson sobre flujos temporales de microestructura.
#[derive(Debug, Clone, Copy)]
pub struct WilsonRGEngine;

impl WilsonRGEngine {
    /// Ejecuta el proceso de coarse-graining diádico decimation $x^{(k)}(t) = \frac{1}{2}(x^{(k-1)}(2t) + x^{(k-1)}(2t+1))$
    /// y calcula las funciones beta de acoplamiento.
    pub fn compute_scale_flow(series: &[f64]) -> WilsonRGScaleReport {
        let mut report = WilsonRGScaleReport::default();
        let len = series.len();
        if len < 16 {
            return report;
        }

        // Buffer local para almacenamiento de las series en cada escala
        // Escala 0: hasta 256 elementos
        let mut current_buf = [0.0; RETURNS_HISTORY_SIZE];
        let init_len = len.min(RETURNS_HISTORY_SIZE);
        current_buf[..init_len].copy_from_slice(&series[(len - init_len)..len]);
        let mut current_len = init_len;

        const LN_2: f64 = 0.6931471805599453;

        for scale in 0..MAX_DYADIC_SCALES {
            if current_len < 4 {
                break;
            }

            // 1. Calcular acoplamiento de autocorrelación lag-1 g1
            let mut mean = 0.0;
            for i in 0..current_len {
                mean += current_buf[i];
            }
            mean /= current_len as f64;

            let mut var = 0.0;
            let mut cov1 = 0.0;
            let mut m4 = 0.0;

            for i in 0..current_len {
                let diff = current_buf[i] - mean;
                let diff_sq = diff * diff;
                var += diff_sq;
                m4 += diff_sq * diff_sq;
            }
            var /= current_len as f64;
            m4 /= current_len as f64;

            for i in 0..(current_len - 1) {
                let d0 = current_buf[i] - mean;
                let d1 = current_buf[i + 1] - mean;
                cov1 += d0 * d1;
            }
            cov1 /= (current_len - 1) as f64;

            let g1 = if var > 1e-12 {
                (cov1 / var).clamp(-1.0, 1.0)
            } else {
                0.0
            };

            let g2 = if var > 1e-12 {
                (m4 / (var * var) - 3.0).clamp(-3.0, 10.0)
            } else {
                0.0
            };

            report.g1_couplings[scale] = g1;
            report.g2_couplings[scale] = g2;

            // 2. Coarse-graining para el siguiente nivel: momentum-shell decimation
            let next_len = current_len / 2;
            let mut next_buf = [0.0; RETURNS_HISTORY_SIZE];
            for i in 0..next_len {
                next_buf[i] = 0.5 * (current_buf[2 * i] + current_buf[2 * i + 1]);
            }
            current_buf = next_buf;
            current_len = next_len;
        }

        // 3. Calcular funciones beta beta_1(k) = (g1(k+1) - g1(k)) / ln(2)
        let mut beta_sum = 0.0;
        let valid_steps = 4.min(MAX_DYADIC_SCALES - 1);
        for k in 0..(MAX_DYADIC_SCALES - 1) {
            let flow = (report.g1_couplings[k + 1] - report.g1_couplings[k]) / LN_2;
            report.beta1_flows[k] = flow;
            if k < valid_steps {
                beta_sum += flow;
            }
        }

        report.effective_beta1 = beta_sum / (valid_steps as f64);

        // 4. Clasificación física del operador
        report.operator_class = if report.effective_beta1 > 0.05 {
            WilsonOperatorClass::Relevant
        } else if report.effective_beta1 < -0.05 {
            WilsonOperatorClass::Irrelevant
        } else {
            WilsonOperatorClass::Marginal
        };

        report
    }
}

// ============================================================================
// CONTINUOUS MAXIMAL LYAPUNOV EXPONENT ESTIMATOR
// ============================================================================

/// Estimador del exponente de Lyapunov continuo $\lambda_{\max}$ en espacio de fases de Hilbert.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousLyapunovEstimator {
    /// Buffer de vectores de estado inmersos $Y(t) \in \mathbb{R}^m$.
    trajectory_buffer: [[f64; LYAPUNOV_EMBED_DIM]; LYAPUNOV_BUFFER_SIZE],
    /// Puntero de escritura circular en el buffer de trayectorias.
    head: usize,
    /// Conteo de vectores registrados.
    count: usize,
    /// Exponente de Lyapunov suavizado $\lambda_{\max}$.
    smoothed_lambda: f64,
}

impl Default for ContinuousLyapunovEstimator {
    fn default() -> Self {
        Self {
            trajectory_buffer: [[0.0; LYAPUNOV_EMBED_DIM]; LYAPUNOV_BUFFER_SIZE],
            head: 0,
            count: 0,
            smoothed_lambda: 0.0,
        }
    }
}

impl ContinuousLyapunovEstimator {
    /// Crea una nueva instancia del estimador.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra un nuevo vector de estado inmerso $Y(t) = [x_t, x_{t-1}, x_{t-2}, x_{t-3}]$
    /// y actualiza el exponente de divergencia orbital.
    pub fn update(&mut self, state_vector: [f64; LYAPUNOV_EMBED_DIM]) -> LyapunovDiagnostic {
        let current_idx = self.head;
        self.trajectory_buffer[current_idx] = state_vector;
        self.head = (self.head + 1) % LYAPUNOV_BUFFER_SIZE;
        if self.count < LYAPUNOV_BUFFER_SIZE {
            self.count += 1;
        }

        if self.count < 16 {
            return LyapunovDiagnostic::default();
        }

        // Algoritmo de Rosenstein / Wolf en tiempo real:
        // Buscar el vecino más cercano en el pasado excluyendo la ventana de Theiler (W=4)
        let current_pt = state_vector;
        let mut min_dist = f64::MAX;
        let mut nearest_idx = usize::MAX;

        const THEILER_WINDOW: usize = 4;
        let effective_size = self.count;

        for offset in THEILER_WINDOW..(effective_size - 2) {
            let past_idx = (current_idx + LYAPUNOV_BUFFER_SIZE - offset) % LYAPUNOV_BUFFER_SIZE;
            let past_pt = self.trajectory_buffer[past_idx];

            let mut dist_sq = 0.0;
            for dim in 0..LYAPUNOV_EMBED_DIM {
                let diff = current_pt[dim] - past_pt[dim];
                dist_sq += diff * diff;
            }

            if dist_sq > 1e-12 && dist_sq < min_dist {
                min_dist = dist_sq;
                nearest_idx = past_idx;
            }
        }

        if nearest_idx == usize::MAX {
            return LyapunovDiagnostic {
                lambda_max: self.smoothed_lambda,
                horizon_compression: (1.0 / (1.0 + self.smoothed_lambda.max(0.0))).clamp(0.10, 1.0),
                is_chaotic: self.smoothed_lambda > 0.02,
                phase_space_diameter: 0.0,
            };
        }

        let d0 = min_dist.sqrt().max(1e-6);

        // Medir separación evolucionada tras 1 paso temporal
        let past_next_idx = (nearest_idx + 1) % LYAPUNOV_BUFFER_SIZE;
        let past_next_pt = self.trajectory_buffer[past_next_idx];

        let mut evolved_dist_sq = 0.0;
        for dim in 0..LYAPUNOV_EMBED_DIM {
            let diff = current_pt[dim] - past_next_pt[dim];
            evolved_dist_sq += diff * diff;
        }
        let d1 = evolved_dist_sq.sqrt().max(1e-6);

        // Tasa local de divergencia: ln(d1 / d0)
        let local_lyapunov = (d1 / d0).ln().clamp(-5.0, 5.0);

        // Actualización EWMA con alpha = 0.05
        self.smoothed_lambda = 0.95 * self.smoothed_lambda + 0.05 * local_lyapunov;

        let horizon_comp = (1.0 / (1.0 + self.smoothed_lambda.max(0.0))).clamp(0.05, 1.0);
        let is_chaotic = self.smoothed_lambda > 0.02;

        LyapunovDiagnostic {
            lambda_max: self.smoothed_lambda,
            horizon_compression: horizon_comp,
            is_chaotic,
            phase_space_diameter: d0,
        }
    }
}

// ============================================================================
// RANDOM MATRIX & WILSON UNIVERSE ENGINE (UNIFICADOR SISTÉMICO)
// ============================================================================

/// Motor unificado que integra Denoising de Marchenko-Pastur, Grupo de Renormalización de Wilson
/// y Dinámica del Exponente de Lyapunov para hasta 16 activos concurrentes.
#[derive(Debug, Clone)]
pub struct RandomMatrixWilsonEngine {
    /// Número de activos activos en el universo.
    n_assets: usize,
    /// Historial de retornos recientes por activo [asset_idx][sample_idx].
    returns_history: [[f64; RETURNS_HISTORY_SIZE]; MAX_ASSETS],
    /// Punteros circulares de escritura por activo.
    history_heads: [usize; MAX_ASSETS],
    /// Longitud de datos acumulados por activo.
    history_counts: [usize; MAX_ASSETS],
    /// Último resultado de denoising de Marchenko-Pastur.
    last_denoised_result: Option<DenoisedCorrelationResult>,
    /// Reportes de Wilson RG por activo.
    wilson_reports: [WilsonRGScaleReport; MAX_ASSETS],
    /// Estimadores de Lyapunov por activo.
    lyapunov_estimators: [ContinuousLyapunovEstimator; MAX_ASSETS],
    /// Últimos diagnósticos de Lyapunov por activo.
    lyapunov_diagnostics: [LyapunovDiagnostic; MAX_ASSETS],
}

impl RandomMatrixWilsonEngine {
    /// Inicializa el motor para un universo de `n_assets` activos ($1 \le N \le 16$).
    pub fn new(n_assets: usize) -> Self {
        let dim = n_assets.clamp(1, MAX_ASSETS);
        Self {
            n_assets: dim,
            returns_history: [[0.0; RETURNS_HISTORY_SIZE]; MAX_ASSETS],
            history_heads: [0; MAX_ASSETS],
            history_counts: [0; MAX_ASSETS],
            last_denoised_result: None,
            wilson_reports: [WilsonRGScaleReport::default(); MAX_ASSETS],
            lyapunov_estimators: [ContinuousLyapunovEstimator::new(); MAX_ASSETS],
            lyapunov_diagnostics: [LyapunovDiagnostic::default(); MAX_ASSETS],
        }
    }

    /// Ingesta un nuevo retorno streaming para el activo `asset_idx`.
    pub fn push_return(&mut self, asset_idx: usize, ret_val: f64) {
        if asset_idx >= self.n_assets {
            return;
        }

        let val = if ret_val.is_finite() { ret_val } else { 0.0 };

        // 1. Actualizar historial de retornos
        let head = self.history_heads[asset_idx];
        self.returns_history[asset_idx][head] = val;
        self.history_heads[asset_idx] = (head + 1) % RETURNS_HISTORY_SIZE;
        if self.history_counts[asset_idx] < RETURNS_HISTORY_SIZE {
            self.history_counts[asset_idx] += 1;
        }

        // 2. Actualizar estimador de Lyapunov con vector de retornos retardados [r_t, r_{t-1}, r_{t-2}, r_{t-3}]
        let count = self.history_counts[asset_idx];
        if count >= LYAPUNOV_EMBED_DIM {
            let mut state = [0.0; LYAPUNOV_EMBED_DIM];
            for lag in 0..LYAPUNOV_EMBED_DIM {
                let idx = (self.history_heads[asset_idx] + RETURNS_HISTORY_SIZE - 1 - lag) % RETURNS_HISTORY_SIZE;
                state[lag] = self.returns_history[asset_idx][idx];
            }
            let diag = self.lyapunov_estimators[asset_idx].update(state);
            self.lyapunov_diagnostics[asset_idx] = diag;
        }

        // 3. Actualizar Grupo de Renormalización de Wilson periódicamente (cada 8 ticks)
        if count >= 32 && (count % 8 == 0) {
            let mut ordered = [0.0; RETURNS_HISTORY_SIZE];
            let start = (self.history_heads[asset_idx] + RETURNS_HISTORY_SIZE - count) % RETURNS_HISTORY_SIZE;
            for i in 0..count {
                ordered[i] = self.returns_history[asset_idx][(start + i) % RETURNS_HISTORY_SIZE];
            }
            self.wilson_reports[asset_idx] = WilsonRGEngine::compute_scale_flow(&ordered[..count]);
        }
    }

    /// Recalcula la matriz empírica de correlación y ejecuta el Denoising cuántico de Marchenko-Pastur.
    pub fn update_correlation_denoising(&mut self) -> Option<DenoisedCorrelationResult> {
        let min_count = self.history_counts[..self.n_assets].iter().copied().min().unwrap_or(0);
        if min_count < 16 {
            return None;
        }

        let t_samples = min_count;

        // Calcular matriz empírica de correlación
        let mut corr = [[0.0; MAX_ASSETS]; MAX_ASSETS];
        let mut means = [0.0; MAX_ASSETS];
        let mut stds = [1.0; MAX_ASSETS];

        // Calcular medias y desviaciones estándar
        for i in 0..self.n_assets {
            let mut m = 0.0;
            let start = (self.history_heads[i] + RETURNS_HISTORY_SIZE - t_samples) % RETURNS_HISTORY_SIZE;
            for k in 0..t_samples {
                m += self.returns_history[i][(start + k) % RETURNS_HISTORY_SIZE];
            }
            m /= t_samples as f64;
            means[i] = m;

            let mut var = 0.0;
            for k in 0..t_samples {
                let diff = self.returns_history[i][(start + k) % RETURNS_HISTORY_SIZE] - m;
                var += diff * diff;
            }
            var /= t_samples as f64;
            stds[i] = var.sqrt().max(1e-8);
        }

        // Calcular correlaciones cruzadas Pearson
        for i in 0..self.n_assets {
            corr[i][i] = 1.0;
            let start_i = (self.history_heads[i] + RETURNS_HISTORY_SIZE - t_samples) % RETURNS_HISTORY_SIZE;

            for j in (i + 1)..self.n_assets {
                let start_j = (self.history_heads[j] + RETURNS_HISTORY_SIZE - t_samples) % RETURNS_HISTORY_SIZE;
                let mut cov = 0.0;
                for k in 0..t_samples {
                    let di = self.returns_history[i][(start_i + k) % RETURNS_HISTORY_SIZE] - means[i];
                    let dj = self.returns_history[j][(start_j + k) % RETURNS_HISTORY_SIZE] - means[j];
                    cov += di * dj;
                }
                cov /= t_samples as f64;
                let r = (cov / (stds[i] * stds[j])).clamp(-1.0, 1.0);
                corr[i][j] = r;
                corr[j][i] = r;
            }
        }

        // Ejecutar denoising de Marchenko-Pastur
        let result = MarchenkoPasturFilter::denoise_correlation(&corr, self.n_assets, t_samples);
        self.last_denoised_result = Some(result);
        Some(result)
    }

    /// Obtiene la correlación filtrada entre dos activos.
    #[inline]
    pub fn get_denoised_correlation(&self, i: usize, j: usize) -> f64 {
        if i >= self.n_assets || j >= self.n_assets {
            return if i == j { 1.0 } else { 0.0 };
        }
        if let Some(ref res) = self.last_denoised_result {
            res.cleaned_matrix[i][j]
        } else {
            if i == j { 1.0 } else { 0.0 }
        }
    }

    /// Clasificación del operador Wilson RG para el activo dado.
    #[inline]
    pub fn get_wilson_class(&self, asset_idx: usize) -> WilsonOperatorClass {
        if asset_idx < self.n_assets {
            self.wilson_reports[asset_idx].operator_class
        } else {
            WilsonOperatorClass::Marginal
        }
    }

    /// Retorna la función beta efectiva del activo.
    #[inline]
    pub fn get_wilson_beta1(&self, asset_idx: usize) -> f64 {
        if asset_idx < self.n_assets {
            self.wilson_reports[asset_idx].effective_beta1
        } else {
            0.0
        }
    }

    /// Retorna el diagnóstico de Lyapunov para el activo dado.
    #[inline]
    pub fn get_lyapunov_diagnostic(&self, asset_idx: usize) -> LyapunovDiagnostic {
        if asset_idx < self.n_assets {
            self.lyapunov_diagnostics[asset_idx]
        } else {
            LyapunovDiagnostic::default()
        }
    }

    /// Retorna el SNR global del mercado (energía de señal / traza total).
    #[inline]
    pub fn get_signal_to_noise_ratio(&self) -> f64 {
        if let Some(ref res) = self.last_denoised_result {
            res.signal_to_noise_ratio
        } else {
            0.0
        }
    }

    /// Verifica si un activo tiene tendencia macro relevante (aprobado para ejecución tendencial).
    #[inline]
    pub fn is_trend_relevant(&self, asset_idx: usize) -> bool {
        self.get_wilson_class(asset_idx) == WilsonOperatorClass::Relevant
    }

    /// Factor de compresión de horizonte basado en caos de Lyapunov.
    #[inline]
    pub fn get_horizon_compression(&self, asset_idx: usize) -> f64 {
        self.get_lyapunov_diagnostic(asset_idx).horizon_compression
    }
}
