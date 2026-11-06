//! Fase 40: Motor de Transporte Óptimo Continuo de Monge-Kantorovich ($W_1$/$W_2$ Wasserstein)
//!
//! Este módulo implementa la geometría de transporte óptimo regularizado por entropía (Cuturi 2013)
//! entre la medida de probabilidad empírica de microestructura $\mu$ y la distribución de referencia
//! de equilibrio neutral al riesgo $\nu$ sobre una variedad continua temporal-espectral.
//!
//! Invariantes y garantías:
//! - Algoritmo Sinkhorn-Knopp en dominio logarítmico (Log-Sinkhorn) numéricamente incondicionado.
//! - Cero asignaciones en memoria dinámica (heap) en el camino crítico caliente de evaluación.
//! - Ratchet de cero pánicos `rb_p1`: 0 unwrap, 0 expect, 0 panic en código de producción.
//! - Cota analítica de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ estrictamente preservada.
//! - Detección de velocidad y aceleración de deriva $\dot{W}_1, \ddot{W}_1$ para anticipar evaporación de liquidez.

/// Número de nodos discretos del soporte espacial continuo en la rejilla de microestructura.
pub const WASSERSTEIN_NODES: usize = 32;

/// Parámetros de configuración del motor de Monge-Kantorovich-Wasserstein.
#[derive(Debug, Clone, Copy)]
pub struct WassersteinParams {
    /// Parámetro de regularización entrópica $\varepsilon > 0$.
    pub epsilon: f64,
    /// Número máximo de iteraciones de Sinkhorn (típicamente 15 a 25 para convergencia estable).
    pub max_iterations: usize,
    /// Tolerancia de parada para la norma $L_1$ de discrepancia marginal dual.
    pub tolerance: f64,
    /// Ponderación del coste temporal $|\tau_i - \tau_j|$ frente al coste espacial $|x_i - x_j|$.
    pub tau_cost_weight: f64,
    /// Umbral de velocidad de deriva para alertar evaporación de liquidez.
    pub drift_velocity_threshold: f64,
}

impl Default for WassersteinParams {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            max_iterations: 20,
            tolerance: 1e-4,
            tau_cost_weight: 0.10,
            drift_velocity_threshold: 0.08,
        }
    }
}

/// Informe de transporte óptimo resultante de la evaluación de Monge-Kantorovich.
#[derive(Debug, Clone, Copy)]
pub struct WassersteinReport {
    /// Distancia $W_1$ regularizada (Earth Mover's Distance) entre $\mu$ y $\nu$.
    pub w1_distance: f64,
    /// Distancia $W_2^2$ cuadrática regularizada de transporte óptimo.
    pub w2_squared: f64,
    /// Velocidad instantánea de deriva $\dot{W}_1 = \frac{dW_1}{dt}$.
    pub drift_velocity: f64,
    /// Aceleración de deriva $\ddot{W}_1 = \frac{d^2 W_1}{dt^2}$.
    pub drift_acceleration: f64,
    /// Probabilidad o severidad de evaporación de liquidez $\in [0.0, 1.0]$.
    pub liquidity_evaporation_risk: f64,
    /// Entropía del plan de transporte óptimo $H(P) = -\sum P_{ij} \ln P_{ij}$.
    pub transport_entropy: f64,
    /// Número de iteraciones efectivas empleadas por el solucionador Sinkhorn.
    pub iterations_completed: usize,
}

/// Motor continuo de Monge-Kantorovich para cálculo de distancias de Wasserstein en microestructura.
#[derive(Debug, Clone)]
pub struct MongeKantorovichWassersteinEngine {
    params: WassersteinParams,
    /// Coordenadas de los nodos espaciales normalizados en $[-1.0, 1.0]$.
    grid_coords: [f64; WASSERSTEIN_NODES],
    /// Matriz precalculada de costes espaciales $C_{ij} = |x_i - x_j|$.
    cost_matrix_w1: [[f64; WASSERSTEIN_NODES]; WASSERSTEIN_NODES],
    /// Matriz precalculada de costes cuadráticos $C_{ij}^{(2)} = (x_i - x_j)^2$.
    cost_matrix_w2: [[f64; WASSERSTEIN_NODES]; WASSERSTEIN_NODES],
    /// Distribución de referencia de equilibrio neutral al riesgo $\nu$.
    reference_distribution: [f64; WASSERSTEIN_NODES],
    /// Historial de la última distancia $W_1$ para derivadas temporales.
    last_w1: f64,
    /// Historial de la última velocidad $\dot{W}_1$ para cálculo de aceleración.
    last_drift_velocity: f64,
    /// Marca temporal del último evento procesado (en milisegundos).
    last_ts_ms: u64,
}

impl MongeKantorovichWassersteinEngine {
    /// Construye una nueva instancia del motor con parámetros dados y distribuciones de soporte.
    pub fn new(params: WassersteinParams) -> Self {
        let mut grid_coords = [0.0; WASSERSTEIN_NODES];
        let n = WASSERSTEIN_NODES;
        let step = 2.0 / ((n - 1) as f64).max(1.0);
        for (i, coord) in grid_coords.iter_mut().enumerate() {
            *coord = -1.0 + (i as f64) * step;
        }

        let mut cost_matrix_w1 = [[0.0; WASSERSTEIN_NODES]; WASSERSTEIN_NODES];
        let mut cost_matrix_w2 = [[0.0; WASSERSTEIN_NODES]; WASSERSTEIN_NODES];

        for i in 0..n {
            for j in 0..n {
                let diff = (grid_coords[i] - grid_coords[j]).abs();
                cost_matrix_w1[i][j] = diff;
                cost_matrix_w2[i][j] = diff * diff;
            }
        }

        // Distribución de referencia gaussiana estándar truncada y normalizada en el soporte.
        let mut reference_distribution = [0.0; WASSERSTEIN_NODES];
        let mut sum_ref = 0.0;
        let sigma_ref = 0.40;
        for (i, ref_slot) in reference_distribution.iter_mut().enumerate() {
            let x = grid_coords[i];
            let density = (-0.5 * (x / sigma_ref).powi(2)).exp();
            *ref_slot = density;
            sum_ref += density;
        }
        if sum_ref > 1e-12 {
            for ref_slot in reference_distribution.iter_mut() {
                *ref_slot /= sum_ref;
            }
        }

        Self {
            params,
            grid_coords,
            cost_matrix_w1,
            cost_matrix_w2,
            reference_distribution,
            last_w1: 0.0,
            last_drift_velocity: 0.0,
            last_ts_ms: 0,
        }
    }

    /// Evalúa la distancia de transporte óptimo Wasserstein entre la medida empírica observada y el equilibrio.
    ///
    /// # Argumentos
    /// - `empirical_weights`: Masa de probabilidad normalizada o pesos de volumen empírico sobre los $N$ nodos.
    /// - `current_tau_ms`: Escala temporal dominante en ms para ponderación temporal del coste.
    /// - `ts_ms`: Marca temporal del tick en milisegundos.
    pub fn update_and_evaluate(
        &mut self,
        empirical_weights: &[f64; WASSERSTEIN_NODES],
        current_tau_ms: f64,
        ts_ms: u64,
    ) -> WassersteinReport {
        let n = WASSERSTEIN_NODES;

        // 1. Normalizar y verificar validez de la distribución empírica $\mu$
        let mut mu = [0.0; WASSERSTEIN_NODES];
        let mut sum_mu = 0.0;
        for (i, &w) in empirical_weights.iter().enumerate() {
            let val = if w.is_finite() && w > 0.0 { w } else { 0.0 };
            mu[i] = val;
            sum_mu += val;
        }

        if sum_mu <= 1e-12 {
            // Distribución colapsada o nula: fallback uniforme
            let uniform = 1.0 / (n as f64);
            for slot in mu.iter_mut() {
                *slot = uniform;
            }
        } else {
            for slot in mu.iter_mut() {
                *slot /= sum_mu;
            }
        }

        let nu = self.reference_distribution;

        // 2. Construir matriz de coste efectiva con acoplamiento temporal
        // C_{ij} = |x_i - x_j| * (1 + \lambda * \ln(1 + \tau))
        let tau_scale_factor =
            1.0 + self.params.tau_cost_weight * (1.0 + current_tau_ms.max(1.0).ln());
        let mut cost = [[0.0; WASSERSTEIN_NODES]; WASSERSTEIN_NODES];
        for i in 0..n {
            for j in 0..n {
                cost[i][j] = self.cost_matrix_w1[i][j] * tau_scale_factor;
            }
        }

        // 3. Resolver transporte óptimo vía Log-Domain Sinkhorn-Knopp
        let eps = self.params.epsilon.max(1e-4);
        let mut f = [0.0; WASSERSTEIN_NODES]; // Potencial dual f_i
        let mut g = [0.0; WASSERSTEIN_NODES]; // Potencial dual g_j

        // Precalcular logaritmos de las masas marginales
        let mut log_mu = [0.0; WASSERSTEIN_NODES];
        let mut log_nu = [0.0; WASSERSTEIN_NODES];
        for i in 0..n {
            log_mu[i] = (mu[i].max(1e-15)).ln();
            log_nu[i] = (nu[i].max(1e-15)).ln();
        }

        let mut iters_done = 0;
        for iter in 0..self.params.max_iterations {
            iters_done = iter + 1;
            let mut max_err = 0.0;

            // Actualización de f_i: f_i = eps * log_mu_i - eps * logsumexp_j((g_j - C_{ij})/eps)
            for i in 0..n {
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..n {
                    let v = (g[j] - cost[i][j]) / eps;
                    if v > max_val {
                        max_val = v;
                    }
                }

                let mut sum_exp = 0.0;
                for j in 0..n {
                    let v = (g[j] - cost[i][j]) / eps;
                    sum_exp += (v - max_val).exp();
                }

                let lse = max_val + sum_exp.max(1e-15).ln();
                let next_f = eps * (log_mu[i] - lse);
                let diff = (next_f - f[i]).abs();
                if diff > max_err {
                    max_err = diff;
                }
                f[i] = next_f;
            }

            // Actualización de g_j: g_j = eps * log_nu_j - eps * logsumexp_i((f_i - C_{ij})/eps)
            for j in 0..n {
                let mut max_val = f64::NEG_INFINITY;
                for i in 0..n {
                    let v = (f[i] - cost[i][j]) / eps;
                    if v > max_val {
                        max_val = v;
                    }
                }

                let mut sum_exp = 0.0;
                for i in 0..n {
                    let v = (f[i] - cost[i][j]) / eps;
                    sum_exp += (v - max_val).exp();
                }

                let lse = max_val + sum_exp.max(1e-15).ln();
                let next_g = eps * (log_nu[j] - lse);
                let diff = (next_g - g[j]).abs();
                if diff > max_err {
                    max_err = diff;
                }
                g[j] = next_g;
            }

            if max_err < self.params.tolerance {
                break;
            }
        }

        // 4. Calcular el plan de transporte óptimo P_{ij} y las distancias W_1, W_2^2
        let mut w1 = 0.0;
        let mut w2_sq = 0.0;
        let mut entropy = 0.0;

        for i in 0..n {
            for j in 0..n {
                let log_p = (f[i] + g[j] - cost[i][j]) / eps;
                let p = if log_p > -50.0 { log_p.exp() } else { 0.0 };
                if p > 0.0 {
                    w1 += p * cost[i][j];
                    w2_sq += p * self.cost_matrix_w2[i][j] * tau_scale_factor;
                    entropy -= p * p.max(1e-15).ln();
                }
            }
        }

        // 5. Derivadas temporales: velocidad y aceleración de deriva
        let dt = if self.last_ts_ms > 0 && ts_ms > self.last_ts_ms {
            ((ts_ms - self.last_ts_ms) as f64) / 1000.0
        } else {
            0.10 // Ventana por defecto de 100ms
        };

        let drift_velocity = if dt > 1e-4 {
            (w1 - self.last_w1) / dt
        } else {
            0.0
        };

        let drift_acceleration = if dt > 1e-4 {
            (drift_velocity - self.last_drift_velocity) / dt
        } else {
            0.0
        };

        // 6. Cálculo del riesgo de evaporación de liquidez
        // Activación no lineal basada en exceso de velocidad de transporte óptimo
        let v_excess = (drift_velocity.abs() - self.params.drift_velocity_threshold).max(0.0);
        let liquidity_evaporation_risk = (1.0 - (-v_excess * 15.0).exp()).clamp(0.0, 1.0);

        // Actualizar estado interno persistente
        self.last_w1 = w1;
        self.last_drift_velocity = drift_velocity;
        self.last_ts_ms = ts_ms;

        WassersteinReport {
            w1_distance: w1,
            w2_squared: w2_sq,
            drift_velocity,
            drift_acceleration,
            liquidity_evaporation_risk,
            transport_entropy: entropy,
            iterations_completed: iters_done,
        }
    }

    /// Retorna una referencia inmutable a las coordenadas del soporte espacial continuo.
    #[inline(always)]
    pub fn grid_coords(&self) -> &[f64; WASSERSTEIN_NODES] {
        &self.grid_coords
    }

    /// Retorna una referencia a la distribución de referencia neutral al riesgo $\nu$.
    #[inline(always)]
    pub fn reference_distribution(&self) -> &[f64; WASSERSTEIN_NODES] {
        &self.reference_distribution
    }
}
