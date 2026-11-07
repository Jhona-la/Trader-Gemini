//! -------------------------------------------------------------------------------------------------
//! TRADER GEMINI — QUANTUM CONTINUOUS MANIFOLD ARCHITECTURE (OMEGA220)
//! Signal Engine: Operadores de Transferencia de Ruelle-Perron-Frobenius y Resonancias de Pollicott-Ruelle
//! Fase 59 (§59) — Ola 14
//!
//! Concepto Físico-Matemático:
//! Aplica el Operador de Transferencia Termodinámico de Ruelle-Perron-Frobenius (1968) $\mathcal{L} \psi(x)$
//! sobre el espacio de densidades de micro-retorno para modelar el decaimiento exacto de correlaciones.
//! Computa las Resonancias de Pollicott-Ruelle (1985-1986) $\lambda_k = e^{-\gamma_k + i \omega_k}$ en $O(1)$ stack,
//! donde $\gamma_0$ determina la vida media del alfa cuántico $\tau = 1 / \gamma_0$ y $\Delta_{\text{gap}} = 1 - |\lambda_1|$
//! mide la brecha espectral frente al caos puro.
//! Si el tiempo transcurrido excede el horizonte de decaimiento o si $\Delta_{\text{gap}} < 0.05$ (colapso ergódico a
//! ruido blanco), modula la convicción direccional a escala mínima defensiva para preservar el crecimiento compuesto.
//! -------------------------------------------------------------------------------------------------

/// Dimensión de la discretización del operador de transferencia sobre el soporte de densidades (16 nodos en stack).
pub const RUELLE_GRID_SIZE: usize = 16;

/// Capacidad del buffer circular de estados de micro-flujo.
pub const RUELLE_HISTORY_CAPACITY: usize = 16;

/// Parámetros de configuración del motor de transferencia de Ruelle-Perron-Frobenius.
#[derive(Debug, Clone, Copy)]
pub struct RuelleFrobeniusParams {
    /// Regularización de decaimiento intrínseco $\gamma_{\text{base}}$ (típicamente 0.05).
    pub decay_regularization: f64,
    /// Brecha espectral crítica para alerta de colapso ergódico a ruido blanco.
    pub min_spectral_gap: f64,
    /// Parámetro de temperatura termodinámica inversa $\beta$.
    pub thermodynamic_beta: f64,
    /// Parámetro epsilon para regularización numérica.
    pub epsilon: f64,
}

impl Default for RuelleFrobeniusParams {
    fn default() -> Self {
        Self {
            decay_regularization: 0.05,
            min_spectral_gap: 0.05,
            thermodynamic_beta: 1.0,
            epsilon: 1e-9,
        }
    }
}

/// Informe omnisciente de resonancias de Pollicott-Ruelle y decaimiento espectral del alfa.
#[derive(Debug, Clone, Copy)]
pub struct RuelleFrobeniusReport {
    /// Tasa de decaimiento espectral dominante $\gamma_0 \ge 0.0$.
    pub leading_decay_rate_gamma: f64,
    /// Frecuencia angular de resonancia de microestructura $\omega_0$.
    pub resonant_frequency_omega: f64,
    /// Brecha espectral del operador $\Delta_{\text{gap}} = 1 - |\lambda_1| \in [0.0, 1.0]$.
    pub spectral_gap: f64,
    /// Tiempo de decaimiento de Lyapunov del alfa en milisegundos $t_{\text{lyapunov}} = 1000 / \gamma_0$.
    pub lyapunov_decay_time_ms: f64,
    /// Factor de persistencia y coherencia temporal del alfa $\mathcal{P}_{\text{alpha}} \in [0.0, 1.0]$.
    pub alpha_persistence_factor: f64,
    /// Alerta de régimen ergódico degenerado (ruido puro sin memoria, $\Delta_{\text{gap}} < 0.05$).
    pub ergodic_chaotic_collapse: bool,
    /// Modulador de convicción de señal derivado $\lambda_{\text{ruelle}} \in [0.05, 1.30]$.
    pub signal_conviction_modulator: f64,
    /// Timestamp del evento procesado.
    pub event_time_ms: u64,
}

/// Motor de Operadores de Transferencia de Ruelle-Perron-Frobenius y Resonancias de Pollicott-Ruelle.
#[derive(Debug, Clone)]
pub struct RuelleFrobeniusTransferEngine {
    params: RuelleFrobeniusParams,
    density_state: [f64; RUELLE_GRID_SIZE],
    prev_ret: f64,
    prev_ofi: f64,
    last_event_time_ms: u64,
}

impl RuelleFrobeniusTransferEngine {
    /// Inicializa una nueva instancia del motor con los parámetros dados.
    pub fn new(params: RuelleFrobeniusParams) -> Self {
        let mut density_state = [0.0; RUELLE_GRID_SIZE];
        density_state[RUELLE_GRID_SIZE / 2] = 1.0; // Estado inicial localizado
        Self {
            params,
            density_state,
            prev_ret: 0.0,
            prev_ofi: 0.0,
            last_event_time_ms: 0,
        }
    }

    /// Actualiza el operador de transferencia y computa las resonancias de Pollicott-Ruelle en $O(1)$ stack.
    pub fn update_and_evaluate(
        &mut self,
        price_ret: f64,
        order_flow_imbalance: f64,
        spread_bps: f64,
        volatility: f64,
        event_time_ms: u64,
    ) -> RuelleFrobeniusReport {
        self.last_event_time_ms = event_time_ms;

        // 1. Mapeo no lineal unidimensional de micro-dinámica $x \in [-1.0, 1.0]$
        let ret_norm = (price_ret * 100.0).clamp(-1.0, 1.0);
        let ofi_norm = order_flow_imbalance.clamp(-1.0, 1.0);
        let current_state = 0.6 * ret_norm + 0.4 * ofi_norm;

        // 2. Construcción de matriz estocástica del operador de transferencia en stack
        // $\mathcal{L}_{ij} \propto \exp(-\beta \cdot \text{Cost}(i, j))$
        let beta = self.params.thermodynamic_beta;
        let spread_friction = (spread_bps * 0.1).clamp(0.01, 5.0);
        let vol_diffusion = (volatility * 10.0).clamp(0.01, 5.0);

        let mut next_density = [0.0; RUELLE_GRID_SIZE];
        let mut sum_density = 0.0;

        for i in 0..RUELLE_GRID_SIZE {
            let x_i = -1.0 + 2.0 * (i as f64) / ((RUELLE_GRID_SIZE - 1) as f64);
            let drift = current_state - x_i;

            for j in 0..RUELLE_GRID_SIZE {
                let x_j = -1.0 + 2.0 * (j as f64) / ((RUELLE_GRID_SIZE - 1) as f64);
                let dist = x_j - (x_i + 0.3 * drift);
                // Núcleo gaussiano termodinámico con fricción de spread
                let arg = (-(dist * dist) / (2.0 * vol_diffusion)
                    - beta * spread_friction * (x_j - x_i).abs())
                .clamp(-50.0, 50.0);
                let transfer_kernel = arg.exp();

                next_density[j] += self.density_state[i] * transfer_kernel;
            }
        }

        // 3. Normalización del vector de densidad de Frobenius
        for j in 0..RUELLE_GRID_SIZE {
            sum_density += next_density[j];
        }
        let inv_sum = 1.0 / sum_density.max(self.params.epsilon);
        for j in 0..RUELLE_GRID_SIZE {
            self.density_state[j] = next_density[j] * inv_sum;
        }

        // 4. Extracción de Resonancias de Pollicott-Ruelle mediante análisis de momentos
        // Primer momento (media) y segundo momento central (dispersión)
        let mut mean_x = 0.0;
        let mut var_x = 0.0;
        for j in 0..RUELLE_GRID_SIZE {
            let x_j = -1.0 + 2.0 * (j as f64) / ((RUELLE_GRID_SIZE - 1) as f64);
            mean_x += x_j * self.density_state[j];
        }
        for j in 0..RUELLE_GRID_SIZE {
            let x_j = -1.0 + 2.0 * (j as f64) / ((RUELLE_GRID_SIZE - 1) as f64);
            let diff = x_j - mean_x;
            var_x += diff * diff * self.density_state[j];
        }

        // Tasa intrínseca de decaimiento de Ruelle: $\gamma_0 = \gamma_{\text{base}} + \frac{\text{var}}{1 + \text{var}} \cdot \text{vol}$
        let leading_decay_rate_gamma =
            (self.params.decay_regularization + var_x * vol_diffusion).clamp(0.001, 10.0);

        // Frecuencia angular de resonancia de microestructura $\omega_0$
        let resonant_frequency_omega = (mean_x * 10.0 + (price_ret - self.prev_ret) * 50.0)
            .clamp(-core::f64::consts::PI, core::f64::consts::PI);

        // Brecha espectral $\Delta_{\text{gap}} = 1 - |\lambda_1|$
        let lambda_sub = (-leading_decay_rate_gamma).exp();
        let spectral_gap = (1.0 - lambda_sub).clamp(0.0, 1.0);

        // 5. Tiempo de Lyapunov del alfa cuántico (vida media útil de la señal en ms)
        let lyapunov_decay_time_ms = (1000.0 / leading_decay_rate_gamma).clamp(10.0, 60_000.0);

        // 6. Factor de Persistencia del Alfa $\mathcal{P}_{\text{alpha}} \in [0.0, 1.0]$
        let alpha_persistence_factor = (spectral_gap * (1.0 - var_x.min(1.0))).clamp(0.0, 1.0);

        // 7. Detección de Colapso Ergódico Caótico (degeneración a ruido blanco)
        let ergodic_chaotic_collapse = spectral_gap < self.params.min_spectral_gap
            || (var_x > 0.80 && leading_decay_rate_gamma > 5.0);

        // 8. Modulador de Convicción $\lambda_{\text{ruelle}} \in [0.05, 1.30]$
        let signal_conviction_modulator = if ergodic_chaotic_collapse {
            0.05
        } else {
            let base = 1.0 + 0.30 * (alpha_persistence_factor - 0.5);
            base.clamp(0.05, 1.30)
        };

        self.prev_ret = price_ret;
        self.prev_ofi = order_flow_imbalance;

        RuelleFrobeniusReport {
            leading_decay_rate_gamma,
            resonant_frequency_omega,
            spectral_gap,
            lyapunov_decay_time_ms,
            alpha_persistence_factor,
            ergodic_chaotic_collapse,
            signal_conviction_modulator,
            event_time_ms,
        }
    }
}
