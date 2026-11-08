//! Fase 65: Dinámica de Solitones No-Lineales de Korteweg-de Vries y Ondas Dispersivas de N-Solitón.
//!
//! Este módulo implementa la ecuación integrable de Korteweg-de Vries (KdV):
//! $$\partial_t u + 6 u \partial_x u + \beta \partial_{xxx} u = 0$$
//! resolviendo el balance no-lineal convectivo vs dispersivo en la presión de liquidez continua.
//! Identifica trenes de solitones coherentes que propagan momento cuántico sin disipación.
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! 1. Solución canónica de 1-solitón: $u(x, t) = \frac{v}{2} \text{sech}^2\left(\frac{\sqrt{v}}{2}(x - vt - x_0)\right)$.
//! 2. Invariantes de movimiento de Gardner-Miura: Masa $I_1 = \int u$, Energía $I_2 = \int u^2$, Hamiltoniano $I_3$.
//! 3. Balance de KdV: $\mathcal{R}_{\text{kdv}} = \frac{\|6 u \partial_x u\|}{\beta \|\partial_{xxx} u\| + \epsilon} \approx 1.0$.
//! 4. Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y memoria acotada en stack $O(1)$.

/// Dimensión de la malla espacial acotada para el perfil KdV.
pub const KDV_GRID_POINTS: usize = 16;

/// Parámetros de configuración del motor de solitones KdV.
#[derive(Debug, Clone, Copy)]
pub struct KdvSolitonParams {
    /// Coeficiente de dispersión de tercer orden $\beta > 0$.
    pub dispersion_beta: f64,
    /// Epsilon para divisiones numéricas seguras.
    pub epsilon: f64,
    /// Umbral de balance no-lineal/dispersivo para clasificar un solitón coherente.
    pub soliton_balance_min: f64,
    pub soliton_balance_max: f64,
}

impl Default for KdvSolitonParams {
    fn default() -> Self {
        Self {
            dispersion_beta: 0.15,
            epsilon: 1e-12,
            soliton_balance_min: 0.65,
            soliton_balance_max: 1.55,
        }
    }
}

/// Informe omnisciente de la dinámica solitónica de Korteweg-de Vries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KdvSolitonReport {
    /// Amplitud de pico del solitón dominante $A = v/2 \ge 0$.
    pub soliton_amplitude: f64,
    /// Velocidad de propagación del solitón $v = 2A$.
    pub soliton_propagation_velocity: f64,
    /// Ratio de balance convectivo / dispersivo $\mathcal{R}_{\text{kdv}} \ge 0$.
    pub nonlinearity_dispersion_balance_ratio: f64,
    /// Hamiltoniano de KdV conservado $I_3$.
    pub kdv_hamiltonian_energy: f64,
    /// Indicador de solitón coherente no disipativo activo.
    pub coherent_soliton_detected: bool,
    /// Señal direccional de impulso solitónico sintetizada $\in [-1.0, 1.0]$.
    pub soliton_impulse_signal: f64,
}

impl Default for KdvSolitonReport {
    fn default() -> Self {
        Self {
            soliton_amplitude: 0.0,
            soliton_propagation_velocity: 0.0,
            nonlinearity_dispersion_balance_ratio: 1.0,
            kdv_hamiltonian_energy: 0.0,
            coherent_soliton_detected: false,
            soliton_impulse_signal: 0.0,
        }
    }
}

/// Motor de Dinámica Solitónica de Korteweg-de Vries.
#[derive(Debug, Clone)]
pub struct KdvSolitonDispersionEngine {
    params: KdvSolitonParams,
    spatial_profile: [f64; KDV_GRID_POINTS],
    cursor: usize,
    count: usize,
}

impl KdvSolitonDispersionEngine {
    /// Crea una nueva instancia del motor con parámetros específicos.
    pub fn new(params: KdvSolitonParams) -> Self {
        Self {
            params,
            spatial_profile: [0.0; KDV_GRID_POINTS],
            cursor: 0,
            count: 0,
        }
    }

    /// Actualiza el perfil espacial e integra el balance de Korteweg-de Vries.
    ///
    /// # Argumentos:
    /// - `price_return`: Retorno continuo normalizado $\Delta p / p$.
    /// - `order_flow_imbalance`: OFI normalizado $\in [-1, 1]$.
    /// - `spread_bps`: Spread actual en puntos básicos $\ge 0$.
    /// - `instantaneous_vol`: Volatilidad continua actual $\sigma \ge 0$.
    pub fn update(
        &mut self,
        price_return: f64,
        order_flow_imbalance: f64,
        spread_bps: f64,
        instantaneous_vol: f64,
    ) -> KdvSolitonReport {
        let safe_ret = if price_return.is_finite() {
            price_return.clamp(-0.50, 0.50)
        } else {
            0.0
        };

        let safe_ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 100.0)
        } else {
            1.0
        };

        let safe_vol = if instantaneous_vol.is_finite() && instantaneous_vol > 0.0 {
            instantaneous_vol.clamp(0.001, 1.0)
        } else {
            0.01
        };

        // 1. Inyección de la excitación de presión de liquidez al buffer circular
        // u = (ret / vol) + ofi / (1 + 0.1 * spread)
        let exc = (safe_ret / safe_vol) + safe_ofi / (1.0 + 0.1 * safe_spread);
        self.spatial_profile[self.cursor] = exc.clamp(-10.0, 10.0);
        self.cursor = (self.cursor + 1) % KDV_GRID_POINTS;
        if self.count < KDV_GRID_POINTS {
            self.count += 1;
        }

        if self.count < 5 {
            return KdvSolitonReport::default();
        }

        // 2. Cálculo de derivadas espaciales discretas \partial_x u y \partial_{xxx} u
        // Diferencias centradas periódicas de orden 2
        let n = self.count;
        let mut max_abs_u = 0.0;
        let mut peak_sign = 1.0;
        let mut conv_norm = 0.0;
        let mut disp_norm = 0.0;
        let mut hamiltonian_i3 = 0.0;

        for i in 0..n {
            let u_i = self.spatial_profile[i];
            let abs_u = u_i.abs();
            if abs_u > max_abs_u {
                max_abs_u = abs_u;
                peak_sign = if u_i >= 0.0 { 1.0 } else { -1.0 };
            }

            let next1 = self.spatial_profile[(i + 1) % n];
            let prev1 = self.spatial_profile[(i + n - 1) % n];
            let next2 = self.spatial_profile[(i + 2) % n];
            let prev2 = self.spatial_profile[(i + n - 2) % n];

            // Primera derivada centrada: \partial_x u = (u_{i+1} - u_{i-1}) / 2
            let du = (next1 - prev1) * 0.5;

            // Tercera derivada centrada: \partial_{xxx} u = (u_{i+2} - 2 u_{i+1} + 2 u_{i-1} - u_{i-2}) / 2
            let d3u = (next2 - 2.0 * next1 + 2.0 * prev1 - prev2) * 0.5;

            // Término convectivo: 6 * u * du
            let conv_term = 6.0 * u_i * du;
            // Término dispersivo: \beta * d3u
            let disp_term = self.params.dispersion_beta * d3u;

            conv_norm += conv_term * conv_term;
            disp_norm += disp_term * disp_term;

            hamiltonian_i3 += 2.0 * u_i * u_i * u_i - self.params.dispersion_beta * du * du;
        }

        let conv_rms = (conv_norm / n as f64).sqrt();
        let disp_rms = (disp_norm / n as f64).sqrt();

        // 3. Balance No-Lineal / Dispersivo de Korteweg-de Vries
        let balance_ratio = (conv_rms / (disp_rms + self.params.epsilon)).clamp(0.0, 10.0);

        // 4. Parámetros del solitón canónico
        let amplitude = max_abs_u;
        let velocity = 2.0 * amplitude;

        let coherent = amplitude >= 0.20
            && balance_ratio >= self.params.soliton_balance_min
            && balance_ratio <= self.params.soliton_balance_max;

        // 5. Señal direccional de impulso solitónico
        let signal_magnitude = if coherent {
            (amplitude / (1.0 + amplitude)).clamp(0.10, 1.0)
        } else {
            (amplitude / (2.0 + amplitude)).clamp(0.0, 0.50)
        };
        let impulse_signal = (peak_sign * signal_magnitude).clamp(-1.0, 1.0);

        KdvSolitonReport {
            soliton_amplitude: amplitude,
            soliton_propagation_velocity: velocity,
            nonlinearity_dispersion_balance_ratio: balance_ratio,
            kdv_hamiltonian_energy: hamiltonian_i3,
            coherent_soliton_detected: coherent,
            soliton_impulse_signal: impulse_signal,
        }
    }
}
