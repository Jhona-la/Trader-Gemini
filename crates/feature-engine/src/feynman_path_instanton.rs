//! 🌌 FORMALISMO DE INTEGRALES DE CAMINO DE FEYNMAN, INSTANTONES Y TUNELAMIENTO CUÁNTICO (FASE 11)
//!
//! ## Fundamento Físico y Matemático
//!
//! En un universo multivariante continuo temporal-espectral, la evolución temporal de los
//! precios y la liquidez no es una simple caminata aleatoria browniana ni un proceso de difusión
//! lineal. Por el contrario, la probabilidad de transición entre un estado de libro de órdenes
//! $(x_a, t_a)$ y un estado $(x_b, t_b)$ se expresa mediante la **Integral de Camino de Feynman**
//! en tiempo imaginario (rotación de Wick $\tau = i t$):
//!
//!   $$\mathcal{Z}[x_b, \tau_b | x_a, \tau_a] = \int_{x(\tau_a)=x_a}^{x(\tau_b)=x_b} \mathcal{D}[x(\tau)] \exp\left( -\frac{1}{\hbar_{\text{eff}}} \mathcal{S}_E[x] \right)$$
//!
//! donde la **Acción Euclidiana** viene dada por:
//!   $$\mathcal{S}_E[x] = \int_{\tau_a}^{\tau_b} \left[ \frac{1}{2} m \left(\frac{dx}{d\tau}\right)^2 + V(x) \right] d\tau$$
//!
//! ### 1. Muros de Liquidez y Potencial de Doble Pozo
//! El libro de órdenes L2/L3 genera barreras de potencial metaestables modeladas canónicamente por un
//! **Potencial de Doble Pozo asimétrico**:
//!   $$V(x) = \frac{\lambda}{4} (x^2 - a^2)^2 - \epsilon x$$
//! donde:
//! - $\pm a$ son las posiciones de equilibrio de liquidez (ej. Bid dominante en $-a$ vs Ask dominante en $+a$).
//! - $\lambda > 0$ modula la rigidez del muro institucional / densidad de profundidad de libro.
//! - $\epsilon$ representa la inclinación externa inducida por el flujo de órdenes neto (OFI/CVD).
//!
//! ### 2. Soluciones Solitónicas: Instantones de Mercado
//! El instantón es la solución clásica $\bar{x}(\tau)$ a las ecuaciones de movimiento euclidianas
//! $m \frac{d^2 x}{d\tau^2} = V'(x)$ que conecta los dos pozos de potencial (efecto túnel entre niveles):
//!   $$\bar{x}(\tau) = a \tanh\left( \sqrt{\frac{\lambda}{2m}} a (\tau - \tau_0) \right)$$
//! La acción del instantón clásico es finita y fija la escala de probabilidad del salto:
//!   $$\mathcal{S}_0 = \frac{4}{3} a^3 \sqrt{2 m \lambda}$$
//!
//! ### 3. Tasa de Tunelamiento Cuántico de Gamow / WKB
//! La probabilidad de que una ráfaga de flujo de órdenes atraviese una pared de liquidez sin ser absorbida
//! se evalúa mediante la integral semiclásica WKB entre los puntos de retorno $x_1, x_2$ donde $V(x) \ge E$:
//!   $$B(E) = \int_{x_1}^{x_2} \sqrt{2m (V(x) - E)} \, dx$$
//!   $$\mathcal{T}_{\text{WKB}} = \exp\left( - \frac{2 B(E)}{\hbar_{\text{eff}}} \right) \in [0, 1]$$
//!
//! ### 4. Métrica Holográfica de Poincaré (AdS/CFT)
//! El horizonte temporal diádico $\tau$ se mapea como la coordenada radial holográfica $z$ del espacio
//! anti-de Sitter ($\text{AdS}_{d+1}$):
//!   $$ds^2 = \frac{L^2}{z^2} (dz^2 + \eta_{\mu\nu} dx^\mu dx^\nu)$$
//! La entropía holográfica de Ryu-Takayanagi cuantifica el flujo de información cuántica entre
//! microestructura instantánea ($z \to 0$) y la deriva macroeconómica ($z \to \infty$).

use std::f64;

pub const MAX_PATH_POINTS: usize = 32;

/// Clasificación física del régimen instantónico y de tunelamiento.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstantonRegime {
    /// Rango atrapado: Energía cinética insuficiente para penetrar el muro de liquidez (Mean-Reverting).
    MetastableTrapped,
    /// Tunelamiento inminente: Fluctuaciones cuánticas penetran la barrera de libro (Breakout Hazard).
    QuantumTunnelingImminent,
    /// Ruptura clásica: Energía supera la cima del potencial (Breakout explosivo confirmado).
    ClassicalBreakout,
}

/// Parámetros de la física cuántica de instantones y potencial de libro.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstantonParams {
    /// Masa inercial efectiva del precio / liquidez (default: 1.0)
    pub effective_mass: f64,
    /// Constante de Planck efectiva de microestructura (escala de ruido L3, default: 0.05)
    pub hbar_eff: f64,
    /// Acoplamiento cuartico del pozo de potencial lambda (default: 2.0)
    pub lambda_coupling: f64,
    /// Semidistancia entre mínimos de potencial 'a' (default: 1.0)
    pub well_separation_a: f64,
}

impl Default for InstantonParams {
    fn default() -> Self {
        Self {
            effective_mass: 1.0,
            hbar_eff: 0.05,
            lambda_coupling: 2.0,
            well_separation_a: 1.0,
        }
    }
}

impl InstantonParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.effective_mass.is_finite() || self.effective_mass <= 0.0 || self.effective_mass > 1000.0 {
            return Err(format!("Invalid effective_mass: {}", self.effective_mass));
        }
        if !self.hbar_eff.is_finite() || self.hbar_eff <= 1e-6 || self.hbar_eff > 10.0 {
            return Err(format!("Invalid hbar_eff: {}", self.hbar_eff));
        }
        if !self.lambda_coupling.is_finite() || self.lambda_coupling <= 0.0 || self.lambda_coupling > 1000.0 {
            return Err(format!("Invalid lambda_coupling: {}", self.lambda_coupling));
        }
        if !self.well_separation_a.is_finite() || self.well_separation_a <= 0.0 || self.well_separation_a > 1000.0 {
            return Err(format!("Invalid well_separation_a: {}", self.well_separation_a));
        }
        Ok(())
    }
}

/// Diagnóstico de tunelamiento cuántico a través de muros de liquidez.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstantonTunnelingReport {
    /// Transmitancia WKB Gamow \in [0, 1]
    pub tunneling_probability: f64,
    /// Acción euclidiana calculada bajo la barrera
    pub barrier_action_b: f64,
    /// Altura neta de la barrera de potencial Delta V
    pub barrier_height: f64,
    /// Energía cinética normalizada de la orden/flujo E
    pub kinetic_energy: f64,
    /// Clasificación física del régimen
    pub regime: InstantonRegime,
    /// Objetivo direccional proyectado tras el tunelamiento (+a o -a)
    pub projected_target_x: f64,
    /// Entropía holográfica de entrelazamiento AdS/CFT
    pub holographic_entropy: f64,
}

/// Motor de Integrales de Camino de Feynman e Instantones de Mercado.
#[derive(Debug, Clone, Copy)]
pub struct FeynmanInstantonEngine {
    pub params: InstantonParams,
}

impl Default for FeynmanInstantonEngine {
    fn default() -> Self {
        Self::new(InstantonParams::default())
    }
}

impl FeynmanInstantonEngine {
    /// Crea un nuevo motor cuántico con parámetros especificados.
    pub fn new(params: InstantonParams) -> Self {
        let valid_params = if params.validate().is_ok() {
            params
        } else {
            InstantonParams::default()
        };
        Self { params: valid_params }
    }

    /// Evalúa el potencial de doble pozo $V(x) = \frac{\lambda}{4} (x^2 - a^2)^2 - \epsilon x$.
    #[inline]
    pub fn evaluate_potential(&self, x: f64, epsilon_bias: f64) -> f64 {
        let a = self.params.well_separation_a;
        let diff = x * x - a * a;
        0.25 * self.params.lambda_coupling * diff * diff - epsilon_bias * x
    }

    /// Evalúa la derivada del potencial $V'(x) = \lambda x (x^2 - a^2) - \epsilon$.
    #[inline]
    pub fn evaluate_potential_derivative(&self, x: f64, epsilon_bias: f64) -> f64 {
        let a = self.params.well_separation_a;
        self.params.lambda_coupling * x * (x * x - a * a) - epsilon_bias
    }

    /// Calcula la acción euclidiana discretizada $\mathcal{S}_E[x]$ sobre una trayectoria temporal.
    ///
    /// $$\mathcal{S}_E = \sum_{k=0}^{K-2} \left[ \frac{1}{2} m \frac{(x_{k+1} - x_k)^2}{\Delta \tau} + \frac{V(x_{k+1}) + V(x_k)}{2} \Delta \tau \right]$$
    pub fn compute_euclidean_action(
        &self,
        path: &[f64],
        delta_tau: f64,
        epsilon_bias: f64,
    ) -> Option<f64> {
        let k = path.len();
        if k < 2 || !delta_tau.is_finite() || delta_tau <= 0.0 || !epsilon_bias.is_finite() {
            return None;
        }

        let m = self.params.effective_mass;
        let mut total_action = 0.0_f64;

        for i in 0..(k - 1) {
            let x0 = path[i];
            let x1 = path[i + 1];
            if !x0.is_finite() || !x1.is_finite() {
                return None;
            }

            let dx = x1 - x0;
            let kinetic = 0.5 * m * (dx * dx) / delta_tau;
            let v0 = self.evaluate_potential(x0, epsilon_bias);
            let v1 = self.evaluate_potential(x1, epsilon_bias);
            let potential = 0.5 * (v0 + v1) * delta_tau;

            total_action += kinetic + potential;
        }

        if total_action.is_finite() {
            Some(total_action)
        } else {
            None
        }
    }

    /// Calcula la acción analítica del instantón continuo $\mathcal{S}_0 = \frac{4}{3} a^3 \sqrt{2 m \lambda}$.
    #[inline]
    pub fn analytical_instanton_action(&self) -> f64 {
        let a = self.params.well_separation_a;
        let m = self.params.effective_mass;
        let lambda = self.params.lambda_coupling;
        (4.0 / 3.0) * a.powi(3) * (2.0 * m * lambda).sqrt()
    }

    /// Evalúa la probabilidad de tunelamiento Gamow / WKB a través de la barrera de liquidez.
    ///
    /// # Parámetros
    /// - `current_x`: Posición actual del precio normalizado (ej. en el pozo izquierdo $-a$).
    /// - `order_flow_velocity`: Velocidad/impulso del flujo de órdenes instantáneo (OFI normalizado).
    /// - `epsilon_bias`: Asimetría de libro (deriva externa del flujo).
    pub fn evaluate_tunneling(
        &self,
        current_x: f64,
        order_flow_velocity: f64,
        epsilon_bias: f64,
    ) -> Option<InstantonTunnelingReport> {
        if !current_x.is_finite() || !order_flow_velocity.is_finite() || !epsilon_bias.is_finite() {
            return None;
        }

        let a = self.params.well_separation_a;
        let m = self.params.effective_mass;
        let hbar = self.params.hbar_eff;

        // Energía cinética efectiva del flujo de órdenes
        let kinetic_e = 0.5 * m * order_flow_velocity * order_flow_velocity;

        // La altura de la barrera central (en x = 0 aproximadamente)
        let v_peak = self.evaluate_potential(0.0, epsilon_bias);
        let v_current = self.evaluate_potential(current_x, epsilon_bias);
        let barrier_height = (v_peak - v_current).max(0.0);

        let energy_level = v_current + kinetic_e;

        // Si la energía supera la cúspide de la barrera -> Ruptura clásica garantizada
        if energy_level >= v_peak {
            let target = if current_x < 0.0 { a } else { -a };
            let holo_s = self.compute_holographic_entropy(a * 2.0);
            return Some(InstantonTunnelingReport {
                tunneling_probability: 1.0,
                barrier_action_b: 0.0,
                barrier_height,
                kinetic_energy: kinetic_e,
                regime: InstantonRegime::ClassicalBreakout,
                projected_target_x: target,
                holographic_entropy: holo_s,
            });
        }

        // Puntos de retorno clásicos x1, x2 bajo la barrera donde V(x) == energy_level
        // En el potencial simétrico no sesgado, x_turn = sqrt(a^2 - sqrt(4 E / lambda))
        let (x1, x2) = self.find_turning_points(energy_level, epsilon_bias)?;

        if x2 <= x1 {
            let target = if current_x < 0.0 { a } else { -a };
            let holo_s = self.compute_holographic_entropy(a * 2.0);
            return Some(InstantonTunnelingReport {
                tunneling_probability: 1.0,
                barrier_action_b: 0.0,
                barrier_height,
                kinetic_energy: kinetic_e,
                regime: InstantonRegime::ClassicalBreakout,
                projected_target_x: target,
                holographic_entropy: holo_s,
            });
        }

        // Integración semiclásica WKB por cuadratura de Gauss-Legendre de 8 puntos
        let action_b = self.integrate_wkb_action(x1, x2, energy_level, epsilon_bias);

        // Probabilidad de tunelamiento Gamow T = exp(-2 B / hbar)
        let exponent = (-2.0 * action_b / hbar).clamp(-700.0, 0.0);
        let prob = exponent.exp();

        let regime = if prob < 0.20 {
            InstantonRegime::MetastableTrapped
        } else if prob > 0.85 {
            InstantonRegime::ClassicalBreakout
        } else {
            InstantonRegime::QuantumTunnelingImminent
        };

        let target = if current_x < 0.0 { a } else { -a };
        let holo_s = self.compute_holographic_entropy((x2 - x1).abs().max(1e-4));

        Some(InstantonTunnelingReport {
            tunneling_probability: prob,
            barrier_action_b: action_b,
            barrier_height,
            kinetic_energy: kinetic_e,
            regime,
            projected_target_x: target,
            holographic_entropy: holo_s,
        })
    }

    /// Calcula la entropía holográfica de entrelazamiento AdS/CFT $S = \frac{c}{3} \ln(L / \epsilon_{\text{UV}})$.
    #[inline]
    pub fn compute_holographic_entropy(&self, spatial_separation: f64) -> f64 {
        let l = spatial_separation.abs().max(1e-6);
        let eps_uv = self.params.hbar_eff.max(1e-5);
        let central_charge = 1.0; // CFT2 central charge unitario
        (central_charge / 3.0) * (l / eps_uv).ln().max(0.0)
    }

    /// Búsqueda robusta en stack de los dos puntos de retorno clásicos $x_1 < x_2$ donde $V(x) = E$.
    fn find_turning_points(&self, target_energy: f64, epsilon_bias: f64) -> Option<(f64, f64)> {
        let a = self.params.well_separation_a;
        // Búsqueda bisección para x1 en [-a, 0] y x2 en [0, a]
        let mut left_low = -a;
        let mut left_high = 0.0;
        let v_left_low = self.evaluate_potential(left_low, epsilon_bias);
        let v_center = self.evaluate_potential(0.0, epsilon_bias);

        if target_energy < v_left_low || target_energy >= v_center {
            // Fuera de la barrera de doble pozo
            return None;
        }

        // Bisección para punto de retorno izquierdo x1
        for _ in 0..24 {
            let mid = 0.5 * (left_low + left_high);
            let v_mid = self.evaluate_potential(mid, epsilon_bias);
            if v_mid < target_energy {
                left_low = mid;
            } else {
                left_high = mid;
            }
        }
        let x1 = 0.5 * (left_low + left_high);

        // Bisección para punto de retorno derecho x2
        let mut right_low = 0.0;
        let mut right_high = a;
        for _ in 0..24 {
            let mid = 0.5 * (right_low + right_high);
            let v_mid = self.evaluate_potential(mid, epsilon_bias);
            if v_mid > target_energy {
                right_low = mid;
            } else {
                right_high = mid;
            }
        }
        let x2 = 0.5 * (right_low + right_high);

        if x1.is_finite() && x2.is_finite() && x1 < x2 {
            Some((x1, x2))
        } else {
            None
        }
    }

    /// Cuadratura de Gauss-Legendre de 8 puntos en stack para $B = \int_{x_1}^{x_2} \sqrt{2m (V(x) - E)} \, dx$.
    fn integrate_wkb_action(
        &self,
        x1: f64,
        x2: f64,
        energy: f64,
        epsilon_bias: f64,
    ) -> f64 {
        // Nodos y pesos normalizados en [-1, 1] de Gauss-Legendre de orden 8
        const NODES: [f64; 8] = [
            -0.9602898564975363,
            -0.7966664774136267,
            -0.5255324099163290,
            -0.1834346424956498,
            0.1834346424956498,
            0.5255324099163290,
            0.7966664774136267,
            0.9602898564975363,
        ];
        const WEIGHTS: [f64; 8] = [
            0.1012285362903763,
            0.2223810344533745,
            0.3137066458778873,
            0.3626837833783620,
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
        ];

        let m = self.params.effective_mass;
        let half_len = 0.5 * (x2 - x1);
        let mid = 0.5 * (x1 + x2);
        let mut sum = 0.0_f64;

        for i in 0..8 {
            let x = mid + half_len * NODES[i];
            let v = self.evaluate_potential(x, epsilon_bias);
            let diff = (v - energy).max(0.0);
            let integrand = (2.0 * m * diff).sqrt();
            sum += WEIGHTS[i] * integrand;
        }

        (sum * half_len).max(0.0)
    }

    /// Propagador de Fokker-Planck forward de 1 paso en espacio de fases:
    /// Calcula la media y varianza evolucionada de la distribución del precio tras $\Delta t$.
    pub fn propagate_fokker_planck(
        &self,
        mean_x: f64,
        var_x: f64,
        drift_velocity: f64,
        diffusion_coeff_d: f64,
        delta_t: f64,
        epsilon_bias: f64,
    ) -> Option<(f64, f64)> {
        if !mean_x.is_finite()
            || !var_x.is_finite()
            || var_x < 0.0
            || !drift_velocity.is_finite()
            || !diffusion_coeff_d.is_finite()
            || diffusion_coeff_d < 0.0
            || !delta_t.is_finite()
            || delta_t <= 0.0
        {
            return None;
        }

        // Fuerza conservativa determinista F(x) = -V'(x)
        let force = -self.evaluate_potential_derivative(mean_x, epsilon_bias);
        let m = self.params.effective_mass;
        let acceleration = force / m;

        // Evolución de Taylor de la media: x(t+dt) = x + v*dt + 0.5*a*dt^2
        let next_mean = mean_x + drift_velocity * delta_t + 0.5 * acceleration * delta_t * delta_t;

        // Expansión difusiva de la varianza: Var(t+dt) = Var + 2*D*dt
        let next_var = var_x + 2.0 * diffusion_coeff_d * delta_t;

        if next_mean.is_finite() && next_var.is_finite() {
            Some((next_mean, next_var))
        } else {
            None
        }
    }
}
