//! ♟️ TEORÍA DE JUEGOS DE CAMPO MEDIO (MEAN FIELD GAMES) Y PSICOLOGÍA CUÁNTICA DE COLUSIÓN (OLA OMEGA220 — FASE 38)
//!
//! ## Fundamento Matemático: Sistema Acoplado Lasry-Lions (HJB + FPK)
//!
//! En un mercado continuo de alta frecuencia con millones de participantes interconectados
//! (minoristas con sesgos cognitivos de Kahneman-Tversky, algoritmos depredadores institucionales y market makers),
//! el equilibrio estratégico colectivo se modela mediante el sistema continuo de EDPs de
//! Mean Field Games (Pierre-Louis Lions & Jean-Michel Lasry, 2006):
//!
//! ### 1. Ecuación de Fokker-Planck-Kolmogorov (FPK) Anterógrada (Distribución de Masa)
//! Modela la evolución de la densidad de probabilidad $m(x, t)$ de las posiciones o creencias del mercado:
//!   $$\partial_t m - \frac{\sigma^2}{2} \partial_{xx} m + \partial_x (m \cdot u(x, t)) = 0$$
//! donde $u(x, t) = \nabla_p H(x, \nabla v)$ representa la velocidad de deriva óptima de la multitud.
//! Conservación estricta de probabilidad: $\int_{-1}^1 m(x, t) dx = 1$.
//!
//! ### 2. Ecuación de Hamilton-Jacobi-Bellman (HJB) Retrógrada (Valor del Agente Representativo)
//! Modela la función de valor óptimo $v(x, t)$ de un operador racional ante el campo medio:
//!   $$-\partial_t v - \frac{\sigma^2}{2} \partial_{xx} v + H(x, \nabla v) - f(x, m) = 0$$
//! donde el Hamiltoniano con fricción de ejecución es:
//!   $$H(p) = \frac{p^2}{2\gamma} - c_{\mathrm{fee}} |p|$$
//! y el potencial de interacción psicológica es:
//!   $$f(x, m) = -\alpha_{\mathrm{crowd}} \cdot m(x, t)^\theta + \beta_{\mathrm{hunt}} \cdot \Phi_{\mathrm{trap}}(x)$$
//!
//! ### 3. Fenomenología de Colusión y Caza de Paradas (Stop-Hunting)
//! - **Congestión del Rebaño ($\kappa_{\mathrm{crowd}}$)**: Medida por la concentración de masa $\int m^2 dx$.
//!   Cuando $\kappa_{\mathrm{crowd}} \to 1$, el mercado entra en un estado metaestable de estampida vulnerable a squeezes.
//! - **Agotamiento del Squeeze ($P_{\mathrm{exhaustion}}$)**: Probabilidad de que la presión compradora/vendedora
//!   alcance el punto de saturación terminal tras barrer los cúmulos de stop-loss.
//! - **Alpha Anti-Rebaño ($v_{\mathrm{anti}}$)**: Convicción direccional asimétrica para posicionarse en el
//!   equilibrio de Nash opuesto al colapso de la masa minorista.

use std::f64;

/// Número de nodos espaciales discretizados sobre el soporte $x \in [-1.0, 1.0]$.
pub const MFG_SPATIAL_NODES: usize = 32;

/// Parámetros configurables del motor de Mean Field Games y Psicología de Masas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanFieldGamesParams {
    /// Coeficiente de difusión estocástica $\sigma^2 / 2$ (volatilidad intrínseca del campo).
    pub diffusion_coeff: f64,
    /// Coeficiente de aversión al riesgo / inercia de fricción $\gamma > 0$.
    pub friction_gamma: f64,
    /// Costo de transacción o fricción de fee $c_{\mathrm{fee}} \ge 0$.
    pub transaction_fee: f64,
    /// Penalización por aglomeración de rebaño $\alpha_{\mathrm{crowd}} \ge 0$.
    pub crowd_penalty: f64,
    /// Sensibilidad no lineal de aglomeración $\theta \ge 1.0$.
    pub crowd_exponent: f64,
    /// Intensidad del potencial de caza de paradas $\beta_{\mathrm{hunt}} \ge 0$.
    pub hunt_intensity: f64,
    /// Paso de tiempo interno $\Delta t$ para la integración continua de EDPs.
    pub time_step_dt: f64,
}

impl Default for MeanFieldGamesParams {
    fn default() -> Self {
        Self {
            diffusion_coeff: 0.05,
            friction_gamma: 1.20,
            transaction_fee: 0.0004,
            crowd_penalty: 0.40,
            crowd_exponent: 1.50,
            hunt_intensity: 0.60,
            time_step_dt: 0.02,
        }
    }
}

impl MeanFieldGamesParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.diffusion_coeff.is_finite()
            || self.diffusion_coeff <= 0.0
            || self.diffusion_coeff > 1.0
        {
            return Err("MeanFieldGamesParams: diffusion_coeff fuera de rango (0.0, 1.0]");
        }
        if !self.friction_gamma.is_finite()
            || self.friction_gamma <= 0.1
            || self.friction_gamma > 10.0
        {
            return Err("MeanFieldGamesParams: friction_gamma fuera de rango [0.1, 10.0]");
        }
        if !self.transaction_fee.is_finite()
            || self.transaction_fee < 0.0
            || self.transaction_fee > 0.05
        {
            return Err("MeanFieldGamesParams: transaction_fee fuera de rango [0.0, 0.05]");
        }
        if !self.crowd_penalty.is_finite() || self.crowd_penalty < 0.0 || self.crowd_penalty > 5.0 {
            return Err("MeanFieldGamesParams: crowd_penalty fuera de rango [0.0, 5.0]");
        }
        if !self.crowd_exponent.is_finite()
            || self.crowd_exponent < 1.0
            || self.crowd_exponent > 3.0
        {
            return Err("MeanFieldGamesParams: crowd_exponent fuera de rango [1.0, 3.0]");
        }
        if !self.hunt_intensity.is_finite()
            || self.hunt_intensity < 0.0
            || self.hunt_intensity > 5.0
        {
            return Err("MeanFieldGamesParams: hunt_intensity fuera de rango [0.0, 5.0]");
        }
        if !self.time_step_dt.is_finite() || self.time_step_dt <= 0.0 || self.time_step_dt > 0.5 {
            return Err("MeanFieldGamesParams: time_step_dt fuera de rango (0.0, 0.5]");
        }
        Ok(())
    }
}

/// Informe omnisciente del estado del campo medio y métricas de colusión.
#[derive(Debug, Clone, Copy)]
pub struct MeanFieldGamesReport {
    /// Índice de riesgo de estampida / aglomeración del rebaño $\kappa_{\mathrm{crowd}} \in [0.0, 1.0]$.
    pub crowd_herding_risk: f64,
    /// Probabilidad analítica de agotamiento del squeeze $P_{\mathrm{exhaustion}} \in [0.0, 1.0]$.
    pub squeeze_exhaustion_prob: f64,
    /// Posición espacial óptima del equilibrio de Nash $x^* \in [-1.0, 1.0]$.
    pub nash_equilibrium_coordinate: f64,
    /// Señal alfa anti-rebaño normalizada en $[-1.0, 1.0]$ (positiva = compra contraria / squeeze up).
    pub anti_crowd_alpha: f64,
    /// Dirección del gradiente de stop-hunting institucional: $+1.0$ (al alza), $-1.0$ (a la baja), $0.0$ (neutro).
    pub predatory_hunt_direction: f64,
    /// Centro de masa de la multitud $\bar{x} = \int x m(x) dx \in [-1.0, 1.0]$.
    pub crowd_center_of_mass: f64,
    /// Dispersión o varianza espacial de la masa $\sigma_m^2$.
    pub crowd_spatial_variance: f64,
}

/// Motor continuo de Teoría de Juegos de Campo Medio y Psicología de Masas.
#[derive(Debug, Clone)]
pub struct MeanFieldGamesPsychologyEngine {
    params: MeanFieldGamesParams,
    /// Densidad de masa continua $m(x_i)$ discretizada en 32 nodos.
    density: [f64; MFG_SPATIAL_NODES],
    /// Función de valor $v(x_i)$ del agente representativo en 32 nodos.
    value_function: [f64; MFG_SPATIAL_NODES],
    /// Coordenadas espaciales $x_i \in [-1.0, 1.0]$.
    grid_x: [f64; MFG_SPATIAL_NODES],
    /// Espaciado espacial uniforme $\Delta x$.
    dx: f64,
    has_history: bool,
    last_report: Option<MeanFieldGamesReport>,
}

impl Default for MeanFieldGamesPsychologyEngine {
    fn default() -> Self {
        Self::new(MeanFieldGamesParams::default())
    }
}

impl MeanFieldGamesPsychologyEngine {
    /// Inicializa el motor con una distribución gaussiana uniforme centrada en $x = 0$.
    pub fn new(params: MeanFieldGamesParams) -> Self {
        let n = MFG_SPATIAL_NODES;
        let mut grid_x = [0.0; MFG_SPATIAL_NODES];
        let dx = 2.0 / (n as f64 - 1.0);

        for (i, x) in grid_x.iter_mut().enumerate() {
            *x = -1.0 + i as f64 * dx;
        }

        let mut density = [0.0; MFG_SPATIAL_NODES];
        let mut sum = 0.0;
        for (i, val) in density.iter_mut().enumerate() {
            let x = grid_x[i];
            let g = (-x * x / 0.20).exp();
            *val = g;
            sum += g * dx;
        }
        if sum > 1e-12 {
            for val in density.iter_mut() {
                *val /= sum;
            }
        }

        let value_function = [0.0; MFG_SPATIAL_NODES];

        Self {
            params,
            density,
            value_function,
            grid_x,
            dx,
            has_history: false,
            last_report: None,
        }
    }

    /// Obtiene los parámetros actuales.
    pub fn params(&self) -> &MeanFieldGamesParams {
        &self.params
    }

    /// Actualiza los parámetros del motor.
    pub fn set_params(&mut self, params: MeanFieldGamesParams) -> Result<(), &'static str> {
        params.validate()?;
        self.params = params;
        Ok(())
    }

    /// Obtiene el último informe computado.
    pub fn last_report(&self) -> Option<&MeanFieldGamesReport> {
        self.last_report.as_ref()
    }

    /// Evalúa el potencial de trampa de stop-loss $\Phi_{\mathrm{trap}}(x)$
    /// centrado en los niveles típicos de parada minorista ($|x| \approx 0.60..0.85$).
    #[inline(always)]
    fn trap_potential(&self, x: f64, order_flow_bias: f64) -> f64 {
        // Zonas de liquidación asimétricas inducidas por sesgo de flujo
        let upper_trap = (-((x - 0.70) / 0.15).powi(2)).exp() * (1.0 + order_flow_bias).max(0.0);
        let lower_trap = (-((x + 0.70) / 0.15).powi(2)).exp() * (1.0 - order_flow_bias).max(0.0);
        upper_trap - lower_trap
    }

    /// Evalúa el Hamiltoniano de mercado $H(p) = \frac{p^2}{2\gamma} - c_{\mathrm{fee}} |p|$.
    #[inline(always)]
    fn hamiltonian(&self, p: f64) -> f64 {
        let p_sq = p * p;
        let fee_drag = self.params.transaction_fee * p.abs();
        (0.5 * p_sq / self.params.friction_gamma - fee_drag).max(0.0)
    }

    /// Evalúa la derivada del Hamiltoniano $\nabla_p H(p) = \frac{p}{\gamma} - c_{\mathrm{fee}} \operatorname{sgn}(p)$.
    #[inline(always)]
    fn hamiltonian_derivative(&self, p: f64) -> f64 {
        let sign = if p > 1e-6 {
            1.0
        } else if p < -1e-6 {
            -1.0
        } else {
            0.0
        };
        p / self.params.friction_gamma - self.params.transaction_fee * sign
    }

    /// Actualiza el sistema continuo de EDPs (HJB retrógrado + FPK anterógrado)
    /// integrando la información en tiempo real de desequilibrio de flujo y volatilidad.
    ///
    /// Ejecución $O(1)$ sin heap allocations ni pánicos.
    pub fn update_step(
        &mut self,
        order_flow_imbalance: f64,
        observed_volatility: f64,
        dt_seconds: f64,
    ) -> MeanFieldGamesReport {
        let n = MFG_SPATIAL_NODES;
        let dx = self.dx;
        let dx_sq = dx * dx;

        let dt = if dt_seconds.is_finite() && dt_seconds > 1e-4 {
            dt_seconds.min(self.params.time_step_dt * 2.0)
        } else {
            self.params.time_step_dt
        };

        let ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let diff_coeff = if observed_volatility.is_finite() && observed_volatility > 0.0 {
            (self.params.diffusion_coeff * (1.0 + observed_volatility * 10.0)).clamp(0.01, 0.40)
        } else {
            self.params.diffusion_coeff
        };

        // 1. Calcular gradientes espaciales de la función de valor actual: \nabla v(x_i)
        let mut grad_v = [0.0; MFG_SPATIAL_NODES];
        for (i, grad_slot) in grad_v.iter_mut().enumerate().take(n - 1).skip(1) {
            *grad_slot = (self.value_function[i + 1] - self.value_function[i - 1]) / (2.0 * dx);
        }
        grad_v[0] = (self.value_function[1] - self.value_function[0]) / dx;
        grad_v[n - 1] = (self.value_function[n - 1] - self.value_function[n - 2]) / dx;

        // 2. Velocidad óptima de deriva del rebaño: u(x_i) = \nabla_p H(\nabla v) + ofi * bias
        let mut drift_u = [0.0; MFG_SPATIAL_NODES];
        for (i, u_slot) in drift_u.iter_mut().enumerate() {
            let u_opt = self.hamiltonian_derivative(grad_v[i]);
            *u_slot = (u_opt + ofi * 0.5).clamp(-2.0, 2.0);
        }

        // 3. Paso anterógrado de Fokker-Planck-Kolmogorov (FPK) con Upwind y Neumann
        let mut next_density = self.density;
        for i in 1..n - 1 {
            let d2m = (self.density[i + 1] - 2.0 * self.density[i] + self.density[i - 1]) / dx_sq;

            // Flujo advectivo upwind
            let u = drift_u[i];
            let adv = if u > 0.0 {
                u * (self.density[i] - self.density[i - 1]) / dx
            } else {
                u * (self.density[i + 1] - self.density[i]) / dx
            };

            let dm_dt = diff_coeff * d2m - adv;
            next_density[i] = (self.density[i] + dt * dm_dt).max(1e-8);
        }

        // Condiciones de frontera Neumann (flujo cero)
        next_density[0] = next_density[1];
        next_density[n - 1] = next_density[n - 2];

        // Normalización estricta de masa: \int m(x) dx = 1
        let mut mass_sum = 0.0;
        for val in next_density.iter() {
            mass_sum += *val * dx;
        }
        if mass_sum > 1e-12 {
            for val in next_density.iter_mut() {
                *val /= mass_sum;
            }
        }
        self.density = next_density;

        // 4. Paso retrógrado de Hamilton-Jacobi-Bellman (HJB)
        let mut next_value = self.value_function;
        for i in 1..n - 1 {
            let d2v = (self.value_function[i + 1] - 2.0 * self.value_function[i]
                + self.value_function[i - 1])
                / dx_sq;
            let h = self.hamiltonian(grad_v[i]);

            // Acoplamiento psicológico f(x, m)
            let m_val = self.density[i];
            let crowding_disutility =
                self.params.crowd_penalty * m_val.powf(self.params.crowd_exponent);
            let trap = self.params.hunt_intensity * self.trap_potential(self.grid_x[i], ofi);
            let f_coupling = -crowding_disutility + trap;

            let dv_dt = -diff_coeff * d2v + h - f_coupling;
            next_value[i] = self.value_function[i] + dt * dv_dt;
        }
        next_value[0] = next_value[1];
        next_value[n - 1] = next_value[n - 2];
        self.value_function = next_value;

        // 5. Extracción de Métricas Omniscientes del Campo Medio
        // a) Centro de masa y varianza espacial
        let mut center_of_mass = 0.0;
        let mut herding_integral = 0.0;
        for (i, &dens) in self.density.iter().enumerate() {
            center_of_mass += self.grid_x[i] * dens * dx;
            herding_integral += dens * dens * dx;
        }

        let mut variance = 0.0;
        for (i, &dens) in self.density.iter().enumerate() {
            let diff = self.grid_x[i] - center_of_mass;
            variance += diff * diff * dens * dx;
        }

        // b) Riesgo de aglomeración / estampida \kappa \in [0, 1]
        // Para distribución uniforme en [-1, 1], \int m^2 dx = 0.50. Concentración máxima > 5.0.
        let crowd_herding_risk = ((herding_integral - 0.50) / 3.0).clamp(0.0, 1.0);

        // c) Probabilidad de agotamiento de squeeze P_{exhaustion}
        // Medida por la masa acumulada en las colas (|x| > 0.65)
        let mut tail_mass = 0.0;
        for (i, &dens) in self.density.iter().enumerate() {
            if self.grid_x[i].abs() > 0.65 {
                tail_mass += dens * dx;
            }
        }
        let squeeze_exhaustion_prob =
            (1.0 / (1.0 + (-12.0 * (tail_mass - 0.35)).exp())).clamp(0.0, 1.0);

        // d) Coordenada óptima esperada del equilibrio de Nash mediante softmax (Boltzmann rational inattention)
        let mut max_val = f64::NEG_INFINITY;
        for &val in self.value_function.iter() {
            if val > max_val {
                max_val = val;
            }
        }
        let beta = 4.0;
        let mut z_sum = 0.0;
        let mut x_star_sum = 0.0;
        for (i, &val) in self.value_function.iter().enumerate() {
            let w = (beta * (val - max_val)).exp();
            z_sum += w;
            x_star_sum += self.grid_x[i] * w;
        }
        let nash_equilibrium_coordinate = if z_sum > 1e-12 {
            x_star_sum / z_sum
        } else {
            0.0
        };

        // e) Dirección del gradiente depredador de paradas
        let predatory_hunt_direction = if ofi > 0.20 {
            1.0
        } else if ofi < -0.20 {
            -1.0
        } else {
            0.0
        };

        // f) Convicción alpha anti-rebaño:
        // Contracción hacia el equilibrio de Nash ponderada por el riesgo de aglomeración
        let raw_alpha = (nash_equilibrium_coordinate - center_of_mass) * (1.0 + crowd_herding_risk);
        let anti_crowd_alpha = (raw_alpha * 2.0).tanh();

        self.has_history = true;

        let report = MeanFieldGamesReport {
            crowd_herding_risk,
            squeeze_exhaustion_prob,
            nash_equilibrium_coordinate,
            anti_crowd_alpha,
            predatory_hunt_direction,
            crowd_center_of_mass: center_of_mass,
            crowd_spatial_variance: variance,
        };

        self.last_report = Some(report);
        report
    }

    /// Reinicia el motor al estado basal no sesgado.
    pub fn reset(&mut self) {
        *self = Self::new(self.params);
    }
}
