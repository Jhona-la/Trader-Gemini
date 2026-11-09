//! -------------------------------------------------------------------------------------------------
//! TRADER GEMINI — QUANTUM CONTINUOUS MANIFOLD ARCHITECTURE (OMEGA220)
//! Signal Engine: Termodinámica Estocástica de Jarzynski-Crooks y Producción de Entropía en Alta Frecuencia
//! Fase 56 (§56) — Ola 13
//!
//! Concepto Físico-Matemático:
//! A escala microscópica de alta frecuencia (< 10ms), el flujo de órdenes y la liquidez se encuentran
//! fuera del equilibrio termodinámico.
//! Aplica la Igualdad de Jarzynski $\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$ y el Teorema de
//! Fluctuación de Crooks $P_F(W) / P_R(-W) = \exp(\beta(W - \Delta F))$ para cuantificar el trabajo disipado
//! irreversible $W_{\text{diss}} = \langle W \rangle - \Delta F \ge 0$ y la tasa de producción de entropía
//! $\Sigma_{\text{entropy}} = \beta W_{\text{diss}}$.
//! Un colapso de reversibilidad temporal $\mathcal{R}_{\text{rev}} \to 0$ ($\Sigma > 2.50$) detecta shocks
//! irreversibles de liquidez (cascadas de liquidación forzada), modulando la convicción en $O(1)$ stack.
//! -------------------------------------------------------------------------------------------------

/// Capacidad del buffer circular de trayectorias microscópicas de trabajo disipado.
pub const JARZYNSKI_WINDOW_CAPACITY: usize = 32;

/// Constante de Boltzmann efectiva del libro de órdenes ($k_B = 1.0$ en unidades normalizadas).
pub const DEFAULT_KB_BOLTZMANN: f64 = 1.0;

/// Umbral crítico de producción de entropía microscópica para colapso irreversible.
pub const CRITICAL_ENTROPY_PRODUCTION: f64 = 2.50;

/// Muestra de trabajo microscópico hacia adelante y reverso en el espacio de fases del libro.
#[derive(Debug, Clone, Copy, Default)]
pub struct JarzynskiTrajectorySample {
    pub work_forward: f64,
    pub work_reverse: f64,
    pub timestamp_ms: u64,
}

/// Parámetros de configuración del motor de termodinámica estocástica Jarzynski-Crooks.
#[derive(Debug, Clone, Copy)]
pub struct JarzynskiCrooksParams {
    /// Escala de temperatura efectiva del libro $T_0$ (típicamente 1.0).
    pub effective_temp_scale: f64,
    /// Umbral crítico de producción de entropía para alerta de irreversibilidad.
    pub critical_entropy_threshold: f64,
    /// Tasa de relajación temporal hacia el equilibrio $\gamma_{\text{relax}}$.
    pub relaxation_rate: f64,
    /// Mínimo de muestras requeridas en el buffer para evaluación estadística confiable.
    pub min_samples: usize,
    /// Parámetro de regularización numérica epsilon.
    pub epsilon: f64,
}

impl Default for JarzynskiCrooksParams {
    fn default() -> Self {
        Self {
            effective_temp_scale: 1.0,
            critical_entropy_threshold: CRITICAL_ENTROPY_PRODUCTION,
            relaxation_rate: 0.10,
            min_samples: 4,
            epsilon: 1e-9,
        }
    }
}

/// Informe omnisciente de termodinámica estocástica fuera del equilibrio y producción de entropía.
#[derive(Debug, Clone, Copy)]
pub struct JarzynskiCrooksReport {
    /// Diferencia de energía libre de Helmholtz $\Delta F = -\frac{1}{\beta} \ln \langle e^{-\beta W} \rangle$.
    pub free_energy_delta: f64,
    /// Trabajo medio realizado sobre el libro $\langle W \rangle$.
    pub mean_work: f64,
    /// Trabajo disipado irreversible $W_{\text{diss}} = \langle W \rangle - \Delta F \ge 0$.
    pub dissipated_work: f64,
    /// Tasa de producción de entropía estocástica $\Sigma_{\text{entropy}} = \beta W_{\text{diss}} \ge 0$.
    pub entropy_production: f64,
    /// Índice de reversibilidad temporal $\mathcal{R}_{\text{rev}} = \exp(-\Sigma_{\text{entropy}}) \in [0.0, 1.0]$.
    pub reversibility_index: f64,
    /// Razón de fluctuación de Crooks $P_F(W) / P_R(-W) = \exp(\beta(W - \Delta F))$.
    pub crooks_fluctuation_ratio: f64,
    /// Alerta de régimen irreversible violento (shock de liquidez o absorción unilateral).
    pub irreversible_entropy_collapse: bool,
    /// Modulador de convicción de señal derivado $\lambda_{\text{jc}} \in [0.05, 1.35]$.
    pub signal_conviction_modulator: f64,
    /// Temperatura efectiva del libro $T_{\text{eff}}$.
    pub effective_temperature: f64,
    /// Timestamp del evento procesado.
    pub event_time_ms: u64,
}

/// Motor de Termodinámica Estocástica de Jarzynski-Crooks y Producción de Entropía.
#[derive(Debug, Clone)]
pub struct JarzynskiCrooksThermodynamicsEngine {
    params: JarzynskiCrooksParams,
    samples: [JarzynskiTrajectorySample; JARZYNSKI_WINDOW_CAPACITY],
    sample_idx: usize,
    sample_count: usize,
    prev_ofi: f64,
    last_event_time_ms: u64,
}

impl JarzynskiCrooksThermodynamicsEngine {
    /// Inicializa una nueva instancia del motor con los parámetros dados.
    pub fn new(params: JarzynskiCrooksParams) -> Self {
        Self {
            params,
            samples: [JarzynskiTrajectorySample::default(); JARZYNSKI_WINDOW_CAPACITY],
            sample_idx: 0,
            sample_count: 0,
            prev_ofi: 0.0,
            last_event_time_ms: 0,
        }
    }

    /// Actualiza el estado termodinámico del libro y evalúa la producción de entropía en $O(1)$ stack.
    pub fn update_and_evaluate(
        &mut self,
        price_ret: f64,
        order_flow_imbalance: f64,
        spread_bps: f64,
        volatility: f64,
        event_time_ms: u64,
    ) -> JarzynskiCrooksReport {
        self.last_event_time_ms = event_time_ms;

        // 1. Temperatura efectiva del libro $T_{\text{eff}} = T_0 \cdot \sqrt{\text{spread}} \cdot (1 + \sigma)$
        let spread_factor = (spread_bps * 0.0001).max(0.01).sqrt();
        let vol_factor = 1.0 + (volatility * 10.0).clamp(0.0, 10.0);
        let effective_temperature = (self.params.effective_temp_scale * spread_factor * vol_factor)
            .max(self.params.epsilon);

        // Parámetro termodinámico inverso $\beta = 1 / (k_B T_{\text{eff}})$
        let beta = 1.0 / (DEFAULT_KB_BOLTZMANN * effective_temperature);

        // 2. Trabajo mecánico infinitesimal realizado por el flujo agresivo sobre el libro
        // $W = \text{OFI} \cdot \Delta p + \frac{1}{2} \text{spread} \cdot |\text{OFI}|$
        let mechanical_work = order_flow_imbalance * (price_ret * 100.0)
            + 0.5 * (spread_bps * 0.0001) * order_flow_imbalance.abs();

        // Trabajo reverso simulado (inversión temporal del flujo bajo paridad P-T)
        let delta_ofi = order_flow_imbalance - self.prev_ofi;
        let reverse_work = -order_flow_imbalance * (price_ret * 100.0)
            + 0.5 * (spread_bps * 0.0001) * delta_ofi.abs();

        self.prev_ofi = order_flow_imbalance;

        // 3. Almacenamiento en el buffer circular
        self.samples[self.sample_idx] = JarzynskiTrajectorySample {
            work_forward: mechanical_work,
            work_reverse: reverse_work,
            timestamp_ms: event_time_ms,
        };
        self.sample_idx = (self.sample_idx + 1) % JARZYNSKI_WINDOW_CAPACITY;
        if self.sample_count < JARZYNSKI_WINDOW_CAPACITY {
            self.sample_count += 1;
        }

        // 4. Evaluación de la Igualdad de Jarzynski: $\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$
        let mut sum_exp_neg_beta_w = 0.0;
        let mut sum_work = 0.0;
        let n = self.sample_count.max(1);

        for i in 0..self.sample_count {
            let w = self.samples[i].work_forward;
            sum_work += w;
            // Clamping del exponente para evitar desbordamiento numérico
            let exp_arg = (-beta * w).clamp(-50.0, 50.0);
            sum_exp_neg_beta_w += exp_arg.exp();
        }

        let mean_exp_w = (sum_exp_neg_beta_w / (n as f64)).max(self.params.epsilon);
        let free_energy_delta = -(1.0 / beta) * mean_exp_w.ln();
        let mean_work = sum_work / (n as f64);

        // 5. Segunda Ley de la Termodinámica Estocástica: $W_{\text{diss}} = \langle W \rangle - \Delta F \ge 0$
        let dissipated_work = (mean_work - free_energy_delta).max(0.0);

        // 6. Producción de Entropía Microscópica: $\Sigma_{\text{entropy}} = \beta W_{\text{diss}} \ge 0$
        let entropy_production = (beta * dissipated_work).clamp(0.0, 50.0);

        // 7. Índice de Reversibilidad Temporal $\mathcal{R}_{\text{rev}} = \exp(-\Sigma_{\text{entropy}}) \in [0.0, 1.0]$
        let reversibility_index = (-entropy_production).exp().clamp(0.0, 1.0);

        // 8. Razón de Crooks $P_F(W) / P_R(-W) = \exp(\beta(W - \Delta F))$
        let crooks_exponent = (beta * (mechanical_work - free_energy_delta)).clamp(-50.0, 50.0);
        let crooks_fluctuation_ratio = crooks_exponent.exp();

        // 9. Detección de Colapso Irreversible de Entropía
        let irreversible_entropy_collapse =
            entropy_production >= self.params.critical_entropy_threshold;

        // 10. Modulador de Convicción $\lambda_{\text{jc}} \in [0.05, 1.35]$
        let signal_conviction_modulator = if irreversible_entropy_collapse {
            0.05
        } else {
            let base = 1.0 + 0.35 * (reversibility_index - 0.5);
            base.clamp(0.05, 1.35)
        };

        JarzynskiCrooksReport {
            free_energy_delta,
            mean_work,
            dissipated_work,
            entropy_production,
            reversibility_index,
            crooks_fluctuation_ratio,
            irreversible_entropy_collapse,
            signal_conviction_modulator,
            effective_temperature,
            event_time_ms,
        }
    }
}
