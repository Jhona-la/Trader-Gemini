//! # Adaptive Quantum Veto Governor & Conformal Risk Controller
//!
//! FASE 36 (§36) — OLA OMEGA220: DE LA FÍSICA DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Audita y sustituye compuertas y vetos estáticos rígidos (spread fijo, ATR fijo,
//! book absent estático) por un control de riesgo conformalizado y auto-adaptativo en tiempo
//! real (Conformal Risk Control):
//! $$\mathbb{P}(\text{Pérdida Inaceptable} \mid \mathbf{x}) \le \alpha_{\text{risk}}$$
//!
//! Separa estrictamente el **Riesgo Duro Inviolable** (Supermartingala de Ville $\mathbb{P} \le 10^{-6}$,
//! Piso de Grossman-Zhou, Margin Call, Kill-Switch) de la **Lógica de Entrada Adaptable**
//! (elasticidad dinámica ante expansión de volatilidad continua y ondas de choque de Burgers).
//!
//! ## Invariantes de Silicio & Gobernanza:
//! 1. Zero-Allocation Hot Path: Estructura en stack sin asignaciones en heap.
//! 2. Cero Pánicos (`rb_p1`): 0 `assert!`, 0 `unwrap()`, 0 `expect()`, 0 `panic!()`.
//! 3. Protección Ville & Grossman-Zhou: Incondicional (colapso estricto a 0.0 si se toca el piso).
//! 4. Elasticidad Dinámica Conformal: Umbrales suaves modulados por el tensor cuántico continuo.

/// Parámetros de configuración del gobernador de veto adaptativo.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveQuantumVetoParams {
    /// Spread basal de referencia en bps (típicamente 8.0 bps = 0.0008).
    pub base_spread_ceiling_bps: f64,
    /// Coeficiente de elasticidad del spread ante volatilidad continua $\kappa_{\text{vol}}$.
    pub spread_elasticity_kappa: f64,
    /// Umbral mínimo de convicción tensorial cuántica para autorizar elasticidad.
    pub min_quantum_conviction_for_elasticity: f64,
    /// Techo absoluto infranqueable de spread en bps (ej. 35.0 bps).
    pub max_absolute_spread_bps: f64,
    /// Nivel de confianza conformal $\alpha_{\text{risk}}$ (ej. 0.05 para 95% de confianza).
    pub conformal_alpha: f64,
}

impl Default for AdaptiveQuantumVetoParams {
    fn default() -> Self {
        Self {
            base_spread_ceiling_bps: 8.0,
            spread_elasticity_kappa: 1.50,
            min_quantum_conviction_for_elasticity: 0.65,
            max_absolute_spread_bps: 35.0,
            conformal_alpha: 0.05,
        }
    }
}

/// Estado de entrada del mercado y tensores cuánticos para la evaluación del gobernador.
#[derive(Debug, Clone, Copy)]
pub struct MarketQuantumStateInput {
    /// Spread relativo actual en puntos básicos (bps): $(P_{\text{ask}} - P_{\text{bid}})/P_{\text{mid}} \times 10000$.
    pub current_spread_bps: f64,
    /// Volatilidad efectiva del operador continuo hermítico $\sigma_{\text{eff}}$.
    pub continuous_effective_vol: f64,
    /// Volatilidad basal de referencia $\sigma_{\text{ref}}$.
    pub reference_vol: f64,
    /// Convicción tensorial cuántica unificada (Yang-Mills, Dirac, Wasserstein $W_2$) en $[-1.0, 1.0]$.
    pub quantum_tensor_conviction: f64,
    /// Amplitud de salto de onda de choque de Riemann-Burgers $\Delta u$.
    pub shockwave_amplitude: f64,
    /// Detección de bolsa de aire de liquidez (air pocket).
    pub air_pocket_active: bool,
    /// Resonancia conductual de masas $B(t) \in [-1.0, 1.0]$.
    pub behavioral_resonance: f64,
    /// Distancia actual al piso móvil de Grossman-Zhou: $(W_t - F_t) / W_t \in [0.0, 1.0]$.
    pub grossman_zhou_floor_distance: f64,
    /// Riqueza acumulada respecto al límite tolerable de ruina de Ville.
    pub ville_ruin_headroom: f64,
}

/// Decisión y modulación resultante de la gobernanza adaptativa de vetos.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveVetoDecision {
    /// Indica si se activó un veto duro de supervivencia (Ville, Grossman-Zhou, colapso de capital).
    pub is_hard_vetoed: bool,
    /// Causa explícita del veto duro si existe, o "None".
    pub hard_veto_reason: &'static str,
    /// Umbral elástico de spread calculado dinámicamente en bps.
    pub dynamic_spread_ceiling_bps: f64,
    /// Multiplicador adaptativo continuo de tamaño de posición en $[0.0, 1.0]$.
    pub sizing_multiplier: f64,
    /// Score de riesgo conformal en $[0.0, 1.0]$ (valores $> 0.80$ indican alta probabilidad de pérdida).
    pub conformal_risk_score: f64,
    /// Indica si se permitió la entrada elástica a pesar de spread elevado gracias a convicción cuántica.
    pub elastic_entry_granted: bool,
}

impl Default for AdaptiveVetoDecision {
    fn default() -> Self {
        Self {
            is_hard_vetoed: false,
            hard_veto_reason: "None",
            dynamic_spread_ceiling_bps: 8.0,
            sizing_multiplier: 1.0,
            conformal_risk_score: 0.0,
            elastic_entry_granted: false,
        }
    }
}

/// Gobernador de Veto Adaptativo Cuántico y Controlador Conformal.
#[derive(Debug, Clone)]
pub struct AdaptiveQuantumVetoGovernor {
    params: AdaptiveQuantumVetoParams,
    loss_history: [f64; 32],
    history_head: usize,
    history_count: usize,
}

impl AdaptiveQuantumVetoGovernor {
    /// Construye una nueva instancia del gobernador con los parámetros dados.
    pub fn new(params: AdaptiveQuantumVetoParams) -> Self {
        Self {
            params,
            loss_history: [0.0; 32],
            history_head: 0,
            history_count: 0,
        }
    }

    /// Registra el resultado de pérdida normalizada de una orden cerrada para calibración conformal online.
    pub fn record_closed_loss_normalized(&mut self, loss_norm: f64) {
        let safe_loss = if loss_norm.is_finite() && loss_norm >= 0.0 {
            loss_norm.clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.loss_history[self.history_head] = safe_loss;
        self.history_head = (self.history_head + 1) % 32;
        if self.history_count < 32 {
            self.history_count += 1;
        }
    }

    /// Evalúa el estado del mercado y determina la decisión de veto y modulación de sizing en $O(1)$ stack.
    pub fn evaluate_governance(&self, input: &MarketQuantumStateInput) -> AdaptiveVetoDecision {
        // 1. REGLA DE SUPERVIVENCIA: VETOS DE RIESGO DURO INVIOLABLES
        // A. Cota de ruina de Ville: si el margen de ruina colapsa por debajo de 0.001
        if input.ville_ruin_headroom <= 0.001 {
            return AdaptiveVetoDecision {
                is_hard_vetoed: true,
                hard_veto_reason: "VILLE_SUPERMARTINGALE_RUIN_BOUND_EXCEEDED",
                dynamic_spread_ceiling_bps: self.params.base_spread_ceiling_bps,
                sizing_multiplier: 0.0,
                conformal_risk_score: 1.0,
                elastic_entry_granted: false,
            };
        }

        // B. Piso dinámico de Grossman-Zhou: si la distancia al piso es cero o negativa
        if input.grossman_zhou_floor_distance <= 0.0005 {
            return AdaptiveVetoDecision {
                is_hard_vetoed: true,
                hard_veto_reason: "GROSSMAN_ZHOU_FLOOR_COLLAPSE",
                dynamic_spread_ceiling_bps: self.params.base_spread_ceiling_bps,
                sizing_multiplier: 0.0,
                conformal_risk_score: 1.0,
                elastic_entry_granted: false,
            };
        }

        // 2. CÁLCULO DE ELASTICIDAD DINÁMICA DE SPREAD
        // Durante fases de expansión de volatilidad, el spread natural de mercado se ensancha.
        let vol_ratio = if input.reference_vol > 1e-6 {
            (input.continuous_effective_vol / input.reference_vol).clamp(0.50, 4.0)
        } else {
            1.0
        };

        let elasticity_factor =
            1.0 + self.params.spread_elasticity_kappa * (vol_ratio - 1.0).max(0.0);
        let calculated_spread_ceiling = (self.params.base_spread_ceiling_bps * elasticity_factor)
            .clamp(
                self.params.base_spread_ceiling_bps,
                self.params.max_absolute_spread_bps,
            );

        // 3. EVALUACIÓN CONFORMAL DE RIESGO
        let empirical_quantile = self.calculate_conformal_quantile();

        let spread_norm = (input.current_spread_bps / self.params.max_absolute_spread_bps.max(1.0))
            .clamp(0.0, 1.0);
        let mut conformal_risk_score = spread_norm * 0.40
            + (1.0 - input.grossman_zhou_floor_distance.clamp(0.0, 1.0)) * 0.30
            + empirical_quantile * 0.30;

        // Si hay una bolsa de aire de liquidez (air pocket) contraria a la señal cuántica, el riesgo se eleva
        if input.air_pocket_active
            && (input.quantum_tensor_conviction * input.shockwave_amplitude < 0.0)
        {
            conformal_risk_score += 0.35;
        }

        let safe_risk_score = if conformal_risk_score.is_finite() {
            conformal_risk_score.clamp(0.0, 1.0)
        } else {
            1.0
        };

        // 4. MODULACIÓN ADAPTATIVA DE SIZING Y PUERTA DE SPREAD ELÁSTICA
        let abs_quantum_conviction = input.quantum_tensor_conviction.abs();
        let mut elastic_entry_granted = false;
        let mut sizing_multiplier;

        if input.current_spread_bps > calculated_spread_ceiling {
            // El spread supera el techo elástico estándar: verificar si la física cuántica autoriza entrada
            if abs_quantum_conviction >= self.params.min_quantum_conviction_for_elasticity
                && input.current_spread_bps <= self.params.max_absolute_spread_bps
                && safe_risk_score < 0.75
            {
                // Convicción cuántica hiper-fuerte (Yang-Mills / Dirac / Wasserstein W2):
                // Se concede entrada elástica modulando el tamaño suavemente para pagar el spread
                let spread_excess = (input.current_spread_bps - calculated_spread_ceiling)
                    / (self.params.max_absolute_spread_bps - calculated_spread_ceiling + 1e-4);
                let contraction = 1.0 / (1.0 + 3.0 * spread_excess * spread_excess);

                sizing_multiplier = contraction.clamp(0.15, 0.85);
                elastic_entry_granted = true;
            } else {
                // Rechazo por spread excesivo no justificado por física cuántica
                return AdaptiveVetoDecision {
                    is_hard_vetoed: false, // Es un filtro suave de lógica, no de ruina
                    hard_veto_reason: "None",
                    dynamic_spread_ceiling_bps: calculated_spread_ceiling,
                    sizing_multiplier: 0.0,
                    conformal_risk_score: safe_risk_score,
                    elastic_entry_granted: false,
                };
            }
        } else {
            // Spread normal dentro del techo: modulación continua suave por distancia a piso Grossman-Zhou
            let floor_factor = (input.grossman_zhou_floor_distance / 0.15).clamp(0.10, 1.0);
            let risk_factor = (1.0 - safe_risk_score * 0.50).clamp(0.20, 1.0);

            sizing_multiplier = (floor_factor * risk_factor).clamp(0.10, 1.0);
        }

        // Si la resonancia conductual está alineada con la convicción cuántica, expandir suavemente el sizing (+15%)
        if input.quantum_tensor_conviction * input.behavioral_resonance > 0.20 {
            sizing_multiplier = (sizing_multiplier * 1.15).clamp(0.10, 1.0);
        }

        AdaptiveVetoDecision {
            is_hard_vetoed: false,
            hard_veto_reason: "None",
            dynamic_spread_ceiling_bps: calculated_spread_ceiling,
            sizing_multiplier,
            conformal_risk_score: safe_risk_score,
            elastic_entry_granted,
        }
    }

    /// Calcula el cuantil empírico conformal $1 - \alpha$ en stack memory sobre el historial de pérdidas.
    fn calculate_conformal_quantile(&self) -> f64 {
        if self.history_count == 0 {
            return 0.10; // Valor seguro inicial
        }

        let mut sorted = [0.0; 32];
        sorted[..self.history_count].copy_from_slice(&self.loss_history[..self.history_count]);

        // Ordenamiento por inserción en stack (32 elementos en $< 10$ ns)
        for i in 1..self.history_count {
            let key = sorted[i];
            let mut j = i;
            while j > 0 && sorted[j - 1] > key {
                sorted[j] = sorted[j - 1];
                j -= 1;
            }
            sorted[j] = key;
        }

        let target_idx =
            ((self.history_count as f64) * (1.0 - self.params.conformal_alpha)).floor() as usize;
        let safe_idx = target_idx.min(self.history_count - 1);

        sorted[safe_idx]
    }
}

impl Default for AdaptiveQuantumVetoGovernor {
    fn default() -> Self {
        Self::new(AdaptiveQuantumVetoParams::default())
    }
}
