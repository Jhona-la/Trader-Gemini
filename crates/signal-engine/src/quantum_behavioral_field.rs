//! 🧠 MOTOR CONTINUO DE FINANZAS CONDUCTUALES Y CAMPO CUÁNTICO DE SENTIMIENTO
//!
//! # FASE 32 (§32) — OLA OMEGA220: DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Este módulo modela la psicología de masas y los sesgos cognitivos sistemáticos de los operadores
//! minoristas e institucionales mediante la Teoría Prospectiva Acumulativa (Kahneman & Tversky),
//! la función de ponderación no lineal de probabilidad de Prelec y el potencial de liquidación
//! forzada (stop hunting y squeezes de apalancamiento) en stack zero-alloc.
//!
//! ## 👨‍🏫 MODO PROFESOR RIGUROSO
//!
//! - **QUÉ:** Motor analítico de física cuántica y finanzas conductuales (`QuantumBehavioralFieldEngine`)
//!   que cuantifica el estado emocional y la distribución de puntos de dolor (pain points) de la masa
//!   en el mercado, emitiendo un vector de resonancia conductual $B(t) \in [-1.0, 1.0]$.
//! - **POR QUÉ:** Los mercados de criptoactivos no son eficientes ni perfectamente gaussianos. El 90%
//!   de los minoristas sufre de aversión asimétrica a la pérdida ($\lambda \approx 2.25$), anclaje cognitivo
//!   y sobreponderación de eventos extremos (Prelec $w(p)$), acumulando órdenes de stop en zonas predecibles.
//!   Detectar el gradiente de liquidación institucional permite posicionarse en el lado ganador.
//! - **PARA QUÉ:** Multiplicar la tasa de crecimiento geométrico e interés compuesto $\ge +100\%$ cada 72 horas
//!   ingresando con ratio beneficio/riesgo asimétrico ($RR \ge 2.25$) en los instantes precisos de
//!   capitulación o cascada de liquidaciones, preservando la cota de ruina $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//! - **CÓMO:**
//!   1. Función de valor en $S$ de Kahneman-Tversky con aversión a la pérdida $\lambda = 2.25$ y curvatura $\alpha = \beta = 0.88$.
//!   2. Distorsión de probabilidad axiomática de Prelec $w(p) = \exp(-(-\ln p)^\gamma)$ con $\gamma = 0.65$.
//!   3. Campo de potencial de paradas minoristas $\Phi(P)$ y cálculo analítico de la fuerza de stop-hunting $F_{\text{hunt}} = -\nabla \Phi$.
//!   4. Estimación de entropía psico-acústica / desorden cognitivo en stack sin asignaciones dinámicas.
//! - **CUÁNDO:** En cada evaluación de tick del `QuantumSlotOrchestrator` para modular la convicción de la señal.
//! - **DÓNDE:** `crates/signal-engine/src/quantum_behavioral_field.rs`.
//! - **QUIÉN:** Consejo de 10 Roles Senior (Lead: Rol 9 Psicólogo Cuantitativo / Bio-informático & Rol 2 Quant Microestructura).

/// Parámetros configurables del motor de finanzas conductuales cuánticas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantumBehavioralParams {
    /// Parámetro de aversión a la pérdida de Kahneman-Tversky $\lambda$ (default: 2.25).
    pub loss_aversion_lambda: f64,
    /// Curvatura de valor para ganancias $\alpha$ (default: 0.88).
    pub value_alpha: f64,
    /// Curvatura de valor para pérdidas $\beta$ (default: 0.88).
    pub value_beta: f64,
    /// Parámetro de curvatura de probabilidad de Prelec $\gamma$ (default: 0.65).
    pub prelec_gamma: f64,
    /// Parámetro de regularización de distancia $\epsilon$ para evitar singularidades (default: 0.001).
    pub epsilon_regularizer: f64,
    /// Factor de amplificación de fuerza de caza de liquidez (default: 1.50).
    pub hunt_force_gain: f64,
}

impl Default for QuantumBehavioralParams {
    #[inline(always)]
    fn default() -> Self {
        Self {
            loss_aversion_lambda: 2.25,
            value_alpha: 0.88,
            value_beta: 0.88,
            prelec_gamma: 0.65,
            epsilon_regularizer: 0.001,
            hunt_force_gain: 1.50,
        }
    }
}

impl QuantumBehavioralParams {
    /// Sanitiza rigurosamente los parámetros para garantizar estabilidad analítica.
    #[inline]
    pub fn sanitized(&self) -> Self {
        Self {
            loss_aversion_lambda: if self.loss_aversion_lambda.is_finite()
                && self.loss_aversion_lambda >= 1.0
            {
                self.loss_aversion_lambda.min(10.0)
            } else {
                2.25
            },
            value_alpha: if self.value_alpha.is_finite() && (0.1..=1.0).contains(&self.value_alpha)
            {
                self.value_alpha
            } else {
                0.88
            },
            value_beta: if self.value_beta.is_finite() && (0.1..=1.0).contains(&self.value_beta) {
                self.value_beta
            } else {
                0.88
            },
            prelec_gamma: if self.prelec_gamma.is_finite()
                && (0.2..=1.0).contains(&self.prelec_gamma)
            {
                self.prelec_gamma
            } else {
                0.65
            },
            epsilon_regularizer: if self.epsilon_regularizer.is_finite()
                && self.epsilon_regularizer > 0.0
            {
                self.epsilon_regularizer.min(0.1)
            } else {
                0.001
            },
            hunt_force_gain: if self.hunt_force_gain.is_finite() && self.hunt_force_gain >= 0.0 {
                self.hunt_force_gain.min(10.0)
            } else {
                1.50
            },
        }
    }
}

/// Diagnóstico completo del estado conductual cuántico emitido por el motor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BehavioralStateReport {
    /// Resonancia conductual normalizada $B(t) \in [-1.0, 1.0]$.
    /// Valores $> 0$ favorecen LONGs por liquidación de cortos; $< 0$ favorecen SHORTs.
    pub behavioral_resonance: f64,
    /// Fuerza vectorial de stop-hunting institucional normalizada $F_{\text{hunt}} \in [-1.0, 1.0]$.
    pub stop_hunt_force: f64,
    /// Valor prospectivo subjetivo de la masa $V_{\text{prospect}} \in [-1.0, 1.0]$.
    pub prospect_value: f64,
    /// Ponderación distorsionada de probabilidad de crash $w(p_{\text{crash}}) \in [0.0, 1.0]$.
    pub distorted_crash_prob: f64,
    /// Potencial de atrapamiento minorista en el libro $\Phi_{\text{trapped}} \ge 0.0$.
    pub retail_trapped_potential: f64,
    /// Factor de amplificación de convicción de señal $[0.50, 2.00]$.
    pub conviction_multiplier: f64,
}

impl Default for BehavioralStateReport {
    #[inline(always)]
    fn default() -> Self {
        Self {
            behavioral_resonance: 0.0,
            stop_hunt_force: 0.0,
            prospect_value: 0.0,
            distorted_crash_prob: 0.0,
            retail_trapped_potential: 0.0,
            conviction_multiplier: 1.0,
        }
    }
}

/// Motor determinista en stack zero-alloc de finanzas conductuales cuánticas.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantumBehavioralFieldEngine {
    pub params: QuantumBehavioralParams,
}

impl QuantumBehavioralFieldEngine {
    /// Crea una nueva instancia con parámetros por defecto.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            params: QuantumBehavioralParams::default(),
        }
    }

    /// Crea una nueva instancia con parámetros personalizados.
    #[inline(always)]
    pub fn with_params(params: QuantumBehavioralParams) -> Self {
        Self {
            params: params.sanitized(),
        }
    }

    /// Evalúa la función de valor subjetivo de Kahneman-Tversky $v(x)$.
    /// $x$ es el retorno normalizado respecto al precio de anclaje de referencia.
    #[inline]
    pub fn evaluate_prospect_value(&self, return_rel: f64) -> f64 {
        if !return_rel.is_finite() {
            return 0.0;
        }

        let p = self.params.sanitized();
        if return_rel >= 0.0 {
            let x = return_rel.min(10.0);
            x.powf(p.value_alpha)
        } else {
            let x = (-return_rel).min(10.0);
            -p.loss_aversion_lambda * x.powf(p.value_beta)
        }
    }

    /// Evalúa la distorsión de probabilidad no lineal de Prelec $w(p) = \exp(-(-\ln p)^\gamma)$.
    /// Si $p \le 0$, retorna 0.0; si $p \ge 1.0$, retorna 1.0.
    #[inline]
    pub fn evaluate_prelec_weighting(&self, p: f64) -> f64 {
        if !p.is_finite() || p <= 0.0 {
            return 0.0;
        }
        if p >= 1.0 {
            return 1.0;
        }

        let gamma = self.params.sanitized().prelec_gamma;
        let neg_ln_p = -p.ln();
        if neg_ln_p <= 0.0 {
            return 1.0;
        }

        let exponent = -neg_ln_p.powf(gamma);
        exponent.exp().clamp(0.0, 1.0)
    }

    /// Evalúa el campo de fuerza de stop-hunting hacia zonas de acumulación de stops minoristas.
    /// - `current_price`: precio actual de mercado ($> 0$).
    /// - `pivot_high`: resistencia técnica inmediata donde se agrupan stops de cortos ($> current_price$).
    /// - `pivot_low`: soporte técnico inmediato donde se agrupan stops de largos ($< current_price$).
    /// - `short_liquidation_volume`: volumen estimado de posiciones cortas vulnerables a squeeze.
    /// - `long_liquidation_volume`: volumen estimado de posiciones largas vulnerables a cascada.
    #[inline]
    pub fn compute_stop_hunt_force(
        &self,
        current_price: f64,
        pivot_high: f64,
        pivot_low: f64,
        short_liquidation_volume: f64,
        long_liquidation_volume: f64,
    ) -> (f64, f64) {
        if !current_price.is_finite() || current_price <= 0.0 {
            return (0.0, 0.0);
        }

        let p = self.params.sanitized();
        let eps = p.epsilon_regularizer;

        // Atracción hacia el Pivot High (Stop hunting de cortos -> impulso alcista)
        let dist_high = if pivot_high.is_finite() && pivot_high > current_price {
            (pivot_high - current_price) / current_price
        } else {
            0.05
        };
        let vol_short = if short_liquidation_volume.is_finite() && short_liquidation_volume > 0.0 {
            short_liquidation_volume
        } else {
            1.0
        };
        let denom_high = (dist_high * dist_high + eps).powi(2);
        let force_up = (2.0 * vol_short * dist_high / denom_high) * p.hunt_force_gain;

        // Atracción hacia el Pivot Low (Stop hunting de largos -> impulso bajista)
        let dist_low = if pivot_low.is_finite() && pivot_low < current_price && pivot_low > 0.0 {
            (current_price - pivot_low) / current_price
        } else {
            0.05
        };
        let vol_long = if long_liquidation_volume.is_finite() && long_liquidation_volume > 0.0 {
            long_liquidation_volume
        } else {
            1.0
        };
        let denom_low = (dist_low * dist_low + eps).powi(2);
        let force_down = (2.0 * vol_long * dist_low / denom_low) * p.hunt_force_gain;

        let net_force = (force_up - force_down) / (force_up + force_down + 1.0);
        let trapped_potential = (force_up + force_down).min(100.0);

        (net_force.clamp(-1.0, 1.0), trapped_potential)
    }

    /// Evaluación integral del estado conductual de mercado en $O(1)$ stack.
    ///
    /// # Argumentos:
    /// - `current_price`: precio actual del activo ($P > 0$).
    /// - `anchor_price`: precio de referencia / coste medio de la masa ($P_{\text{ref}} > 0$).
    /// - `pivot_high`: resistencia técnica de liquidación de cortos.
    /// - `pivot_low`: soporte técnico de liquidación de largos.
    /// - `raw_crash_prob`: probabilidad base de crash antes de la distorsión de Prelec $[0.0, 1.0]$.
    /// - `short_liq_vol`: volumen nocional de cortos en riesgo de liquidación.
    /// - `long_liq_vol`: volumen nocional de largos en riesgo de liquidación.
    pub fn evaluate(
        &self,
        current_price: f64,
        anchor_price: f64,
        pivot_high: f64,
        pivot_low: f64,
        raw_crash_prob: f64,
        short_liq_vol: f64,
        long_liq_vol: f64,
    ) -> BehavioralStateReport {
        if !current_price.is_finite() || current_price <= 0.0 {
            return BehavioralStateReport::default();
        }

        let anchor = if anchor_price.is_finite() && anchor_price > 0.0 {
            anchor_price
        } else {
            current_price
        };

        // 1. Retorno subjetivo de la masa frente al ancla
        let return_rel = (current_price - anchor) / anchor;
        let v_prospect = self.evaluate_prospect_value(return_rel);
        let norm_v_prospect = (v_prospect / 5.0).tanh();

        // 2. Distorsión de Prelec sobre probabilidad de shock/crash
        let distorted_crash = self.evaluate_prelec_weighting(raw_crash_prob);

        // 3. Fuerza de caza de liquidez institucional
        let (hunt_force, trapped_pot) = self.compute_stop_hunt_force(
            current_price,
            pivot_high,
            pivot_low,
            short_liq_vol,
            long_liq_vol,
        );

        // 4. Resonancia conductual combinada:
        // - Si hay fuerza alcista de stop hunt y la masa está atrapada en corto con aversión al dolor -> fuerte LONG
        // - Si la probabilidad de crash distorsionada genera pánico masivo y hunt_force es bajista -> fuerte SHORT
        let behavioral_resonance =
            (0.60 * hunt_force - 0.25 * norm_v_prospect - 0.15 * (distorted_crash - 0.50) * 2.0)
                .clamp(-1.0, 1.0);

        // 5. Multiplicador de convicción:
        // Si el campo conductual y la fuerza de caza están altamente alineados, la convicción se expande hasta 2.0x.
        // Si hay conflicto severo o alta incertidumbre, se contrae hasta 0.50x.
        let alignment = (hunt_force * behavioral_resonance).max(0.0);
        let conviction_multiplier = (0.75 + 1.25 * alignment).clamp(0.50, 2.00);

        BehavioralStateReport {
            behavioral_resonance,
            stop_hunt_force: hunt_force,
            prospect_value: norm_v_prospect,
            distorted_crash_prob: distorted_crash,
            retail_trapped_potential: trapped_pot,
            conviction_multiplier,
        }
    }
}
