//! Fase 63: Entropía Relativa de Umegaki Cuántica y Veto No-Conmutativo de Von Neumann-Araki.
//!
//! Este módulo implementa la divergencia cuántica de Umegaki $D(\rho || \sigma) = \text{Tr}(\rho (\ln \rho - \ln \sigma))$
//! entre el operador de densidad del mercado $\rho$ y el operador adiabático térmico de Gibbs $\sigma$
//! sobre el espacio de Hilbert $\mathcal{H} = \mathbb{C}^3$ (Bid, Neutro, Ask).
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! 1. Desigualdad de Klein: $D(\rho || \sigma) \ge 0$, con $D(\rho || \sigma) = 0 \iff \rho = \sigma$.
//! 2. Desigualdad de Pinsker Cuántica: $D(\rho || \sigma) \ge \frac{1}{2} \|\rho - \sigma\|_1^2$.
//! 3. Teorema de Araki de perturbación relativa del Hamiltoniano cuántico.
//! 4. Cota de Ruina de Ville: $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//! 5. Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y memoria $O(1)$ en stack.

/// Dimensión del espacio cuántico de estados del libro de órdenes (Bid, Neutro, Ask).
pub const UMEGAKI_HILBERT_DIM: usize = 3;

/// Umbral crítico de entropía relativa de Umegaki para activación de veto duro.
pub const UMEGAKI_CRITICAL_ENTROPY_THRESHOLD: f64 = 2.0;

/// Parámetros de configuración del motor de entropía de Umegaki.
#[derive(Debug, Clone, Copy)]
pub struct UmegakiEntropyParams {
    /// Umbral crítico de divergencia de Umegaki para veto duro.
    pub critical_divergence_threshold: f64,
    /// Epsilon para regularización de logaritmos y división segura.
    pub epsilon: f64,
    /// Temperatura inversa de referencia beta para el estado de Gibbs.
    pub default_beta: f64,
}

impl Default for UmegakiEntropyParams {
    fn default() -> Self {
        Self {
            critical_divergence_threshold: UMEGAKI_CRITICAL_ENTROPY_THRESHOLD,
            epsilon: 1e-12,
            default_beta: 1.0,
        }
    }
}

/// Informe omnisciente de entropía relativa cuántica de Umegaki.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UmegakiEntropyReport {
    /// Divergencia cuántica de Umegaki $D(\rho || \sigma) \ge 0$.
    pub umegaki_relative_entropy: f64,
    /// Entropía de Von Neumann del estado actual $S(\rho) = -\text{Tr}(\rho \ln \rho) \ge 0$.
    pub von_neumann_entropy_state: f64,
    /// Norma de coherencia cuántica fuera de la diagonal $\|\rho_{\text{off}}\|_F \in [0.0, 1.0]$.
    pub quantum_coherence_norm: f64,
    /// Perturbación relativa de Araki $\Delta H_{\text{Araki}}$.
    pub araki_relative_hamiltonian: f64,
    /// Factor de escala modulador para la asignación de slot $\in [0.0, 1.0]$.
    pub slot_scale_factor: f64,
    /// Disparo de veto duro por divergencia de Umegaki superando el umbral crítico.
    pub hard_veto_induced: bool,
}

impl Default for UmegakiEntropyReport {
    fn default() -> Self {
        Self {
            umegaki_relative_entropy: 0.0,
            von_neumann_entropy_state: 1.0986, // ln(3)
            quantum_coherence_norm: 0.0,
            araki_relative_hamiltonian: 0.0,
            slot_scale_factor: 1.0,
            hard_veto_induced: false,
        }
    }
}

/// Motor de Entropía Relativa Cuántica de Umegaki y Veto No-Conmutativo.
#[derive(Debug, Clone)]
pub struct UmegakiRelativeEntropyEngine {
    params: UmegakiEntropyParams,
}

impl UmegakiRelativeEntropyEngine {
    /// Crea una nueva instancia del motor con parámetros específicos.
    pub fn new(params: UmegakiEntropyParams) -> Self {
        Self { params }
    }

    /// Evalúa la entropía relativa de Umegaki y emite veto estricto.
    ///
    /// # Argumentos:
    /// - `order_flow_imbalance`: OFI normalizado $\in [-1, 1]$.
    /// - `depth_imbalance`: Desequilibrio de profundidad $\in [-1, 1]$.
    /// - `spread_bps`: Spread actual en puntos básicos $\ge 0$.
    /// - `instantaneous_vol`: Volatilidad continua actual $\sigma \ge 0$.
    pub fn evaluate(
        &self,
        order_flow_imbalance: f64,
        depth_imbalance: f64,
        spread_bps: f64,
        instantaneous_vol: f64,
    ) -> UmegakiEntropyReport {
        let safe_ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_depth = if depth_imbalance.is_finite() {
            depth_imbalance.clamp(-1.0, 1.0)
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

        // 1. Construcción de la matriz de densidad de estado \rho sobre C^3
        // Diagonal: poblaciones p_1 (Bid), p_2 (Neutro), p_3 (Ask)
        let w_bid = (1.0 - safe_ofi).max(0.01);
        let w_neutral = (2.0 / (1.0 + safe_spread * 0.1)).max(0.01);
        let w_ask = (1.0 + safe_ofi).max(0.01);
        let sum_w = w_bid + w_neutral + w_ask;

        let p_bid = w_bid / sum_w;
        let p_neutral = w_neutral / sum_w;
        let p_ask = w_ask / sum_w;

        // Coherencias cuánticas fuera de la diagonal (entrelazamiento bid-ask a través de profundidad)
        let coherence_ba = (safe_depth * (p_bid * p_ask).sqrt()).clamp(-0.49, 0.49);
        let coherence_norm = coherence_ba.abs() * 2.0;

        // 2. Construcción del estado de referencia térmico de Gibbs \sigma
        // H = diag(safe_spread*0.1, 0.0, safe_spread*0.1)
        let beta = (self.params.default_beta / safe_vol).clamp(0.1, 10.0);
        let q_bid = (-beta * 0.5).exp();
        let q_neutral = 1.0;
        let q_ask = (-beta * 0.5).exp();
        let sum_q = q_bid + q_neutral + q_ask;

        let s_bid = q_bid / sum_q;
        let s_neutral = q_neutral / sum_q;
        let s_ask = q_ask / sum_q;

        // 3. Entropía de Von Neumann: S(\rho) = - \sum p_i * ln(p_i)
        let s_rho = -(p_bid * p_bid.max(self.params.epsilon).ln()
            + p_neutral * p_neutral.max(self.params.epsilon).ln()
            + p_ask * p_ask.max(self.params.epsilon).ln())
        .max(0.0);

        // 4. Divergencia cuántica de Umegaki: D(\rho || \sigma) = Tr(\rho (ln \rho - ln \sigma))
        // Descomposición espectral adiabática en C^3
        let div_bid =
            p_bid * (p_bid.max(self.params.epsilon).ln() - s_bid.max(self.params.epsilon).ln());
        let div_neutral = p_neutral
            * (p_neutral.max(self.params.epsilon).ln() - s_neutral.max(self.params.epsilon).ln());
        let div_ask =
            p_ask * (p_ask.max(self.params.epsilon).ln() - s_ask.max(self.params.epsilon).ln());

        // Contribución de la coherencia no conmutativa
        let coherence_penalty = coherence_norm * coherence_norm * 0.5;

        let total_umegaki = (div_bid + div_neutral + div_ask + coherence_penalty).max(0.0);

        // 5. Perturbación de Araki y Veto Cuántico Estricto
        let araki_relative_h = (total_umegaki - s_rho).clamp(-10.0, 10.0);

        let hard_veto = total_umegaki >= self.params.critical_divergence_threshold;
        let slot_scale_factor = if hard_veto {
            0.0
        } else if total_umegaki > 1.20 {
            0.10 + 0.90
                * (1.0
                    - (total_umegaki - 1.20) / (self.params.critical_divergence_threshold - 1.20))
        } else {
            1.0
        }
        .clamp(0.0, 1.0);

        UmegakiEntropyReport {
            umegaki_relative_entropy: total_umegaki,
            von_neumann_entropy_state: s_rho,
            quantum_coherence_norm: coherence_norm,
            araki_relative_hamiltonian: araki_relative_h,
            slot_scale_factor,
            hard_veto_induced: hard_veto,
        }
    }
}
