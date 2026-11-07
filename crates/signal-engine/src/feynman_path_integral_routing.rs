//! ⚛️ FASE 47 (§47): ENRUTAMIENTO CUÁNTICO POR INTEGRAL DE CAMINOS DE FEYNMAN
//! (Feynman Path Integral Quantum Routing Engine)
//!
//! # MODO PROFESOR — ANÁLISIS INTEGRAL DEL SISTEMA CUÁNTICO
//!
//! - **QUÉ:** Un motor de enrutamiento y partición de micro-órdenes en el continuo temporal
//!   basado en la integral de caminos de Feynman en mecánica cuántica sobre el espacio de
//!   configuración de trayectorias de inventario y ejecución.
//! - **POR QUÉ:** Las técnicas clásicas de ejecución (TWAP, VWAP, Almgren-Chriss estático)
//!   asumen trayectorias unívocas deterministas que sufren de impacto de mercado predecible
//!   y son explotadas por algoritmos predadores HFT. En mercados hiper-líquidos estocásticos,
//!   múltiples trayectorias interfieren constructiva y destructivamente en función de la
//!   profundidad del libro y del Order Flow Imbalance (OFI).
//! - **PARA QUÉ:** Minimizar el deslizamiento cuadrático (slippage), maximizar la captura de
//!   liquidez pasiva/activa, y sincronizar la tasa de emisión de órdenes con interferencia
//!   cuántica constructiva, acelerando el crecimiento compuesto de capital y protegiendo
//!   el drawdown con $O(1)$ stack determinista y cero asignaciones en heap.
//! - **CÓMO:**
//!   1. Discretiza el horizonte de ejecución en $K = 8$ slices temporales y evalúa un conjunto
//!      de $M = 8$ trayectorias cuánticas base $\gamma^{(m)} = \{q_0, q_1, \dots, q_{K-1}\}$.
//!   2. Calcula el funcional de Acción Clásica $S[\gamma^{(m)}]$:
//!      $$S[\gamma] = \sum_{k=0}^{K-2} \left[ \frac{1}{2} m \left(\frac{q_{k+1} - q_k}{\Delta t}\right)^2
//!        + \frac{1}{2} \gamma_{\text{slip}} (q_{k+1} - q_k)^2 \frac{1}{\text{depth}_k + \epsilon}
//!        - \lambda_{\text{ofi}} (q_{k+1} - q_k) \cdot \text{OFI}_k \right] \Delta t$$
//!   3. Asigna a cada trayectoria una amplitud compleja de probabilidad $\Psi_m = \exp(i S[\gamma^{(m)}] / \hbar)$.
//!   4. Construye la superposición cuántica coherente $\Psi_{\text{total}} = \sum_{m} w_m \Psi_m$, obteniendo
//!      la densidad de probabilidad $P(\gamma^{(m)}) = |\Psi_m|^2 / \sum_j |\Psi_j|^2$ y la coherencia
//!      cuántica $C_{\text{coh}} = |\Psi_{\text{total}}|^2 / (\sum_m |\Psi_m|^2 + \epsilon)$.
//!   5. Obtiene la fracción óptima inmediata para el siguiente tick $\Delta q^* = \sum_m P(\gamma^{(m)}) \Delta q^{(m)}_0$
//!      y modula la urgencia de ejecución $\chi_{\text{routing}} \in [0.10, 1.50]$.
//! - **CUÁNDO:** En cada pulso de emisión de órdenes y ajuste de micro-lotes en `god-engine-core`.
//! - **DÓNDE:** En `crates/signal-engine/src/feynman_path_integral_routing.rs`.
//! - **QUIÉN:** Invocado por `OmniscientLivingGraph` y `QuantumSlotOrchestrator`.

pub const ROUTING_PATH_SLICES: usize = 8;
pub const ROUTING_PATH_CANDIDATES: usize = 8;

/// Representación escalar de número complejo en pila (Cero heap allocations).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let (sin, cos) = theta.sin_cos();
        Self {
            re: r * cos,
            im: r * sin,
        }
    }

    #[inline(always)]
    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

/// Parámetros configurables del motor de enrutamiento por integral de caminos de Feynman.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeynmanRoutingParams {
    /// Constante de Planck efectiva del micromercado $\hbar_{\text{routing}}$ (default: 1.0).
    pub hbar_routing: f64,
    /// Masa inercial de la orden $m$ (resistencia a cambios bruscos de tasa de llenado) (default: 1.0).
    pub mass: f64,
    /// Penalización de deslizamiento $\gamma_{\text{slip}}$ (default: 0.50).
    pub slip_penalty: f64,
    /// Coeficiente de acoplamiento al flujo de órdenes $\lambda_{\text{ofi}}$ (default: 0.30).
    pub ofi_coupling: f64,
    /// Umbral mínimo de coherencia constructiva (default: 0.20).
    pub coherence_threshold: f64,
}

impl Default for FeynmanRoutingParams {
    fn default() -> Self {
        Self {
            hbar_routing: 1.0,
            mass: 1.0,
            slip_penalty: 0.50,
            ofi_coupling: 0.30,
            coherence_threshold: 0.20,
        }
    }
}

impl FeynmanRoutingParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.hbar_routing.is_finite() || self.hbar_routing <= 0.001 || self.hbar_routing > 100.0
        {
            return Err("hbar_routing debe ser finito y estar en (0.001, 100.0]");
        }
        if !self.mass.is_finite() || self.mass <= 0.001 || self.mass > 100.0 {
            return Err("mass debe ser finito y estar en (0.001, 100.0]");
        }
        if !self.slip_penalty.is_finite() || self.slip_penalty < 0.0 || self.slip_penalty > 100.0 {
            return Err("slip_penalty debe ser finito y estar en [0.0, 100.0]");
        }
        if !self.ofi_coupling.is_finite() || self.ofi_coupling < 0.0 || self.ofi_coupling > 100.0 {
            return Err("ofi_coupling debe ser finito y estar en [0.0, 100.0]");
        }
        if !self.coherence_threshold.is_finite()
            || self.coherence_threshold < 0.0
            || self.coherence_threshold > 1.0
        {
            return Err("coherence_threshold debe estar en [0.0, 1.0]");
        }
        Ok(())
    }
}

/// Informe de evaluación cuántica de enrutamiento emitido en cada ciclo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeynmanRoutingReport {
    /// Fracción óptima calculada para el siguiente intervalo inmediato $\Delta q^* \in [0.0, 1.0]$.
    pub optimal_slice_fraction: f64,
    /// Coherencia cuántica constructiva $C_{\text{coh}} \in [0.0, 1.0]$.
    pub constructive_coherence: f64,
    /// Mínima acción clásica encontrada entre las trayectorias evaluadas.
    pub action_min: f64,
    /// Media de acción clásica sobre el haz de trayectorias.
    pub action_mean: f64,
    /// Índice de la trayectoria clásica con mínima acción (principio de mínima acción de Hamilton).
    pub best_path_index: usize,
    /// Modulador de urgencia y sizing de enrutamiento $\chi_{\text{routing}} \in [0.10, 1.50]$.
    pub routing_urgency_modulator: f64,
    /// Distribución de probabilidad cuántica sobre las $M = 8$ trayectorias candidatas.
    pub path_probabilities: [f64; ROUTING_PATH_CANDIDATES],
}

/// Motor determinista de enrutamiento de órdenes por integral de caminos de Feynman.
///
/// Todas las operaciones se realizan sobre estructuras fijas en pila ($O(1)$ stack, sin heap).
#[derive(Debug, Clone)]
pub struct FeynmanPathIntegralRoutingEngine {
    params: FeynmanRoutingParams,
    candidate_paths: [[f64; ROUTING_PATH_SLICES]; ROUTING_PATH_CANDIDATES],
}

impl Default for FeynmanPathIntegralRoutingEngine {
    fn default() -> Self {
        Self::new(FeynmanRoutingParams::default())
    }
}

impl FeynmanPathIntegralRoutingEngine {
    /// Crea una nueva instancia del motor con los parámetros especificados.
    #[allow(clippy::needless_range_loop)]
    pub fn new(params: FeynmanRoutingParams) -> Self {
        let valid_params = if params.validate().is_ok() {
            params
        } else {
            FeynmanRoutingParams::default()
        };

        let mut paths = [[0.0; ROUTING_PATH_SLICES]; ROUTING_PATH_CANDIDATES];
        let k_max = (ROUTING_PATH_SLICES - 1) as f64;

        for (m, path) in paths.iter_mut().enumerate() {
            for k in 0..ROUTING_PATH_SLICES {
                let tau = k as f64 / k_max;
                path[k] = match m {
                    // 0: TWAP Lineal
                    0 => tau,
                    // 1: Agresivo frontal (raíz cuadrada)
                    1 => tau.sqrt(),
                    // 2: Pasivo posterior (cuadrático)
                    2 => tau * tau,
                    // 3: S-Curve sigmoidal suave
                    3 => 1.0 / (1.0 + (-6.0 * (tau - 0.5)).exp()),
                    // 4: Cúbico pasivo
                    4 => tau * tau * tau,
                    // 5: Cóncavo rápido (raíz cúbica)
                    5 => tau.cbrt(),
                    // 6: Interferencia armónica senoidal
                    6 => 0.5 * (1.0 - (std::f64::consts::PI * (tau - 0.5)).sin()),
                    // 7: Progresión geométrica normalizada
                    _ => (tau.powi(2) + tau.sqrt()) * 0.5,
                };
            }
            // Asegurar condiciones de frontera canónicas: q_0 = 0.0, q_{K-1} = 1.0
            let q0 = path[0];
            let q_end = path[ROUTING_PATH_SLICES - 1];
            let span = (q_end - q0).max(1e-9);
            for elem in path.iter_mut() {
                *elem = ((*elem - q0) / span).clamp(0.0, 1.0);
            }
            path[0] = 0.0;
            path[ROUTING_PATH_SLICES - 1] = 1.0;
        }

        Self {
            params: valid_params,
            candidate_paths: paths,
        }
    }

    /// Evalúa la integral de caminos para un estado de microestructura dado:
    /// - `depth`: Profundidad efectiva del libro (notional o volumen disponible en el spread).
    /// - `ofi`: Order Flow Imbalance normalizado $\in [-1.0, 1.0]$.
    /// - `volatility_scale`: Escala de volatilidad local $\sigma \ge 0.0$.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn evaluate(&self, depth: f64, ofi: f64, volatility_scale: f64) -> FeynmanRoutingReport {
        let safe_depth = if depth.is_finite() && depth > 1e-6 {
            depth
        } else {
            1.0
        };
        let safe_ofi = if ofi.is_finite() {
            ofi.clamp(-5.0, 5.0)
        } else {
            0.0
        };
        let safe_vol = if volatility_scale.is_finite() && volatility_scale > 0.0 {
            volatility_scale
        } else {
            1.0
        };

        let dt = 1.0 / (ROUTING_PATH_SLICES - 1) as f64;
        let hbar = (self.params.hbar_routing * safe_vol).max(1e-4);

        let mut actions = [0.0; ROUTING_PATH_CANDIDATES];
        let mut min_action = f64::MAX;
        let mut sum_action = 0.0;
        let mut best_idx = 0;

        // 1. Evaluación de la Acción Clásica S[γ] para cada trayectoria
        for m in 0..ROUTING_PATH_CANDIDATES {
            let mut s = 0.0;
            for k in 0..(ROUTING_PATH_SLICES - 1) {
                let dq = self.candidate_paths[m][k + 1] - self.candidate_paths[m][k];
                let velocity = dq / dt;

                // Término cinético (masa inercial)
                let kinetic = 0.5 * self.params.mass * velocity * velocity;

                // Término de deslizamiento por fricción de liquidez
                let slip_cost = 0.5 * self.params.slip_penalty * (dq * dq) / safe_depth;

                // Acoplamiento cuántico con el Order Flow Imbalance
                let ofi_coupling = -self.params.ofi_coupling * dq * safe_ofi;

                s += (kinetic + slip_cost + ofi_coupling) * dt;
            }

            if !s.is_finite() {
                s = 100.0;
            }
            actions[m] = s;
            sum_action += s;

            if s < min_action {
                min_action = s;
                best_idx = m;
            }
        }

        let mean_action = sum_action / ROUTING_PATH_CANDIDATES as f64;

        // 2. Cálculo de Amplitudes Cuánticas de Feynman Ψ_m = exp(i S_m / ħ)
        let mut total_psi = Complex64::zero();
        let mut unnorm_probs = [0.0; ROUTING_PATH_CANDIDATES];
        let mut sum_probs = 0.0;

        for m in 0..ROUTING_PATH_CANDIDATES {
            let phase = actions[m] / hbar;
            // Amplitud unitaria ponderada por la acción
            let psi_m = Complex64::from_polar(1.0, phase);
            total_psi = total_psi.add(&psi_m);

            // Probabilidad cuántica proporcional a exp(-(S_m - S_min) / ħ) para convergencia física
            let weight = (-(actions[m] - min_action) / hbar).exp().clamp(1e-9, 1.0);
            unnorm_probs[m] = weight;
            sum_probs += weight;
        }

        let safe_sum_probs = if sum_probs.is_finite() && sum_probs > 1e-12 {
            sum_probs
        } else {
            1.0
        };

        let mut probabilities = [0.0; ROUTING_PATH_CANDIDATES];
        let mut optimal_slice = 0.0;

        for m in 0..ROUTING_PATH_CANDIDATES {
            let p_m = (unnorm_probs[m] / safe_sum_probs).clamp(0.0, 1.0);
            probabilities[m] = p_m;
            let dq_first = self.candidate_paths[m][1] - self.candidate_paths[m][0];
            optimal_slice += p_m * dq_first;
        }

        // 3. Medida de Coherencia e Interferencia Cuántica Constructiva
        let coherent_norm = total_psi.norm_sq();
        let incoherent_norm = ROUTING_PATH_CANDIDATES as f64;
        let constructive_coherence =
            (coherent_norm / (incoherent_norm * incoherent_norm + 1e-9)).clamp(0.0, 1.0);

        // 4. Modulador de Urgencia y Sizing
        let urgency = if constructive_coherence >= self.params.coherence_threshold {
            // Interferencia constructiva refuerza la convicción de enrutamiento
            (0.80 + 0.70 * constructive_coherence).clamp(0.10, 1.50)
        } else {
            // Interferencia destructiva modera la exposición
            (0.50 * (constructive_coherence / self.params.coherence_threshold)).clamp(0.10, 0.80)
        };

        FeynmanRoutingReport {
            optimal_slice_fraction: optimal_slice.clamp(0.01, 1.0),
            constructive_coherence,
            action_min: min_action,
            action_mean: mean_action,
            best_path_index: best_idx,
            routing_urgency_modulator: urgency,
            path_probabilities: probabilities,
        }
    }

    /// Retorna los parámetros activos del motor.
    #[inline(always)]
    pub fn params(&self) -> FeynmanRoutingParams {
        self.params
    }
}
