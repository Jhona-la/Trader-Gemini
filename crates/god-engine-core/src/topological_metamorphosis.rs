//! ⚛️ FASE 48 (§48): METAMORFOSIS TOPOLÓGICA DEL GRAFO VIVO AUTÓNOMO
//! (Autonomous Living Graph Topological Metamorphosis)
//!
//! # MODO PROFESOR — ANÁLISIS INTEGRAL DEL SISTEMA CUÁNTICO
//!
//! - **QUÉ:** Un motor de plasticidad neuronal y reorganización topológica continua para la
//!   matriz de adyacencia $\mathbf{A} \in \mathbb{R}^{12 \times 12}$ del Grafo Omnisciente Vivo.
//! - **POR QUÉ:** Las influencias relativas e interdependencias dinámicas entre los 12 subsistemas
//!   principales del motor (Hodge-Poincaré, Navier-Stokes, Instantones Yang-Mills, Ville Supermartingale,
//!   Feynman Routing, Mean Field Games Psychology, Dirac-Clifford, MPS DMRG, SRB Chaos, Shockwaves,
//!   Hawkes-Bessel, Conformal Reversion) no son estáticas. En condiciones cambiantes de mercado,
//!   ciertas aristas sinápticas amplifican o atenúan su acoplamiento causal.
//! - **PARA QUÉ:** Permitir la auto-evolución continua de las sinapsis del Grafo Vivo, podando
//!   aristas ruidosas ($|A_{ij}| < \theta_{\text{prune}} \implies 0$) y reforzando las conexiones
//!   que producen convergencia predictiva superior, preservando la estabilidad asintótica de Lyapunov
//!   vía normalización de Gershgorin ($\|A\|_{\infty} \le 1.0$), en $O(1)$ stack sin heap allocations.
//! - **CÓMO:**
//!   1. Mantiene una matriz de adyacencia $\mathbf{A} \in \mathbb{R}^{12 \times 12}$ en pila.
//!   2. En cada ciclo, recibe el vector de activación de subsistemas $s \in [-1, 1]^{12}$ y el
//!      desempeño reciente de la cartera (e.g. Sharpe ratio normalizado o delta de utilidad logarítmica).
//!   3. Aplica la regla de aprendizaje Hebbiano con decaimiento homeostático:
//!      $$\Delta A_{ij} = \eta \cdot \left( s_i s_j - \lambda_{\text{decay}} A_{ij} \right) \cdot \Phi(\text{perf})$$
//!   4. Aplica poda dura (pruning) para eliminar aristas por debajo del umbral de ruido $\theta_{\text{prune}}$.
//!   5. Proyecta las filas sobre la bola unitaria inducida por la norma $\|\cdot\|_{\infty}$
//!      para asegurar que el radio espectral $\rho(\mathbf{A}) \le 1.0$ (estabilidad de Lyapunov).
//!   6. Calcula el modulador de plasticidad $\mu_{\text{plasticity}} \in [0.10, 1.50]$.
//! - **CUÁNDO:** Invocado en cada iteración del Grafo Omnisciente (`OmniscientLivingGraph`).
//! - **DÓNDE:** En `crates/god-engine-core/src/topological_metamorphosis.rs`.
//! - **QUIÉN:** Integrado directamente en el núcleo de `OmniscientLivingGraph`.

pub const METAMORPHOSIS_NODES: usize = 12;

/// Nombres descriptivos de los 12 nodos canónicos del Grafo Vivo.
pub const NODE_NAMES: [&str; METAMORPHOSIS_NODES] = [
    "HodgePoincare",
    "NavierStokesReynolds",
    "YangMillsInstantons",
    "VilleSupermartingale",
    "FeynmanRouting",
    "MfgPsychology",
    "DiracCliffordSpinor",
    "MpsDmrgEntanglement",
    "SrbChaosEntropy",
    "RiemannBurgersShock",
    "HawkesBesselPoint",
    "ConformalReversion",
];

/// Parámetros de control de plasticidad y estabilidad para la metamorfosis topológica.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologicalMetamorphosisParams {
    /// Tasa de aprendizaje Hebbiano $\eta$ (default: 0.05).
    pub learning_rate: f64,
    /// Tasa de decaimiento homeostático $\lambda_{\text{decay}}$ (default: 0.01).
    pub decay_rate: f64,
    /// Umbral de poda de aristas ruidosas $\theta_{\text{prune}}$ (default: 0.02).
    pub pruning_threshold: f64,
    /// Margen de estabilidad espectral de Gershgorin $\le 1.0$ (default: 0.95).
    pub stability_margin: f64,
}

impl Default for TopologicalMetamorphosisParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            decay_rate: 0.01,
            pruning_threshold: 0.02,
            stability_margin: 0.95,
        }
    }
}

impl TopologicalMetamorphosisParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 || self.learning_rate > 1.0
        {
            return Err("learning_rate debe estar en (0.0, 1.0]");
        }
        if !self.decay_rate.is_finite() || self.decay_rate < 0.0 || self.decay_rate > 1.0 {
            return Err("decay_rate debe estar en [0.0, 1.0]");
        }
        if !self.pruning_threshold.is_finite()
            || self.pruning_threshold < 0.0
            || self.pruning_threshold > 0.50
        {
            return Err("pruning_threshold debe estar en [0.0, 0.50]");
        }
        if !self.stability_margin.is_finite()
            || self.stability_margin <= 0.10
            || self.stability_margin > 1.0
        {
            return Err("stability_margin debe estar en (0.10, 1.0]");
        }
        Ok(())
    }
}

/// Informe de estado topológico emitido tras cada paso de adaptación.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologicalMetamorphosisReport {
    /// Número de aristas activas (peso no nulo tras podado).
    pub active_edge_count: usize,
    /// Cota superior de Gershgorin para el radio espectral $\rho(\mathbf{A}) \le \max_i \sum_j |A_{ij}|$.
    pub spectral_radius_bound: f64,
    /// Peso medio absoluto de las conexiones activas.
    pub mean_connectivity: f64,
    /// Peso máximo absoluto de arista en la matriz $\max_{ij} |A_{ij}|$.
    pub max_edge_weight: f64,
    /// Modulador de plasticidad global para el sistema $\in [0.10, 1.50]$.
    pub plasticity_modulator: f64,
    /// Instantánea de la matriz de adyacencia $\mathbf{A} \in \mathbb{R}^{12 \times 12}$.
    pub adjacency_matrix: [[f64; METAMORPHOSIS_NODES]; METAMORPHOSIS_NODES],
}

/// Motor de metamorfosis topológica adaptativa para el Grafo Omnisciente.
#[derive(Debug, Clone)]
pub struct TopologicalMetamorphosisEngine {
    params: TopologicalMetamorphosisParams,
    adjacency: [[f64; METAMORPHOSIS_NODES]; METAMORPHOSIS_NODES],
    total_updates: u64,
}

impl Default for TopologicalMetamorphosisEngine {
    fn default() -> Self {
        Self::new(TopologicalMetamorphosisParams::default())
    }
}

impl TopologicalMetamorphosisEngine {
    /// Inicializa el motor con una topología inicial estructurada según afinidad física.
    pub fn new(params: TopologicalMetamorphosisParams) -> Self {
        let valid_params = if params.validate().is_ok() {
            params
        } else {
            TopologicalMetamorphosisParams::default()
        };

        let mut adj = [[0.0; METAMORPHOSIS_NODES]; METAMORPHOSIS_NODES];

        // Topología inicial equilibrada con conexiones canónicas prioritarias
        for i in 0..METAMORPHOSIS_NODES {
            for j in 0..METAMORPHOSIS_NODES {
                if i == j {
                    // Autoconexión amortiguada
                    adj[i][j] = 0.20;
                } else {
                    // Acoplamiento inicial moderado entre subsistemas afines
                    let diff = (i as isize - j as isize).abs();
                    if diff == 1 || diff == (METAMORPHOSIS_NODES as isize - 1) {
                        adj[i][j] = 0.15;
                    } else {
                        adj[i][j] = 0.05;
                    }
                }
            }
        }

        let mut engine = Self {
            params: valid_params,
            adjacency: adj,
            total_updates: 0,
        };
        engine.enforce_stability_and_pruning();
        engine
    }

    /// Actualiza la matriz de adyacencia usando el vector de activaciones $s \in [-1, 1]^{12}$
    /// y el desempeño reciente de la cartera.
    #[inline]
    pub fn update(
        &mut self,
        activations: &[f64; METAMORPHOSIS_NODES],
        performance_score: f64,
    ) -> TopologicalMetamorphosisReport {
        self.total_updates = self.total_updates.wrapping_add(1);

        // 1. Sanitizar activaciones y puntaje
        let mut safe_acts = [0.0; METAMORPHOSIS_NODES];
        for i in 0..METAMORPHOSIS_NODES {
            let a = activations[i];
            safe_acts[i] = if a.is_finite() {
                a.clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }

        let safe_perf = if performance_score.is_finite() {
            performance_score.clamp(-10.0, 10.0)
        } else {
            0.0
        };

        // Función de modulación sigmoidal: $\Phi(\text{perf}) = \frac{1}{1 + e^{-\text{perf}}} \in [0, 1]$
        let phi_perf = 1.0 / (1.0 + (-safe_perf).exp());

        let eta = self.params.learning_rate;
        let lambda = self.params.decay_rate;

        // 2. Regla de plasticidad Hebbiano-Lyapunov
        for i in 0..METAMORPHOSIS_NODES {
            let s_i = safe_acts[i];
            for j in 0..METAMORPHOSIS_NODES {
                let s_j = safe_acts[j];
                let current = self.adjacency[i][j];

                // Hebbiano (co-activación) menos decaimiento
                let delta = eta * (s_i * s_j - lambda * current) * phi_perf;
                self.adjacency[i][j] = current + delta;
            }
        }

        // 3. Poda de ruido y garantía de estabilidad espectral de Lyapunov
        self.enforce_stability_and_pruning()
    }

    /// Aplica poda por umbral y normalización de Gershgorin en cada fila para garantizar
    /// que el radio espectral $\rho(\mathbf{A}) \le \text{stability\_margin} \le 1.0$.
    fn enforce_stability_and_pruning(&mut self) -> TopologicalMetamorphosisReport {
        let thresh = self.params.pruning_threshold;
        let max_row_sum = self.params.stability_margin;

        let mut active_count = 0;
        let mut max_weight = 0.0f64;
        let mut total_abs_weight = 0.0f64;
        let mut max_row_l1 = 0.0f64;

        for i in 0..METAMORPHOSIS_NODES {
            let mut row_l1 = 0.0;

            // Paso 1: Poda
            for j in 0..METAMORPHOSIS_NODES {
                let val = self.adjacency[i][j];
                if val.abs() < thresh || !val.is_finite() {
                    self.adjacency[i][j] = 0.0;
                } else {
                    row_l1 += self.adjacency[i][j].abs();
                }
            }

            // Paso 2: Normalización de Gershgorin si excede la cota de estabilidad
            if row_l1 > max_row_sum && row_l1 > 1e-9 {
                let scale = max_row_sum / row_l1;
                for j in 0..METAMORPHOSIS_NODES {
                    self.adjacency[i][j] *= scale;
                }
                row_l1 = max_row_sum;
            }

            if row_l1 > max_row_l1 {
                max_row_l1 = row_l1;
            }

            for j in 0..METAMORPHOSIS_NODES {
                let w = self.adjacency[i][j].abs();
                if w > 0.0 {
                    active_count += 1;
                    total_abs_weight += w;
                    if w > max_weight {
                        max_weight = w;
                    }
                }
            }
        }

        let mean_conn = if active_count > 0 {
            total_abs_weight / active_count as f64
        } else {
            0.0
        };

        // Modulador de plasticidad global
        let plasticity_mod = (0.50
            + 0.50 * (active_count as f64 / (METAMORPHOSIS_NODES * METAMORPHOSIS_NODES) as f64)
            + 0.30 * max_row_l1)
            .clamp(0.10, 1.50);

        TopologicalMetamorphosisReport {
            active_edge_count: active_count,
            spectral_radius_bound: max_row_l1,
            mean_connectivity: mean_conn,
            max_edge_weight: max_weight,
            plasticity_modulator: plasticity_mod,
            adjacency_matrix: self.adjacency,
        }
    }

    /// Retorna los parámetros activos del motor.
    #[inline(always)]
    pub fn params(&self) -> TopologicalMetamorphosisParams {
        self.params
    }

    /// Retorna la matriz de adyacencia actual.
    #[inline(always)]
    pub fn adjacency(&self) -> &[[f64; METAMORPHOSIS_NODES]; METAMORPHOSIS_NODES] {
        &self.adjacency
    }
}
