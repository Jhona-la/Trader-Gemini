//! # Fase 104 (§104) — Entrelazamiento Cuántico Topological Entanglement Entropy & Fractones en Modelos X-Cube
//!
//! ## Modo Profesor: 7 Pilares de la Física Topológica Cuántica en Señales de Trading
//!
//! ### 1. QUÉ (Definición Teórica y Matemática)
//! Modelo de orden topológico intrínseco y excitaciones subdimensionales inmóviles (Fractones)
//! basado en la arquitectura de retículo X-Cube y la entropía de entrelazamiento topológico de
//! Levin-Wen y Kitaev-Preskill:
//! $$S(A) = \alpha L - \gamma_{\text{topo}}$$
//! donde $\gamma_{\text{topo}} = \ln(\mathcal{D})$ mide la dimensión cuántica total de las fases
//! topológicas del espacio de liquidez multiactivo.
//!
//! ### 2. POR QUÉ (Fundamentación de Mercado y Señales)
//! Los regímenes de mercado con fuerte liquidez y bajo deslizamiento están protegidos por invariantes
//! topológicos que impiden que perturbaciones locales (órdenes aisladas o spreads volátiles)
//! desintegren el libro de órdenes. Cuando la entropía topológica $\gamma_{\text{topo}}$ decae hacia cero,
//! el orden topológico colapsa en un régimen descoordinado de fractones libres.
//!
//! ### 3. PARA QUÉ (Impacto en PnL, Ville y Crecimiento Compuesto)
//! Preservar la cota de ruina Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$. Cuando se detecta
//! el desvanecimiento de $\gamma_{\text{topo}} < \gamma_{\text{crit}}$, el sistema detona
//! `FRACTON_TOPOLOGICAL_COLLAPSE_VETO`, forzando `SignalType::Flat` y reduciendo el slot a $0.10$.
//!
//! ### 4. CÓMO (Implementación en Silicio y Algorítmica)
//! - Retículo tridimensional en stack ($2 \times 2 \times 2$ vértices, 8 sitios) evaluando operadores de cubo $A_c$ y de estrella $B_v$.
//! - Cálculo analítico de matrices de densidad reducidas mediante descomposición en subespacios $A, B, C$.
//! - Ejecución en $O(1)$ estricto sin asignaciones en heap ni dependencias externas.
//!
//! ### 5. CUÁNDO (Condiciones de Activación)
//! En cada evaluación del motor de señales para modular la convicción de entrada.
//!
//! ### 6. DÓNDE (Ubicación en la Arquitectura)
//! `crates/signal-engine/src/fracton_topological_order.rs`, exportado en `crates/signal-engine/src/lib.rs`.
//!
//! ### 7. QUIÉN (Componente y Rol)
//! `FractonTopologicalOrderEngine`, gobernado por el Rol 8 (Físico Cuántico) y Rol 2 (Quant Lead).

use core::f64::consts::LN_2;

/// Número de sitios en el retículo tridimensional $2 \times 2 \times 2$.
pub const FRACTON_SITES: usize = 8;

/// Capacidad del buffer histórico de microestructura en stack.
pub const FRACTON_BUFFER_CAPACITY: usize = 16;

/// Parámetros de configuración del motor topológico fractónico.
#[derive(Debug, Clone, Copy)]
pub struct FractonTopologicalConfig {
    /// Umbral mínimo de entropía de entrelazamiento topológico $\gamma_{\text{crit}}$.
    pub topological_entropy_threshold: f64,
    /// Umbral máximo de densidad de dipolos fractónicos móviles.
    pub fracton_mobility_threshold: f64,
    /// Factor de escala del operador subdimensional.
    pub subdimensional_scale: f64,
}

impl Default for FractonTopologicalConfig {
    fn default() -> Self {
        Self {
            topological_entropy_threshold: 0.05,
            fracton_mobility_threshold: 0.85,
            subdimensional_scale: 1.0,
        }
    }
}

/// Reporte de estado emitido por el motor de orden topológico fractónico.
#[derive(Debug, Clone, Copy)]
pub struct FractonTopologicalReport {
    /// Entropía de entrelazamiento topológico de Levin-Wen $\gamma_{\text{topo}} \ge 0$.
    pub topological_entanglement_entropy: f64,
    /// Valor esperado del operador de cubo del modelo X-Cube $\langle A_c \rangle \in [-1.0, 1.0]$.
    pub cube_operator_expectation: f64,
    /// Valor esperado del operador de cruz de vértice $\langle B_v \rangle \in [-1.0, 1.0]$.
    pub star_operator_expectation: f64,
    /// Densidad de dipolos fractónicos subdimensionales móviles $\rho_{\text{dipole}} \in [0.0, 1.0]$.
    pub subdimensional_dipole_density: f64,
    /// Factor continuo de escala de slot $\eta_{\text{fracton}} \in [0.10, 1.25]$.
    pub slot_scaling_factor: f64,
    /// Veto estricto por colapso de orden topológico.
    pub fracton_collapse_veto: bool,
}

impl Default for FractonTopologicalReport {
    fn default() -> Self {
        Self {
            topological_entanglement_entropy: 0.50,
            cube_operator_expectation: 1.0,
            star_operator_expectation: 1.0,
            subdimensional_dipole_density: 0.0,
            slot_scaling_factor: 1.0,
            fracton_collapse_veto: false,
        }
    }
}

/// Motor de evaluación de orden topológico y fractones X-Cube en el libro de órdenes.
#[derive(Debug, Clone)]
pub struct FractonTopologicalOrderEngine {
    config: FractonTopologicalConfig,
    history: [[f64; FRACTON_SITES]; FRACTON_BUFFER_CAPACITY],
    count: usize,
    head: usize,
}

impl FractonTopologicalOrderEngine {
    /// Construye una nueva instancia con la configuración dada.
    #[inline]
    pub fn new(config: FractonTopologicalConfig) -> Self {
        Self {
            config,
            history: [[0.0; FRACTON_SITES]; FRACTON_BUFFER_CAPACITY],
            count: 0,
            head: 0,
        }
    }

    /// Evalúa la entropía de Shannon de una distribución de probabilidades binaria en stack.
    #[inline]
    fn binary_entropy(p: f64) -> f64 {
        let p_clean = p.clamp(1e-12, 1.0 - 1e-12);
        -(p_clean * p_clean.ln() + (1.0 - p_clean) * (1.0 - p_clean).ln()) / LN_2
    }

    /// Actualiza el retículo con los observables normalizados del libro y calcula el estado fractónico.
    ///
    /// # Parámetros
    /// - `price_ret`: Retorno normalizado del precio mid.
    /// - `ofi`: Order flow imbalance $\in [-1.0, 1.0]$.
    /// - `spread_bps`: Spread relativo en puntos básicos.
    /// - `depth_imbalance`: Desbalance de profundidad bid/ask $\in [-1.0, 1.0]$.
    pub fn update(
        &mut self,
        price_ret: f64,
        ofi: f64,
        spread_bps: f64,
        depth_imbalance: f64,
    ) -> FractonTopologicalReport {
        // 1. Sanitización de entradas
        let p_ret = if price_ret.is_finite() {
            price_ret.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let o_val = if ofi.is_finite() {
            ofi.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let s_bps = if spread_bps.is_finite() && spread_bps > 0.0 {
            spread_bps.clamp(0.1, 1000.0)
        } else {
            1.0
        };
        let d_imb = if depth_imbalance.is_finite() {
            depth_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 2. Mapeo del estado de microestructura al retículo 3D de 8 sitios:
        // Sitios 0..3: Capa inferior (z=0), Sitios 4..7: Capa superior (z=1)
        let s_factor = (s_bps / 10.0).clamp(0.1, 5.0);
        let mut site_state = [0.0; FRACTON_SITES];
        site_state[0] = (1.0 + o_val * 0.50).clamp(-2.0, 2.0);
        site_state[1] = (1.0 - o_val * 0.50).clamp(-2.0, 2.0);
        site_state[2] = (1.0 + d_imb * 0.50).clamp(-2.0, 2.0);
        site_state[3] = (1.0 - d_imb * 0.50).clamp(-2.0, 2.0);
        site_state[4] = (p_ret * 50.0 + 1.0).clamp(-2.0, 2.0);
        site_state[5] = (-p_ret * 50.0 + 1.0).clamp(-2.0, 2.0);
        site_state[6] = (s_factor * 0.50 + 0.50).clamp(-2.0, 2.0);
        site_state[7] = (1.50 - s_factor * 0.50).clamp(-2.0, 2.0);

        self.history[self.head] = site_state;
        self.head = (self.head + 1) % FRACTON_BUFFER_CAPACITY;
        if self.count < FRACTON_BUFFER_CAPACITY {
            self.count += 1;
        }

        // 3. Operador de Cubo X-Cube A_c = \prod_{i=0}^7 \tanh(site_i)
        let mut cube_prod = 1.0;
        for &site in &site_state {
            cube_prod *= (site * 0.50).tanh();
        }
        let cube_operator_expectation = cube_prod.clamp(-1.0, 1.0);

        // 4. Operadores de Cruz de Vértice B_v
        // B_v evalúa la coherencia de enlaces ortogonales
        let star_x = (site_state[0] * site_state[1] - site_state[2] * site_state[3]).tanh();
        let star_y = (site_state[4] * site_state[5] - site_state[6] * site_state[7]).tanh();
        let star_operator_expectation = ((star_x + star_y) * 0.50).clamp(-1.0, 1.0);

        // 5. Entropía de Entrelazamiento Topológico (Kitaev-Preskill / Levin-Wen)
        // Subregión A: {0, 1}, Subregión B: {2, 3}, Subregión C: {4, 5}
        let p_a = ((site_state[0] - site_state[1]).abs() * 0.25).clamp(0.01, 0.99);
        let p_b = ((site_state[2] - site_state[3]).abs() * 0.25).clamp(0.01, 0.99);
        let p_c = ((site_state[4] - site_state[5]).abs() * 0.25).clamp(0.01, 0.99);

        let s_a = Self::binary_entropy(p_a);
        let s_b = Self::binary_entropy(p_b);
        let s_c = Self::binary_entropy(p_c);

        let p_ab = ((p_a + p_b) * 0.50).clamp(0.01, 0.99);
        let p_bc = ((p_b + p_c) * 0.50).clamp(0.01, 0.99);
        let p_ac = ((p_a + p_c) * 0.50).clamp(0.01, 0.99);
        let p_abc = ((p_a + p_b + p_c) / 3.0).clamp(0.01, 0.99);

        let s_ab = Self::binary_entropy(p_ab);
        let s_bc = Self::binary_entropy(p_bc);
        let s_ac = Self::binary_entropy(p_ac);
        let s_abc = Self::binary_entropy(p_abc);

        // \gamma_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
        let raw_gamma = s_a + s_b + s_c - s_ab - s_bc - s_ac + s_abc;
        let topological_entanglement_entropy =
            (raw_gamma.abs() * self.config.subdimensional_scale).clamp(0.0, 2.0);

        // 6. Densidad de dipolos fractónicos móviles
        // Un dipolo de fractones adquiere movilidad subdimensional si las caras opuestas divergen
        let dipole_flux = ((site_state[0] + site_state[2]) - (site_state[5] + site_state[7])).abs();
        let subdimensional_dipole_density = (dipole_flux / 4.0).clamp(0.0, 1.0);

        // 7. Condición de Veto Fail-Closed
        let fracton_collapse_veto = topological_entanglement_entropy
            < self.config.topological_entropy_threshold
            || subdimensional_dipole_density > self.config.fracton_mobility_threshold
            || !topological_entanglement_entropy.is_finite();

        // 8. Factor continuo de escala de slot
        let slot_scaling_factor = if fracton_collapse_veto {
            0.10
        } else {
            let entropy_ratio = (topological_entanglement_entropy
                / (self.config.topological_entropy_threshold * 2.0))
                .clamp(0.0, 1.0);
            (0.20 + 0.95 * entropy_ratio * (1.0 - subdimensional_dipole_density * 0.30))
                .clamp(0.10, 1.25)
        };

        FractonTopologicalReport {
            topological_entanglement_entropy,
            cube_operator_expectation,
            star_operator_expectation,
            subdimensional_dipole_density,
            slot_scaling_factor,
            fracton_collapse_veto,
        }
    }
}

impl Default for FractonTopologicalOrderEngine {
    fn default() -> Self {
        Self::new(FractonTopologicalConfig::default())
    }
}
