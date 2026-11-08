//! Fase 64: Redes Tensoriales PEPS y Curvatura de Ricci en Hipergrafos.
//!
//! Este módulo implementa la contracción en stack $O(1)$ de redes tensoriales proyectadas 2D
//! (Projected Entangled Pair States - PEPS) sobre una malla planar $2 \times 3$, junto con el
//! cálculo de la curvatura de Ollivier-Ricci $\kappa(x, y) = 1 - W_1(m_x, m_y)/d(x, y)$ sobre
//! hipergrafos de microestructura continua.
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! 1. Ley de Área 2D para PEPS: $S(A) \le c \cdot |\partial A|$.
//! 2. Curvatura de Ollivier-Ricci positiva ($\kappa > 0$) implica contracción y estabilidad;
//!    curvatura fuertemente negativa ($\kappa < -0.80$) indica dispersión hiperbólica de volatilidad.
//! 3. Dimensión espectral del hipergrafo $d_s \in (0, 4]$.
//! 4. Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y memoria acotada en stack $O(1)$.

/// Dimensiones de la malla planar PEPS (filas x columnas).
pub const PEPS_ROWS: usize = 2;
pub const PEPS_COLS: usize = 3;

/// Dimensión de enlace virtual del tensor PEPS.
pub const PEPS_BOND_DIM: usize = 2;

/// Parámetros de configuración del motor PEPS y Ricci.
#[derive(Debug, Clone, Copy)]
pub struct PepsRicciParams {
    /// Umbral de curvatura negativa de Ollivier-Ricci para alerta de dispersión hiperbólica.
    pub critical_negative_ricci: f64,
    /// Epsilon para divisiones numéricas seguras.
    pub epsilon: f64,
    /// Regularizador de transporte de Wasserstein discreto.
    pub wasserstein_reg: f64,
}

impl Default for PepsRicciParams {
    fn default() -> Self {
        Self {
            critical_negative_ricci: -0.80,
            epsilon: 1e-12,
            wasserstein_reg: 0.10,
        }
    }
}

/// Informe omnisciente de la red tensorial PEPS y curvatura de Ricci.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PepsHypergraphRicciReport {
    /// Entropía de entrelazamiento de la red PEPS 2D $S_{\text{PEPS}} \ge 0$.
    pub peps_entanglement_entropy: f64,
    /// Curvatura media de Ollivier-Ricci $\kappa \in [-2.0, 1.0]$.
    pub ollivier_ricci_curvature: f64,
    /// Dimensión espectral estimada del hipergrafo $d_s > 0$.
    pub hypergraph_spectral_dimension: f64,
    /// Flujo de área de frontera PEPS $|\partial A|$.
    pub boundary_area_flux: f64,
    /// Factor de escala modulador para la asignación de slot $\in [0.10, 1.0]$.
    pub slot_scale_factor: f64,
    /// Alerta de dispersión hiperbólica por colapso de curvatura de Ricci.
    pub hyperbolic_dispersion_detected: bool,
}

impl Default for PepsHypergraphRicciReport {
    fn default() -> Self {
        Self {
            peps_entanglement_entropy: 0.6931, // ln(2)
            ollivier_ricci_curvature: 0.10,
            hypergraph_spectral_dimension: 2.0,
            boundary_area_flux: 1.0,
            slot_scale_factor: 1.0,
            hyperbolic_dispersion_detected: false,
        }
    }
}

/// Motor de Redes Tensoriales PEPS 2D y Curvatura de Ricci en Hipergrafos.
#[derive(Debug, Clone)]
pub struct PepsHypergraphRicciEngine {
    params: PepsRicciParams,
    // Tensor de nodos 2x3 con observables proyectados en stack O(1)
    node_weights: [[f64; PEPS_COLS]; PEPS_ROWS],
    initialized: bool,
}

impl PepsHypergraphRicciEngine {
    /// Crea una nueva instancia del motor con parámetros específicos.
    pub fn new(params: PepsRicciParams) -> Self {
        Self {
            params,
            node_weights: [[1.0 / 6.0; PEPS_COLS]; PEPS_ROWS],
            initialized: false,
        }
    }

    /// Actualiza la contracción PEPS y evalúa la curvatura de Ricci del hipergrafo.
    ///
    /// # Argumentos:
    /// - `price_ret`: Retorno continuo normalizado.
    /// - `spread_bps`: Spread relativo actual $\ge 0$.
    /// - `ofi`: Desequilibrio de flujo de órdenes $\in [-1, 1]$.
    /// - `depth_ratio`: Ratio de profundidad ask/bid $\ge 0$.
    /// - `instantaneous_vol`: Volatilidad continua actual $\sigma \ge 0$.
    pub fn update(
        &mut self,
        price_ret: f64,
        spread_bps: f64,
        ofi: f64,
        depth_ratio: f64,
        instantaneous_vol: f64,
    ) -> PepsHypergraphRicciReport {
        let safe_ret = if price_ret.is_finite() {
            price_ret.clamp(-0.50, 0.50)
        } else {
            0.0
        };

        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 100.0)
        } else {
            1.0
        };

        let safe_ofi = if ofi.is_finite() {
            ofi.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_depth = if depth_ratio.is_finite() && depth_ratio > 0.0 {
            depth_ratio.clamp(0.01, 100.0)
        } else {
            1.0
        };

        let safe_vol = if instantaneous_vol.is_finite() && instantaneous_vol > 0.0 {
            instantaneous_vol.clamp(0.001, 1.0)
        } else {
            0.01
        };

        // 1. Asignación de pesos a la malla planar PEPS (2 filas x 3 columnas)
        // Fila 0: Microestructura inmediata (retorno, ofi, spread)
        // Fila 1: Dinámica difusiva (volatilidad, desequilibrio de profundidad, energía cinética)
        let w00 = (1.0 + safe_ret / (safe_vol + self.params.epsilon)).max(0.01);
        let w01 = (1.0 + safe_ofi).max(0.01);
        let w02 = (safe_spread * 0.1).max(0.01);

        let w10 = (safe_vol * 10.0).max(0.01);
        let w11 = safe_depth.max(0.01);
        let w12 =
            (safe_ret * safe_ret / (2.0 * safe_vol * safe_vol + self.params.epsilon)).max(0.01);

        let sum_total = w00 + w01 + w02 + w10 + w11 + w12;
        let inv_sum = 1.0 / sum_total.max(self.params.epsilon);

        self.node_weights[0][0] = w00 * inv_sum;
        self.node_weights[0][1] = w01 * inv_sum;
        self.node_weights[0][2] = w02 * inv_sum;
        self.node_weights[1][0] = w10 * inv_sum;
        self.node_weights[1][1] = w11 * inv_sum;
        self.node_weights[1][2] = w12 * inv_sum;
        self.initialized = true;

        // 2. Contracción PEPS aproximada y Entropía de Entrelazamiento
        // Bipartición entre Fila 0 y Fila 1 (frontera de longitud 3)
        let mut row0_weight = 0.0;
        let mut row1_weight = 0.0;
        for j in 0..PEPS_COLS {
            row0_weight += self.node_weights[0][j];
            row1_weight += self.node_weights[1][j];
        }

        let p0 = row0_weight.clamp(self.params.epsilon, 1.0 - self.params.epsilon);
        let p1 = row1_weight.clamp(self.params.epsilon, 1.0 - self.params.epsilon);
        let peps_entropy = -(p0 * p0.ln() + p1 * p1.ln()).max(0.0);

        // Flujo de área de frontera PEPS: correlación entre nodos vecinos cruzando la frontera
        let mut boundary_flux = 0.0;
        for j in 0..PEPS_COLS {
            boundary_flux += (self.node_weights[0][j] - self.node_weights[1][j]).abs();
        }

        // 3. Curvatura de Ollivier-Ricci en el Hipergrafo
        // Transporte de Wasserstein W_1 entre la distribución de Fila 0 y Fila 1
        let mut w1_dist = 0.0;
        let mut cum_diff = 0.0;
        for j in 0..PEPS_COLS {
            cum_diff += self.node_weights[0][j] - self.node_weights[1][j];
            w1_dist += cum_diff.abs();
        }
        let geodetic_dist = 1.0 + safe_spread * 0.05;
        let ricci_curvature =
            (1.0 - (w1_dist / geodetic_dist.max(self.params.epsilon))).clamp(-2.0, 1.0);

        // 4. Dimensión Espectral del Hipergrafo d_s = 2 * (1 + ricci)
        let spectral_dim = (2.0 * (1.0 + 0.5 * ricci_curvature)).clamp(0.5, 4.0);

        // 5. Veto y modulación de escala de slot ante dispersión hiperbólica
        let hyperbolic = ricci_curvature <= self.params.critical_negative_ricci;
        let slot_scale_factor = if hyperbolic {
            0.15
        } else if ricci_curvature < -0.30 {
            0.50 + 0.50
                * ((ricci_curvature - self.params.critical_negative_ricci)
                    / (-0.30 - self.params.critical_negative_ricci))
        } else {
            1.0
        }
        .clamp(0.10, 1.0);

        PepsHypergraphRicciReport {
            peps_entanglement_entropy: peps_entropy,
            ollivier_ricci_curvature: ricci_curvature,
            hypergraph_spectral_dimension: spectral_dim,
            boundary_area_flux: boundary_flux,
            slot_scale_factor,
            hyperbolic_dispersion_detected: hyperbolic,
        }
    }
}
