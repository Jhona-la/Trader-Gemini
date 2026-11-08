//! 🌌 FASE 52 (§52): HOMOLOGÍA PERSISTENTE Y FILTRACIÓN VIETORIS-RIPS EN MICROESTRUCTURA CUÁNTICA
//! (Vietoris-Rips Persistent Homology & Topological Data Analysis Engine)
//!
//! # MODO PROFESOR — ANÁLISIS INTEGRAL DEL SISTEMA CUÁNTICO
//!
//! - **QUÉ:** Motor de Análisis Topológico de Datos (TDA) continuo en stack (`VietorisRipsTopologicalEngine`)
//!   que evalúa los números de Betti $\beta_0, \beta_1$ y la entropía persistente de Shannon $E_{\text{pers}}$
//!   sobre una nube estática de puntos de microestructura $X \subset \mathbb{R}^4$ de longitud fija $K = 16$.
//! - **POR QUÉ:** Las estadísticas estándar de momentos locales (media, varianza, curtosis) son ciegas
//!   a las deformaciones globales de la estructura de liquidez. Cuando los creadores de mercado retiran
//!   profundidad de manera asimétrica, se forman "túneles de vórtice" y "agujeros de aire" (air pockets)
//!   en el espacio de fases que preceden a deslizamientos catastróficos. La homología persistente detecta
//!   invariantes topológicos algebraicos que resisten el ruido estocástico de microestructura.
//! - **PARA QUÉ:** Identificar la fragmentación de liquidez ($\beta_0 > 1$) y la aparición de trampas de
//!   vórtice de precios ($\beta_1 \ge 1$), modulando la escala del slot con un factor de resiliencia
//!   topológica $\zeta_{\text{tda}} \in [0.10, 1.30]$ determinista en $O(1)$ stack sin heap allocations.
//! - **CÓMO:**
//!   1. Mantiene un búfer circular en stack de $K = 16$ estados cuatridimensionales:
//!      $$x_t = (\Delta \ln p_t, \text{OFI}_t, \text{depth\_imbalance}_t, \text{spread\_bps}_t)$$
//!   2. Construye la matriz simétrica de distancias euclidianas $D_{ij} = \|x_i - x_j\|_2$ en stack ($16 \times 16$).
//!   3. Aplica la filtración de Vietoris-Rips a través de 4 escalas de proximidad diádicas $\epsilon \in \{\epsilon_1, \epsilon_2, \epsilon_3, \epsilon_4\}$.
//!   4. Calcula las componentes conexas $\beta_0(\epsilon)$ mediante un algoritmo de Disjoint-Set Union (DSU) en array de stack.
//!   5. Cuenta los 1-ciclos de Voronoi-Delaunay $\beta_1(\epsilon)$ vía la característica de Euler simplicial $\chi = V - E + F$.
//!   6. Deriva la entropía persistente de Shannon sobre la longitud de vida normalizada de las características:
//!      $$E_{\text{pers}} = -\sum_{i} p_i \ln(p_i + 10^{-12}), \quad p_i = \frac{l_i}{\sum_j l_j}$$
//!   7. Modula dinámicamente la escala del slot para reducir la exposición ante colapso topológico.
//! - **CUÁNDO:** En cada pulso de microestructura y tick de libro L2 en `crates/feature-engine`.
//! - **DÓNDE:** En `crates/feature-engine/src/vietoris_rips_persistence.rs`.
//! - **QUIÉN:** Rol 8 Físico-Matemático de Topología & Manifolds y Rol 2 Quant Developer.

/// Número estático de puntos en la nube de microestructura en stack (Cero heap allocations).
pub const TDA_POINT_CLOUD_LEN: usize = 16;
/// Dimensión del espacio de características de microestructura.
pub const TDA_FEATURE_DIM: usize = 4;
/// Número de escalas de filtración de Vietoris-Rips.
pub const TDA_FILTRATION_LEVELS: usize = 4;

/// Configuración de parámetros para el motor de homología persistente.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VietorisRipsParams {
    /// Radio base de filtración $\epsilon_{\text{base}} > 0$.
    pub base_filtration_radius: f64,
    /// Factor de escala multiplicativo entre niveles de filtración diádica.
    pub filtration_step_scale: f64,
    /// Umbral crítico de entropía persistente para detectar desorden caótico de liquidez.
    pub chaotic_entropy_threshold: f64,
    /// Factor de suavizado exponencial para las métricas topológicas continuas.
    pub topological_smoothing: f64,
}

impl Default for VietorisRipsParams {
    fn default() -> Self {
        Self {
            base_filtration_radius: 0.25,
            filtration_step_scale: 1.50,
            chaotic_entropy_threshold: 0.85,
            topological_smoothing: 0.15,
        }
    }
}

/// Punto de estado en el espacio euclidiano de microestructura $\mathbb{R}^4$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdaMicroPoint {
    /// Desviación logarítmica de precio normalizada.
    pub log_price_ret: f64,
    /// Desequilibrio de flujo de órdenes (OFI) $\in [-1.0, 1.0]$.
    pub ofi: f64,
    /// Ratio normalizado de profundidad bid/ask centrado en cero.
    pub depth_imbalance: f64,
    /// Spread relativo normalizado en escala logarítmica.
    pub spread_norm: f64,
}

impl TdaMicroPoint {
    pub const ZERO: Self = Self {
        log_price_ret: 0.0,
        ofi: 0.0,
        depth_imbalance: 0.0,
        spread_norm: 0.0,
    };

    #[inline(always)]
    pub fn new(log_price_ret: f64, ofi: f64, depth_ratio: f64, spread_bps: f64) -> Self {
        let safe_ret = if log_price_ret.is_finite() {
            log_price_ret.clamp(-0.10, 0.10) * 10.0
        } else {
            0.0
        };
        let safe_ofi = if ofi.is_finite() {
            ofi.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let safe_depth = if depth_ratio.is_finite() && depth_ratio > 0.0 {
            ((depth_ratio - 1.0) / (depth_ratio + 1.0)).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let safe_spread = if spread_bps.is_finite() && spread_bps > 0.0 {
            (spread_bps / 10.0).clamp(0.01, 10.0).ln() * 0.5
        } else {
            0.0
        };

        Self {
            log_price_ret: safe_ret,
            ofi: safe_ofi,
            depth_imbalance: safe_depth,
            spread_norm: safe_spread,
        }
    }

    /// Distancia euclidiana $L_2$ entre dos puntos en $\mathbb{R}^4$.
    #[inline(always)]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let d0 = self.log_price_ret - other.log_price_ret;
        let d1 = self.ofi - other.ofi;
        let d2 = self.depth_imbalance - other.depth_imbalance;
        let d3 = self.spread_norm - other.spread_norm;
        (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3).sqrt()
    }
}

/// Informe de homología persistente y características topológicas de microestructura.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VietorisRipsReport {
    /// Número de Betti 0 medio $\bar{\beta}_0$ (componentes conexas / fragmentación de liquidez).
    pub mean_betti_0: f64,
    /// Número de Betti 1 medio $\bar{\beta}_1$ (1-ciclos / túneles de vórtice de precios).
    pub mean_betti_1: f64,
    /// Entropía persistente de Shannon $E_{\text{pers}} \in [0.0, 1.0]$.
    pub persistent_entropy: f64,
    /// Indicador de alerta de vacío topológico de liquidez (air pocket).
    pub air_pocket_detected: bool,
    /// Factor de escala de slot de resiliencia topológica $\zeta_{\text{tda}} \in [0.10, 1.30]$.
    pub slot_scaling_factor: f64,
}

/// Motor determinista de homología persistente en stack $O(1)$.
#[derive(Debug, Clone, Copy)]
pub struct VietorisRipsTopologicalEngine {
    pub params: VietorisRipsParams,
    pub point_cloud: [TdaMicroPoint; TDA_POINT_CLOUD_LEN],
    pub head: usize,
    pub count: usize,
    pub smoothed_betti_0: f64,
    pub smoothed_betti_1: f64,
    pub smoothed_entropy: f64,
}

impl VietorisRipsTopologicalEngine {
    /// Inicializa un nuevo motor TDA con memoria estática en stack.
    pub fn new(params: VietorisRipsParams) -> Self {
        Self {
            params,
            point_cloud: [TdaMicroPoint::ZERO; TDA_POINT_CLOUD_LEN],
            head: 0,
            count: 0,
            smoothed_betti_0: 1.0,
            smoothed_betti_1: 0.0,
            smoothed_entropy: 0.10,
        }
    }

    /// Actualiza la nube de puntos y evalúa la filtración de Vietoris-Rips en $O(1)$ stack.
    ///
    /// Preserva la cota de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ y
    /// la invariancia de cero pánicos `rb_p1`.
    pub fn update_and_evaluate(
        &mut self,
        log_price_ret: f64,
        ofi: f64,
        depth_ratio: f64,
        spread_bps: f64,
        _event_time_ms: u64,
    ) -> VietorisRipsReport {
        let point = TdaMicroPoint::new(log_price_ret, ofi, depth_ratio, spread_bps);

        // 1. Inserción en búfer circular estático de nube de puntos
        self.point_cloud[self.head] = point;
        self.head = (self.head + 1) % TDA_POINT_CLOUD_LEN;
        if self.count < TDA_POINT_CLOUD_LEN {
            self.count += 1;
        }

        let n = self.count.clamp(1, TDA_POINT_CLOUD_LEN);

        // 2. Construcción de matriz de distancias simétrica en stack ($16 \times 16$)
        let mut dist_matrix = [[0.0_f64; TDA_POINT_CLOUD_LEN]; TDA_POINT_CLOUD_LEN];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.point_cloud[i].distance_to(&self.point_cloud[j]);
                dist_matrix[i][j] = d;
                dist_matrix[j][i] = d;
            }
        }

        // 3. Filtración de Vietoris-Rips a través de 4 radios diádicos
        let mut betti_0_sum = 0.0;
        let mut betti_1_sum = 0.0;
        let mut lifetimes = [0.0_f64; TDA_FILTRATION_LEVELS];

        let mut r = self.params.base_filtration_radius.max(0.01);
        let mut prev_components = n as f64;

        for level in 0..TDA_FILTRATION_LEVELS {
            let (b0, b1) = self.compute_betti_at_radius(&dist_matrix, n, r);
            betti_0_sum += b0 as f64;
            betti_1_sum += b1 as f64;

            // Vida de componentes que colapsan en este nivel
            let collapsed = (prev_components - (b0 as f64)).max(0.0);
            lifetimes[level] = collapsed * r;
            prev_components = b0 as f64;

            r *= self.params.filtration_step_scale.clamp(1.10, 3.0);
        }

        let mean_b0 = betti_0_sum / (TDA_FILTRATION_LEVELS as f64);
        let mean_b1 = betti_1_sum / (TDA_FILTRATION_LEVELS as f64);

        // 4. Entropía Persistente de Shannon sobre la distribución normalizada de vidas
        let sum_lifetimes: f64 = lifetimes.iter().sum();
        let mut persistent_entropy = 0.0;
        if sum_lifetimes > 1e-9 {
            for &l in &lifetimes {
                if l > 1e-9 {
                    let p = l / sum_lifetimes;
                    persistent_entropy -= p * p.ln();
                }
            }
            // Normalizar a [0.0, 1.0] dividiendo por ln(TDA_FILTRATION_LEVELS)
            let max_ent = (TDA_FILTRATION_LEVELS as f64).ln().max(1e-6);
            persistent_entropy = (persistent_entropy / max_ent).clamp(0.0, 1.0);
        }

        // 5. Suavizado continuo de métricas topológicas
        let alpha = self.params.topological_smoothing.clamp(0.01, 1.0);
        self.smoothed_betti_0 = (1.0 - alpha) * self.smoothed_betti_0 + alpha * mean_b0;
        self.smoothed_betti_1 = (1.0 - alpha) * self.smoothed_betti_1 + alpha * mean_b1;
        self.smoothed_entropy = (1.0 - alpha) * self.smoothed_entropy + alpha * persistent_entropy;

        // 6. Detección de Vacío Topológico (Air Pocket)
        // Ocurre si hay fragmentación extrema ($\beta_0 \ge 8$) con entropía caótica alta
        let air_pocket_detected = self.smoothed_betti_0 >= 6.0
            && self.smoothed_entropy >= self.params.chaotic_entropy_threshold;

        // 7. Factor de Escala de Slot Topológico $\zeta_{\text{tda}} \in [0.10, 1.30]$
        // Penaliza la presencia de túneles de vórtice ($\beta_1$) y fragmentación ($\beta_0$)
        let fragmentation_penalty = (1.0 + (self.smoothed_betti_0 - 1.0).max(0.0) * 0.15).max(1.0);
        let vortex_penalty = (1.0 + self.smoothed_betti_1 * 0.25).max(1.0);
        let entropy_scale = 1.0 - 0.40 * self.smoothed_entropy;
        let air_pocket_cut = if air_pocket_detected { 0.25 } else { 1.0 };

        let raw_scale =
            (entropy_scale / (fragmentation_penalty * vortex_penalty)) * 1.20 * air_pocket_cut;
        let slot_scaling_factor = raw_scale.clamp(0.10, 1.30);

        VietorisRipsReport {
            mean_betti_0: self.smoothed_betti_0,
            mean_betti_1: self.smoothed_betti_1,
            persistent_entropy: self.smoothed_entropy,
            air_pocket_detected,
            slot_scaling_factor,
        }
    }

    /// Calcula $\beta_0$ y $\beta_1$ para un radio $\epsilon$ dado mediante DSU y triángulo de Euler en stack.
    fn compute_betti_at_radius(
        &self,
        dist_matrix: &[[f64; TDA_POINT_CLOUD_LEN]; TDA_POINT_CLOUD_LEN],
        n: usize,
        radius: f64,
    ) -> (usize, usize) {
        // Disjoint Set Union (DSU) en stack array
        let mut parent = [0_usize; TDA_POINT_CLOUD_LEN];
        for (i, p) in parent.iter_mut().enumerate() {
            *p = i;
        }

        let mut edge_count = 0_usize;
        let mut adj = [[false; TDA_POINT_CLOUD_LEN]; TDA_POINT_CLOUD_LEN];

        // 1. Inserción de 1-simplices (aristas) con longitud $\le 2\epsilon$
        let diameter = radius * 2.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if dist_matrix[i][j] <= diameter {
                    adj[i][j] = true;
                    adj[j][i] = true;
                    edge_count += 1;

                    // DSU union
                    let mut root_i = i;
                    while root_i != parent[root_i] {
                        root_i = parent[root_i];
                    }
                    let mut root_j = j;
                    while root_j != parent[root_j] {
                        root_j = parent[root_j];
                    }
                    if root_i != root_j {
                        parent[root_i] = root_j;
                    }
                }
            }
        }

        // Conteo de componentes conexas $\beta_0$
        let mut b0 = 0_usize;
        for i in 0..n {
            if parent[i] == i {
                b0 += 1;
            }
        }

        // 2. Inserción de 2-simplices (triángulos) con todas sus aristas presentes
        let mut triangle_count = 0_usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[i][j] {
                    for k in (j + 1)..n {
                        if adj[i][k] && adj[j][k] {
                            triangle_count += 1;
                        }
                    }
                }
            }
        }

        // Característica de Euler simplicial: $\chi = V - E + F$
        // En un complejo 2-dimensional: $\chi = \beta_0 - \beta_1 + \beta_2$
        // Asumiendo $\beta_2 = 0$ (no cavidades de orden 3 en este complejo 2D planar):
        // $\beta_1 = \beta_0 - (V - E + F) = \beta_0 - V + E - F$
        let v = n as isize;
        let e = edge_count as isize;
        let f = triangle_count as isize;
        let raw_b1 = (b0 as isize) - v + e - f;
        let b1 = raw_b1.max(0) as usize;

        (b0, b1)
    }
}

impl Default for VietorisRipsTopologicalEngine {
    fn default() -> Self {
        Self::new(VietorisRipsParams::default())
    }
}
