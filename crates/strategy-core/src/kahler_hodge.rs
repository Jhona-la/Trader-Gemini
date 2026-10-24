//! 🌌 TEORÍA DE VARIEDADES KÄHLERIANAS Y CONJETURA DE HODGE (OLA Ω139)
//!
//! ## Fundamentación Matemática y Física Teórica
//!
//! En un universo financiero multivariante continuo con $N$ activos ($N \le 32$), el espacio
//! de fases de cada activo $j$ se representa de forma natural como una coordenada compleja:
//!   $$z_j(t) = x_j(t) + i \cdot y_j(t)$$
//! donde:
//! - $x_j(t) = \ln(P_j(t)) - \overline{\ln(P_j)}$: Coordenada de potencial conservativo (log-precio centrado).
//! - $y_j(t) = \tanh(\text{OFI}_j(t) / \sigma_j)$: Coordenada de momento conjugado (desequilibrio de flujo L2/L3).
//!
//! ### 1. Potencial y Métrica de Kähler
//! El espacio complejo $\mathbb{C}^N$ está dotado de un potencial de Kähler estrictamente plurisubarmónico:
//!   $$\mathcal{K}(z, \bar{z}) = \frac{1}{2} \sum_{j=1}^N |z_j|^2 + \frac{1}{2} \sum_{j < k} c_{jk} (z_j \bar{z}_k + \bar{z}_j z_k)$$
//! donde $c_{jk} \in (-1, 1)$ es el tensor de co-movimiento o correlación en tiempo real.
//!
//! La métrica de Kähler hermitiana asociada es:
//!   $$g_{j\bar{k}} = \frac{\partial^2 \mathcal{K}}{\partial z_j \partial \bar{z}_k} = \delta_{jk} + c_{jk} (1 - \delta_{jk})$$
//!
//! Y la 2-forma fundamental de Kähler cerrada ($d\omega = 0$):
//!   $$\omega = \frac{i}{2} \sum_{j,k} g_{j\bar{k}} dz_j \wedge d\bar{z}_k$$
//!
//! ### 2. Conjetura de Hodge y Descomposición de Formas Armónicas (1, 1)
//! Por el Teorema de Descomposición de Hodge sobre variedades de Kähler:
//!   $$H^2(M, \mathbb{C}) \cong H^{2,0} \oplus H^{1,1} \oplus H^{0,2}$$
//! Las 2-formas de tipo $(1, 1)$ representan ciclos algebraicos cerrados de cointegración:
//!   $$\Omega_{jk}^{(1,1)} = \frac{i}{2} (dz_j \wedge d\bar{z}_k - dz_k \wedge d\bar{z}_j)$$
//!
//! La energía de Hodge $(1, 1)$, $E_{1,1}$, cuantifica la coherencia algebraica global de paridad.
//! Una alta energía $E_{1,1}$ garantiza que las relaciones de arbitraje multiactivo son estables,
//! mientras que una disipación de $E_{1,1}$ advierte de rupturas estructurales y contagio.
//!
//! ### 3. Curvatura de Ricci y Geometría Calabi-Yau
//! La curvatura escalar de Ricci $R_{\text{Kahler}} = \text{Tr}(g^{-1} R_{j\bar{k}})$ caracteriza el régimen:
//! - $R \approx 0$: **Régimen Calabi-Yau** (Ricci-flat, equilibrio armónico perfecto).
//! - $R < 0$: **Régimen Hiperbólico** (expansión divergente de volatilidad y spreads).
//! - $R > 0$: **Régimen Elíptico / Esférico** (compresión y reversión a la media).
//!
//! ### 4. Corrientes Holomorfas Restauradoras $\mathcal{J}_{\text{Kahler}, j}$
//!   $$\mathcal{J}_{\text{Kahler}, j} = \tanh\left(-\alpha \cdot \text{Re}\left(\frac{\partial \mathcal{K}}{\partial \bar{z}_j}\right) - \beta \cdot \text{Im}\left(\frac{\partial \mathcal{K}}{\partial \bar{z}_j}\right)\right)$$

use crate::QuantumStrategy;
use omniscient_registry::OmniscientRegistry;
use std::f64;
use std::sync::Arc;

pub const MAX_KAHLER_ASSETS: usize = 32;

/// 🌌 OLA Ω139: Parámetros configurables y desacoplados para la variedad Kähleriana
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KahlerHodgeParams {
    /// Peso de la componente real (precio / potencial) en la corriente restauradora (default: 1.0)
    pub alpha_price_weight: f64,
    /// Peso de la componente imaginaria (flujo / momento) en la corriente restauradora (default: 0.5)
    pub beta_flow_weight: f64,
    /// Factor de adaptación online LMS para la matriz de correlación hermitiana (default: 0.01)
    pub adaptation_rate: f64,
    /// Tolerancia de curvatura para identificar régimen Calabi-Yau $|R| \le \epsilon$ (default: 0.05)
    pub calabi_yau_tolerance: f64,
}

impl Default for KahlerHodgeParams {
    fn default() -> Self {
        Self {
            alpha_price_weight: 1.0,
            beta_flow_weight: 0.5,
            adaptation_rate: 0.01,
            calabi_yau_tolerance: 0.05,
        }
    }
}

impl KahlerHodgeParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.alpha_price_weight.is_finite() || !(0.01..=10.0).contains(&self.alpha_price_weight) {
            return Err(format!("Invalid alpha_price_weight: {}", self.alpha_price_weight));
        }
        if !self.beta_flow_weight.is_finite() || !(0.0..=10.0).contains(&self.beta_flow_weight) {
            return Err(format!("Invalid beta_flow_weight: {}", self.beta_flow_weight));
        }
        if !self.adaptation_rate.is_finite() || !(1e-5..=0.20).contains(&self.adaptation_rate) {
            return Err(format!("Invalid adaptation_rate: {}", self.adaptation_rate));
        }
        if !self.calabi_yau_tolerance.is_finite() || !(1e-4..=1.0).contains(&self.calabi_yau_tolerance) {
            return Err(format!("Invalid calabi_yau_tolerance: {}", self.calabi_yau_tolerance));
        }
        Ok(())
    }
}

/// 🌌 OLA Ω139: Métricas instantáneas de la variedad de Kähler y cohomología de Hodge
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KahlerHodgeMetrics {
    /// Escalar de curvatura de Ricci $R_{\text{Kahler}} \in \mathbb{R}$
    pub ricci_scalar: f64,
    /// Energía de formas armónicas $(1, 1)$ $E_{1,1} \ge 0$ (intensidad de cointegración algebraica)
    pub hodge_1_1_energy: f64,
    /// Valor instantáneo del potencial de Kähler $\mathcal{K}(z, \bar{z}) \ge 0$
    pub kahler_potential: f64,
    /// Distancia al vacío de Calabi-Yau $|R_{\text{Kahler}}| \ge 0$
    pub calabi_yau_distance: f64,
    /// Indicador continuo de equilibrio Calabi-Yau $\in [0.0, 1.0]$ (1.0 = plano / Ricci-flat)
    pub calabi_yau_flatness: f64,
    /// Número de activos válidos en la variedad
    pub active_assets: usize,
}

impl Default for KahlerHodgeMetrics {
    fn default() -> Self {
        Self {
            ricci_scalar: 0.0,
            hodge_1_1_energy: 0.0,
            kahler_potential: 0.0,
            calabi_yau_distance: 0.0,
            calabi_yau_flatness: 1.0,
            active_assets: 0,
        }
    }
}

/// Motor cuántico de Variedades de Kähler y Conjetura de Hodge en stack zero-alloc
#[derive(Debug, Clone)]
pub struct KahlerHodgeEngine {
    /// Número de activos gestionados (≤ MAX_KAHLER_ASSETS)
    pub num_assets: usize,
    /// Parámetros analíticos configurables
    pub params: KahlerHodgeParams,
    /// Métricas instantáneas calculadas en el último tick
    pub metrics: KahlerHodgeMetrics,
    /// Corrientes restauradoras holomorfas por activo
    pub currents: [f64; MAX_KAHLER_ASSETS],
    /// Matriz de co-movimiento hermitiana (correlación cruzada)
    pub corr_matrix: [[f64; MAX_KAHLER_ASSETS]; MAX_KAHLER_ASSETS],
    /// Últimos log-precios observados
    pub last_ln_prices: [f64; MAX_KAHLER_ASSETS],
    /// Último retorno no nulo observado por activo (G8-OMEGA-03: sincronizador Hayashi-Yoshida por evento)
    pub last_nonzero_ret: [f64; MAX_KAHLER_ASSETS],
    /// Secuencia monótona de innovaciones de precio no nulas por activo
    pub ret_seq: [u64; MAX_KAHLER_ASSETS],
    /// Secuencias consumidas por par (para i < j: [i][j] = seq_i consumido, [j][i] = seq_j consumido)
    pub last_pair_seq: [[u64; MAX_KAHLER_ASSETS]; MAX_KAHLER_ASSETS],
    /// Conteo de observaciones válidas
    pub count: u64,
    /// Registro omnisciente
    pub registry: Option<Arc<OmniscientRegistry>>,
}

impl Default for KahlerHodgeEngine {
    fn default() -> Self {
        Self::new(MAX_KAHLER_ASSETS)
    }
}

impl KahlerHodgeEngine {
    /// Crea un nuevo motor de variedades Kählerianas para $N$ activos.
    pub fn new(num_assets: usize) -> Self {
        let n = num_assets.clamp(2, MAX_KAHLER_ASSETS);
        let mut corr = [[0.0; MAX_KAHLER_ASSETS]; MAX_KAHLER_ASSETS];
        for i in 0..MAX_KAHLER_ASSETS {
            corr[i][i] = 1.0;
        }
        let default_params = KahlerHodgeParams::default();
        Self {
            num_assets: n,
            params: default_params,
            metrics: KahlerHodgeMetrics::default(),
            currents: [0.0; MAX_KAHLER_ASSETS],
            corr_matrix: corr,
            last_ln_prices: [0.0; MAX_KAHLER_ASSETS],
            last_nonzero_ret: [0.0; MAX_KAHLER_ASSETS],
            ret_seq: [0; MAX_KAHLER_ASSETS],
            last_pair_seq: [[0; MAX_KAHLER_ASSETS]; MAX_KAHLER_ASSETS],
            count: 0,
            registry: None,
        }
    }

    /// OLA Ω139: Acceso inmutable en tiempo O(1) a las métricas de Kähler y Hodge
    #[inline]
    pub fn current_metrics(&self) -> KahlerHodgeMetrics {
        self.metrics
    }

    /// OLA Ω139: Acceso inmutable en tiempo O(1) a las corrientes holomorfas restauradoras
    #[inline]
    pub fn current_currents(&self) -> [f64; MAX_KAHLER_ASSETS] {
        self.currents
    }

    /// Constructor con parámetros personalizados
    pub fn with_params(mut self, params: KahlerHodgeParams) -> Self {
        if params.validate().is_ok() {
            self.params = params;
        }
        self
    }

    /// Actualización dinámica de parámetros
    pub fn update_params(&mut self, params: KahlerHodgeParams) {
        if params.validate().is_ok() {
            self.params = params;
        }
    }

    /// Actualiza el estado multiactivo continuo y evalúa la geometría de Kähler y formas de Hodge en stack.
    ///
    /// # Argumentos
    /// - `prices`: Precios spot/mark actuales para cada activo.
    /// - `ofis`: Desequilibrio de flujo de órdenes (Order Flow Imbalance) L2/L3 actual.
    ///
    /// # Retorno
    /// Tupla con `(metrics, currents)`.
    pub fn update_and_calculate_manifold(
        &mut self,
        prices: &[f64],
        ofis: &[f64],
    ) -> (KahlerHodgeMetrics, [f64; MAX_KAHLER_ASSETS]) {
        let limit = self.num_assets.min(prices.len()).min(ofis.len()).min(MAX_KAHLER_ASSETS);
        let mut valid_indices = [0usize; MAX_KAHLER_ASSETS];
        let mut n_valid = 0usize;

        let mut x = [0.0_f64; MAX_KAHLER_ASSETS];
        let mut y = [0.0_f64; MAX_KAHLER_ASSETS];
        let mut returns = [0.0_f64; MAX_KAHLER_ASSETS];

        for i in 0..limit {
            let p = prices[i];
            let ofi = ofis[i];
            if p.is_finite() && p > 0.0 && ofi.is_finite() {
                valid_indices[n_valid] = i;
                let ln_p = p.ln();
                let prev_p = self.last_ln_prices[i];
                let ret = if self.count > 0 && prev_p != 0.0 {
                    ln_p - prev_p
                } else {
                    0.0
                };
                if ret.abs() > 1e-15 {
                    self.last_nonzero_ret[i] = ret;
                    self.ret_seq[i] = self.ret_seq[i].wrapping_add(1);
                }
                returns[i] = ret;
                x[i] = if ret.abs() > 1e-15 {
                    ret
                } else {
                    self.last_nonzero_ret[i]
                }; // Coordenada de flujo potencial centrada en innovación reciente
                y[i] = ofi.clamp(-10.0, 10.0).tanh(); // Coordenada de momento conjugado
                n_valid += 1;
            } else {
                self.last_ln_prices[i] = 0.0;
                self.last_nonzero_ret[i] = 0.0;
            }
        }

        if n_valid < 2 {
            return (KahlerHodgeMetrics::default(), [0.0; MAX_KAHLER_ASSETS]);
        }

        // 1. Actualización adaptativa LMS de la métrica hermitiana (matriz de correlación cruzada)
        // G8-OMEGA-03: Sincronización asíncrona por par (Hayashi-Yoshida). Un par (i, j) actualiza
        // su co-movimiento cuando AMBOS activos han registrado una nueva innovación de precio desde
        // su última actualización conjunta, funcionando idénticamente en modo síncrono y por-evento.
        if self.count > 0 {
            let gamma = self.params.adaptation_rate;
            for vi in 0..n_valid {
                let i = valid_indices[vi];
                for vj in (vi + 1)..n_valid {
                    let j = valid_indices[vj];
                    let seq_i = self.ret_seq[i];
                    let seq_j = self.ret_seq[j];
                    if seq_i > self.last_pair_seq[i][j] && seq_j > self.last_pair_seq[j][i] {
                        let ri = self.last_nonzero_ret[i];
                        let rj = self.last_nonzero_ret[j];
                        let err = ri * rj - self.corr_matrix[i][j];
                        if err.is_finite() {
                            let step = gamma * err;
                            let new_c = (self.corr_matrix[i][j] + step).clamp(-0.99, 0.99);
                            self.corr_matrix[i][j] = new_c;
                            self.corr_matrix[j][i] = new_c;
                        }
                        self.last_pair_seq[i][j] = seq_i;
                        self.last_pair_seq[j][i] = seq_j;
                    }
                }
            }
        }

        // 2. Cómputo del Potencial de Kähler K(z, \bar{z})
        // K = 1/2 sum |z_i|^2 + sum_{i < j} c_{ij} Re(z_i \bar{z}_j)
        // donde Re(z_i \bar{z}_j) = x_i x_j + y_i y_j
        let mut potential_sum = 0.0_f64;
        let mut offdiag_c2_sum = 0.0_f64;
        for vi in 0..n_valid {
            let i = valid_indices[vi];
            let xi = x[i];
            let yi = y[i];
            potential_sum += 0.5 * (xi * xi + yi * yi);

            for vj in (vi + 1)..n_valid {
                let j = valid_indices[vj];
                let xj = x[j];
                let yj = y[j];
                let cij = self.corr_matrix[i][j];

                potential_sum += cij * (xi * xj + yi * yj);
                offdiag_c2_sum += cij * cij;
            }
        }

        // 3. Cómputo de la Curvatura de Ricci Escalar
        // Para métrica de Kähler g = I + C:
        // ln det(g) ≈ - 1/2 Tr(C^2) = - sum_{i < j} c_{ij}^2
        // R_{Kahler} = - laplaciano_z ln det(g) proporcional a - sum c_{ij}^2
        // Normalizamos por el número de pares activos
        let num_pairs = (n_valid * (n_valid - 1)) / 2;
        let mean_c2 = if num_pairs > 0 {
            offdiag_c2_sum / (num_pairs as f64)
        } else {
            0.0
        };
        // Escalar de Ricci característico: R < 0 si correlaciones son expansivas/fuertes
        let ricci_scalar = -mean_c2 * 2.0;
        let cy_dist = ricci_scalar.abs();
        let cy_tol = self.params.calabi_yau_tolerance;
        let cy_flatness = (1.0 - (cy_dist / cy_tol)).clamp(0.0, 1.0);

        // 4. Cómputo de Formas Armónicas de Hodge (1, 1)
        // Omega_{ij}^{(1,1)} = 1/2 (x_i y_j - x_j y_i) (área simpléctica en plano complejo)
        let mut total_hodge_1_1 = 0.0_f64;
        for vi in 0..n_valid {
            let i = valid_indices[vi];
            let xi = x[i];
            let yi = y[i];
            for vj in (vi + 1)..n_valid {
                let j = valid_indices[vj];
                let xj = x[j];
                let yj = y[j];

                // Producto exterior simpléctico cruzado (1,1)
                let symplectic_area = 0.5 * (xi * yj - xj * yi);
                total_hodge_1_1 += symplectic_area * symplectic_area;
            }
        }
        let hodge_1_1_energy = if num_pairs > 0 {
            total_hodge_1_1 / (num_pairs as f64)
        } else {
            0.0
        };

        // 5. Cómputo de Corrientes Holomorfas Restauradoras J_{Kahler, i}
        // J_i = - nabla_i K
        // dK/dx_i = x_i + sum_{j != i} c_{ij} x_j
        // dK/dy_i = y_i + sum_{j != i} c_{ij} y_j
        let mut currents = [0.0_f64; MAX_KAHLER_ASSETS];
        let alpha = self.params.alpha_price_weight;
        let beta = self.params.beta_flow_weight;

        for vi in 0..n_valid {
            let i = valid_indices[vi];
            let xi = x[i];
            let yi = y[i];

            let mut grad_x = xi;
            let mut grad_y = yi;

            for vj in 0..n_valid {
                if vi == vj {
                    continue;
                }
                let j = valid_indices[vj];
                let cij = self.corr_matrix[i][j];
                grad_x += cij * x[j];
                grad_y += cij * y[j];
            }

            // Fuerza restauradora covariante negativa: empuja hacia el equilibrio K = 0
            let force = - (alpha * grad_x + beta * grad_y);
            currents[i] = force.clamp(-5.0, 5.0).tanh();
        }

        // 6. Actualizar memoria y conteo de ticks
        for vi in 0..n_valid {
            let i = valid_indices[vi];
            self.last_ln_prices[i] = prices[i].ln();
        }
        self.count += 1;

        let metrics = KahlerHodgeMetrics {
            ricci_scalar,
            hodge_1_1_energy,
            kahler_potential: potential_sum,
            calabi_yau_distance: cy_dist,
            calabi_yau_flatness: cy_flatness,
            active_assets: n_valid,
        };

        self.metrics = metrics;
        self.currents = currents;

        (metrics, currents)
    }
}

impl QuantumStrategy for KahlerHodgeEngine {
    fn name(&self) -> &str {
        "KahlerHodgeEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() { None } else { Some(symbol) };
        let cid_opt = Some(coin_id);
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };

        let current = r
            .get_scoped_parameter(sym_opt, cid_opt, "kahler_restoring_current", "KahlerHodgeEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        if !current.is_finite() {
            return 0.0;
        }

        current.clamp(-1.0, 1.0)
    }
}
