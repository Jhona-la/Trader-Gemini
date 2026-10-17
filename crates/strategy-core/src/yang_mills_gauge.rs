//! ⚡ TEORÍA DE GAUGE YANG-MILLS & CURVATURA DE ARBITRAJE MULTIACTIVO (OLA Ω36)
//!
//! ## Fundamento Teórico y Geometría Diferencial
//!
//! En un universo financiero multivariante continuo con $N$ activos, los tipos de cambio
//! o relaciones de paridad entre activos definen una 1-forma de conexión $A$ sobre un
//! fibrado principal $G = \mathbb{R}^+$ (álgebra de Lie abelianizada $\mathfrak{g} \cong \mathbb{R}$
//! en espacio logarítmico).
//!
//! Para cualquier par de activos $(i, j)$, la conexión de paridad logarítmica es:
//!   $$A_{ij} = \ln(P_i) - \beta_{ij} \ln(P_j)$$
//!
//! donde $\beta_{ij}$ es el ratio de cobertura o elasticidad de paridad.
//!
//! En un mercado en perfecto equilibrio libre de arbitraje, la holonomía alrededor de cualquier
//! bucle cerrado es idénticamente nula (curvatura plana). Para un 3-ciclo ordenado (plaqueta triangular)
//! $i \to j \to k \to i$, la 2-forma de curvatura de Yang-Mills (bucle de Wilson) es:
//!   $$F_{ijk} = A_{ij} + A_{jk} + A_{ki}$$
//!
//! Propiedades invariantes fundamentales:
//! 1. **Invariancia Gauge**: Bajo cualquier reescalado de numerario local $P_i \mapsto \lambda_i P_i$,
//!    la curvatura $F_{ijk}$ es estrictamente invariante si los $\beta$ satisfacen la condición
//!    de paridad cerrada $\beta_{ij} \beta_{jk} \beta_{ki} = 1$.
//! 2. **Antisimetría y Permutación Cíclica**:
//!    $$F_{ijk} = F_{jki} = F_{kij} = -F_{ikj} = -F_{kji} = -F_{jik}$$
//! 3. **Densidad de Acción de Yang-Mills**:
//!    $$\mathcal{S}_{\text{YM}} = \frac{1}{2} \sum_{i < j < k} |F_{ijk}|^2$$
//!    Mide la energía de dislocación o tensión de arbitraje global en el universo multiactivo.
//! 4. **Corriente Restauradora de Gauge**:
//!    $$\mathcal{J}_i = \frac{1}{\binom{N-1}{2}} \sum_{j < k, j \neq i, k \neq i} F_{ijk}$$
//!    Proporciona la fuerza de gradiente topológico para restaurar la paridad del activo $i$.

use crate::{QuantumStrategy, TradeHorizon};
use omniscient_registry::OmniscientRegistry;
use std::f64;
use std::sync::Arc;

pub const MAX_GAUGE_ASSETS: usize = 32;

/// ⚡ OLA Ω122 / Ω138: Parámetros desacoplados y configurables para la teoría de gauge Yang-Mills SU(N)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YangMillsParams {
    /// Tasa de adaptación estocástica LMS normalizada $\gamma$ (default: 0.005)
    pub adaptation_rate: f64,
    /// Cota inferior admisible para ratios de conexión $\beta_{ij}$ (default: 0.01)
    pub beta_min: f64,
    /// Cota superior admisible para ratios de conexión $\beta_{ij}$ (default: 100.0)
    pub beta_max: f64,
    /// Factor de escala para corrientes restauradoras $\mathcal{J}_i$ en el argumento tanh (default: 1.0)
    pub current_scale: f64,
    /// OLA Ω138: Constante de acoplamiento de gauge no-abeliano $g$ para el conmutador $[A_i, A_j]$ (default: 0.05)
    pub gauge_coupling_g: f64,
}

impl Default for YangMillsParams {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.005,
            beta_min: 0.01,
            beta_max: 100.0,
            current_scale: 1.0,
            gauge_coupling_g: 0.05,
        }
    }
}

impl YangMillsParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.adaptation_rate.is_finite() || !(1e-5..=0.1).contains(&self.adaptation_rate) {
            return Err(format!("Invalid adaptation_rate: {}", self.adaptation_rate));
        }
        if !self.beta_min.is_finite() || !(0.0001..=1.0).contains(&self.beta_min) {
            return Err(format!("Invalid beta_min: {}", self.beta_min));
        }
        if !self.beta_max.is_finite() || !(1.0..=10000.0).contains(&self.beta_max) || self.beta_max <= self.beta_min {
            return Err(format!("Invalid beta_max: {}", self.beta_max));
        }
        if !self.current_scale.is_finite() || !(0.01..=10.0).contains(&self.current_scale) {
            return Err(format!("Invalid current_scale: {}", self.current_scale));
        }
        if !self.gauge_coupling_g.is_finite() || !(0.0..=1.0).contains(&self.gauge_coupling_g) {
            return Err(format!("Invalid gauge_coupling_g: {}", self.gauge_coupling_g));
        }
        Ok(())
    }
}

/// OLA Ω138: Métricas de curvatura gauge Yang-Mills SU(N) y holonomía de bucles de Wilson
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YangMillsMetrics {
    /// Densidad media de acción de Wilson $\bar{\mathcal{S}}_{\text{YM}} \ge 0$
    pub action_density: f64,
    /// Traza media normalizada del bucle de Wilson $\bar{W} \in [-1.0, 1.0]$ (1.0 = equilibrio gauge perfecto)
    pub wilson_loop: f64,
    /// Densidad de holonomía $\bar{H} = 1.0 - \bar{W} \in [0.0, 2.0]$
    pub holonomy_density: f64,
    /// Norma media del conmutador no-abeliano $[A \wedge A]$
    pub non_abelian_commutator_norm: f64,
    /// Número de triángulos activos computados
    pub active_triangles: usize,
}

impl Default for YangMillsMetrics {
    fn default() -> Self {
        Self {
            action_density: 0.0,
            wilson_loop: 1.0,
            holonomy_density: 0.0,
            non_abelian_commutator_norm: 0.0,
            active_triangles: 0,
        }
    }
}

/// Estado y estimador de curvatura de Yang-Mills sobre el universo multiactivo.
#[derive(Debug, Clone)]
pub struct YangMillsGaugeEngine {
    /// Número de activos activos en el fibrado (≤ MAX_GAUGE_ASSETS)
    pub num_assets: usize,
    /// Conexiones de paridad estimadas (ratios beta de paridad)
    pub beta_matrix: [[f64; MAX_GAUGE_ASSETS]; MAX_GAUGE_ASSETS],
    /// Tasa de adaptación RLS para betas de conexión
    pub adaptation_rate: f64,
    /// OLA Ω122: Parámetros desacoplados y configurables
    pub params: YangMillsParams,
    /// Últimos log-precios observados
    pub last_ln_prices: [f64; MAX_GAUGE_ASSETS],
    /// Último retorno no nulo observado por activo (G8-OMEGA-04: sincronizador Hayashi-Yoshida por evento)
    pub last_nonzero_ret: [f64; MAX_GAUGE_ASSETS],
    /// Secuencia monótona de innovaciones de precio no nulas por activo
    pub ret_seq: [u64; MAX_GAUGE_ASSETS],
    /// Secuencias consumidas por par (para i < j: [i][j] = seq_i consumido, [j][i] = seq_j consumido)
    pub last_pair_seq: [[u64; MAX_GAUGE_ASSETS]; MAX_GAUGE_ASSETS],
    /// Conteo de observaciones válidas
    pub count: u64,
    /// Registro omnisciente para deliberación y lectura de parámetros continuos
    pub registry: Option<Arc<OmniscientRegistry>>,
    /// OLA Ω138: Métricas instantáneas de curvatura no-abeliana y bucles de Wilson
    pub metrics: YangMillsMetrics,
    /// OLA Ω138: Últimas corrientes restauradoras calculadas en stack
    pub currents: [f64; MAX_GAUGE_ASSETS],
}

impl Default for YangMillsGaugeEngine {
    fn default() -> Self {
        Self::new(MAX_GAUGE_ASSETS)
    }
}

impl YangMillsGaugeEngine {
    /// Crea un nuevo motor de curvatura de gauge para $N$ activos.
    pub fn new(num_assets: usize) -> Self {
        let n = num_assets.clamp(3, MAX_GAUGE_ASSETS);
        let mut beta = [[1.0; MAX_GAUGE_ASSETS]; MAX_GAUGE_ASSETS];
        for i in 0..MAX_GAUGE_ASSETS {
            beta[i][i] = 1.0;
        }
        let default_params = YangMillsParams::default();
        Self {
            num_assets: n,
            beta_matrix: beta,
            adaptation_rate: default_params.adaptation_rate,
            params: default_params,
            last_ln_prices: [0.0; MAX_GAUGE_ASSETS],
            last_nonzero_ret: [0.0; MAX_GAUGE_ASSETS],
            ret_seq: [0; MAX_GAUGE_ASSETS],
            last_pair_seq: [[0; MAX_GAUGE_ASSETS]; MAX_GAUGE_ASSETS],
            count: 0,
            registry: None,
            metrics: YangMillsMetrics::default(),
            currents: [0.0; MAX_GAUGE_ASSETS],
        }
    }

    /// OLA Ω138: Acceso inmutable en tiempo O(1) a las métricas de curvatura y bucles de Wilson
    #[inline]
    pub fn current_metrics(&self) -> YangMillsMetrics {
        self.metrics
    }

    /// OLA Ω138: Acceso inmutable en tiempo O(1) a las corrientes restauradoras J_i
    #[inline]
    pub fn current_currents(&self) -> [f64; MAX_GAUGE_ASSETS] {
        self.currents
    }

    /// OLA Ω122: Constructor con parámetros desacoplados específicos
    pub fn with_params(mut self, params: YangMillsParams) -> Self {
        if params.validate().is_ok() {
            self.adaptation_rate = params.adaptation_rate;
            self.params = params;
        }
        self
    }

    /// OLA Ω122: Actualización dinámica de parámetros
    pub fn update_params(&mut self, params: YangMillsParams) {
        if params.validate().is_ok() {
            self.adaptation_rate = params.adaptation_rate;
            self.params = params;
        }
    }

    /// Configura la tasa de adaptación adaptativa de los coeficientes de conexión $\beta_{ij}$.
    pub fn with_adaptation_rate(mut self, rate: f64) -> Self {
        if rate.is_finite() && rate > 0.0 {
            let r = rate.clamp(1e-5, 0.1);
            self.adaptation_rate = r;
            self.params.adaptation_rate = r;
        }
        self
    }

    /// Configura el ratio de elasticidad de conexión beta entre dos activos i y j.
    pub fn with_beta(mut self, i: usize, j: usize, beta: f64) -> Self {
        if i < MAX_GAUGE_ASSETS && j < MAX_GAUGE_ASSETS && beta.is_finite() && beta > 0.0 {
            let b = beta.clamp(self.params.beta_min, self.params.beta_max);
            self.beta_matrix[i][j] = b;
            self.beta_matrix[j][i] = (1.0 / b).clamp(self.params.beta_min, self.params.beta_max);
        }
        self
    }

    /// Actualiza los precios de los activos y calcula la curvatura de Yang-Mills.
    ///
    /// # Retorno
    /// Tupla con:
    /// - `action_density`: Densidad media de acción gauge por plaqueta triangular $\bar{\mathcal{S}}_{\text{YM}} \ge 0$.
    /// - `restoring_currents`: Vector de corrientes restauradoras $\mathcal{J}_i \in [-1, 1]$ para cada activo.
    pub fn update_and_calculate_curvature(
        &mut self,
        prices: &[f64],
    ) -> (f64, [f64; MAX_GAUGE_ASSETS]) {
        let limit = self.num_assets.min(prices.len()).min(MAX_GAUGE_ASSETS);
        let mut valid_indices = [0usize; MAX_GAUGE_ASSETS];
        let mut ln_p = [0.0_f64; MAX_GAUGE_ASSETS];
        let mut n_valid = 0usize;

        for i in 0..limit {
            let p = prices[i];
            if p.is_finite() && p > 0.0 {
                valid_indices[n_valid] = i;
                ln_p[i] = p.ln();
                n_valid += 1;
            } else {
                // R7-R3-D-2 / linea_g:316: Resetear last_ln_prices para activos caducados/inválidos
                // para evitar que al regresar arrastren un salto espurio de retorno acumulado.
                self.last_ln_prices[i] = 0.0;
                self.last_nonzero_ret[i] = 0.0;
            }
        }

        if n_valid < 3 {
            return (0.0, [0.0; MAX_GAUGE_ASSETS]);
        }

        // 1. Precomputar innovaciones/retornos para activos válidos antes de cualquier mutación de estado
        // R7-R3-B-1: En frío (count == 0) o tras expiración de TTL (prev_p_i == 0.0), el activo
        // carece de precio previo; su retorno de innovación es estrictamente 0.0.
        // Asignar el nivel p_i inyectaba log(P) ~ 11.5 al LMS, corrompiendo beta y la curvatura.
        let mut returns = [0.0_f64; MAX_GAUGE_ASSETS];
        for vi in 0..n_valid {
            let i = valid_indices[vi];
            let p_i = ln_p[i];
            let prev_p_i = self.last_ln_prices[i];
            let ret = if self.count > 0 && prev_p_i != 0.0 {
                p_i - prev_p_i
            } else {
                0.0
            };
            if ret.abs() > 1e-15 {
                self.last_nonzero_ret[i] = ret;
                self.ret_seq[i] = self.ret_seq[i].wrapping_add(1);
            }
            returns[i] = ret;
        }

        // 2. Actualización adaptativa online de coeficientes beta sobre pares válidos (G8-OMEGA-04)
        // Sincronización por par Hayashi-Yoshida: un par (i, j) actualiza beta_matrix[i][j]
        // únicamente cuando AMBOS activos han registrado un retorno no nulo nuevo desde su última
        // actualización conjunta. Preserva R6-C6 (si un activo permanece estacionario, su seq no
        // avanza y beta no muta) y permite aprender co-movimiento en llamadas por-evento.
        if self.count > 0 {
            let gamma = self.params.adaptation_rate;
            let b_min = self.params.beta_min;
            let b_max = self.params.beta_max;
            for vi in 0..n_valid {
                let i = valid_indices[vi];
                for vj in (vi + 1)..n_valid {
                    let j = valid_indices[vj];
                    let seq_i = self.ret_seq[i];
                    let seq_j = self.ret_seq[j];
                    if seq_i > self.last_pair_seq[i][j] && seq_j > self.last_pair_seq[j][i] {
                        let r_i = self.last_nonzero_ret[i];
                        let r_j = self.last_nonzero_ret[j];

                        let err = r_i - self.beta_matrix[i][j] * r_j;
                        if err.is_finite() && r_j.abs() > 1e-12 {
                            let step = gamma * ((err * r_j) / (1.0 + r_j * r_j));
                            let new_beta = (self.beta_matrix[i][j] + step).clamp(b_min, b_max);
                            self.beta_matrix[i][j] = new_beta;
                            self.beta_matrix[j][i] = (1.0 / new_beta).clamp(b_min, b_max);
                        }
                        self.last_pair_seq[i][j] = seq_i;
                        self.last_pair_seq[j][i] = seq_j;
                    }
                }
            }
        }

        // 3. Cómputo de la curvatura 2-forma F_{ijk} para todos los 3-ciclos válidos (vi < vj < vk)
        let mut total_action = 0.0_f64;
        let mut total_wilson = 0.0_f64;
        let mut total_commutator = 0.0_f64;
        let mut currents = [0.0_f64; MAX_GAUGE_ASSETS];
        let mut cycle_counts = [0.0_f64; MAX_GAUGE_ASSETS];
        let mut total_cycles = 0usize;

        for vi in 0..n_valid {
            let i = valid_indices[vi];
            let r_i = if returns[i].abs() > 1e-15 {
                returns[i]
            } else {
                self.last_nonzero_ret[i]
            };
            for vj in (vi + 1)..n_valid {
                let j = valid_indices[vj];
                let r_j = if returns[j].abs() > 1e-15 {
                    returns[j]
                } else {
                    self.last_nonzero_ret[j]
                };
                for vk in (vj + 1)..n_valid {
                    let k = valid_indices[vk];
                    let r_k = if returns[k].abs() > 1e-15 {
                        returns[k]
                    } else {
                        self.last_nonzero_ret[k]
                    };

                    // Conexión de paridad A_{ab} sobre log-retornos o innovaciones
                    let a_ij = r_i - self.beta_matrix[i][j] * r_j;
                    let a_jk = r_j - self.beta_matrix[j][k] * r_k;
                    let a_ki = r_k - self.beta_matrix[k][i] * r_i;

                    // Holonomía longitudinal de bucle de Wilson F_loop = A_ij + A_jk + A_ki
                    let f_loop = a_ij + a_jk + a_ki;

                    // Dislocación transversal de cointegración triangular en el 2-símplex (F2.3b-07):
                    // Para 3 activos en numerario común, la estructura triangular se manifiesta cuando uno
                    // de los activos es la paridad cruzada o cointegrada de los otros dos:
                    // d_1 = r_i - β_ij·r_j - β_ik·r_k (k y j componen a i)
                    // d_2 = r_j - β_ji·r_i - β_jk·r_k (i y k componen a j)
                    // d_3 = r_k - β_ki·r_i - β_kj·r_j (i y j componen a k)
                    // El residual mínimo de cointegración selecciona la orientación geométrica del cruce.
                    let b_ij = self.beta_matrix[i][j];
                    let b_ik = self.beta_matrix[i][k];
                    let b_jk = self.beta_matrix[j][k];
                    let b_ji = self.beta_matrix[j][i];
                    let b_ki = self.beta_matrix[k][i];
                    let b_kj = self.beta_matrix[k][j];

                    let d1 = r_i - b_ij * r_j - b_ik * r_k;
                    let d2 = r_j - b_ji * r_i - b_jk * r_k;
                    let d3 = r_k - b_ki * r_i - b_kj * r_j;

                    let (d_cross, d_grad_i, d_grad_j, d_grad_k) = if d1.abs() <= d2.abs() && d1.abs() <= d3.abs() {
                        (d1, 1.0, -b_ij, -b_ik)
                    } else if d2.abs() <= d3.abs() {
                        (d2, -b_ji, 1.0, -b_jk)
                    } else {
                        (d3, -b_ki, -b_kj, 1.0)
                    };

                    // OLA Ω138: Curvatura gauge Yang-Mills no-abeliana F_{ijk} = F_loop + g · d_cross · (A_ij - A_jk)
                    let g = self.params.gauge_coupling_g;
                    let comm_term = if g > 0.0 && d_cross.is_finite() {
                        g * d_cross * (a_ij - a_jk)
                    } else {
                        0.0
                    };
                    let f_ym = f_loop + comm_term;

                    // Traza del bucle de Wilson: W = cos(F_ym)
                    let w_loop = if f_ym.is_finite() { f_ym.cos() } else { 1.0 };
                    // Densidad de acción Yang-Mills combinada de Wilson (longitudinal + transversal):
                    // S_plaqueta = 2.0 * (1 - cos(F_ym)) + d_cross^2 (para |F| << 1, 2(1 - cos F) ≈ F^2)
                    let action_plaqueta = if f_ym.is_finite() && d_cross.is_finite() {
                        2.0 * (1.0 - w_loop) + d_cross * d_cross
                    } else if f_ym.is_finite() {
                        2.0 * (1.0 - w_loop)
                    } else {
                        0.0
                    };

                    if action_plaqueta.is_finite() {
                        total_action += action_plaqueta;
                        total_wilson += w_loop;
                        total_commutator += (a_ij * a_jk + a_jk * a_ki + a_ki * a_ij).abs();
                        total_cycles += 1;

                        // Corrientes restauradoras invariantes como gradiente negativo covariante de la acción de Wilson:
                        // J = - ∇ S_Wilson = - sin(F_ym) · ∇ F - d_cross · ∇ d
                        let sin_f = if f_ym.is_finite() { f_ym.sin() } else { 0.0 };
                        let j_i = -sin_f * (1.0 - b_ki) - d_cross * d_grad_i;
                        let j_j = -sin_f * (1.0 - b_ij) - d_cross * d_grad_j;
                        let j_k = -sin_f * (1.0 - b_jk) - d_cross * d_grad_k;

                        currents[i] += j_i;
                        currents[j] += j_j;
                        currents[k] += j_k;

                        cycle_counts[i] += 1.0;
                        cycle_counts[j] += 1.0;
                        cycle_counts[k] += 1.0;
                    }
                }
            }
        }

        // 4. Actualizar memoria de precios y conteo causal SOLO al finalizar la curvatura (R6-C6)
        for vi in 0..n_valid {
            let i = valid_indices[vi];
            self.last_ln_prices[i] = ln_p[i];
        }
        self.count += 1;

        // Normalización de corrientes por el número de ciclos incidentes y clamp suave a [-1.0, 1.0]
        let c_scale = if self.params.current_scale.is_finite() && self.params.current_scale > 0.0 {
            self.params.current_scale
        } else {
            1.0
        };
        for vi in 0..n_valid {
            let i = valid_indices[vi];
            if cycle_counts[i] > 0.0 {
                currents[i] = ((currents[i] / cycle_counts[i]) * c_scale).tanh();
            }
        }

        // Densidad media de acción gauge por plaqueta triangular (magnitud intensiva e invariante a N)
        let (action_density, wilson_loop, holonomy_density, commutator_norm) = if total_cycles > 0 {
            let denom = total_cycles as f64;
            let a_dens = 0.5 * (total_action / denom);
            let w_loop = (total_wilson / denom).clamp(-1.0, 1.0);
            let h_dens = (1.0 - w_loop).max(0.0);
            let comm_norm = total_commutator / denom;
            (a_dens, w_loop, h_dens, comm_norm)
        } else {
            (0.0, 1.0, 0.0, 0.0)
        };

        self.metrics = YangMillsMetrics {
            action_density,
            wilson_loop,
            holonomy_density,
            non_abelian_commutator_norm: commutator_norm,
            active_triangles: total_cycles,
        };
        self.currents = currents;

        (action_density, currents)
    }

    /// Calcula la holonomía exacta de un 3-ciclo específico (i, j, k).
    #[inline]
    pub fn loop_holonomy(&self, i: usize, j: usize, k: usize) -> f64 {
        if i >= self.num_assets || j >= self.num_assets || k >= self.num_assets {
            return 0.0;
        }
        let ln_p = &self.last_ln_prices;
        let a_ij = ln_p[i] - self.beta_matrix[i][j] * ln_p[j];
        let a_jk = ln_p[j] - self.beta_matrix[j][k] * ln_p[k];
        let a_ki = ln_p[k] - self.beta_matrix[k][i] * ln_p[i];
        a_ij + a_jk + a_ki
    }
}

impl QuantumStrategy for YangMillsGaugeEngine {
    fn name(&self) -> &str {
        "YangMillsGaugeEngine"
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
        // R6-C11 / OLA Ω42: Preservar el ámbito por coin_id independientemente de si symbol está vacío,
        // evitando que caiga al ámbito global ("") sujeto a condición de carrera del último escritor.
        let cid_opt = Some(coin_id);
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };

        let current = r
            .get_scoped_parameter(sym_opt, cid_opt, "yang_mills_current", "YangMillsGaugeEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        if !current.is_finite() {
            return 0.0;
        }

        // Corriente gauge restauradora J_i:
        // Si J_i > 0, fuerza restauradora alcista; si J_i < 0, fuerza bajista.
        current.clamp(-1.0, 1.0)
    }

    fn is_operable_for_coin(&self, coin_id: usize, symbol: &str) -> bool {
        let sym_opt = if symbol.is_empty() { None } else { Some(symbol) };
        let cid_opt = Some(coin_id);
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return false,
        };

        let absent = r
            .get_scoped_parameter(sym_opt, cid_opt, "yang_mills_current_absent", "YangMillsGaugeEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.0) > 0.5;
        if absent {
            return false;
        }

        let active_triangles = r
            .get_scoped_parameter(sym_opt, cid_opt, "yang_mills_active_triangles", "YangMillsGaugeEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let has_current = r
            .get_scoped_parameter(sym_opt, cid_opt, "yang_mills_current", "YangMillsGaugeEngine")
            .map(|p| p.get_value())
            .filter(|v| v.is_finite())
            .is_some();

        // G8-OMEGA-05 / LG8-145-01: En producción `self.num_assets` se inicializa a la capacidad (30);
        // exigir `active_triangles >= 1.0` garantiza que en replay monoactivo o con < 3 precios
        // frescos (donde active_triangles == 0.0) Yang-Mills se declare inoperable y no diluya.
        self.num_assets >= 3 && has_current && active_triangles >= 1.0
    }

    fn horizon(&self) -> TradeHorizon {
        TradeHorizon::Continuous
    }
}
