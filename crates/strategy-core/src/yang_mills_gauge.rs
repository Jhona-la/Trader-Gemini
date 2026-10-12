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

/// Estado y estimador de curvatura de Yang-Mills sobre el universo multiactivo.
#[derive(Debug, Clone)]
pub struct YangMillsGaugeEngine {
    /// Número de activos activos en el fibrado (≤ MAX_GAUGE_ASSETS)
    pub num_assets: usize,
    /// Conexiones de paridad estimadas (ratios beta de paridad)
    pub beta_matrix: [[f64; MAX_GAUGE_ASSETS]; MAX_GAUGE_ASSETS],
    /// Tasa de adaptación RLS para betas de conexión
    pub adaptation_rate: f64,
    /// Últimos log-precios observados
    pub last_ln_prices: [f64; MAX_GAUGE_ASSETS],
    /// Conteo de observaciones válidas
    pub count: u64,
    /// Registro omnisciente para deliberación y lectura de parámetros continuos
    pub registry: Option<Arc<OmniscientRegistry>>,
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
        Self {
            num_assets: n,
            beta_matrix: beta,
            adaptation_rate: 0.005,
            last_ln_prices: [0.0; MAX_GAUGE_ASSETS],
            count: 0,
            registry: None,
        }
    }

    /// Configura la tasa de adaptación adaptativa de los coeficientes de conexión $\beta_{ij}$.
    pub fn with_adaptation_rate(mut self, rate: f64) -> Self {
        if rate.is_finite() && rate > 0.0 {
            self.adaptation_rate = rate.clamp(1e-5, 0.1);
        }
        self
    }

    /// Configura el ratio de elasticidad de conexión beta entre dos activos i y j.
    pub fn with_beta(mut self, i: usize, j: usize, beta: f64) -> Self {
        if i < MAX_GAUGE_ASSETS && j < MAX_GAUGE_ASSETS && beta.is_finite() && beta > 0.0 {
            let b = beta.clamp(0.01, 100.0);
            self.beta_matrix[i][j] = b;
            self.beta_matrix[j][i] = (1.0 / b).clamp(0.01, 100.0);
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
            returns[i] = if self.count > 0 && prev_p_i != 0.0 {
                p_i - prev_p_i
            } else {
                0.0
            };
        }

        // 2. Actualización adaptativa online de coeficientes beta sobre pares válidos
        // Paso de gradiente exacto LMS normalizado: d/dβ 1/2(r_i - β r_j)^2 = - (r_i - β r_j) * r_j
        // step = gamma * (err * r_j) / (1.0 + r_j^2)
        // R6-C6: Si r_j == 0 (el activo j no cotizó en este tick), step == 0 exactamente,
        // eliminando cualquier deriva espuria o dependencia del camino de llegada de ticks.
        if self.count > 0 {
            let gamma = self.adaptation_rate;
            for vi in 0..n_valid {
                let i = valid_indices[vi];
                let r_i = returns[i];
                for vj in (vi + 1)..n_valid {
                    let j = valid_indices[vj];
                    let r_j = returns[j];

                    let err = r_i - self.beta_matrix[i][j] * r_j;
                    if err.is_finite() && r_j.abs() > 1e-12 {
                        let step = gamma * ((err * r_j) / (1.0 + r_j * r_j));
                        let new_beta = (self.beta_matrix[i][j] + step).clamp(0.01, 100.0);
                        self.beta_matrix[i][j] = new_beta;
                        self.beta_matrix[j][i] = (1.0 / new_beta).clamp(0.01, 100.0);
                    }
                }
            }
        }

        // 3. Cómputo de la curvatura 2-forma F_{ijk} para todos los 3-ciclos válidos (vi < vj < vk)
        let mut total_action = 0.0_f64;
        let mut currents = [0.0_f64; MAX_GAUGE_ASSETS];
        let mut cycle_counts = [0.0_f64; MAX_GAUGE_ASSETS];
        let mut total_cycles = 0usize;

        for vi in 0..n_valid {
            let i = valid_indices[vi];
            let r_i = returns[i];
            for vj in (vi + 1)..n_valid {
                let j = valid_indices[vj];
                let r_j = returns[j];
                for vk in (vj + 1)..n_valid {
                    let k = valid_indices[vk];
                    let r_k = returns[k];

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

                    // Densidad de acción Yang-Mills combinada (longitudinal + transversal):
                    // S_plaqueta = 1/2 (F_loop^2 + d_cross^2)
                    let f_sq = if f_loop.is_finite() && d_cross.is_finite() {
                        f_loop * f_loop + d_cross * d_cross
                    } else if f_loop.is_finite() {
                        f_loop * f_loop
                    } else {
                        0.0
                    };

                    if f_sq.is_finite() {
                        total_action += f_sq;
                        total_cycles += 1;

                        // Corrientes restauradoras invariantes como gradiente negativo de la acción:
                        // J = - ∇ S
                        let j_i = -f_loop * (1.0 - b_ki) - d_cross * d_grad_i;
                        let j_j = -f_loop * (1.0 - b_ij) - d_cross * d_grad_j;
                        let j_k = -f_loop * (1.0 - b_jk) - d_cross * d_grad_k;

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
        for vi in 0..n_valid {
            let i = valid_indices[vi];
            if cycle_counts[i] > 0.0 {
                currents[i] = (currents[i] / cycle_counts[i]).tanh();
            }
        }

        // Densidad media de acción gauge por plaqueta triangular (magnitud intensiva e invariante a N)
        let action_density = if total_cycles > 0 {
            0.5 * (total_action / (total_cycles as f64))
        } else {
            0.0
        };

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

    fn horizon(&self) -> TradeHorizon {
        TradeHorizon::Continuous
    }
}
