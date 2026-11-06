//! 🌌 TEORÍA GAUGE NO ABELIANA SU(3) CROMODINÁMICA MULTIACTIVO (OLA OMEGA220 — FASE 37)
//!
//! ## Fundamento Teórico: Yang-Mills SU(3) en el Espacio de Hilbert Financiero
//!
//! En un universo multivariante de $N \ge 3$ activos, las interacciones inter-activos
//! no son abelianas (conmutativas). La secuencia temporal de rotaciones de capital entre
//! una tríada fundamental (e.g. $c_0 = \text{BTC}$, $c_1 = \text{ETH}$, $c_2 = \text{SOL/Alt}$)
//! exhibe no conmutatividad estricta: rotar de BTC a ETH y luego a SOL genera una curvatura
//! topológica distinta a rotar de BTC a SOL y luego a ETH:
//!   $$[A_\mu, A_\nu] \neq 0$$
//!
//! Modelamos la tríada de activos como un multiplete de color fundamental en la representación
//! de $\mathrm{SU}(3)$ mediado por 8 "gluones" de mercado asociados al álgebra de Lie $\mathfrak{su}(3)$.
//!
//! ### 1. Matrices de Gell-Mann $\lambda_a$ ($a \in \{1, \dots, 8\}$)
//! Generadores hermíticos y de traza nula: $\operatorname{Tr}(\lambda_a) = 0$, $\operatorname{Tr}(\lambda_a \lambda_b) = 2 \delta_{ab}$.
//!
//! ### 2. Constantes de Estructura Antisimétricas $f^{abc}$
//! Relaciones de conmutación de los generadores $T_a = \lambda_a / 2$:
//!   $$[T_a, T_b] = i \sum_{c=1}^8 f^{abc} T_c$$
//!
//! ### 3. 1-Forma de Conexión de Gauge $A_\mu = \sum_a A_\mu^a T_a$ ($\mu \in \{0, 1\}$)
//! - $A_0^a$: Componente temporal del potencial gauge derivada de la proyección de retornos y paridades.
//! - $A_1^a$: Componente espacial del potencial gauge derivada del desequilibrio de flujo de órdenes (OFI/OBI).
//!
//! ### 4. Tensor de Intensidad de Campo Cromodinámico $F_{\mu\nu}^a$
//!   $$F_{01}^a = \partial_0 A_1^a - \partial_1 A_0^a + g \sum_{b,c=1}^8 f^{abc} A_0^b A_1^c$$
//! donde el término cuadrático representa la **auto-interacción gluónica no abeliana**
//! de la liquidez entre pares.
//!
//! ### 5. Densidad de Acción Invariante de Yang-Mills
//!   $$\mathcal{L}_{\mathrm{YM}} = \frac{1}{4} \sum_{a=1}^8 (F_{01}^a)^2$$
//!
//! ### 6. Fuerza Cromodinámica de Lorentz-Wong sobre cada Activo
//!   $$K_i = \sum_{a=1}^8 F_{01}^a \cdot (\lambda_a)_{ii}, \quad \sum_{i=0}^2 K_i = 0$$
//! Garantiza la conservación estricta de la carga de color (suma nula de fuerzas ficticias).

use crate::QuantumStrategy;
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;

/// Número de generadores del álgebra de Lie su(3).
pub const SU3_GENERATORS: usize = 8;
/// Dimensión fundamental de la representación de color (tríada).
pub const SU3_COLORS: usize = 3;

/// Raíz de 3 para las matrices de Gell-Mann y constantes de estructura.
const SQRT_3: f64 = 1.7320508075688772;
const HALF_SQRT_3: f64 = 0.8660254037844386;

/// Parámetros configurables del motor cromodinámico de gauge SU(3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su3GaugeParams {
    /// Constante de acoplamiento de gauge no abeliano $g$ (interacción gluónica).
    pub gauge_coupling_g: f64,
    /// Factor de memoria exponencial para derivadas temporales $\alpha \in (0, 1)$.
    pub memory_decay: f64,
    /// Umbral mínimo de curvatura de Yang-Mills para declarar dislocación activa.
    pub curvature_threshold: f64,
    /// Escala de normalización para la fuerza cromodinámica en señales alpha.
    pub force_scale: f64,
}

impl Default for Su3GaugeParams {
    fn default() -> Self {
        Self {
            gauge_coupling_g: 0.15,
            memory_decay: 0.85,
            curvature_threshold: 0.02,
            force_scale: 2.50,
        }
    }
}

impl Su3GaugeParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.gauge_coupling_g.is_finite()
            || self.gauge_coupling_g < 0.0
            || self.gauge_coupling_g > 2.0
        {
            return Err("Su3GaugeParams: gauge_coupling_g fuera de rango [0.0, 2.0]");
        }
        if !self.memory_decay.is_finite() || self.memory_decay <= 0.0 || self.memory_decay >= 1.0 {
            return Err("Su3GaugeParams: memory_decay fuera de rango (0.0, 1.0)");
        }
        if !self.curvature_threshold.is_finite() || self.curvature_threshold <= 0.0 {
            return Err("Su3GaugeParams: curvature_threshold debe ser positivo");
        }
        if !self.force_scale.is_finite() || self.force_scale <= 0.0 {
            return Err("Su3GaugeParams: force_scale debe ser positivo");
        }
        Ok(())
    }
}

/// Estado de entrada de un activo para la proyección cromodinámica.
#[derive(Debug, Clone, Copy, Default)]
pub struct AssetColorState {
    /// Retorno normalizado o log-precio relativo.
    pub normalized_return: f64,
    /// Desequilibrio de flujo de órdenes (Order Flow Imbalance OFI / OBI) $\in [-1, 1]$.
    pub order_flow_imbalance: f64,
    /// Peso de capital o volumen relativo $\ge 0$.
    pub relative_weight: f64,
}

/// Informe omnisciente de la geometría gauge SU(3) y fuerzas de Yang-Mills.
#[derive(Debug, Clone, Copy)]
pub struct Su3GaugeReport {
    /// Densidad de acción de Yang-Mills $\mathcal{L}_{\mathrm{YM}} \ge 0$.
    pub yang_mills_action: f64,
    /// Invariante de Casimir cuadrático $C_2 = \sum_a (A_0^a)^2 + (A_1^a)^2$.
    pub casimir_invariant: f64,
    /// Carga topológica instantánea de Chern-Simons $Q_{\mathrm{top}} = \sum_a F_{01}^a A_0^a$.
    pub topological_charge: f64,
    /// Fuerzas cromodinámicas resultantes sobre cada color de la tríada: $\sum K_i = 0$.
    pub color_forces: [f64; SU3_COLORS],
    /// Intensidades del tensor de curvatura $F_{01}^a$ para los 8 generadores.
    pub field_strength_f01: [f64; SU3_GENERATORS],
    /// Magnitud del término no conmutativo de auto-interacción gluónica: $\| g [A_0, A_1] \|$.
    pub non_abelian_curl_norm: f64,
    /// Señales alfa resultantes normalizadas en $[-1.0, 1.0]$ para la tríada.
    pub alpha_signals: [f64; SU3_COLORS],
    /// Indica si la densidad de acción supera el umbral crítico de dislocación.
    pub active_dislocation: bool,
}

/// Motor cromodinámico de gauge no abeliano $\mathrm{SU}(3)$ sobre tríadas multiactivo.
#[derive(Debug, Clone)]
pub struct ChromodynamicGaugeSu3Engine {
    params: Su3GaugeParams,
    prev_a0: [f64; SU3_GENERATORS],
    prev_a1: [f64; SU3_GENERATORS],
    has_history: bool,
    last_report: Option<Su3GaugeReport>,
}

impl Default for ChromodynamicGaugeSu3Engine {
    fn default() -> Self {
        Self::new(Su3GaugeParams::default())
    }
}

impl ChromodynamicGaugeSu3Engine {
    /// Construye una nueva instancia del motor con los parámetros especificados.
    pub fn new(params: Su3GaugeParams) -> Self {
        Self {
            params,
            prev_a0: [0.0; SU3_GENERATORS],
            prev_a1: [0.0; SU3_GENERATORS],
            has_history: false,
            last_report: None,
        }
    }

    /// Obtiene los parámetros actuales.
    pub fn params(&self) -> &Su3GaugeParams {
        &self.params
    }

    /// Actualiza los parámetros del motor.
    pub fn set_params(&mut self, params: Su3GaugeParams) -> Result<(), &'static str> {
        params.validate()?;
        self.params = params;
        Ok(())
    }

    /// Obtiene el último informe computado, si existe.
    pub fn last_report(&self) -> Option<&Su3GaugeReport> {
        self.last_report.as_ref()
    }

    /// Evalúa las constantes de estructura totalmente antisimétricas $f^{abc}$ de $\mathfrak{su}(3)$.
    ///
    /// Índices $a, b, c \in \{0, \dots, 7\}$ (correspondientes a los generadores 1 a 8).
    #[inline(always)]
    pub fn structure_constant_f(a: usize, b: usize, c: usize) -> f64 {
        if a == b || b == c || a == c || a >= 8 || b >= 8 || c >= 8 {
            return 0.0;
        }

        // Ordenamiento canónico y paridad de permutación
        let mut idx = [a, b, c];
        let mut sign = 1.0;

        if idx[0] > idx[1] {
            idx.swap(0, 1);
            sign = -sign;
        }
        if idx[1] > idx[2] {
            idx.swap(1, 2);
            sign = -sign;
        }
        if idx[0] > idx[1] {
            idx.swap(0, 1);
            sign = -sign;
        }

        // Generadores 1 a 8 corresponden a índices 0 a 7
        let val = match (idx[0], idx[1], idx[2]) {
            (0, 1, 2) => 1.0,         // f^{123} = 1
            (0, 3, 6) => 0.5,         // f^{147} = 1/2
            (0, 4, 5) => -0.5,        // f^{156} = -1/2
            (1, 3, 5) => 0.5,         // f^{246} = 1/2
            (1, 4, 6) => 0.5,         // f^{257} = 1/2
            (2, 3, 4) => 0.5,         // f^{345} = 1/2
            (2, 5, 6) => -0.5,        // f^{367} = -1/2
            (3, 4, 7) => HALF_SQRT_3, // f^{458} = sqrt(3)/2
            (5, 6, 7) => HALF_SQRT_3, // f^{678} = sqrt(3)/2
            _ => 0.0,
        };

        sign * val
    }

    /// Proyecta los vectores de retorno y de flujo de la tríada sobre los 8 generadores de Gell-Mann
    /// para obtener los potenciales gauge $(A_0^a, A_1^a)$.
    ///
    /// Retorna $(A_0, A_1)$ sin asignaciones de heap.
    #[inline(always)]
    pub fn project_gauge_potentials(
        ret: [f64; SU3_COLORS],
        flow: [f64; SU3_COLORS],
    ) -> ([f64; SU3_GENERATORS], [f64; SU3_GENERATORS]) {
        let mut a0 = [0.0; SU3_GENERATORS];
        let mut a1 = [0.0; SU3_GENERATORS];

        let r0 = ret[0];
        let r1 = ret[1];
        let r2 = ret[2];

        let f0 = flow[0];
        let f1 = flow[1];
        let f2 = flow[2];

        // lambda_1 = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
        a0[0] = r0 * r1;
        a1[0] = f0 * f1;

        // lambda_2 = [[0, -i, 0], [i, 0, 0], [0, 0, 0]]
        // Término antisimétrico cruzado entre retorno y flujo
        a0[1] = r0 * f1 - r1 * f0;
        a1[1] = -(f0 * r1 - f1 * r0);

        // lambda_3 = [[1, 0, 0], [0, -1, 0], [0, 0, 0]]
        // Desbalance de retorno y flujo entre color 0 (BTC) y color 1 (ETH)
        a0[2] = 0.5 * (r0 * r0.abs() - r1 * r1.abs());
        a1[2] = 0.5 * (f0 * f0.abs() - f1 * f1.abs());

        // lambda_4 = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
        a0[3] = r0 * r2;
        a1[3] = f0 * f2;

        // lambda_5 = [[0, 0, -i], [0, 0, 0], [i, 0, 0]]
        a0[4] = r0 * f2 - r2 * f0;
        a1[4] = -(f0 * r2 - f2 * r0);

        // lambda_6 = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
        a0[5] = r1 * r2;
        a1[5] = f1 * f2;

        // lambda_7 = [[0, 0, 0], [0, 0, -i], [0, i, 0]]
        a0[6] = r1 * f2 - r2 * f1;
        a1[6] = -(f1 * r2 - f2 * r1);

        // lambda_8 = (1/sqrt(3)) * [[1, 0, 0], [0, 1, 0], [0, 0, -2]]
        // Desbalance de hiper-carga de color entre tríada principal y altcoin
        a0[7] = (0.5 / SQRT_3) * (r0 * r0.abs() + r1 * r1.abs() - 2.0 * r2 * r2.abs());
        a1[7] = (0.5 / SQRT_3) * (f0 * f0.abs() + f1 * f1.abs() - 2.0 * f2 * f2.abs());

        (a0, a1)
    }

    /// Actualiza y evalúa la geometría de gauge no abeliana para una tríada de activos.
    ///
    /// Computa el tensor de Yang-Mills $F_{01}^a$, la densidad de acción $\mathcal{L}_{\mathrm{YM}}$,
    /// las fuerzas cromodinámicas $K_i$ y las señales de arbitraje cuántico en $O(1)$ sin asignaciones de memoria.
    pub fn update_triad(
        &mut self,
        assets: &[AssetColorState; SU3_COLORS],
        dt_seconds: f64,
    ) -> Su3GaugeReport {
        let dt = if dt_seconds.is_finite() && dt_seconds > 1e-4 {
            dt_seconds.min(60.0)
        } else {
            1.0
        };

        // 1. Ponderar retornos y flujos por pesos de capital
        let mut ret = [0.0; SU3_COLORS];
        let mut flow = [0.0; SU3_COLORS];

        for (i, asset) in assets.iter().enumerate() {
            let w = if asset.relative_weight.is_finite() && asset.relative_weight > 0.0 {
                asset.relative_weight.min(10.0)
            } else {
                1.0
            };
            let norm_ret = if asset.normalized_return.is_finite() {
                asset.normalized_return.clamp(-5.0, 5.0)
            } else {
                0.0
            };
            let ofi = if asset.order_flow_imbalance.is_finite() {
                asset.order_flow_imbalance.clamp(-1.0, 1.0)
            } else {
                0.0
            };

            ret[i] = norm_ret * w.sqrt();
            flow[i] = ofi * w.sqrt();
        }

        // 2. Proyección sobre generadores de Gell-Mann: (A_0, A_1)
        let (current_a0, current_a1) = Self::project_gauge_potentials(ret, flow);

        // 3. Derivadas temporales de flujo d_a1 y paridades d_a0
        let mut d_a1 = [0.0; SU3_GENERATORS];
        let mut d_a0 = [0.0; SU3_GENERATORS];

        if self.has_history {
            for a in 0..SU3_GENERATORS {
                d_a1[a] = (current_a1[a] - self.prev_a1[a]) / dt;
                d_a0[a] = (current_a0[a] - self.prev_a0[a]) / dt;
            }
        } else {
            // En el primer paso, la derivada temporal se inicializa con el valor instantáneo
            d_a1[..SU3_GENERATORS].copy_from_slice(&current_a1[..SU3_GENERATORS]);
            d_a0[..SU3_GENERATORS].copy_from_slice(&current_a0[..SU3_GENERATORS]);
        }

        // 4. Término no abeliano: g * f^{abc} A_0^b A_1^c
        let g = self.params.gauge_coupling_g;
        let mut non_abelian_term = [0.0; SU3_GENERATORS];
        let mut curl_sq = 0.0;

        for (a, term) in non_abelian_term.iter_mut().enumerate() {
            let mut sum_bc = 0.0;
            for (b, &a0_b) in current_a0.iter().enumerate() {
                for (c, &a1_c) in current_a1.iter().enumerate() {
                    let fabc = Self::structure_constant_f(a, b, c);
                    if fabc.abs() > 1e-6 {
                        sum_bc += fabc * a0_b * a1_c;
                    }
                }
            }
            *term = g * sum_bc;
            curl_sq += *term * *term;
        }
        let non_abelian_curl_norm = curl_sq.sqrt();

        // 5. Tensor de curvatura de Yang-Mills F_{01}^a = \partial_0 A_1^a - \partial_1 A_0^a + g f^{abc} A_0^b A_1^c
        // En aproximación de microestructura: \partial_0 A_1 es el impulso de flujo neto y d_a0 * 0.1 es el arrastre
        let mut f01 = [0.0; SU3_GENERATORS];
        let mut action_sum = 0.0;
        let mut casimir_sum = 0.0;
        let mut topo_sum = 0.0;

        for (a, f_val) in f01.iter_mut().enumerate() {
            *f_val = d_a1[a] - 0.1 * d_a0[a] + non_abelian_term[a];
            action_sum += *f_val * *f_val;
            casimir_sum += current_a0[a] * current_a0[a] + current_a1[a] * current_a1[a];
            topo_sum += *f_val * current_a0[a];
        }

        let yang_mills_action = 0.25 * action_sum;
        let casimir_invariant = casimir_sum;
        let topological_charge = topo_sum;

        // 6. Fuerzas cromodinámicas de Lorentz-Wong:
        // K_i = \sum_a F_{01}^a * (\lambda_a)_{ii}
        // (\lambda_3)_{00} = 1, (\lambda_3)_{11} = -1, (\lambda_3)_{22} = 0
        // (\lambda_8)_{00} = 1/sqrt(3), (\lambda_8)_{11} = 1/sqrt(3), (\lambda_8)_{22} = -2/sqrt(3)
        let k0 = f01[2] + (1.0 / SQRT_3) * f01[7];
        let k1 = -f01[2] + (1.0 / SQRT_3) * f01[7];
        let k2 = -(2.0 / SQRT_3) * f01[7];

        let color_forces = [k0, k1, k2];

        // 7. Señales alfa resultantes normalizadas mediante tanh
        let scale = self.params.force_scale;
        let mut alpha_signals = [0.0; SU3_COLORS];
        for i in 0..SU3_COLORS {
            alpha_signals[i] = (scale * color_forces[i]).tanh();
        }

        let active_dislocation = yang_mills_action >= self.params.curvature_threshold;

        // 8. Actualizar memoria interna con decaimiento exponencial
        let decay = self.params.memory_decay;
        for (a, prev_0) in self.prev_a0.iter_mut().enumerate() {
            *prev_0 = decay * *prev_0 + (1.0 - decay) * current_a0[a];
            self.prev_a1[a] = decay * self.prev_a1[a] + (1.0 - decay) * current_a1[a];
        }
        self.has_history = true;

        let report = Su3GaugeReport {
            yang_mills_action,
            casimir_invariant,
            topological_charge,
            color_forces,
            field_strength_f01: f01,
            non_abelian_curl_norm,
            alpha_signals,
            active_dislocation,
        };

        self.last_report = Some(report);
        report
    }

    /// Reinicia el historial temporal del motor.
    pub fn reset(&mut self) {
        self.prev_a0 = [0.0; SU3_GENERATORS];
        self.prev_a1 = [0.0; SU3_GENERATORS];
        self.has_history = false;
        self.last_report = None;
    }
}

impl QuantumStrategy for ChromodynamicGaugeSu3Engine {
    fn name(&self) -> &str {
        "ChromodynamicGaugeSu3Engine"
    }

    fn init(&mut self, _registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.reset();
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        // Devuelve la convicción promedio ponderada del activo líder (Color 0)
        if let Some(ref report) = self.last_report {
            report.alpha_signals[0]
        } else {
            0.0
        }
    }

    fn evaluate_for_coin(&self, coin_id: usize, _symbol: &str) -> f64 {
        if let Some(ref report) = self.last_report {
            let color_idx = coin_id % SU3_COLORS;
            report.alpha_signals[color_idx]
        } else {
            0.0
        }
    }

    fn is_operable_for_coin(&self, _coin_id: usize, _symbol: &str) -> bool {
        // Operable cuando el historial de gauge se encuentra activo
        self.has_history
    }
}
