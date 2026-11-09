//! # Fase 75: Entropía de Ryu-Takayanagi y Superficies Extremales de Entrelazamiento Holográfico
//!
//! Modela la entropía de entrelazamiento cuántico $S_A$ para subsistemas de liquidez y activos en el borde
//! mediante la fórmula geométrica de Ryu-Takayanagi (RT):
//! $$S_A = \frac{\text{Área}(\gamma_A)}{4 G_N^{(3)}} = \frac{c}{3} \ln\left( \frac{L_A}{\epsilon} \right)$$
//! donde $\gamma_A$ es la superficie mínima extremal homóloga en el bulto hiperbólico $AdS_3$,
//! $c = \frac{3 L_{AdS}}{2 G_N}$ es la carga central de Brown-Henneaux y $\epsilon$ es el corte UV.
//!
//! Evalúa además la transición de fase topológica de Información Mutua Holográfica:
//! $$I(A:B) = S_A + S_B - S_{A \cup B} \ge 0$$
//! detectando cuándo la superficie extremal sufre desconexión topológica (análogo de Hawking-Page),
//! fragmentando la coherencia del mercado.
//!
//! ## Invariantes del Sistema:
//! - Cero asignaciones en heap en la ruta crítica ($O(1)$ stack determinista).
//! - Cero pánicos (`rb_p1`): ausencia total de `unwrap()`, `expect()`, `assert!()` o `panic!()`.

use core::f64;

/// Parámetros configurables del motor de entrelazamiento de Ryu-Takayanagi.
#[derive(Debug, Clone, Copy)]
pub struct RyuTakayanagiParams {
    /// Carga central de Brown-Henneaux $c = \frac{3 L}{2 G_N} > 0$. Típicamente $c \approx 12.0$.
    pub brown_henneaux_central_charge: f64,
    /// Longitud de corte UV en el borde $\epsilon > 0$.
    pub uv_cutoff_epsilon: f64,
    /// Umbral mínimo de información mutua holográfica para preservar la conectividad de $\gamma_{A \cup B}$.
    pub min_mutual_information_threshold: f64,
    /// Umbral máximo de entropía de entrelazamiento admisible antes de saturación térmica.
    pub max_entanglement_entropy_threshold: f64,
    /// Factor de escala mínimo defensivo ante veto duro.
    pub min_slot_scale: f64,
    /// Regularizador numérico epsilon $> 0$.
    pub epsilon: f64,
}

impl Default for RyuTakayanagiParams {
    fn default() -> Self {
        Self {
            brown_henneaux_central_charge: 12.0,
            uv_cutoff_epsilon: 0.05,
            min_mutual_information_threshold: 0.10,
            max_entanglement_entropy_threshold: 35.0,
            min_slot_scale: 0.15,
            epsilon: 1e-12,
        }
    }
}

/// Informe omnisciente de entropía holográfica y conectividad de superficies extremales.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RyuTakayanagiReport {
    /// Entropía de entrelazamiento holográfico del subsistema $A$ (activo primario / BTC): $S_A \ge 0$.
    pub entanglement_entropy_sa: f64,
    /// Entropía de entrelazamiento holográfico del subsistema $B$ (ensamble cruzado / ETH/SOL): $S_B \ge 0$.
    pub entanglement_entropy_sb: f64,
    /// Entropía de entrelazamiento conjunta del sistema total $S_{A \cup B} \ge 0$.
    pub entanglement_entropy_union: f64,
    /// Información Mutua Holográfica $I(A:B) = S_A + S_B - S_{A \cup B} \ge 0$.
    pub holographic_mutual_information: f64,
    /// Indicador booleano de conectividad de la superficie extremal mínima de Ryu-Takayanagi en el bulto.
    pub extremal_surface_connected: bool,
    /// Factor modulador continuo de slot $\in [0.0, 1.25]$. Colapsa a 0.0 bajo veto duro.
    pub slot_scale_factor: f64,
    /// Veto duro inducido por desconexión de Ryu-Takayanagi (`RYU_TAKAYANAGI_ENTANGLEMENT_VETO`).
    pub hard_veto_induced: bool,
}

impl Default for RyuTakayanagiReport {
    fn default() -> Self {
        Self {
            entanglement_entropy_sa: 1.0,
            entanglement_entropy_sb: 1.0,
            entanglement_entropy_union: 1.5,
            holographic_mutual_information: 0.5,
            extremal_surface_connected: true,
            slot_scale_factor: 1.0,
            hard_veto_induced: false,
        }
    }
}

/// Motor de Entrelazamiento Holográfico de Ryu-Takayanagi en stack $O(1)$.
#[derive(Debug, Clone)]
pub struct RyuTakayanagiEntanglementEngine {
    params: RyuTakayanagiParams,
}

impl RyuTakayanagiEntanglementEngine {
    /// Construye una nueva instancia inicializada con los parámetros holográficos.
    pub fn new(params: RyuTakayanagiParams) -> Self {
        Self { params }
    }

    /// Actualiza la entropía de entrelazamiento holográfico entre el subsistema primario y el mercado cruzado.
    ///
    /// # Argumentos:
    /// - `btc_dispersion`: Dispersión de volatilidad o escala espacial característica de BTC $> 0$.
    /// - `alt_dispersion`: Dispersión de volatilidad o escala espacial característica de los ALTs $> 0$.
    /// - `cross_correlation`: Correlación cruzada entre BTC y el ensamble $[-1.0, 1.0]$.
    /// - `spread_bps`: Spread medio en puntos básicos $\ge 0$.
    pub fn update(
        &mut self,
        btc_dispersion: f64,
        alt_dispersion: f64,
        cross_correlation: f64,
        spread_bps: f64,
    ) -> RyuTakayanagiReport {
        let safe_disp_a = if btc_dispersion.is_finite() && btc_dispersion > 0.0 {
            btc_dispersion.clamp(0.001, 10.0)
        } else {
            0.02
        };
        let safe_disp_b = if alt_dispersion.is_finite() && alt_dispersion > 0.0 {
            alt_dispersion.clamp(0.001, 10.0)
        } else {
            0.02
        };
        let safe_corr = if cross_correlation.is_finite() {
            cross_correlation.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.01, 100.0)
        } else {
            1.0
        };

        let c = self.params.brown_henneaux_central_charge.clamp(1.0, 50.0);
        let eps = self.params.uv_cutoff_epsilon.clamp(0.001, 0.5);

        // 1. Longitudes de coherencia características L_A y L_B de los subsistemas en el borde:
        // En física de condensados y CFT, la longitud de correlación es inversamente proporcional a la dispersión térmica
        let length_a = (5.0 / (safe_disp_a * 2.0 + 0.1) + 2.0).clamp(2.0, 50.0);
        let length_b = (5.0 / (safe_disp_b * 2.0 + 0.1) + 2.0).clamp(2.0, 50.0);

        // 2. Entropías de Ryu-Takayanagi individuales:
        // S_A = (c / 3) * \ln(L_A / \epsilon)
        let sa = ((c / 3.0) * (length_a / eps).max(1.0).ln()).clamp(0.0, 100.0);
        let sb = ((c / 3.0) * (length_b / eps).max(1.0).ln()).clamp(0.0, 100.0);

        // 3. Separación holográfica efectiva D entre los intervalos A y B:
        // D se dilata fuertemente con la descorrelación y el ensanchamiento del spread (fricción de puente)
        let separation_d = ((1.0 - safe_corr.abs()) * 10.0 + safe_spread * 0.20 + 0.05).clamp(0.01, 50.0);

        // 4. Competencia de Superficies Extremales en el Bulto AdS_3 y Razón Cruzada (Cross-Ratio):
        // En AdS_3 / CFT_2, la información mutua holográfica es UV-finita (independiente de \epsilon):
        // \eta = \frac{L_A \cdot L_B}{(L_A + L_B + D) \cdot D}
        // Si \eta > 1.0: Fase Conexa entrelazada (\gamma_{A \cup B} se conecta en el bulto).
        // Si \eta <= 1.0: Fase Disconexa (Hawking-Page / transición de entrelazamiento de Ryu-Takayanagi).
        let cross_ratio = (length_a * length_b) / (((length_a + length_b + separation_d) * separation_d).max(self.params.epsilon));

        let (extremal_surface_connected, holographic_mutual_information) = if cross_ratio > 1.0 {
            let mi = ((c / 3.0) * cross_ratio.ln()).clamp(0.0, 50.0);
            (true, mi)
        } else {
            (false, 0.0)
        };

        let s_union = (sa + sb - holographic_mutual_information).max(0.0);

        // 6. Inducción de Veto Duro de Ryu-Takayanagi:
        // Se dispara si la superficie extremal se desconecta (Hawking-Page fragmentation)
        // o si la información mutua colapsa por debajo del umbral mínimo de entrelazamiento
        // o si la entropía satura térmicamente por encima del umbral máximo
        let hard_veto_induced = !extremal_surface_connected
            || holographic_mutual_information < self.params.min_mutual_information_threshold
            || sa > self.params.max_entanglement_entropy_threshold
            || sb > self.params.max_entanglement_entropy_threshold
            || safe_spread > 25.0;

        // 7. Factor de Escala de Slot:
        let slot_scale_factor = if hard_veto_induced {
            0.0
        } else {
            // Modulación constructiva en presencia de un puente de entrelazamiento robusto en el bulto
            let mi_excess = (holographic_mutual_information / 2.0).clamp(0.0, 1.0);
            (0.85 + 0.35 * mi_excess).clamp(self.params.min_slot_scale, 1.20)
        };

        RyuTakayanagiReport {
            entanglement_entropy_sa: sa,
            entanglement_entropy_sb: sb,
            entanglement_entropy_union: s_union,
            holographic_mutual_information,
            extremal_surface_connected,
            slot_scale_factor,
            hard_veto_induced,
        }
    }
}
