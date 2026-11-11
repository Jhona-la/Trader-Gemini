//! Fase 110 (§110): Modelos Cuánticos Dobles de Kitaev $\mathcal{D}(G)$ & Condensación Anyónica No Abeliana
//!
//! Implementa el modelo de red topológica exactamente soluble de Kitaev para el grupo cuántico doble
//! $\mathcal{D}(G)$ basado en el grupo de simetría de permutación $G = S_3$ ($|S_3| = 6$) sobre los
//! 3 canales de la tríada de activos continuos.
//!
//! # Estructura del Grupo $S_3$:
//! - 0: $e$ (Identidad)
//! - 1: $(12)$ (Transposición 1-2)
//! - 2: $(13)$ (Transposición 1-3)
//! - 3: $(23)$ (Transposición 2-3)
//! - 4: $(123)$ (Ciclo 3)
//! - 5: $(132)$ (Ciclo 3 inverso)
//!
//! # Invariantes de Silicio:
//! - Cero asignaciones de memoria dinámica en el hot-path ($O(1)$ estricto).
//! - Invariante `rb_p1` (0 panics, 0 unwraps, 0 expects).
//! - Tabla de Cayley de $S_3$ precalculada en stack.

/// Orden del grupo de permutaciones S3
pub const S3_ORDER: usize = 6;

/// Tabla de Cayley del grupo de permutaciones S3
/// Elementos: 0: e, 1: (12), 2: (13), 3: (23), 4: (123), 5: (132)
pub const S3_MULTIPLICATION_TABLE: [[usize; S3_ORDER]; S3_ORDER] = [
    [0, 1, 2, 3, 4, 5], // e * g
    [1, 0, 4, 5, 2, 3], // (12) * g
    [2, 5, 0, 4, 3, 1], // (13) * g
    [3, 4, 5, 0, 1, 2], // (23) * g
    [4, 3, 1, 2, 5, 0], // (123) * g
    [5, 2, 3, 1, 0, 4], // (132) * g
];

/// Inversos de cada elemento en S3: g * g^{-1} = e
pub const S3_INVERSES: [usize; S3_ORDER] = [0, 1, 2, 3, 5, 4];

/// Configuración determinista para el motor Cuántico Doble de Kitaev
#[derive(Debug, Clone, Copy)]
pub struct KitaevQuantumDoubleConfig {
    /// Umbral mínimo de fidelidad topológica del estado fundamental
    pub min_ground_state_fidelity: f64,
    /// Penalización de des-confinamiento ante anomalías de curvatura de gauge
    pub deconfinement_penalty_weight: f64,
    /// Factor de escala mínimo ante veto
    pub scale_min: f64,
    /// Factor de escala máximo en régimen condensado
    pub scale_max: f64,
}

impl Default for KitaevQuantumDoubleConfig {
    fn default() -> Self {
        Self {
            min_ground_state_fidelity: 0.40,
            deconfinement_penalty_weight: 1.50,
            scale_min: 0.10,
            scale_max: 1.25,
        }
    }
}

/// Informe de evaluación del motor Cuántico Doble de Kitaev
#[derive(Debug, Clone, Copy)]
pub struct KitaevQuantumDoubleReport {
    /// Fidelidad del estado fundamental topológico $\mathcal{F}_{\mathcal{D}(S_3)} \in [0, 1]$
    pub ground_state_fidelity: f64,
    /// Parámetro de orden topológico de Wilson $\mathcal{O}_{\text{Kitaev}}$
    pub topological_order_parameter: f64,
    /// Valor esperado del operador de estrella $A_s$ (carga gauge)
    pub star_operator_expectation: f64,
    /// Valor esperado del operador de plaqueta $B_p$ (flujo magnético)
    pub plaquette_operator_expectation: f64,
    /// Señal direccional no abeliana de cinta anyónica $\kappa_{\text{quantum\_double}} \in [-1, 1]$
    pub directional_signal: f64,
    /// Veto fail-closed de des-confinamiento o ruptura de simetría gauge
    pub deconfinement_veto: bool,
    /// Factor continuo de modulación de slot $\eta_{\text{Kitaev}} \in [0.10, 1.25]$
    pub slot_scale: f64,
}

/// Motor de simetría de norma topológica $\mathcal{D}(S_3)$ de Kitaev en stack $O(1)$
#[derive(Debug, Clone)]
pub struct KitaevQuantumDoubleEngine {
    config: KitaevQuantumDoubleConfig,
    /// Distribución de amplitudes del estado gauge en el espacio de Hilbert $\mathbb{C}[S_3]$
    gauge_amplitudes: [f64; S3_ORDER],
    total_updates: u64,
}

impl KitaevQuantumDoubleEngine {
    /// Construye una nueva instancia con configuración dada
    pub fn new(config: KitaevQuantumDoubleConfig) -> Self {
        let mut initial_amplitudes = [0.0; S3_ORDER];
        initial_amplitudes[0] = 1.0; // Estado inicial completamente condensado en la identidad
        Self {
            config,
            gauge_amplitudes: initial_amplitudes,
            total_updates: 0,
        }
    }

    /// Retorna una referencia a la tabla de Cayley
    pub fn multiplication_table(&self) -> &[[usize; S3_ORDER]; S3_ORDER] {
        &S3_MULTIPLICATION_TABLE
    }

    /// Actualiza el estado cuántico doble a partir de la microestructura del mercado.
    ///
    /// # Argumentos:
    /// - `price_ret`: Retorno continuo de precio
    /// - `ofi`: Desequilibrio de flujo de órdenes $\in [-1, 1]$
    /// - `vol_norm`: Volatilidad normalizada $\ge 0$
    /// - `spread_bps`: Spread relativo en bps $\ge 0$
    pub fn update(
        &mut self,
        price_ret: f64,
        ofi: f64,
        vol_norm: f64,
        spread_bps: f64,
    ) -> KitaevQuantumDoubleReport {
        let ret = if price_ret.is_finite() {
            price_ret
        } else {
            0.0
        };
        let flow = if ofi.is_finite() {
            ofi.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let vol = if vol_norm.is_finite() && vol_norm > 0.0 {
            vol_norm.clamp(0.01, 10.0)
        } else {
            1.0
        };
        let spr = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 50.0)
        } else {
            1.0
        };

        self.total_updates = self.total_updates.wrapping_add(1);

        // 1. Proyección de excitaciones de microestructura sobre las clases de conjugación de S3:
        // Clases: [e] (identidad, dim 1), [(12)] (transposiciones, dim 3), [(123)] (ciclos, dim 2)
        let id_weight = (1.0 / (1.0 + vol * 0.15 + (spr / 20.0))).clamp(0.10, 1.0);
        let trans_weight = (flow.abs() * 0.50 + ret.abs() * 10.0).clamp(0.0, 1.0);
        let cycle_weight = (vol * 0.10 + (spr / 30.0)).clamp(0.0, 0.80);

        // Construir amplitudes gauge no abelianas normalizadas
        let mut raw_amps = [0.0f64; S3_ORDER];
        raw_amps[0] = id_weight;
        // Transposiciones (12), (13), (23) sesgadas por la dirección del flujo
        let flow_bias = (flow * 0.20).clamp(-0.30, 0.30);
        raw_amps[1] = (trans_weight * (0.33 + flow_bias)).max(0.0);
        raw_amps[2] = (trans_weight * 0.33).max(0.0);
        raw_amps[3] = (trans_weight * (0.33 - flow_bias)).max(0.0);
        // Ciclos 3 (123) y (132) representan vorticidad / rotación en los canales
        raw_amps[4] = (cycle_weight * 0.50).max(0.0);
        raw_amps[5] = (cycle_weight * 0.50).max(0.0);

        // Normalización L2 del vector de estado en C[S3]
        let mut norm_sq = 0.0f64;
        for &a in &raw_amps {
            norm_sq += a * a;
        }
        let inv_norm = if norm_sq > 1e-12 {
            1.0 / norm_sq.sqrt()
        } else {
            1.0
        };

        // Mezcla cuántica continua con memoria previa (decoherencia disipativa suave)
        let alpha_mix = 0.30f64;
        for (amp, &raw_amp) in self.gauge_amplitudes.iter_mut().zip(&raw_amps) {
            let target = raw_amp * inv_norm;
            *amp = (1.0 - alpha_mix) * *amp + alpha_mix * target;
        }

        // Renormalizar estado disipativo
        let mut post_norm_sq = 0.0f64;
        for &a in &self.gauge_amplitudes {
            post_norm_sq += a * a;
        }
        let post_inv_norm = if post_norm_sq > 1e-12 {
            1.0 / post_norm_sq.sqrt()
        } else {
            1.0
        };
        for a in self.gauge_amplitudes.iter_mut() {
            *a *= post_inv_norm;
        }

        // 2. Operador de estrella A_s (proyección sobre estado de carga cero/confinado)
        // A_s \psi = \frac{1}{|G|} \sum_{g \in G} L_g \psi
        // El valor esperado es proporcional a la suma simétrica de amplitudes
        let mut sum_amps = 0.0f64;
        for &a in &self.gauge_amplitudes {
            sum_amps += a;
        }
        let a_s_val = ((sum_amps * sum_amps) / (S3_ORDER as f64)).clamp(0.0, 1.0);

        // 3. Operador de plaqueta B_p (medición de curvatura de gauge nula en la cara)
        // B_p = |<e|\psi>|^2
        let b_p_val = (self.gauge_amplitudes[0] * self.gauge_amplitudes[0]).clamp(0.0, 1.0);

        // 4. Fidelidad del estado fundamental topológico \mathcal{F}_{\mathcal{D}(S3)}
        // Proyección sobre el vacío topológico exacto (|0> = (|e> + \sum_{g} |g>) / \sqrt{|G|})
        let ground_state_fidelity = ((a_s_val + b_p_val) * 0.50).clamp(0.0, 1.0);

        // 5. Parámetro de orden de lazo de Wilson \mathcal{O}_{\text{Kitaev}}
        // \mathcal{O} = \sum_{g \in S3} \text{tr}(\rho(g)) |\psi_g|^2
        // Para la representación bidimensional irreducible estándar de S3:
        // tr(e) = 2, tr(trans) = 0, tr(ciclo) = -1
        let trace_rep = [2.0, 0.0, 0.0, 0.0, -1.0, -1.0];
        let mut wilson_loop = 0.0f64;
        for (&tr, &amp) in trace_rep.iter().zip(&self.gauge_amplitudes) {
            wilson_loop += tr * amp * amp;
        }
        let topo_order = ((wilson_loop + 1.0) / 3.0).clamp(0.0, 1.0);

        // 6. Señal direccional no abeliana de operadores de cinta anyónica
        // La asimetría entre las transposiciones (12) y (23) acoplada a la vorticidad de ciclos
        let trans_asymmetry = self.gauge_amplitudes[1] - self.gauge_amplitudes[3];
        let cycle_vorticity = self.gauge_amplitudes[4] - self.gauge_amplitudes[5];
        let directional_signal =
            ((flow * 0.60) + (trans_asymmetry * 1.50) + (cycle_vorticity * 0.50)).clamp(-1.0, 1.0);

        // 7. Veto fail-closed de des-confinamiento
        let deconfinement_veto = ground_state_fidelity < self.config.min_ground_state_fidelity
            || b_p_val < 0.10
            || !ground_state_fidelity.is_finite();

        // 8. Modulación continua de slot
        let raw_scale = if deconfinement_veto {
            self.config.scale_min
        } else {
            let fidelity_boost = (ground_state_fidelity - self.config.min_ground_state_fidelity)
                / (1.0 - self.config.min_ground_state_fidelity + 1e-6);
            let topo_boost = topo_order * 0.10;
            1.0 + (fidelity_boost * 0.12) + (topo_boost * 0.08)
        };

        let slot_scale = raw_scale.clamp(self.config.scale_min, self.config.scale_max);

        KitaevQuantumDoubleReport {
            ground_state_fidelity,
            topological_order_parameter: topo_order,
            star_operator_expectation: a_s_val,
            plaquette_operator_expectation: b_p_val,
            directional_signal,
            deconfinement_veto,
            slot_scale,
        }
    }
}
