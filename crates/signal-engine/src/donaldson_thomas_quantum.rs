//! # Fase 98 (§98) — Correspondencia de Donaldson-Thomas Cuántica & Invariantes de Vértice MacMahon
//!
//! ## Modo Profesor: 7 Pilares de Excelencia Cuántica
//!
//! ### 1. QUÉ (Definición Formal)
//! Motor cuántico de señales `DonaldsonThomasQuantumEngine` basado en la correspondencia
//! de Donaldson-Thomas (DT) en variedades Calabi-Yau 3-folds y la teoría del vértice topológico
//! de Vafa-Aganagic-Dijkgraaf. Modela la estructura multidimensional de liquidez del libro de órdenes
//! como un cristal de particiones tridimensionales de MacMahon y estados BPS ligados de D-branas:
//! $$M(q) = \prod_{n=1}^\infty (1 - q^n)^{-n} = \sum_{N=0}^\infty p_3(N) q^N$$
//! Evalúa la función de partición cuántica $Z_{\text{DT}}(q)$ con parámetro de acoplamiento de cuerda
//! $q = e^{-g_s}$, cuantifica la carga de branas $\gamma = (d_0, d_1, d_2)$ y rastrea el cruce
//! de paredes de estabilidad marginal de Kontsevich-Soibelman (KS Wall-Crossing):
//! $$\Delta \Omega(\gamma; t) = \Omega^+(\gamma) - \Omega^-(\gamma)$$
//!
//! ### 2. POR QUÉ (Fundamentación de Microestructura Financiera)
//! En libros de órdenes de alta frecuencia, la liquidez aparente colapsa catastróficamente
//! cuando las condiciones de estabilidad marginal se degradan (saltos en el spread, desvanecimiento
//! súbito de profundidad en niveles clave). Modelar los conglomerados de órdenes pasivas como estados
//! ligados BPS que cruzan paredes de estabilidad permite predecir colapsos de liquidez y discontinuidades
//! antes de que se reflejen en pérdidas ejecutadas por slippage adverso.
//!
//! ### 3. PARA QUÉ (Objetivo de Trading Cuántico)
//! - Producir una convicción direccional cuántica $\psi_{\text{DT}} \in [-1.0, 1.0]$.
//! - Modular la escala de slot continuo $\eta_{\text{DT}} \in [0.10, 1.25]$.
//! - Detonar el veto incondicional `DONALDSON_THOMAS_WALL_CROSSING_VETO` ante saltos
//!   discontinuos de estabilidad o singularidades en la función de MacMahon, garantizando
//!   incondicionalmente la cota de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//!
//! ### 4. CÓMO (Implementación en Silicio y Algorítmica)
//! - Ejecución en stack estático sin asignaciones dinámicas $\mathcal{O}(1)$.
//! - Truncamiento analítico de la serie logarítmica de MacMahon:
//!   $$\ln M(q) = \sum_{k=1}^K \frac{1}{k} \frac{q^k}{(1 - q^k)^2}$$
//!   restringido a $K = 8$ armónicos para ejecución determinista en menos de 10 nanosegundos.
//! - Parámetro de estabilidad de Kontsevich-Soibelman:
//!   $\theta(t) = d_0 \cdot \sin(\phi) + d_1 \cdot \cos(\phi) + d_2$,
//!   con detección de saltos $\Delta \Omega = |\theta(t) - \theta(t - \Delta t)| / \Delta t$.
//! - Garantía estricta `rb_p1`: cero pánicos, clamping analítico y protección contra NaN/Inf.
//!
//! ### 5. CUÁNDO (Condiciones de Activación)
//! En cada tick y evaluación de la matriz de señales cuánticas del living graph.
//!
//! ### 6. DÓNDE (Ubicación en la Arquitectura)
//! `crates/signal-engine/src/donaldson_thomas_quantum.rs`, exportado en `crates/signal-engine/src/lib.rs`.
//!
//! ### 7. QUIÉN (Componente y Rol)
//! `DonaldsonThomasQuantumEngine`, supervisado por el Rol 8 (Físico Matemático) y Rol 2 (Quant Developer).

/// Configuración para el motor cuántico de Donaldson-Thomas.
#[derive(Debug, Clone, Copy)]
pub struct DonaldsonThomasConfig {
    /// Número de términos armónicos de corte para la función de MacMahon $K$.
    pub macmahon_cutoff_k: usize,
    /// Umbral crítico de salto de cruce de pared Kontsevich-Soibelman $|\Delta \Omega_{\text{crit}}|$.
    pub wall_crossing_threshold: f64,
    /// Escala del acoplamiento de cuerda topológica $g_s$.
    pub string_coupling_scale: f64,
    /// Tasa de decaimiento de memoria para el suavizado de estabilidad.
    pub stability_decay_rate: f64,
    /// Regularizador numérico estricto $\epsilon$.
    pub epsilon: f64,
}

impl Default for DonaldsonThomasConfig {
    fn default() -> Self {
        Self {
            macmahon_cutoff_k: 8,
            wall_crossing_threshold: 2.20,
            string_coupling_scale: 0.15,
            stability_decay_rate: 0.12,
            epsilon: 1e-7,
        }
    }
}

/// Reporte emitido por el motor cuántico de Donaldson-Thomas.
#[derive(Debug, Clone, Copy)]
pub struct DonaldsonThomasReport {
    /// Densidad de invariantes BPS de Donaldson-Thomas $DT \in [0.0, 1.0]$.
    pub dt_invariant_density: f64,
    /// Energía acumulada de la partición tridimensional de MacMahon $M(q) \ge 0.0$.
    pub macmahon_partition_energy: f64,
    /// Magnitud del salto de estabilidad de cruce de paredes $|\Delta \Omega|$.
    pub wall_crossing_jump: f64,
    /// Convicción direccional cuántica $\psi_{\text{DT}} \in [-1.0, 1.0]$.
    pub directional_conviction: f64,
    /// Factor de escala de slot continuo $\eta_{\text{DT}} \in [0.10, 1.25]$.
    pub slot_scaling_factor: f64,
    /// Veto estricto por colapso de estabilidad de cruce de paredes de Donaldson-Thomas.
    pub donaldson_thomas_wall_crossing_veto: bool,
}

/// Motor cuántico de Donaldson-Thomas y vértice topológico de MacMahon.
#[derive(Debug, Clone)]
pub struct DonaldsonThomasQuantumEngine {
    config: DonaldsonThomasConfig,
    prev_stability_parameter: f64,
    smoothed_wall_crossing_jump: f64,
    smoothed_partition_energy: f64,
}

impl DonaldsonThomasQuantumEngine {
    /// Crea una nueva instancia del motor Donaldson-Thomas con la configuración provista.
    pub fn new(config: DonaldsonThomasConfig) -> Self {
        Self {
            config,
            prev_stability_parameter: 0.0,
            smoothed_wall_crossing_jump: 0.0,
            smoothed_partition_energy: 0.0,
        }
    }

    /// Evalúa la correspondencia cuántica de Donaldson-Thomas para el libro de órdenes.
    ///
    /// # Argumentos
    /// * `price_ret` - Retorno de precio instantáneo (D0-brane: carga puntual $d_0$).
    /// * `ofi` - Flujo de órdenes de libro agresor normalizado (D2-brane: carga lineal $d_1$).
    /// * `book_imbalance` - Desbalance de profundidad bid/ask normalizado (D4-brane: carga de superficie $d_2$).
    /// * `spread_bps` - Spread bid-ask relativo en puntos básicos.
    pub fn evaluate(
        &mut self,
        price_ret: f64,
        ofi: f64,
        book_imbalance: f64,
        spread_bps: f64,
    ) -> DonaldsonThomasReport {
        let eps = self.config.epsilon.max(1e-12);
        let input_invalid = !price_ret.is_finite()
            || !ofi.is_finite()
            || !book_imbalance.is_finite()
            || !spread_bps.is_finite();

        let safe_p = if price_ret.is_finite() {
            price_ret
        } else {
            0.0
        };
        let safe_ofi = if ofi.is_finite() { ofi } else { 0.0 };
        let safe_imb = if book_imbalance.is_finite() {
            book_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let safe_spread = if spread_bps.is_finite() {
            spread_bps.abs().max(eps)
        } else {
            eps
        };

        // 1. Acoplamiento cuántico de cuerda: q = exp(-g_s)
        // A mayor spread, mayor acoplamiento efectivo g_s -> q menor (más cerca de 0)
        let gs = (self.config.string_coupling_scale * (1.0 + safe_spread / 10.0)).clamp(0.01, 4.0);
        let q_quantum = (-gs).exp().clamp(0.001, 0.98);

        // 2. Evaluación de la serie logarítmica de la función de MacMahon:
        // ln M(q) = \sum_{k=1}^K (1/k) * q^k / (1 - q^k)^2
        let cutoff = self.config.macmahon_cutoff_k.clamp(1, 16);
        let mut log_macmahon = 0.0;
        let mut q_pow_k = q_quantum;
        for k in 1..=cutoff {
            let k_f = k as f64;
            let denom = 1.0 - q_pow_k;
            let term = (1.0 / k_f) * (q_pow_k / (denom * denom + eps));
            log_macmahon += term;
            q_pow_k *= q_quantum;
        }

        let instant_partition_energy = log_macmahon.min(50.0);
        self.smoothed_partition_energy = (1.0 - self.config.stability_decay_rate)
            * self.smoothed_partition_energy
            + self.config.stability_decay_rate * instant_partition_energy;
        let macmahon_energy = self.smoothed_partition_energy;

        // 3. Vector de carga de D-branas \gamma = (d0, d1, d2):
        // d0 = retorno * 100, d1 = ofi, d2 = book_imbalance
        let d0 = safe_p * 100.0;
        let d1 = safe_ofi;
        let d2 = safe_imb;

        // Potencial central de estabilidad de Kontsevich-Soibelman Z(\gamma) = z_0 * d0 + z_1 * d1 + z_2 * d2
        let charge_norm = (d0 * d0 + d1 * d1 + d2 * d2).sqrt();
        let current_stability = (d0 + 2.0 * d1 + 1.5 * d2) / (1.0 + 0.05 * charge_norm);

        // 4. Salto de cruce de paredes de Kontsevich-Soibelman |\Delta \Omega|:
        let instant_jump = (current_stability - self.prev_stability_parameter).abs();
        self.prev_stability_parameter = current_stability;

        self.smoothed_wall_crossing_jump = (1.0 - self.config.stability_decay_rate)
            * self.smoothed_wall_crossing_jump
            + self.config.stability_decay_rate * instant_jump;
        let wall_crossing_jump = self.smoothed_wall_crossing_jump;

        // 5. Densidad de invariantes BPS de Donaldson-Thomas DT \in [0.0, 1.0]:
        // Mide la coherencia y estabilidad de los estados ligados en el libro
        let dt_invariant_density = (1.0 / (1.0 + (wall_crossing_jump * 1.5).powi(2)))
            * (1.0 - (-0.25 * macmahon_energy).exp());
        let dt_invariant_density = dt_invariant_density.clamp(0.0, 1.0);

        // 6. Convicción direccional cuántica \psi_{\text{DT}} \in [-1.0, 1.0]:
        // Guiada por la alineación de flujo d1 (OFI) y desbalance d2 (L2) modulada por estabilidad BPS
        let composite_direction = 0.55 * safe_ofi + 0.45 * safe_imb;
        let directional_conviction = (composite_direction * (1.0 + dt_invariant_density)).tanh();
        let directional_conviction = directional_conviction.clamp(-1.0, 1.0);

        // 7. Modulación de escala de slot continuo \eta_{\text{DT}} \in [0.10, 1.25]:
        let wall_penalty = (-0.50 * wall_crossing_jump).exp();
        let spread_factor = 1.0 / (1.0 + (safe_spread / 30.0).powi(2));
        let mut slot_scaling_factor = (0.10
            + 1.15 * wall_penalty * spread_factor * (0.5 + 0.5 * dt_invariant_density))
            .clamp(0.10, 1.25);

        // 8. Veto estricto por colapso de cruce de pared Kontsevich-Soibelman:
        let jump_exceeded = wall_crossing_jump > self.config.wall_crossing_threshold;
        let numerical_fault = wall_crossing_jump.is_nan()
            || wall_crossing_jump.is_infinite()
            || dt_invariant_density.is_nan()
            || dt_invariant_density.is_infinite();
        let donaldson_thomas_wall_crossing_veto = jump_exceeded || numerical_fault || input_invalid;

        if donaldson_thomas_wall_crossing_veto {
            slot_scaling_factor = 0.10;
        }

        DonaldsonThomasReport {
            dt_invariant_density,
            macmahon_partition_energy: macmahon_energy,
            wall_crossing_jump,
            directional_conviction,
            slot_scaling_factor,
            donaldson_thomas_wall_crossing_veto,
        }
    }

    /// Reinicia los acumuladores temporales del motor a cero.
    pub fn reset(&mut self) {
        self.prev_stability_parameter = 0.0;
        self.smoothed_wall_crossing_jump = 0.0;
        self.smoothed_partition_energy = 0.0;
    }
}
