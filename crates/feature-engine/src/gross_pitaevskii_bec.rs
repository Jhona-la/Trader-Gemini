//! Fase 70: Condensados de Bose-Einstein (BEC) y Vórtices Cuantizados de Gross-Pitaevskii.
//!
//! Este módulo implementa la ecuación de campo medio no lineal de Gross-Pitaevskii (GPE)
//! para modelar la condensación macroscópica de liquidez pasiva en el estado fundamental del spread:
//! $$i \hbar \partial_t \psi = \left( -\frac{\hbar^2}{2m} \nabla^2 + V(\mathbf{r}) + g |\psi|^2 \right) \psi$$
//! evaluando la fracción de condensado $N_0 / N$, la velocidad del sonido de Bogoliubov $c_s$,
//! y la emergencia de vórtices cuánticos de circulación cuantizada:
//! $$\Gamma = \oint_{\mathcal{C}} \mathbf{v}_s \cdot d\mathbf{r} = \frac{2\pi \hbar}{m} n, \quad n \in \mathbb{Z}$$
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! 1. Cuantización de Circulación de Onsager-Feynman: $\Gamma = \frac{h}{m} n$.
//! 2. Velocidad de Bogoliubov: $c_s = \sqrt{g n_0 / m} > 0$.
//! 3. Cavitación de Vórtice: El núcleo del vórtice presenta densidad nula $|\psi|^2 \to 0$.
//! 4. Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y memoria acotada en stack $O(1)$.

use core::f64::consts::PI;

/// Constante de Planck reducida efectiva en microespaciotiempo $\hbar_{\text{eff}}$.
pub const HBAR_EFF: f64 = 1.0;

/// Masa de partícula elemental de liquidez $m$.
pub const PARTICLE_MASS_M: f64 = 1.0;

/// Número de sectores angulares para la integración de circulación $\mathcal{C}$.
pub const BEC_ANGULAR_SECTORS: usize = 8;

/// Parámetros de configuración del motor de condensado de Bose-Einstein y Gross-Pitaevskii.
#[derive(Debug, Clone, Copy)]
pub struct GrossPitaevskiiParams {
    /// Constante de interacción no lineal de dos cuerpos $g > 0$.
    pub coupling_g: f64,
    /// Frecuencia angular de la trampa armónica $\omega > 0$.
    pub trap_omega: f64,
    /// Umbral mínimo de fracción de condensado para clasificar régimen superfluido.
    pub min_condensate_fraction_threshold: f64,
    /// Epsilon para divisiones numéricas seguras.
    pub epsilon: f64,
    /// Factor mínimo de escala moduladora de slot en stack.
    pub min_slot_scale: f64,
}

impl Default for GrossPitaevskiiParams {
    fn default() -> Self {
        Self {
            coupling_g: 0.85,
            trap_omega: 1.20,
            min_condensate_fraction_threshold: 0.35,
            epsilon: 1e-12,
            min_slot_scale: 0.15,
        }
    }
}

/// Informe omnisciente del estado cuántico del condensado de Gross-Pitaevskii.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrossPitaevskiiBecReport {
    /// Fracción macroscópica de condensado en el ground state $f_0 = N_0 / N \in [0.0, 1.0]$.
    pub condensate_fraction: f64,
    /// Densidad macroscópica central de partículas $n_0 = |\psi(0)|^2 \ge 0$.
    pub central_density: f64,
    /// Velocidad del sonido cuántico de Bogoliubov $c_s = \sqrt{g n_0 / m} \ge 0$.
    pub bogoliubov_sound_speed: f64,
    /// Longitud de coherencia cuántica (healing length) $\xi = \hbar / \sqrt{2 m g n_0}$.
    pub healing_length: f64,
    /// Carga topológica del vórtice cuántico (winding number) $n \in \mathbb{Z}$.
    pub quantized_vortex_charge: i32,
    /// Circulación superfluida total de Onsager-Feynman $\Gamma = \oint \mathbf{v}_s \cdot d\mathbf{r}$.
    pub superfluid_circulation: f64,
    /// Modulador continuo para asignación de slot $\in [0.15, 1.25]$.
    pub slot_scale_factor: f64,
    /// Detección de vórtice cuántico activo con cavitación de liquidez pasiva.
    pub quantum_vortex_shedding_detected: bool,
}

impl Default for GrossPitaevskiiBecReport {
    fn default() -> Self {
        Self {
            condensate_fraction: 0.85,
            central_density: 1.0,
            bogoliubov_sound_speed: 1.0,
            healing_length: 0.50,
            quantized_vortex_charge: 0,
            superfluid_circulation: 0.0,
            slot_scale_factor: 1.0,
            quantum_vortex_shedding_detected: false,
        }
    }
}

/// Motor de Dinámica No Lineal de Gross-Pitaevskii y Condensados de Bose-Einstein.
#[derive(Debug, Clone)]
pub struct GrossPitaevskiiBecEngine {
    params: GrossPitaevskiiParams,
    prev_x: f64,
    prev_y: f64,
    has_prev: bool,
}

impl GrossPitaevskiiBecEngine {
    /// Construye una nueva instancia en stack $O(1)$ con parámetros dados.
    pub fn new(params: GrossPitaevskiiParams) -> Self {
        Self {
            params,
            prev_x: 0.0,
            prev_y: 0.0,
            has_prev: false,
        }
    }

    /// Actualiza el condensado y evalúa la circulación y vórtices cuánticos de liquidez.
    ///
    /// # Argumentos:
    /// - `price_ret`: Retorno continuo instantáneo de micro-precio.
    /// - `order_flow_imbalance`: OFI normalizado $\in [-1, 1]$.
    /// - `depth_imbalance`: Desbalance de profundidad L2 $\in [-1, 1]$.
    /// - `spread_bps`: Spread actual en puntos básicos $\ge 0$.
    /// - `instantaneous_vol`: Volatilidad continua estimada $\ge 0$.
    pub fn update(
        &mut self,
        price_ret: f64,
        order_flow_imbalance: f64,
        depth_imbalance: f64,
        spread_bps: f64,
        instantaneous_vol: f64,
    ) -> GrossPitaevskiiBecReport {
        let safe_ret = if price_ret.is_finite() {
            price_ret.clamp(-0.5, 0.5)
        } else {
            0.0
        };

        let safe_ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_depth = if depth_imbalance.is_finite() {
            depth_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 100.0)
        } else {
            1.0
        };

        let safe_vol = if instantaneous_vol.is_finite() && instantaneous_vol > 0.0 {
            instantaneous_vol.clamp(0.0001, 1.0)
        } else {
            0.01
        };

        // 1. Densidad macroscópica de partículas en el espacio de fase (x, y):
        // x: momento direccional de precio, y: flujo pasivo acumulado
        let x = (safe_ret * 20.0 + safe_ofi).clamp(-3.0, 3.0);
        let y = (safe_depth * 1.5 - (safe_spread * 0.05 - 0.20)).clamp(-3.0, 3.0);
        let r_sq = x * x + y * y;

        // Potencial armónico de confinamiento en microestructura V(r) = 0.5 * m * \omega^2 * r^2
        let harmonic_pot = 0.5 * PARTICLE_MASS_M * self.params.trap_omega * self.params.trap_omega * r_sq;

        // Densidad de fondo de Thomas-Fermi: n_0 = \max(0, (\mu - V) / g)
        // Potencial químico \mu proporcional a la liquidez efectiva inversa a la volatilidad
        let chemical_mu = (5.0 / (safe_vol * 10.0 + 0.5)).clamp(0.1, 15.0);
        let central_density = ((chemical_mu - harmonic_pot) / self.params.coupling_g.max(self.params.epsilon))
            .clamp(0.01, 20.0);

        // 2. Fracción de Condensado f_0 = 1 - (T / T_c)^3
        // Temperatura efectiva dictada por la fricción de spread y volatilidad térmica
        let temp_ratio = ((safe_spread * 0.10 + safe_vol * 30.0) / chemical_mu.max(self.params.epsilon))
            .clamp(0.0, 2.0);
        let condensate_fraction = (1.0 - temp_ratio * temp_ratio * temp_ratio).clamp(0.0, 1.0);

        // 3. Velocidad del Sonido de Bogoliubov y Longitud de Coherencia (Healing Length):
        // c_s = \sqrt{g n_0 / m}
        let bogoliubov_sound_speed = (self.params.coupling_g * central_density / PARTICLE_MASS_M).sqrt();
        // \xi = \hbar / \sqrt{2 m g n_0}
        let denom_healing = (2.0 * PARTICLE_MASS_M * self.params.coupling_g * central_density).sqrt();
        let healing_length = (HBAR_EFF / denom_healing.max(self.params.epsilon)).clamp(0.01, 10.0);

        // 4. Dinámica Rotacional y Cuantización de Circulación de Onsager-Feynman:
        // Criterio de Feynman para nucleación de vórtice en condensado atrapado:
        // \Omega_c = (\hbar / (m * R^2)) * \ln(R / \xi)
        let trap_radius = 2.0_f64;
        let healing_ratio = (trap_radius / healing_length.max(0.01)).max(1.05);
        let critical_rotation_omega = ((HBAR_EFF / (PARTICLE_MASS_M * trap_radius * trap_radius)) * healing_ratio.ln())
            .clamp(0.05, 5.0);

        let quantized_vortex_charge = if self.has_prev {
            let cross = self.prev_x * y - self.prev_y * x;
            let dot = self.prev_x * x + self.prev_y * y;
            let d_theta = cross.atan2(dot);
            let rot_speed = d_theta.abs();

            if rot_speed >= critical_rotation_omega {
                let sign = if cross >= 0.0 { 1.0 } else { -1.0 };
                let charge = sign * (rot_speed / critical_rotation_omega).round();
                (charge as i32).clamp(-3, 3)
            } else {
                0
            }
        } else {
            0
        };

        self.prev_x = x;
        self.prev_y = y;
        self.has_prev = true;

        // Circulación superfluida \Gamma = (2\pi \hbar / m) * n
        let superfluid_circulation = (2.0 * PI * HBAR_EFF / PARTICLE_MASS_M) * (quantized_vortex_charge as f64);

        // 5. Detección de Cavitación y Vórtice Cuántico Activo:
        // Si hay una carga de vórtice no nula (|n| >= 1) o la fracción de condensado colapsa
        let quantum_vortex_shedding_detected = quantized_vortex_charge != 0
            || (condensate_fraction < self.params.min_condensate_fraction_threshold && safe_spread > 4.0);

        // 6. Factor de Escala de Slot:
        let slot_scale_factor = if quantum_vortex_shedding_detected {
            // Depresión defensiva en presencia de vórtice de circulación (cavitación de liquidez)
            self.params.min_slot_scale
        } else {
            // Modulación constructiva en presencia de un condensado superfluido denso
            (0.85 + 0.35 * condensate_fraction).clamp(self.params.min_slot_scale, 1.25)
        };

        GrossPitaevskiiBecReport {
            condensate_fraction,
            central_density,
            bogoliubov_sound_speed,
            healing_length,
            quantized_vortex_charge,
            superfluid_circulation,
            slot_scale_factor,
            quantum_vortex_shedding_detected,
        }
    }
}
