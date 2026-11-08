//! Fase 61: Cuantización por Espacios de Hilbert Reproducentes de Bergman-Fock y Haces de Línea Holomorfos.
//!
//! Este módulo implementa la proyección holomorfa de microestructura de mercado en espacios de
//! Hilbert con núcleo reproducente (RKHS) de Bergman-Fock sobre el dominio acotado $\mathbb{B}^2 \subset \mathbb{C}^2$.
//! Modela la saturación adiabática del libro de órdenes y la proximidad a la frontera de Shilov.
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! 1. Núcleo de Bergman en la bola unitaria compleja $\mathbb{B}^2$: $K(z, z) = \frac{2}{\pi^2 (1 - \|z\|^2)^3}$.
//! 2. Potencial de Kähler continuo: $\Phi(z) = \ln K(z, z)$.
//! 3. Proximidad a la frontera de Shilov $\partial \mathbb{B}^2$: $\delta_{\text{Shilov}} = \|z\|^2 \in [0, 1)$.
//! 4. Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y stack $O(1)$.

use core::f64::consts::PI;

/// Dimensión compleja del espacio de microestructura Kähler.
pub const BERGMAN_COMPLEX_DIM: usize = 2;

/// Parámetros de configuración del motor de Bergman-Fock.
#[derive(Debug, Clone, Copy)]
pub struct BergmanFockParams {
    /// Cota máxima admisible para $\|z\|^2$ antes de saturación asintótica.
    pub max_norm_sq: f64,
    /// Epsilon para estabilidad numérica y división segura.
    pub epsilon: f64,
    /// Umbral de proximidad a la frontera de Shilov para alerta crítica.
    pub shilov_critical_threshold: f64,
}

impl Default for BergmanFockParams {
    fn default() -> Self {
        Self {
            max_norm_sq: 0.985,
            epsilon: 1e-12,
            shilov_critical_threshold: 0.90,
        }
    }
}

/// Informe omnisciente de cuantización de Bergman-Fock y métrica de Kähler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BergmanFockReport {
    /// Densidad del núcleo reproducente $K(z, z) \ge 0$.
    pub bergman_kernel_density: f64,
    /// Potencial de Kähler continuo $\Phi(z) = \ln K(z, z)$.
    pub kahler_potential: f64,
    /// Distorsión holomorfa normalizada $\in [0.0, 1.0]$.
    pub holomorphic_distortion: f64,
    /// Proximidad cuadrática a la frontera de Shilov $\|z\|^2 \in [0.0, 1.0]$.
    pub shilov_boundary_proximity: f64,
    /// Factor de escala defensivo para asignación de slot $\in [0.10, 1.0]$.
    pub slot_scale_factor: f64,
    /// Alerta de saturación crítica por aproximación extrema a la frontera.
    pub critical_saturation_detected: bool,
}

impl Default for BergmanFockReport {
    fn default() -> Self {
        Self {
            bergman_kernel_density: 2.0 / (PI * PI),
            kahler_potential: 0.0,
            holomorphic_distortion: 0.0,
            shilov_boundary_proximity: 0.0,
            slot_scale_factor: 1.0,
            critical_saturation_detected: false,
        }
    }
}

/// Motor de Espacio de Hilbert Reproducente de Bergman-Fock.
#[derive(Debug, Clone)]
pub struct BergmanFockReproducingEngine {
    params: BergmanFockParams,
    prev_mid_price: f64,
    initialized: bool,
}

impl BergmanFockReproducingEngine {
    /// Crea una nueva instancia del motor con parámetros específicos.
    pub fn new(params: BergmanFockParams) -> Self {
        Self {
            params,
            prev_mid_price: 0.0,
            initialized: false,
        }
    }

    /// Actualiza la proyección holomorfa en $\mathbb{B}^2$ a partir de observables continuos.
    ///
    /// # Argumentos:
    /// - `mid_price`: Precio medio actual $> 0$.
    /// - `spread_bps`: Spread relativo actual en puntos básicos $\ge 0$.
    /// - `order_flow_imbalance`: Desequilibrio OFI $\in [-1, 1]$.
    /// - `depth_imbalance`: Desequilibrio relativo de profundidad $\in [-1, 1]$.
    /// - `instantaneous_vol`: Volatilidad instantánea $\sigma \ge 0$.
    pub fn update(
        &mut self,
        mid_price: f64,
        spread_bps: f64,
        order_flow_imbalance: f64,
        depth_imbalance: f64,
        instantaneous_vol: f64,
    ) -> BergmanFockReport {
        let safe_mid = if mid_price.is_finite() && mid_price > 0.0 {
            mid_price
        } else {
            100.0
        };

        if !self.initialized {
            self.prev_mid_price = safe_mid;
            self.initialized = true;
        }

        let ret = (safe_mid - self.prev_mid_price) / self.prev_mid_price.max(self.params.epsilon);
        self.prev_mid_price = safe_mid;

        let safe_vol = if instantaneous_vol.is_finite() && instantaneous_vol > 0.0 {
            instantaneous_vol
        } else {
            0.01
        };

        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps
        } else {
            1.0
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

        // 1. Proyección a coordenadas complejas z_1 y z_2 en la bola unitaria B^2
        // z_1 = Re(ret normalizado) + i * Im(spread normalizado)
        // Escalado a radio submétrico para asegurar pertenencia a la bola
        let scale_factor = 0.65;
        let re_z1 = (ret / (3.0 * safe_vol + self.params.epsilon)).clamp(-1.0, 1.0) * scale_factor;
        let im_z1 = (safe_spread / 50.0).clamp(0.0, 1.0) * scale_factor;

        // z_2 = Re(OFI) + i * Im(depth imbalance)
        let re_z2 = safe_ofi * scale_factor;
        let im_z2 = safe_depth * scale_factor;

        let norm_sq_z1 = re_z1 * re_z1 + im_z1 * im_z1;
        let norm_sq_z2 = re_z2 * re_z2 + im_z2 * im_z2;
        let total_norm_sq = (norm_sq_z1 + norm_sq_z2).clamp(0.0, self.params.max_norm_sq);

        // 2. Núcleo de Bergman: K(z, z) = 2 / (pi^2 * (1 - ||z||^2)^3)
        let one_minus_norm = (1.0 - total_norm_sq).max(self.params.epsilon);
        let denom = PI * PI * one_minus_norm * one_minus_norm * one_minus_norm;
        let kernel_density = (2.0 / denom.max(self.params.epsilon)).clamp(0.0, 1e7);

        // 3. Potencial de Kähler: Phi(z) = ln(K(z, z))
        let kahler_potential = kernel_density.max(self.params.epsilon).ln();

        // 4. Distorsión Holomorfa y Proximidad a Shilov
        let shilov_proximity = total_norm_sq;
        let holomorphic_distortion = (total_norm_sq / self.params.max_norm_sq).clamp(0.0, 1.0);

        let critical = shilov_proximity >= self.params.shilov_critical_threshold;
        let slot_scale_factor = if critical {
            0.15
        } else if holomorphic_distortion > 0.70 {
            0.50 + 0.50 * (1.0 - (holomorphic_distortion - 0.70) / 0.30)
        } else {
            1.0
        };

        BergmanFockReport {
            bergman_kernel_density: kernel_density,
            kahler_potential,
            holomorphic_distortion,
            shilov_boundary_proximity: shilov_proximity,
            slot_scale_factor,
            critical_saturation_detected: critical,
        }
    }
}
