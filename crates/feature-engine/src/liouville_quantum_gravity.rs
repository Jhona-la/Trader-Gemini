//! # Fase 73: Teoría Conforme de Campos (CFT) de Liouville y Gravedad Cuántica 2D
//!
//! Modela la métrica fluctuante del libro de órdenes como $g_{ab} = e^{2b \phi} \hat{g}_{ab}$,
//! donde $\phi$ es el campo escalar de Liouville regido por la acción de gravedad cuántica:
//! $$S_L[\phi] = \frac{1}{4\pi} \int d^2\xi \sqrt{\hat{g}} \left( \hat{g}^{ab} \partial_a \phi \partial_b \phi + Q \hat{R} \phi + 4\pi \mu e^{2b\phi} \right)$$
//! con carga de fondo $Q = b + 1/b$ y carga central conforme $c_L = 1 + 6Q^2 \ge 25$.
//!
//! ## Invariantes del Sistema:
//! - Cero asignaciones en heap en la ruta crítica ($O(1)$ stack determinista).
//! - Cero pánicos (`rb_p1`): ausencia total de `unwrap()`, `expect()`, `assert!()` o `panic!()`.

use core::f64;

/// Parámetros configurables del motor de Gravedad Cuántica de Liouville.
#[derive(Debug, Clone, Copy)]
pub struct LiouvilleCftParams {
    /// Parámetro de acoplamiento de Liouville $b \in (0, 1)$. Típicamente $b \approx 0.40$.
    pub liouville_b: f64,
    /// Constante cosmológica de Liouville $\mu > 0$.
    pub cosmological_mu: f64,
    /// Umbral crítico de curvatura escalar para declarar singularidad cuántica.
    pub singularity_curvature_threshold: f64,
    /// Factor de escala mínimo defensivo ante singularidad cuántica.
    pub min_slot_scale: f64,
    /// Regularizador numérico epsilon $> 0$.
    pub epsilon: f64,
}

impl Default for LiouvilleCftParams {
    fn default() -> Self {
        Self {
            liouville_b: 0.40,
            cosmological_mu: 1.0,
            singularity_curvature_threshold: 4.50,
            min_slot_scale: 0.15,
            epsilon: 1e-12,
        }
    }
}

/// Informe omnisciente de métrica conforme y geometría de Liouville.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiouvilleQuantumGravityReport {
    /// Amplitud media del campo de Liouville $\phi$.
    pub liouville_field_phi: f64,
    /// Factor conforme de la métrica $e^{2b\phi} \ge 1.0$.
    pub conformal_factor: f64,
    /// Carga central de Liouville $c_L = 1 + 6(b + 1/b)^2 \ge 25.0$.
    pub central_charge_cl: f64,
    /// Dimensión fractal de Hausdorff del espaciotiempo de mercado $d_H \ge 2.0$.
    pub hausdorff_dimension: f64,
    /// Curvatura escalar efectiva de Liouville $\mathcal{R}_L$.
    pub liouville_scalar_curvature: f64,
    /// Modulador continuo de slot scale $\in [0.15, 1.25]$.
    pub slot_scale_factor: f64,
    /// Detección de singularidad de curvatura de Liouville / desgarro métrico.
    pub curvature_singularity_detected: bool,
}

impl Default for LiouvilleQuantumGravityReport {
    fn default() -> Self {
        Self {
            liouville_field_phi: 0.0,
            conformal_factor: 1.0,
            central_charge_cl: 25.0,
            hausdorff_dimension: 2.0,
            liouville_scalar_curvature: 0.0,
            slot_scale_factor: 1.0,
            curvature_singularity_detected: false,
        }
    }
}

/// Motor de Gravedad Cuántica 2D y Teoría Conforme de Liouville en stack $O(1)$.
#[derive(Debug, Clone)]
pub struct LiouvilleCft2dQuantumGravityEngine {
    params: LiouvilleCftParams,
    prev_phi: f64,
    has_prev: bool,
}

impl LiouvilleCft2dQuantumGravityEngine {
    /// Construye una nueva instancia inicializada en el vacío conforme.
    pub fn new(params: LiouvilleCftParams) -> Self {
        Self {
            params,
            prev_phi: 0.0,
            has_prev: false,
        }
    }

    /// Actualiza la geometría conforme de Liouville con la dinámica del libro de órdenes.
    ///
    /// # Argumentos:
    /// - `price_return`: Retorno continuo instantáneo del precio.
    /// - `volume_flow`: Desbalance normalizado de flujo de volumen $[-1.0, 1.0]$.
    /// - `volatility`: Volatilidad efectiva continua instantánea $\ge 0$.
    /// - `spread_bps`: Spread medio en puntos básicos $\ge 0$.
    pub fn update(
        &mut self,
        price_return: f64,
        volume_flow: f64,
        volatility: f64,
        spread_bps: f64,
    ) -> LiouvilleQuantumGravityReport {
        let safe_ret = if price_return.is_finite() { price_return.clamp(-0.5, 0.5) } else { 0.0 };
        let safe_vol_flow = if volume_flow.is_finite() { volume_flow.clamp(-1.0, 1.0) } else { 0.0 };
        let safe_vol = if volatility.is_finite() && volatility >= 0.0 {
            volatility.clamp(0.0001, 1.0)
        } else {
            0.01
        };
        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.01, 100.0)
        } else {
            1.0
        };

        let b = self.params.liouville_b.clamp(0.05, 0.95);
        let mu = self.params.cosmological_mu.max(self.params.epsilon);

        // 1. Carga de fondo Q = b + 1/b y Carga Central c_L = 1 + 6 Q^2
        let q_background = b + 1.0 / b;
        let central_charge_cl = 1.0 + 6.0 * q_background * q_background;

        // 2. Dimensión de Hausdorff d_H de la Gravedad Cuántica 2D:
        // d_H = \frac{2}{b} (\sqrt{1 + b^2/2} + 1)
        let inner_root = (1.0 + 0.5 * b * b).sqrt();
        let hausdorff_dimension = (2.0 / b) * (inner_root + 1.0);

        // 3. Campo de Liouville local \phi:
        // Mapea la energía de deformación de microestructura (fricción de spread y volatilidad térmica)
        let excitation = (safe_spread * 0.08 + safe_vol * 15.0 + safe_ret.abs() * 5.0 + safe_vol_flow.abs() * 0.5)
            / mu;
        let safe_excitation = excitation.max(0.0);
        let phi = (1.0 / (2.0 * b)) * (safe_excitation + 1.0).ln();

        // Factor conforme de la métrica e^{2b\phi}
        let conformal_factor = (2.0 * b * phi).exp().clamp(1.0, 50.0);

        // 4. Laplaciano temporal discreto del campo \nabla^2 \phi:
        let d_phi = if self.has_prev {
            phi - self.prev_phi
        } else {
            0.0
        };
        self.prev_phi = phi;
        self.has_prev = true;

        // Curvatura escalar de Liouville: \mathcal{R}_L = -4\pi \mu e^{2b\phi} + Q \cdot \Delta \phi
        // Normalizada para microestructura:
        let liouville_scalar_curvature = (safe_excitation * 2.0 + q_background * d_phi.abs() * 3.0).clamp(0.0, 20.0);

        // 5. Detección de Singularidad de Curvatura de Liouville:
        let curvature_singularity_detected = liouville_scalar_curvature >= self.params.singularity_curvature_threshold
            || safe_spread > 15.0;

        // 6. Factor de escala modulador de slot:
        let slot_scale_factor = if curvature_singularity_detected {
            self.params.min_slot_scale
        } else {
            // Modulación constructiva en geometría suave de Liouville
            let curvature_ratio = (liouville_scalar_curvature / self.params.singularity_curvature_threshold).clamp(0.0, 1.0);
            (1.15 - 0.40 * curvature_ratio).clamp(self.params.min_slot_scale, 1.25)
        };

        LiouvilleQuantumGravityReport {
            liouville_field_phi: phi,
            conformal_factor,
            central_charge_cl,
            hausdorff_dimension,
            liouville_scalar_curvature,
            slot_scale_factor,
            curvature_singularity_detected,
        }
    }
}
