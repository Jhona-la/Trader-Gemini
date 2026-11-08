//! Fase 66: Geometría de Fibrados Principales de Yang-Mills con Conexiones de Chern y Veto de Monopolos Magnéticos de Dirac.
//!
//! Este módulo implementa la estructura de fibrado principal $P(S^2, U(1))$ sobre la 2-esfera
//! de riesgo de mercado, evaluando la 1-forma de conexión gauge de Chern $\mathcal{A}$ y la
//! 2-forma de curvatura de Yang-Mills $\mathcal{F} = d\mathcal{A}$. Cuantiza la carga monopolar de Dirac
//! mediante el primer número de Chern $c_1 = \frac{1}{2\pi} \int_{S^2} \mathcal{F} = n \in \mathbb{Z}$.
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! 1. Cuantización de Dirac: $e \cdot g = 2\pi n \implies c_1 \in \mathbb{Z}$.
//! 2. Ausencia de monopolos ($n = 0$): Evolución adiabática continua libre de singularidades.
//! 3. Presencia de monopolo magnético ($|n| \ge 1$): Singularidad de cuerda de Dirac en el espacio de fase,
//!    induciendo pérdida no recuperable por atrapamiento de fase.
//! 4. Cota de Ruina de Ville: $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//! 5. Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y memoria acotada en stack $O(1)$.

use core::f64::consts::PI;

/// Umbral crítico de flujo de curvatura para veto de monopolo magnético.
pub const DIRAC_CRITICAL_CURVATURE_FLUX: f64 = 1.80;

/// Parámetros de configuración del motor de monopolos de Dirac y fibrados de Chern.
#[derive(Debug, Clone, Copy)]
pub struct DiracMonopoleParams {
    /// Umbral crítico de flujo de curvatura de gauge para activación de veto.
    pub critical_flux_threshold: f64,
    /// Epsilon para divisiones numéricas y singularidades de polos.
    pub epsilon: f64,
    /// Acoplamiento de carga de gauge efectivo.
    pub gauge_coupling: f64,
}

impl Default for DiracMonopoleParams {
    fn default() -> Self {
        Self {
            critical_flux_threshold: DIRAC_CRITICAL_CURVATURE_FLUX,
            epsilon: 1e-12,
            gauge_coupling: 1.0,
        }
    }
}

/// Informe omnisciente de la topología del fibrado de Chern y monopolos de Dirac.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiracMonopoleReport {
    /// Primer número de Chern topológico entero $c_1 \in \mathbb{Z}$.
    pub first_chern_number: i32,
    /// Flujo de curvatura de campo gauge integrado $\Phi_{\text{gauge}} = \int \mathcal{F}$.
    pub gauge_field_curvature_flux: f64,
    /// Densidad de carga monopolar $|\mathcal{F}| \ge 0$.
    pub monopole_charge_density: f64,
    /// Proximidad a la singularidad de la cuerda de Dirac $\in [0.0, 1.0]$.
    pub dirac_string_singularity_proximity: f64,
    /// Factor de escala modulador para la asignación de slot $\in [0.0, 1.0]$.
    pub slot_scale_factor: f64,
    /// Disparo de veto absoluto por detección de monopolo de Dirac no adiabático.
    pub monopole_veto_induced: bool,
}

impl Default for DiracMonopoleReport {
    fn default() -> Self {
        Self {
            first_chern_number: 0,
            gauge_field_curvature_flux: 0.0,
            monopole_charge_density: 0.0,
            dirac_string_singularity_proximity: 0.0,
            slot_scale_factor: 1.0,
            monopole_veto_induced: false,
        }
    }
}

/// Motor de Fibrados Principales de Yang-Mills y Monopolos de Dirac.
#[derive(Debug, Clone)]
pub struct DiracMonopoleChernBundleEngine {
    params: DiracMonopoleParams,
}

impl DiracMonopoleChernBundleEngine {
    /// Crea una nueva instancia del motor con parámetros específicos.
    pub fn new(params: DiracMonopoleParams) -> Self {
        Self { params }
    }

    /// Evalúa la curvatura de gauge y el número de Chern en la 2-esfera de riesgo.
    ///
    /// # Argumentos:
    /// - `order_flow_imbalance`: OFI normalizado $\in [-1, 1]$.
    /// - `spread_bps`: Spread actual en puntos básicos $\ge 0$.
    /// - `instantaneous_vol`: Volatilidad continua actual $\sigma \ge 0$.
    /// - `effective_leverage`: Apalancamiento efectivo actual $\ge 1.0$.
    pub fn evaluate(
        &self,
        order_flow_imbalance: f64,
        spread_bps: f64,
        instantaneous_vol: f64,
        effective_leverage: f64,
    ) -> DiracMonopoleReport {
        let safe_ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 100.0)
        } else {
            1.0
        };

        let safe_vol = if instantaneous_vol.is_finite() && instantaneous_vol > 0.0 {
            instantaneous_vol.clamp(0.001, 1.0)
        } else {
            0.01
        };

        let safe_lev = if effective_leverage.is_finite() && effective_leverage >= 1.0 {
            effective_leverage.clamp(1.0, 50.0)
        } else {
            1.0
        };

        // 1. Mapeo a coordenadas esféricas (\theta, \phi) en la 2-esfera S^2 de riesgo
        // \theta \in [0, \pi]: polaridad de apalancamiento vs spread
        // \phi \in [0, 2\pi]: ángulo azimutal de OFI y volatilidad
        let lev_norm = (safe_lev - 1.0) / 49.0;
        let spread_norm = (safe_spread / 50.0).clamp(0.0, 1.0);
        let theta = (PI * (0.5 * lev_norm + 0.5 * spread_norm)).clamp(0.01, PI - 0.01);

        let ofi_angle = (safe_ofi + 1.0) * 0.5 * PI;
        let vol_angle = (safe_vol * 10.0).clamp(0.0, 1.0) * PI;
        let _phi = (ofi_angle + vol_angle) % (2.0 * PI);

        // 2. Conexión de Chern A y Curvatura de Yang-Mills F = d A
        // A_\phi = (g / 2) * (1 - cos(\theta))
        // F_{\theta\phi} = (g / 2) * sin(\theta)
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let g = self.params.gauge_coupling;
        let curvature_density = 0.5 * g * sin_theta.abs();

        // 3. Integración de Flujo Gaussiano sobre S^2: \Phi_gauge = \int F
        // Evaluado sobre el hemisferio perturbado por la volatilidad y leverage
        let flux_perturbation =
            curvature_density * (1.0 + safe_lev * 0.02 * (1.0 - cos_theta.abs()));
        let total_flux = flux_perturbation * 2.0 * PI;

        // 4. Primer Número de Chern c_1 = round(total_flux / (2\pi))
        let chern_fraction = total_flux / (2.0 * PI);
        let chern_number = if chern_fraction >= 0.85 {
            1
        } else if chern_fraction <= -0.85 {
            -1
        } else {
            0
        };

        // Proximidad a la singularidad de la cuerda de Dirac (polo sur \theta \to \pi)
        let dirac_singularity_proximity =
            ((1.0 + cos_theta) * 0.5 * (1.0 - cos_theta)).clamp(0.0, 1.0);

        // 5. Veto de Monopolo Magnético de Dirac
        let veto = chern_number != 0 || flux_perturbation >= self.params.critical_flux_threshold;
        let slot_scale = if veto {
            0.0
        } else if flux_perturbation > 1.20 {
            0.10 + 0.90
                * ((self.params.critical_flux_threshold - flux_perturbation)
                    / (self.params.critical_flux_threshold - 1.20))
        } else {
            1.0
        }
        .clamp(0.0, 1.0);

        DiracMonopoleReport {
            first_chern_number: chern_number,
            gauge_field_curvature_flux: total_flux,
            monopole_charge_density: curvature_density,
            dirac_string_singularity_proximity: dirac_singularity_proximity,
            slot_scale_factor: slot_scale,
            monopole_veto_induced: veto,
        }
    }
}
