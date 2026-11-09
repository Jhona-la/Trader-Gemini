//! # Fase 74: Dualidad AdS/CFT Holográfica de Maldacena y Tensión de Cuerda Cuántica
//!
//! Mapea el universo de fluctuaciones de precios y flujo de órdenes en el borde conforme ($d=2$)
//! hacia un espaciotiempo curvo Anti-de Sitter ($AdS_3$) en el bulto gravitatorio:
//! $$ds^2 = \frac{L_{AdS}^2}{z^2} (-dt^2 + dx^2 + dz^2)$$
//! donde la coordenada radial $z$ representa la escala del Grupo de Renormalización (RG flow):
//! $z \to 0$ es la microescala UV, $z \to \infty$ es la escala macro IR.
//!
//! ## Invariantes del Sistema:
//! - Cero asignaciones en heap en la ruta crítica ($O(1)$ stack determinista).
//! - Cero pánicos (`rb_p1`): ausencia total de `unwrap()`, `expect()`, `assert!()` o `panic!()`.

use core::f64;

/// Parámetros configurables del motor holográfico AdS/CFT.
#[derive(Debug, Clone, Copy)]
pub struct AdSCftParams {
    /// Radio de curvatura de Anti-de Sitter $L_{AdS} > 0$.
    pub ads_radius_l: f64,
    /// Dimensión conforme base del operador escalar primario $\Delta_0 \ge 1.0$.
    pub base_conformal_dimension: f64,
    /// Umbral mínimo del acoplamiento de 't Hooft $\lambda_{\text{crit}}$ para validez de supergravedad clásica.
    pub min_thooft_coupling_threshold: f64,
    /// Umbral máximo de penetración radial en el bulto $z_{\text{max}}$.
    pub max_radial_penetration_z: f64,
    /// Regularizador numérico epsilon $> 0$.
    pub epsilon: f64,
}

impl Default for AdSCftParams {
    fn default() -> Self {
        Self {
            ads_radius_l: 1.0,
            base_conformal_dimension: 2.0,
            min_thooft_coupling_threshold: 0.25,
            max_radial_penetration_z: 8.0,
            epsilon: 1e-12,
        }
    }
}

/// Informe omnisciente de geometría holográfica en el bulto AdS y proyección en el borde CFT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdSCftHolographicReport {
    /// Dimensión conforme efectiva del operador primario $\Delta \ge 1.0$.
    pub conformal_dimension_delta: f64,
    /// Acoplamiento de 't Hooft $\lambda = g_{YM}^2 N \ge 0$.
    pub thooft_coupling_lambda: f64,
    /// Tensión de la cuerda cuántica fundamental $T_{\text{string}} = \frac{\sqrt{\lambda}}{2\pi L^2}$.
    pub string_tension: f64,
    /// Profundidad radial de penetración en el bulto $z \in [0.01, 10.0]$ (escala RG).
    pub radial_depth_z: f64,
    /// Modulador continuo de convicción de señal holográfica $\in [0.45, 1.40]$.
    pub holographic_conviction_modulator: f64,
    /// Alerta de pantalla holográfica UV/IR / ruptura de geometría clásica en el bulto.
    pub holographic_rg_screen_alert: bool,
}

impl Default for AdSCftHolographicReport {
    fn default() -> Self {
        Self {
            conformal_dimension_delta: 2.0,
            thooft_coupling_lambda: 1.0,
            string_tension: 0.159,
            radial_depth_z: 1.0,
            holographic_conviction_modulator: 1.0,
            holographic_rg_screen_alert: false,
        }
    }
}

/// Motor Holográfico de Dualidad AdS/CFT en stack $O(1)$.
#[derive(Debug, Clone)]
pub struct AdSCftHolographicEngine {
    params: AdSCftParams,
    prev_radial_z: f64,
    has_prev: bool,
}

impl AdSCftHolographicEngine {
    /// Construye una nueva instancia inicializada en el vacío de Poincaré.
    pub fn new(params: AdSCftParams) -> Self {
        Self {
            params,
            prev_radial_z: 1.0,
            has_prev: false,
        }
    }

    /// Actualiza la proyección holográfica con retornos continuos, volatilidad y persistencia.
    ///
    /// # Argumentos:
    /// - `price_return`: Retorno continuo instantáneo de precio.
    /// - `volatility`: Volatilidad efectiva instantánea $\ge 0$.
    /// - `hurst_exponent`: Exponente continuo de persistencia temporal $H \in (0, 1)$.
    /// - `spread_bps`: Spread medio en puntos básicos $\ge 0$.
    pub fn update(
        &mut self,
        price_return: f64,
        volatility: f64,
        hurst_exponent: f64,
        spread_bps: f64,
    ) -> AdSCftHolographicReport {
        let safe_ret = if price_return.is_finite() { price_return.clamp(-0.5, 0.5) } else { 0.0 };
        let safe_vol = if volatility.is_finite() && volatility >= 0.0 {
            volatility.clamp(0.0001, 1.0)
        } else {
            0.01
        };
        let safe_h = if hurst_exponent.is_finite() {
            hurst_exponent.clamp(0.05, 0.95)
        } else {
            0.50
        };
        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.01, 100.0)
        } else {
            1.0
        };

        let l_ads = self.params.ads_radius_l.clamp(0.1, 10.0);
        let l_sq = l_ads * l_ads;

        // 1. Acoplamiento de 't Hooft \lambda = g_{YM}^2 N
        // En régimen de alta persistencia y liquidez ajustada, el acoplamiento es fuerte (\lambda grande)
        // Spreads amplios o ruido disipativo debilitan el acoplamiento cuántico hacia la zona débil
        let coupling_base = (safe_h * 4.0) / (safe_spread * 0.10 + safe_vol * 10.0 + 0.20);
        let thooft_coupling_lambda = coupling_base.clamp(0.01, 50.0);

        // 2. Tensión de cuerda fundamental T_{\text{string}} = \frac{\sqrt{\lambda}}{2\pi L^2}
        let sqrt_lambda = thooft_coupling_lambda.sqrt();
        let string_tension = (sqrt_lambda / (2.0 * core::f64::consts::PI * l_sq)).clamp(0.001, 10.0);

        // 3. Masa efectiva en el bulto y Dimensión Conforme del Operador Primario \Delta
        // Relación AdS_3: m^2 L^2 = \Delta (\Delta - 2)
        // \Delta = 1 + \sqrt{1 + m^2 L^2}
        let mass_sq_l_sq = ((safe_h - 0.5) * 4.0 + safe_ret.abs() * 10.0).clamp(-0.99, 15.0);
        let conformal_dimension_delta = (1.0 + (1.0 + mass_sq_l_sq).sqrt()).clamp(1.0, 5.0);

        // 4. Coordenada radial z en el bulto de Poincaré (escala de flujo RG):
        // z \propto \sqrt{\nu} / |p| donde la longitud característica crece con el spread
        let z = (safe_spread * 0.20 + safe_vol * 5.0 + 0.10).clamp(0.01, 10.0);

        // Velocidad de penetración radial dz / dt
        let dz = if self.has_prev {
            z - self.prev_radial_z
        } else {
            0.0
        };
        self.prev_radial_z = z;
        self.has_prev = true;

        // 5. Alerta de Pantalla Holográfica UV/IR:
        // Si el acoplamiento de 't Hooft cae por debajo del umbral crítico (régimen no gravitatorio / ruido descorrelacionado)
        // o la profundidad radial penetra más allá del límite del bulto
        let holographic_rg_screen_alert = thooft_coupling_lambda < self.params.min_thooft_coupling_threshold
            || z >= self.params.max_radial_penetration_z
            || dz.abs() > 3.0;

        // 6. Modulador Conforme de Convicción Holográfica:
        let holographic_conviction_modulator = if holographic_rg_screen_alert {
            0.50
        } else {
            // En régimen de 't Hooft fuerte (\lambda >> 1), la supergravedad en el bulto es clásica y altamente predecible.
            let coupling_strength = (thooft_coupling_lambda / 4.0).tanh();
            let delta_distance = (conformal_dimension_delta - self.params.base_conformal_dimension).abs();
            (0.85 + 0.35 * coupling_strength - 0.20 * delta_distance).clamp(0.45, 1.40)
        };

        AdSCftHolographicReport {
            conformal_dimension_delta,
            thooft_coupling_lambda,
            string_tension,
            radial_depth_z: z,
            holographic_conviction_modulator,
            holographic_rg_screen_alert,
        }
    }
}
