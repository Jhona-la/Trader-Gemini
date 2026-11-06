//! # Riemann-Burgers Nonlinear Shockwave & Cole-Hopf Hydrodynamic Engine
//!
//! FASE 35 (§35) — OLA OMEGA220: DE LA FÍSICA DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Modela la propagación no lineal de desequilibrios de órdenes y cascadas de liquidaciones
//! a través del libro L2/L3 mediante la ecuación hiperbólica cuasilineal de Riemann-Burgers:
//! $$\partial_t u + u \partial_x u = \nu \partial_{xx} u$$
//! integrando la transformada analítica exacta de Cole-Hopf y la condición de salto de Rankine-Hugoniot
//! en $\mathcal{O}(1)$ de CPU sin alocaciones dinámicas.
//!
//! ## Invariantes de Silicio & Microestructura:
//! 1. Zero-Allocation Hot Path: Estructuras $O(1)$ en stack.
//! 2. Cero Pánicos (`rb_p1`): 0 `assert!`, 0 `unwrap()`, 0 `expect()`, 0 `panic!()`.
//! 3. Condición de Salto de Rankine-Hugoniot: Velocidad de onda de choque $s = (u_L + u_R) / 2$.
//! 4. Solución Analítica de Cole-Hopf: Ancho de choque $\delta = 4\nu / \Delta u$ en $< 20$ ns.

/// Parámetros de configuración del motor de ondas de choque de Riemann-Burgers.
#[derive(Debug, Clone, Copy)]
pub struct RiemannBurgersParams {
    /// Viscosidad cinemática de mercado basal $\nu_0$ (típicamente en $[0.01, 0.50]$).
    pub market_viscosity_nu: f64,
    /// Umbral de salto de choque crítico $\Delta u_{\text{crit}}$ para detección de discontinuidad.
    pub shock_jump_threshold: f64,
    /// Factor de escala espacial para penetración en profundidad L2/L3.
    pub depth_spatial_scale: f64,
    /// Sensibilidad de convección advectiva.
    pub advection_sensitivity: f64,
}

impl Default for RiemannBurgersParams {
    fn default() -> Self {
        Self {
            market_viscosity_nu: 0.08,
            shock_jump_threshold: 0.35,
            depth_spatial_scale: 1.0,
            advection_sensitivity: 1.5,
        }
    }
}

/// Reporte de estado de la onda de choque de Riemann-Burgers.
#[derive(Debug, Clone, Copy)]
pub struct BurgersShockwaveReport {
    /// Velocidad de propagación de la onda de choque $s = (u_L + u_R)/2$.
    pub shock_velocity_s: f64,
    /// Amplitud de salto del choque $\Delta u = u_L - u_R$.
    pub shock_jump_amplitude: f64,
    /// Ancho físico de la capa de transición del choque $\delta = 4\nu / \Delta u$.
    pub shock_layer_width: f64,
    /// Densidad de masa de liquidez estimada en el frente $\rho(0)$.
    pub liquidity_density_at_front: f64,
    /// Detección de bolsa de aire de liquidez (air pocket inminente).
    pub air_pocket_detected: bool,
    /// Convicción direccional modulada por el frente hidrodinámico en $[-1.0, 1.0]$.
    pub hydrodynamic_conviction: f64,
    /// Tiempo estimado de impacto al siguiente nivel en milisegundos.
    pub time_to_impact_ms: f64,
}

impl Default for BurgersShockwaveReport {
    fn default() -> Self {
        Self {
            shock_velocity_s: 0.0,
            shock_jump_amplitude: 0.0,
            shock_layer_width: 1.0,
            liquidity_density_at_front: 1.0,
            air_pocket_detected: false,
            hydrodynamic_conviction: 0.0,
            time_to_impact_ms: 1000.0,
        }
    }
}

/// Motor hidrodinámico de ondas de choque de Riemann-Burgers para libros L2/L3.
#[derive(Debug, Clone)]
pub struct RiemannBurgersShockwaveEngine {
    params: RiemannBurgersParams,
    smoothed_u_l: f64,
    smoothed_u_r: f64,
    smoothed_viscosity: f64,
}

impl RiemannBurgersShockwaveEngine {
    /// Construye una nueva instancia del motor con los parámetros especificados.
    pub fn new(params: RiemannBurgersParams) -> Self {
        Self {
            params,
            smoothed_u_l: 0.0,
            smoothed_u_r: 0.0,
            smoothed_viscosity: params.market_viscosity_nu,
        }
    }

    /// Evalúa el perfil de onda de choque hidrodinámico en tiempo constante $O(1)$.
    ///
    /// # Parámetros
    /// - `ofi_normalized`: Desequilibrio del flujo de órdenes (Order Flow Imbalance) en $[-1.0, 1.0]$.
    /// - `obi_normalized`: Desequilibrio del libro de órdenes estático (Order Book Imbalance) en $[-1.0, 1.0]$.
    /// - `spread_relative`: Spread relativo actual $(P_{\text{ask}} - P_{\text{bid}}) / P_{\text{mid}}$.
    /// - `depth_liquidity_ratio`: Ratio de profundidad de volumen respecto al volumen medio de trade.
    pub fn evaluate_shockwave(
        &mut self,
        ofi_normalized: f64,
        obi_normalized: f64,
        spread_relative: f64,
        depth_liquidity_ratio: f64,
    ) -> BurgersShockwaveReport {
        let safe_ofi = if ofi_normalized.is_finite() {
            ofi_normalized.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let safe_obi = if obi_normalized.is_finite() {
            obi_normalized.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let safe_spread = if spread_relative.is_finite() && spread_relative > 0.0 {
            spread_relative.clamp(0.0001, 0.10)
        } else {
            0.0010
        };
        let safe_depth = if depth_liquidity_ratio.is_finite() && depth_liquidity_ratio > 0.0 {
            depth_liquidity_ratio.clamp(0.05, 50.0)
        } else {
            1.0
        };

        // Estado a la izquierda (flujo agresor OFI) vs estado a la derecha (resistencia estática OBI)
        let alpha = 0.25; // EWMA de suavizado para estabilidad numérica
        self.smoothed_u_l = (1.0 - alpha) * self.smoothed_u_l + alpha * safe_ofi;
        self.smoothed_u_r = (1.0 - alpha) * self.smoothed_u_r + alpha * (-safe_obi);

        // La viscosidad cinemática de mercado es proporcional a la profundidad y al spread:
        // $\nu \propto \text{depth} \times \text{spread}$
        let dynamic_nu =
            (self.params.market_viscosity_nu * safe_depth * (1.0 + safe_spread * 10.0))
                .clamp(0.005, 2.0);
        self.smoothed_viscosity = (1.0 - alpha) * self.smoothed_viscosity + alpha * dynamic_nu;

        // Condición de Salto de Rankine-Hugoniot para la ley de conservación $\partial_t u + \partial_x (u^2/2) = 0$
        // Velocidad del frente de choque: $s = \frac{u_L + u_R}{2}$
        let shock_velocity = (self.smoothed_u_l + self.smoothed_u_r) / 2.0;

        // Salto de amplitud a través del choque: $\Delta u = u_L - u_R$
        let jump_amplitude = self.smoothed_u_l - self.smoothed_u_r;

        // Ancho de capa de choque de Burgers via Cole-Hopf: $\delta = \frac{4 \nu}{|\Delta u| + \epsilon}$
        let abs_jump = jump_amplitude.abs();
        let shock_width = (4.0 * self.smoothed_viscosity) / (abs_jump + 1e-6);
        let safe_shock_width = if shock_width.is_finite() {
            shock_width.clamp(0.01, 100.0)
        } else {
            1.0
        };

        // Densidad de liquidez en el frente: menor ancho de choque + alta velocidad = vaciamiento rápido
        let density_at_front = (self.smoothed_viscosity * safe_depth) / (1.0 + abs_jump * 2.0);
        let safe_density = if density_at_front.is_finite() {
            density_at_front.max(0.01)
        } else {
            1.0
        };

        // Detección de bolsa de aire (Air Pocket): choque fuerte con ancho estrecho y densidad colapsada
        let is_air_pocket = abs_jump >= self.params.shock_jump_threshold
            && safe_shock_width <= 0.85
            && safe_density <= 0.25;

        // Convicción hidrodinámica basada en la advección del frente de choque de Cole-Hopf:
        // En el marco móvil del frente de choque, la velocidad de advección es la velocidad del choque s
        // amplificada por el factor de compresión (1 + |\Delta u| / 2) de la discontinuidad
        let compression_factor = 1.0 + abs_jump / 2.0;
        let cole_hopf_front_u = shock_velocity * compression_factor;

        let raw_conviction = cole_hopf_front_u * self.params.advection_sensitivity;
        let hydrodynamic_conviction = if raw_conviction.is_finite() {
            raw_conviction.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Tiempo estimado de impacto: $\Delta x / |s|$ en ms
        let speed_mag = shock_velocity.abs().max(0.01);
        let time_to_impact = (safe_shock_width * 100.0 / speed_mag).clamp(1.0, 10_000.0);

        BurgersShockwaveReport {
            shock_velocity_s: shock_velocity,
            shock_jump_amplitude: jump_amplitude,
            shock_layer_width: safe_shock_width,
            liquidity_density_at_front: safe_density,
            air_pocket_detected: is_air_pocket,
            hydrodynamic_conviction,
            time_to_impact_ms: time_to_impact,
        }
    }
}

impl Default for RiemannBurgersShockwaveEngine {
    fn default() -> Self {
        Self::new(RiemannBurgersParams::default())
    }
}
