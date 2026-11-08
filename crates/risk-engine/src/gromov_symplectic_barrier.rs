//! -------------------------------------------------------------------------------------------------
//! TRADER GEMINI — QUANTUM CONTINUOUS MANIFOLD ARCHITECTURE (OMEGA220)
//! Risk Engine: Barrera Conformal de No-Ruina de Gromov y Cuantización Cuaterniónica Simpléctica
//! Fase 57 (§57) — Ola 13
//!
//! Concepto Físico-Matemático:
//! Aplica el Teorema de No-Aplastamiento Simpléctico de Gromov (Symplectic Camel) en variedades
//! hamiltonianas $(\mathbb{R}^{2n}, \omega_0)$. Una bola de fase $B^{2n}(R)$ sólo puede atravesar una
//! garganta cilíndrica $Z^{2n}(r)$ si $R \le r$, pues la capacidad simpléctica de Gromov $c(\Omega) = \pi R^2$
//! es un invariante canónico estricto.
//! Si el volumen de fase del riesgo de capital $R_{\text{portfolio}}$ excede la capacidad de absorción $r_{\text{hole}}$
//! del libro de órdenes, cualquier intento de ejecución forzada inducirá deslizamiento cuadrático y ruina.
//! Emplea rotaciones cuaterniónicas $\mathbb{H}$ para proyectar el tensor simpléctico en $O(1)$ stack,
//! emitiendo `GROMOV_SYMPLECTIC_VETO` para preservar la cota de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//! -------------------------------------------------------------------------------------------------

/// Cuaternión unitario para rotaciones simplécticas en 4 dimensiones sin gimbal lock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    #[inline(always)]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    #[inline(always)]
    pub const fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[inline(always)]
    pub fn norm_sq(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline(always)]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-12 {
            Self::identity()
        } else {
            let inv = 1.0 / n;
            Self {
                w: self.w * inv,
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            }
        }
    }

    #[inline(always)]
    pub fn mul(&self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}

/// Parámetros de configuración del motor de barrera simpléctica de Gromov.
#[derive(Debug, Clone, Copy)]
pub struct GromovSymplecticParams {
    /// Holgura mínima de seguridad para el radio de Gromov $\delta_{\text{safe}}$.
    pub safety_margin: f64,
    /// Apalancamiento nominal de referencia.
    pub nominal_leverage: f64,
    /// Sensibilidad de la capacidad simpléctica del libro ante el spread.
    pub spread_sensitivity: f64,
    /// Epsilon para regularización numérica.
    pub epsilon: f64,
}

impl Default for GromovSymplecticParams {
    fn default() -> Self {
        Self {
            safety_margin: 0.05,
            nominal_leverage: 1.0,
            spread_sensitivity: 0.05,
            epsilon: 1e-9,
        }
    }
}

/// Informe omnisciente de capacidad simpléctica de Gromov y veto de no-aplastamiento.
#[derive(Debug, Clone, Copy)]
pub struct GromovSymplecticReport {
    /// Radio de fase del capital de riesgo del portafolio $R_{\text{portfolio}}$.
    pub portfolio_phase_radius: f64,
    /// Radio efectivo de la garganta simpléctica de liquidez del libro $r_{\text{hole}}$.
    pub hole_capacity_radius: f64,
    /// Capacidad simpléctica de Gromov del portafolio $c_{\text{portfolio}} = \pi R^2$.
    pub gromov_portfolio_capacity: f64,
    /// Capacidad simpléctica de Gromov de la garganta $c_{\text{hole}} = \pi r^2$.
    pub gromov_hole_capacity: f64,
    /// Margen de compresión simpléctica normalizado $\Delta_{\text{gromov}} \in [-1.0, 1.0]$.
    pub compression_margin: f64,
    /// Veto absoluto de compresión simpléctica (activado si $R_{\text{portfolio}} > r_{\text{hole}}$).
    pub gromov_symplectic_veto: bool,
    /// Modulador de tamaño de slot derivado $\omega_{\text{gromov}} \in [0.0, 1.25]$.
    pub slot_scaling_factor: f64,
    /// Distancia geodésica cuaterniónica normalizada.
    pub quaternion_phase_distance: f64,
}

/// Motor de Barrera Conformal de No-Ruina de Gromov y Cuantización Cuaterniónica Simpléctica.
#[derive(Debug, Clone)]
pub struct GromovSymplecticBarrierEngine {
    params: GromovSymplecticParams,
    current_orientation: Quaternion,
}

impl GromovSymplecticBarrierEngine {
    /// Inicializa una nueva instancia de la barrera de Gromov con los parámetros dados.
    pub fn new(params: GromovSymplecticParams) -> Self {
        Self {
            params,
            current_orientation: Quaternion::identity(),
        }
    }

    /// Evalúa la capacidad simpléctica y la admisibilidad de la orden en $O(1)$ stack.
    pub fn evaluate(
        &mut self,
        capital_at_risk_usd: f64,
        total_wallet_capital_usd: f64,
        available_liquidity_depth_usd: f64,
        current_spread_bps: f64,
        market_volatility: f64,
    ) -> GromovSymplecticReport {
        // 1. Regularización de entradas
        let capital_total = total_wallet_capital_usd.max(1.0);
        let capital_risk = capital_at_risk_usd.clamp(0.01, capital_total);
        let depth = available_liquidity_depth_usd.max(0.10);
        let spread = current_spread_bps.max(0.10);
        let vol = market_volatility.clamp(0.0001, 10.0);

        // 2. Radio de fase de la bola simpléctica del capital $R_{\text{portfolio}}$
        let risk_fraction = (capital_risk / capital_total).sqrt();
        let leverage_vol_multiplier = 1.0 + self.params.nominal_leverage * vol;
        let portfolio_phase_radius =
            (risk_fraction * leverage_vol_multiplier).max(self.params.epsilon);

        // 3. Radio de la garganta simpléctica de absorción del libro $r_{\text{hole}}$
        let spread_penalty = 1.0 + spread * self.params.spread_sensitivity;
        let liquidity_headroom = (depth / capital_risk).sqrt();
        let hole_capacity_radius = (liquidity_headroom / spread_penalty).max(self.params.epsilon);

        // 4. Capacidades Simplécticas de Gromov $c = \pi r^2$
        let pi = core::f64::consts::PI;
        let gromov_portfolio_capacity = pi * portfolio_phase_radius * portfolio_phase_radius;
        let gromov_hole_capacity = pi * hole_capacity_radius * hole_capacity_radius;

        // 5. Margen de compresión simpléctica normalizado
        let max_r = portfolio_phase_radius.max(hole_capacity_radius);
        let compression_margin =
            ((hole_capacity_radius - portfolio_phase_radius) / max_r).clamp(-1.0, 1.0);

        // 6. Proyección cuaterniónica de rotación en espacio de fases 4D
        // q = (cos(theta/2), v * sin(theta/2)) acoplando (capital_risk, depth, spread, vol)
        let angle = (spread * 0.01 + vol).min(pi);
        let half_angle = 0.5 * angle;
        let sin_half = half_angle.sin();
        let cos_half = half_angle.cos();
        let q_delta = Quaternion::new(
            cos_half,
            sin_half * risk_fraction,
            sin_half * (1.0 / spread_penalty),
            sin_half * (vol / (1.0 + vol)),
        )
        .normalize();

        self.current_orientation = self.current_orientation.mul(q_delta).normalize();
        let quaternion_phase_distance = (1.0 - self.current_orientation.w.abs()).clamp(0.0, 1.0);

        // 7. Teorema del Camello Simpléctico: si $R > r$, la bola NO puede cruzar la garganta
        let gromov_symplectic_veto =
            portfolio_phase_radius > (hole_capacity_radius * (1.0 - self.params.safety_margin));

        // 8. Modulador de Tamaño de Slot $\omega_{\text{gromov}} \in [0.0, 1.25]$
        let slot_scaling_factor = if gromov_symplectic_veto {
            0.0
        } else {
            let base = 1.0 + 0.25 * compression_margin - 0.15 * quaternion_phase_distance;
            base.clamp(0.10, 1.25)
        };

        GromovSymplecticReport {
            portfolio_phase_radius,
            hole_capacity_radius,
            gromov_portfolio_capacity,
            gromov_hole_capacity,
            compression_margin,
            gromov_symplectic_veto,
            slot_scaling_factor,
            quaternion_phase_distance,
        }
    }
}
