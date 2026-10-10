//! Hidrodinámica Estocástica del Order Book — Ecuaciones de Navier-Stokes y Número de Reynolds
//!
//! Axioma II: O(1) puro, #[inline(always)], zero heap allocations, latencia < 25 ns en CPU.
//!
//! Modela la microestructura del libro de órdenes como un fluido continuo viscoso compresible
//! gobernado por la ecuación de Navier-Stokes estocástica:
//!
//!   ρ (∂u/∂t + u · ∇u) = -∇p + μ ∇²u + f_ext
//!
//! donde:
//!   - ρ: Densidad de masa del libro (profundidad nocional por unidad de precio).
//!   - u: Velocidad de flujo del precio v_t = dP/dt.
//!   - p: Campo de presión de liquidez (desequilibrio BBO y spread bid-ask).
//!   - μ: Viscosidad dinámica del mercado (resistencia disipativa de órdenes límite pasivas).
//!   - ν = μ / ρ: Viscosidad cinemática.
//!   - L: Longitud de escala característica (el spread bid-ask S = P_ask - P_bid).
//!   - f_ext: Fuerza externa impulsada por transacciones agresivas (CVD / flujo taker).
//!
//! El Número de Reynolds Financiero continuo Re cuantifica la relación entre las fuerzas
//! inerciales del flujo agresor y las fuerzas viscosas disipativas del libro:
//!
//!   Re = (Fuerzas Inerciales) / (Fuerzas Viscosas)
//!
//! Clasificación hidrodinámica de régimen:
//!   - Re < 1.0: Régimen Laminar (amortiguamiento viscoso dominante, flujo predecible, cotización Maker óptima).
//!   - 1.0 <= Re < 5.0: Régimen Transicional (inestabilidades de Tollmien-Schlichting, ondas convectivas).
//!   - Re >= 5.0: Régimen Turbulento (cascada inercial de Kolmogorov, vórtices de liquidación, toxicidad taker alta).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HydrodynamicRegime {
    /// Flujo laminar estable: viscosidad dominante, microestructura tranquila
    Laminar,
    /// Flujo transicional: inicio de vórtices y desequilibrio dinámico
    Transitional,
    /// Flujo turbulento: inercia agresiva dominante, cascada de órdenes y alto slippage
    Turbulent,
}

#[derive(Debug, Clone)]
pub struct NavierStokesReynoldsEngine {
    pub prev_mid_price: f64,
    pub prev_event_time_ms: u64,
    pub velocity: f64,
    pub acceleration: f64,
    pub kinematic_viscosity: f64,
    pub reynolds_number: f64,
    pub ewma_reynolds: f64,
    pub energy_dissipation_rate: f64,
    pub laminar_share: f64,
    pub sample_count: u64,
    /// Tiempo de relajación característico del libro de órdenes τ_relax en segundos
    pub relaxation_time_s: f64,
}

impl Default for NavierStokesReynoldsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl NavierStokesReynoldsEngine {
    pub const CRITICAL_REYNOLDS_LAMINAR: f64 = 1.0;
    pub const CRITICAL_REYNOLDS_TURBULENT: f64 = 5.0;
    /// Tiempo de relajación característico estándar de reposición del libro: 200 ms
    pub const DEFAULT_RELAXATION_TIME_S: f64 = 0.200;
    const EWMA_ALPHA: f64 = 0.05;

    pub fn new() -> Self {
        Self {
            prev_mid_price: 0.0,
            prev_event_time_ms: 0,
            velocity: 0.0,
            acceleration: 0.0,
            kinematic_viscosity: 1.0,
            reynolds_number: 0.0,
            ewma_reynolds: 0.0,
            energy_dissipation_rate: 0.0,
            laminar_share: 1.0,
            sample_count: 0,
            relaxation_time_s: Self::DEFAULT_RELAXATION_TIME_S,
        }
    }

    /// Configura un tiempo de relajación característico específico en segundos
    pub fn with_relaxation_time_s(mut self, tau_s: f64) -> Self {
        if tau_s.is_finite() && tau_s > 0.0 {
            self.relaxation_time_s = tau_s.clamp(0.010, 5.0);
        }
        self
    }

    /// Calcula la fracción laminar suave C^∞ a partir del número de Reynolds continuo:
    /// laminar_share = 1.0 / (1.0 + (Re / Re_crit)^2) ∈ [0.0, 1.0].
    #[inline(always)]
    pub fn compute_laminar_share(re: f64) -> f64 {
        let safe_re = if re.is_finite() && re >= 0.0 { re } else { 0.0 };
        let re_ratio = safe_re / Self::CRITICAL_REYNOLDS_LAMINAR;
        (1.0 / (1.0 + re_ratio * re_ratio)).clamp(0.0, 1.0)
    }

    /// Actualiza el estado hidrodinámico en tiempo real ante cada evento de mercado.
    ///
    /// # Argumentos
    /// * `bid_price`: Mejor oferta de compra (BBO bid).
    /// * `ask_price`: Mejor oferta de venta (BBO ask).
    /// * `bid_qty`: Cantidad disponible al bid.
    /// * `ask_qty`: Cantidad disponible al ask.
    /// * `trade_qty`: Volumen ejecutado en el evento actual (0.0 si es evento de profundidad).
    /// * `atr_pct`: Rango verdadero medio relativo (fracción de precio).
    /// * `event_time_ms`: Marca temporal del exchange en milisegundos.
    #[inline(always)]
    pub fn update(
        &mut self,
        bid_price: f64,
        ask_price: f64,
        bid_qty: f64,
        ask_qty: f64,
        trade_qty: f64,
        atr_pct: f64,
        event_time_ms: u64,
    ) -> f64 {
        if !bid_price.is_finite()
            || !ask_price.is_finite()
            || bid_price <= 0.0
            || ask_price <= bid_price
        {
            return self.reynolds_number;
        }

        let mid_price = (bid_price + ask_price) * 0.5;
        let spread = ask_price - bid_price;

        // 1. Escala temporal física causal (dt en segundos)
        let dt = if self.prev_event_time_ms > 0 && event_time_ms > self.prev_event_time_ms {
            ((event_time_ms - self.prev_event_time_ms) as f64 * 0.001).clamp(0.0005, 5.0)
        } else {
            0.050 // Inicialización o eventos intra-ms: 50 ms por defecto
        };

        // 2. Campo de velocidad y aceleración de precio [USD / s] y [USD / s^2]
        let current_velocity = if self.prev_mid_price > 0.0 {
            (mid_price - self.prev_mid_price) / dt
        } else {
            0.0
        };
        let current_accel = (current_velocity - self.velocity) / dt;
        self.velocity = current_velocity;
        self.acceleration = current_accel.clamp(-1e6, 1e6);

        // 3. Profundidad pasiva BBO en términos de valor nocional (amortiguamiento viscoso)
        let valid_bid_qty = if bid_qty.is_finite() && bid_qty > 0.0 { bid_qty } else { 1.0 };
        let valid_ask_qty = if ask_qty.is_finite() && ask_qty > 0.0 { ask_qty } else { 1.0 };
        let passive_depth_usd = (valid_bid_qty * bid_price + valid_ask_qty * ask_price).max(10.0);

        // 4. Inercia del volumen agresor y multiplicador adimensional de agresión
        let aggressive_trade_usd = if trade_qty.is_finite() && trade_qty > 0.0 {
            trade_qty * mid_price
        } else {
            0.0
        };
        let aggression_multiplier = 1.0 + (aggressive_trade_usd / passive_depth_usd).clamp(0.0, 20.0);

        // 5. Escala característica de longitud L y regularización de capa límite [USD]
        let safe_atr = if atr_pct.is_finite() && atr_pct > 0.0 {
            atr_pct.clamp(0.0002, 0.10)
        } else {
            0.0010
        };
        let sigma_min = (safe_atr * 0.05).max(1e-6);
        let characteristic_length = spread.max(mid_price * sigma_min);

        // 6. Tiempo de relajación característico tau_relax [s]
        let tau_relax = if self.relaxation_time_s.is_finite() && self.relaxation_time_s > 0.0 {
            self.relaxation_time_s
        } else {
            Self::DEFAULT_RELAXATION_TIME_S
        };

        // 7. Viscosidad cinemática del libro nu = L^2 / (tau_relax * agresión) en [USD^2 / s]
        // Incorpora el efecto físico de adelgazamiento por corte (shear-thinning):
        // el flujo agresor reduce la viscosidad efectiva disipativa del libro de órdenes.
        let kinematic_visc = (characteristic_length * characteristic_length)
            / (tau_relax * aggression_multiplier);
        self.kinematic_viscosity = kinematic_visc.max(1e-12);

        // 8. Número de Reynolds continuo estrictamente adimensional:
        // Re = (|u| * tau_relax * agresión) / L_eff == (|u| * L_eff) / nu
        // Unidades: ([USD/s] * [s] * [1]) / [USD] = [USD] / [USD] = 1 (Adimensional puro)
        let raw_reynolds = ((self.velocity.abs() * tau_relax * aggression_multiplier)
            / characteristic_length)
            .clamp(0.0, 100.0);
        self.reynolds_number = raw_reynolds;

        // 9. EWMA continua del número de Reynolds
        if self.sample_count == 0 {
            self.ewma_reynolds = raw_reynolds;
        } else {
            self.ewma_reynolds =
                self.ewma_reynolds * (1.0 - Self::EWMA_ALPHA) + raw_reynolds * Self::EWMA_ALPHA;
        }
        self.sample_count = self.sample_count.saturating_add(1);

        // 10. Fracción laminar suave C^∞: 1 / (1 + (Re/Re_crit)^2)
        // C-02 RESOLUCIÓN: Calculada sobre la EWMA continua ewma_reynolds en lugar
        // del valor instantáneo de un solo tick, erradicando falsos colapsos de confianza
        // y saltos artificiales de slippage provocados por ruido de microestructura aislado.
        self.laminar_share = Self::compute_laminar_share(self.ewma_reynolds);

        // 11. Tasa de disipación de energía de Kolmogorov: ε = ν * (u / L)^2 en [USD^2 / s^3]
        let velocity_gradient = (self.velocity / characteristic_length).abs();
        let dissipation = self.kinematic_viscosity * velocity_gradient * velocity_gradient;
        self.energy_dissipation_rate = dissipation.clamp(0.0, 1000.0);

        self.prev_mid_price = mid_price;
        self.prev_event_time_ms = event_time_ms;

        self.reynolds_number
    }

    /// Retorna el régimen hidrodinámico clasificado según la EWMA del número de Reynolds (C-02)
    #[inline(always)]
    pub fn regime(&self) -> HydrodynamicRegime {
        if self.ewma_reynolds < Self::CRITICAL_REYNOLDS_LAMINAR {
            HydrodynamicRegime::Laminar
        } else if self.ewma_reynolds < Self::CRITICAL_REYNOLDS_TURBULENT {
            HydrodynamicRegime::Transitional
        } else {
            HydrodynamicRegime::Turbulent
        }
    }

    /// Retorna si el régimen actual es estrictamente laminar
    #[inline(always)]
    pub fn is_laminar(&self) -> bool {
        self.ewma_reynolds < Self::CRITICAL_REYNOLDS_LAMINAR
    }

    /// Retorna si el régimen actual es turbulento (riesgo inercial extremo)
    #[inline(always)]
    pub fn is_turbulent(&self) -> bool {
        self.ewma_reynolds >= Self::CRITICAL_REYNOLDS_TURBULENT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reynolds_laminar_en_mercado_tranquilo() {
        let mut engine = NavierStokesReynoldsEngine::new();
        let mut re = 0.0;
        let base_p = 50_000.0;
        for i in 0..20 {
            // Precio oscila muy levemente (0.01 USD), spread saludable (1.0 USD), sin trades agresivos
            let p = base_p + (i % 2) as f64 * 0.01;
            re = engine.update(
                p,
                p + 1.0,
                10.0,
                10.0,
                0.0,
                0.0010,
                1_000_000 + i * 100,
            );
        }
        assert!(re < NavierStokesReynoldsEngine::CRITICAL_REYNOLDS_LAMINAR, "Re debe ser laminar: {re}");
        assert_eq!(engine.regime(), HydrodynamicRegime::Laminar);
        assert!(engine.is_laminar());
        assert!(engine.laminar_share > 0.50);
    }

    #[test]
    fn test_reynolds_turbulento_en_salto_violento_con_flujo_agresor() {
        let mut engine = NavierStokesReynoldsEngine::new();
        // Calentar
        engine.update(50_000.0, 50_001.0, 5.0, 5.0, 0.0, 0.0010, 1_000_000);

        // Salto masivo de precio en 10ms (150 USD) con 100x volumen agresor
        let re = engine.update(
            50_150.0,
            50_151.0,
            1.0,
            1.0,
            200.0,
            0.0010,
            1_000_010,
        );

        assert!(re >= NavierStokesReynoldsEngine::CRITICAL_REYNOLDS_TURBULENT, "Re debe ser turbulento: {re}");
        assert_eq!(engine.regime(), HydrodynamicRegime::Turbulent);
        assert!(engine.is_turbulent());
        assert!(engine.laminar_share < 0.10);
    }

    #[test]
    fn test_inmunidad_a_nan_y_precios_no_finitos() {
        let mut engine = NavierStokesReynoldsEngine::new();
        let re_prev = engine.update(50_000.0, 50_001.0, 5.0, 5.0, 0.0, 0.0010, 1_000_000);
        let re_nan = engine.update(f64::NAN, 50_001.0, 5.0, 5.0, 0.0, 0.0010, 1_000_100);
        assert_eq!(re_prev, re_nan);
        assert!(engine.reynolds_number.is_finite());
        assert!(engine.laminar_share.is_finite());
        assert!(engine.energy_dissipation_rate.is_finite());
    }

    #[test]
    fn test_monotonicidad_de_laminar_share() {
        let l0 = NavierStokesReynoldsEngine::compute_laminar_share(0.0);
        assert_eq!(l0, 1.0);

        let l1 = NavierStokesReynoldsEngine::compute_laminar_share(1.0);
        assert_eq!(l1, 0.50);

        let l5 = NavierStokesReynoldsEngine::compute_laminar_share(5.0);
        assert!(l5 < 0.05);
        assert!(l0 > l1 && l1 > l5);
    }
}
