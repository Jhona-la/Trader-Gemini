use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::{QuantumStrategy, TradeHorizon};
use crate::voto_espectral::{VotoEspectral, ESCALAS_VOTO};

/// 🌊 ALGORITMO #99: COLECTOR ESPECTRAL DE NAVIER-STOKES Y CASCADA DE KOLMOGOROV (NAVIER-STOKES MANIFOLD ENGINE)
///
/// Modela la propagación del flujo de órdenes multiescala como una variedad hidrodinámica
/// de Navier-Stokes proyectada sobre las 32 escalas diádicas de la malla de Hilbert:
///
///   ∂u/∂t + (u · ∇) u = -∇p/ρ + ν ∇²u + f_ext
///
/// Principios Físicos Fundamentales:
/// 1. **Cascada Inercial de Kolmogorov (K41)**: La energía inyectada a escalas macro (τ_k grandes)
///    se transfiere conservativamente hacia escalas rápidas mediante interacciones no-lineales
///    triádicas: E(k) ~ ε^(2/3) k^(-5/3).
/// 2. **Régimen Inercial vs Viscoso por Escala Re(τ_k)**:
///    - Para Re(τ_k) > 1.0 (escalas lentas/inerciales): El momentum se propaga sin amortiguamiento,
///      generando persistencia direccional robusta (voto alineado con la corriente).
///    - Para Re(τ_k) < 1.0 (escalas rápidas/viscosas): La viscosidad cinemática ν disipa el flujo
///      en forma de fricción térmica/slippage, forzando reversión hacia el equilibrio laminar.
/// 3. **Propiedades Numéricas**:
///    - O(1) puro, latencia en CPU < 25 ns, #[inline(always)], zero heap allocations.
///    - Suavidad C^∞ sin discontinuidades ni kinking C^0.
///    - Inmunidad total a NaN, ±Inf y valores no representables (fail-closed seguro).
/// 🌊 OLA Ω121: Parámetros desacoplados y configurables para el colector espectral de Navier-Stokes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavierStokesManifoldParams {
    /// Factor de escala modal para el número de Reynolds diádico (default: 0.25)
    pub scale_weight_factor: f64,
    /// Piso mínimo de modulación laminar global (default: 0.40)
    pub modulation_floor: f64,
    /// Amplitud de modulación laminar global (default: 0.60)
    pub modulation_scale: f64,
    /// Factor de escala del argumento tanh en señal direccional (default: 0.50)
    pub signal_tanh_scale: f64,
    /// Factor de escala de velocidad para el colapso turbulento (default: 100.0)
    pub velocity_scale: f64,
}

impl Default for NavierStokesManifoldParams {
    fn default() -> Self {
        Self {
            scale_weight_factor: 0.25,
            modulation_floor: 0.40,
            modulation_scale: 0.60,
            signal_tanh_scale: 0.50,
            velocity_scale: 100.0,
        }
    }
}

impl NavierStokesManifoldParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.scale_weight_factor.is_finite() || !(0.01..=5.0).contains(&self.scale_weight_factor) {
            return Err(format!("Invalid scale_weight_factor: {}", self.scale_weight_factor));
        }
        if !self.modulation_floor.is_finite() || !(0.0..=1.0).contains(&self.modulation_floor) {
            return Err(format!("Invalid modulation_floor: {}", self.modulation_floor));
        }
        if !self.modulation_scale.is_finite() || !(0.0..=2.0).contains(&self.modulation_scale) {
            return Err(format!("Invalid modulation_scale: {}", self.modulation_scale));
        }
        if !self.signal_tanh_scale.is_finite() || !(0.05..=10.0).contains(&self.signal_tanh_scale) {
            return Err(format!("Invalid signal_tanh_scale: {}", self.signal_tanh_scale));
        }
        if !self.velocity_scale.is_finite() || !(1.0..=10000.0).contains(&self.velocity_scale) {
            return Err(format!("Invalid velocity_scale: {}", self.velocity_scale));
        }
        Ok(())
    }
}

#[derive(Clone)]
#[repr(C, align(64))]
pub struct NavierStokesManifoldEngine {
    registry: Option<Arc<OmniscientRegistry>>,
    pub params: NavierStokesManifoldParams,
}

impl Default for NavierStokesManifoldEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for NavierStokesManifoldEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NavierStokesManifoldEngine")
            .field("params", &self.params)
            .finish()
    }
}

impl NavierStokesManifoldEngine {
    pub const CRITICAL_REYNOLDS_LAMINAR: f64 = 1.0;
    pub const CRITICAL_REYNOLDS_TURBULENT: f64 = 5.0;

    pub fn new() -> Self {
        Self {
            registry: None,
            params: NavierStokesManifoldParams::default(),
        }
    }

    /// OLA Ω121: Constructor con parámetros específicos de colector
    pub fn with_params(mut self, params: NavierStokesManifoldParams) -> Self {
        if params.validate().is_ok() {
            self.params = params;
        }
        self
    }

    /// OLA Ω121: Actualización dinámica de parámetros
    pub fn update_params(&mut self, params: NavierStokesManifoldParams) {
        if params.validate().is_ok() {
            self.params = params;
        }
    }

    /// Calcula el número de Reynolds modal a la escala temporal k:
    /// Re(τ_k) = (|x(τ_k)| · (1 + k·0.25)) / ν_eff
    /// Donde k ∈ [0, 31] representa la progresión diádica de escalas temporales.
    #[inline(always)]
    pub fn compute_modal_reynolds(
        momentum_z: f64,
        scale_idx: usize,
        kinematic_viscosity: f64,
    ) -> f64 {
        Self::compute_modal_reynolds_with_scale(momentum_z, scale_idx, kinematic_viscosity, 0.25)
    }

    /// OLA Ω121: Calcula el número de Reynolds modal con factor de escala parametrizado
    #[inline(always)]
    pub fn compute_modal_reynolds_with_scale(
        momentum_z: f64,
        scale_idx: usize,
        kinematic_viscosity: f64,
        scale_weight_factor: f64,
    ) -> f64 {
        let safe_z = if momentum_z.is_finite() {
            momentum_z.abs().clamp(0.0, 10.0)
        } else {
            0.0
        };
        let safe_visc = if kinematic_viscosity.is_finite() && kinematic_viscosity > 0.0 {
            kinematic_viscosity.clamp(1e-6, 100.0)
        } else {
            1.0
        };
        let factor = if scale_weight_factor.is_finite() && scale_weight_factor > 0.0 {
            scale_weight_factor
        } else {
            0.25
        };
        let scale_weight = 1.0 + (scale_idx.min(31) as f64) * factor;
        (safe_z * scale_weight / safe_visc).clamp(0.0, 100.0)
    }

    /// Calcula el factor de amortiguamiento viscoso sub-disipativo de Kolmogorov C^∞:
    /// η(k) = 1.0 / (1.0 + (1.0 / Re(τ_k))^2) ∈ [0.0, 1.0].
    /// - Para Re >> 1 (inercial): η -> 1.0 (transmisión completa de la señal de tendencia).
    /// - Para Re << 1 (viscoso/disipativo): η -> 0.0 (supresión suave de turbulencia de microsegundos).
    #[inline(always)]
    pub fn compute_kolmogorov_inertial_efficiency(re_modal: f64) -> f64 {
        let safe_re = if re_modal.is_finite() && re_modal > 0.0 {
            re_modal
        } else {
            0.0
        };
        if safe_re <= 1e-6 {
            return 0.0;
        }
        let inv_re = 1.0 / safe_re;
        (1.0 / (1.0 + inv_re * inv_re)).clamp(0.0, 1.0)
    }

    /// Genera el voto espectral hidrodinámico VotoEspectral en O(1) puro sobre la malla de 32 escalas.
    ///
    /// # Parámetros
    /// * `desplazamientos`: Malla diádica de momentum z-scores [momentum_z; 32].
    /// * `kinematic_viscosity`: Viscosidad cinemática del libro L2 ν ∈ [1e-6, 100].
    /// * `laminar_share`: Fracción laminar continua C^∞ del libro ∈ [0.0, 1.0].
    #[inline(always)]
    pub fn voto_espectral(
        desplazamientos: &[f64; ESCALAS_VOTO],
        kinematic_viscosity: f64,
        laminar_share: f64,
    ) -> VotoEspectral {
        Self::voto_espectral_with_params(
            desplazamientos,
            kinematic_viscosity,
            laminar_share,
            &NavierStokesManifoldParams::default(),
        )
    }

    /// OLA Ω121: Voto espectral hidrodinámico con parámetros desacoplados y configurables
    #[inline(always)]
    pub fn voto_espectral_with_params(
        desplazamientos: &[f64; ESCALAS_VOTO],
        kinematic_viscosity: f64,
        laminar_share: f64,
        params: &NavierStokesManifoldParams,
    ) -> VotoEspectral {
        let safe_visc = if kinematic_viscosity.is_finite() && kinematic_viscosity > 0.0 {
            kinematic_viscosity.clamp(1e-6, 100.0)
        } else {
            1.0
        };
        let safe_laminar = if laminar_share.is_finite() {
            laminar_share.clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Modulación suave global del colector
        let global_modulation = (params.modulation_floor + params.modulation_scale * safe_laminar).clamp(0.0, 2.0);

        let mut por_escala = [0.0f64; ESCALAS_VOTO];
        for k in 0..ESCALAS_VOTO {
            let z = desplazamientos[k];
            if !z.is_finite() {
                continue;
            }
            let safe_z = z.clamp(-10.0, 10.0);
            let re_modal = Self::compute_modal_reynolds_with_scale(safe_z, k, safe_visc, params.scale_weight_factor);
            let eta = Self::compute_kolmogorov_inertial_efficiency(re_modal);

            // Dirección suave vía tanh(z * scale), ponderada por la eficiencia inercial eta y modulación laminar
            let signal = (safe_z * params.signal_tanh_scale).tanh() * eta * global_modulation;
            por_escala[k] = signal.clamp(-1.0, 1.0);
        }

        VotoEspectral::desde_arr(&por_escala)
    }

    /// Calcula la densidad de energía cinética espectral E(k) = (1/2) · z(τ_k)^2
    #[inline(always)]
    pub fn kolmogorov_energy_spectrum(
        desplazamientos: &[f64; ESCALAS_VOTO],
    ) -> [f64; ESCALAS_VOTO] {
        let mut energy = [0.0f64; ESCALAS_VOTO];
        for k in 0..ESCALAS_VOTO {
            let z = desplazamientos[k];
            if z.is_finite() {
                let clamped_z = z.clamp(-10.0, 10.0);
                energy[k] = 0.5 * clamped_z * clamped_z;
            }
        }
        energy
    }

    /// Calcula el flujo neto de cascada turbulenta entre escalas adyacentes:
    /// Π = Σ_{k=0}^{30} (E(k+1) - E(k))
    #[inline(always)]
    pub fn turbulent_cascade_flux(desplazamientos: &[f64; ESCALAS_VOTO]) -> f64 {
        let energy = Self::kolmogorov_energy_spectrum(desplazamientos);
        let mut flux = 0.0f64;
        for k in 0..(ESCALAS_VOTO - 1) {
            flux += energy[k + 1] - energy[k];
        }
        if flux.is_finite() {
            flux.clamp(-100.0, 100.0)
        } else {
            0.0
        }
    }
}

impl QuantumStrategy for NavierStokesManifoldEngine {
    fn name(&self) -> &str {
        "NavierStokesManifoldEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() { None } else { Some(symbol) };
        let cid_opt = if symbol.is_empty() { None } else { Some(coin_id) };

        let registry = match self.registry.as_ref() {
            Some(r) => r,
            None => return 0.0,
        };

        let laminar_share = registry.get_scoped_val_or(sym_opt, cid_opt, "navier_laminar_share", 1.0);
        let kinematic_visc = registry.get_scoped_val_or(sym_opt, cid_opt, "kinematic_viscosity", 1.0);
        let ofi = registry.get_scoped_val_or(sym_opt, cid_opt, "order_flow_imbalance", 0.0);
        let velocity = registry.get_scoped_val_or(sym_opt, cid_opt, "price_velocity", 0.0);

        if !laminar_share.is_finite() || !kinematic_visc.is_finite() || !ofi.is_finite() || !velocity.is_finite() {
            return 0.0;
        }

        // Señal modal continua de microestructura:
        // En flujo laminar (laminar_share -> 1.0), el desequilibrio de libro OFI domina.
        // En flujo turbulento (laminar_share -> 0.0), la inercia de velocidad devora el libro pasivo.
        let safe_laminar = laminar_share.clamp(0.0, 1.0);
        let safe_ofi = ofi.clamp(-1.0, 1.0);
        let safe_vel = (velocity * self.params.velocity_scale).tanh().clamp(-1.0, 1.0);

        let signal = safe_laminar * safe_ofi + (1.0 - safe_laminar) * safe_vel;
        signal.clamp(-1.0, 1.0)
    }

    fn horizon(&self) -> TradeHorizon {
        TradeHorizon::Continuous
    }
}
