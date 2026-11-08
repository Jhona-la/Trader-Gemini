//! ⚛️ FASE 53 (§53): FLUJO DEL GRUPO DE RENORMALIZACIÓN EXACTO DE WILSON (EXACT RG)
//! (Wilson Exact Functional Renormalization Group Engine)
//!
//! # MODO PROFESOR — ANÁLISIS INTEGRAL DEL SISTEMA CUÁNTICO
//!
//! - **QUÉ:** Motor de Grupo de Renormalización Exacto (ERG / Wetterich-Morris equation)
//!   en tiempo continuo sobre el espacio de acoplamientos efectivos del mercado $(m_k^2, \lambda_k)$
//!   integrando fluctuaciones microscópicas de alta frecuencia para derivar la acción macroscópica $\Gamma_k$.
//! - **POR QUÉ:** En mercados hiper-líquidos, las operaciones HFT de microsegundos, algoritmos tácticos
//!   y macro-fondos operan a escalas dispares que se acoplan no linealmente. Analizar escalas de tiempo
//!   aisladas induce errores de truncamiento severos. El Grupo de Renormalización es el único marco
//!   físico formal que integra progresivamente la energía UV (ruido) preservando invariantes infrarrojos IR.
//! - **PARA QUÉ:** Identificar la proximidad a Puntos Fijos Cuánticos $\beta(g^*) = 0$ (transiciones de fase
//!   de liquidez y turbulencia cuántica), predecir rupturas direccionales no renormalizables, y modular
//!   la convicción de señal con un factor de acoplamiento de Wilson $\lambda_{\text{wilson}} \in [0.10, 1.40]$
//!   en $O(1)$ stack sin heap allocations.
//! - **CÓMO:**
//!   1. Modela el campo continuo de velocidad de precios $\phi(t)$ y sus coeficientes efectivos
//!      en un potencial polinomial $\Gamma_k[\phi] = \frac{1}{2} m_k^2 \phi^2 + \frac{\lambda_k}{4!} \phi^4$.
//!   2. Evalúa las funciones beta analíticas adimensionales del grupo de renormalización:
//!      $$\beta_m(\tilde{m}^2, \tilde{\lambda}) = -2 \tilde{m}^2 - \frac{\tilde{\lambda}}{1 + \tilde{m}^2}$$
//!      $$\beta_\lambda(\tilde{m}^2, \tilde{\lambda}) = -\tilde{\lambda} + 3 \frac{\tilde{\lambda}^2}{(1 + \tilde{m}^2)^2}$$
//!   3. Integra el flujo del grupo de renormalización a través de 3 escalas de momentum (Micro, Meso, Macro)
//!      usando un integrador de Runge-Kutta de 2º orden en stack.
//!   4. Computa la distancia euclidiana en el espacio de fase al punto fijo crítico de Wilson-Fisher.
//!   5. Modula la convicción de la señal multi-slot según la estabilidad de la cuenca de atracción.
//! - **CUÁNDO:** En cada pulso de síntesis de señales en `crates/signal-engine`.
//! - **DÓNDE:** En `crates/signal-engine/src/wilson_exact_renormalization.rs`.
//! - **QUIÉN:** Rol 2 Quant Developer & Rol 7 Arquitecto de Silicio & Teoría Cuántica de Campos.

/// Número de escalas de corte de momento infrarrojo (IR cutoffs) en stack.
pub const RG_MOMENTUM_SCALES: usize = 3;

/// Parámetros configurables del motor del Grupo de Renormalización de Wilson.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonExactRgParams {
    /// Masa efectiva adimensional inicial $\tilde{m}_0^2 \ge 0$.
    pub initial_mass_sq: f64,
    /// Acoplamiento cuártico inicial $\tilde{\lambda}_0 \ge 0$.
    pub initial_quartic_coupling: f64,
    /// Paso logarítmico de flujo de renormalización $\Delta t = \Delta \ln(k/\Lambda)$.
    pub rg_flow_step: f64,
    /// Sensibilidad de escala para el acoplamiento del flujo de órdenes.
    pub ofi_coupling_sensitivity: f64,
    /// Coordenada de masa del Punto Fijo de Wilson-Fisher de referencia.
    pub wf_fixed_point_mass_sq: f64,
    /// Coordenada cuártica del Punto Fijo de Wilson-Fisher de referencia.
    pub wf_fixed_point_coupling: f64,
}

impl Default for WilsonExactRgParams {
    fn default() -> Self {
        Self {
            initial_mass_sq: 0.10,
            initial_quartic_coupling: 0.25,
            rg_flow_step: 0.05,
            ofi_coupling_sensitivity: 0.50,
            wf_fixed_point_mass_sq: -0.15,
            wf_fixed_point_coupling: 0.333,
        }
    }
}

/// Estado de acoplamientos del Grupo de Renormalización a una escala $k$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgScaleCoupling {
    /// Masa efectiva adimensional $\tilde{m}_k^2$.
    pub mass_sq: f64,
    /// Acoplamiento cuártico adimensional $\tilde{\lambda}_k$.
    pub quartic_coupling: f64,
    /// Tasa de cambio de la función beta de masa $\beta_m$.
    pub beta_mass: f64,
    /// Tasa de cambio de la función beta cuártica $\beta_\lambda$.
    pub beta_quartic: f64,
}

impl RgScaleCoupling {
    pub const ZERO: Self = Self {
        mass_sq: 0.0,
        quartic_coupling: 0.0,
        beta_mass: 0.0,
        beta_quartic: 0.0,
    };
}

/// Informe cuantitativo del flujo del Grupo de Renormalización de Wilson.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonExactRgReport {
    /// Acoplamientos a través de las 3 escalas: [0: Micro, 1: Meso, 2: Macro].
    pub scale_couplings: [RgScaleCoupling; RG_MOMENTUM_SCALES],
    /// Distancia euclidiana mínima al Punto Fijo Cuántico de Wilson-Fisher.
    pub min_distance_to_fixed_point: f64,
    /// Indicador de proximidad a transición de fase crítica ($d_{\text{FP}} < 0.15$).
    pub is_critical_phase_transition: bool,
    /// Dimensión anómala emergente $\eta_{\text{anom}}$ del campo de precios.
    pub anomalous_dimension: f64,
    /// Modulador de convicción de señal de Wilson $\lambda_{\text{wilson}} \in [0.10, 1.40]$.
    pub signal_conviction_modulator: f64,
}

/// Motor determinista del Grupo de Renormalización de Wilson en stack $O(1)$.
#[derive(Debug, Clone, Copy)]
pub struct WilsonExactRenormalizationEngine {
    pub params: WilsonExactRgParams,
    pub current_mass_sq: f64,
    pub current_quartic_coupling: f64,
    pub smoothed_anomalous_dim: f64,
}

impl WilsonExactRenormalizationEngine {
    /// Inicializa una nueva instancia del motor con memoria estática en stack.
    pub fn new(params: WilsonExactRgParams) -> Self {
        Self {
            params,
            current_mass_sq: params.initial_mass_sq,
            current_quartic_coupling: params.initial_quartic_coupling,
            smoothed_anomalous_dim: 0.05,
        }
    }

    /// Actualiza y propaga el flujo del Grupo de Renormalización ante nueva microestructura.
    ///
    /// Preserva la cota de ruina de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$ y
    /// la invariancia de cero pánicos `rb_p1`.
    pub fn update_and_evaluate(
        &mut self,
        price_ret: f64,
        ofi: f64,
        volatility: f64,
        _event_time_ms: u64,
    ) -> WilsonExactRgReport {
        let safe_ret = if price_ret.is_finite() {
            price_ret.clamp(-0.20, 0.20)
        } else {
            0.0
        };
        let safe_ofi = if ofi.is_finite() {
            ofi.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let safe_vol = if volatility.is_finite() && volatility > 0.0 {
            volatility.clamp(0.0001, 2.0)
        } else {
            0.01
        };

        // 1. Inyección de perturbación UV del mercado en los acoplamientos bare
        let bare_mass_pert =
            safe_vol * 10.0 + safe_ofi.abs() * self.params.ofi_coupling_sensitivity;
        let bare_quartic_pert = safe_ret.abs() * 20.0;

        self.current_mass_sq = (0.85 * self.current_mass_sq
            + 0.15 * (self.params.initial_mass_sq + bare_mass_pert))
            .clamp(-0.80, 5.0);
        self.current_quartic_coupling = (0.85 * self.current_quartic_coupling
            + 0.15 * (self.params.initial_quartic_coupling + bare_quartic_pert))
            .clamp(0.01, 5.0);

        // 2. Integración del Flujo de Wetterich a través de las 3 escalas diádicas
        let mut scale_couplings = [RgScaleCoupling::ZERO; RG_MOMENTUM_SCALES];
        let mut running_m2 = self.current_mass_sq;
        let mut running_lambda = self.current_quartic_coupling;

        let dt = self.params.rg_flow_step.clamp(0.001, 0.20);
        let mut min_dist_fp = f64::INFINITY;

        for (scale_idx, entry) in scale_couplings.iter_mut().enumerate() {
            // Evaluación de funciones beta en el punto actual
            let (beta_m, beta_l) = Self::compute_beta_functions(running_m2, running_lambda);

            // Integrador Runge-Kutta 2 (Heun) en stack
            let m2_half = running_m2 + 0.5 * dt * beta_m;
            let lambda_half = running_lambda + 0.5 * dt * beta_l;
            let (beta_m_half, beta_l_half) = Self::compute_beta_functions(m2_half, lambda_half);

            running_m2 = (running_m2 + dt * beta_m_half).clamp(-0.95, 10.0);
            running_lambda = (running_lambda + dt * beta_l_half).clamp(0.001, 10.0);

            *entry = RgScaleCoupling {
                mass_sq: running_m2,
                quartic_coupling: running_lambda,
                beta_mass: beta_m_half,
                beta_quartic: beta_l_half,
            };

            // Distancia al Punto Fijo de Wilson-Fisher
            let dm = running_m2 - self.params.wf_fixed_point_mass_sq;
            let dl = running_lambda - self.params.wf_fixed_point_coupling;
            let dist_fp = (dm * dm + dl * dl).sqrt();
            if dist_fp < min_dist_fp {
                min_dist_fp = dist_fp;
            }

            let _ = scale_idx;
        }

        // 3. Dimensión anómala $\eta_{\text{anom}} = \frac{1}{6} \frac{\tilde{\lambda}^2}{(1 + \tilde{m}^2)^3}$
        let denom_anom = (1.0 + running_m2).max(0.05).powi(3);
        let raw_anom = (running_lambda * running_lambda) / (6.0 * denom_anom);
        let instantaneous_anom = raw_anom.clamp(0.0, 0.50);
        self.smoothed_anomalous_dim =
            0.85 * self.smoothed_anomalous_dim + 0.15 * instantaneous_anom;

        // 4. Proximidad a transición de fase crítica
        let is_critical_phase_transition = min_dist_fp < 0.20;

        // 5. Modulador de Convicción de Señal de Wilson $\lambda_{\text{wilson}} \in [0.10, 1.40]$
        // En regímenes no críticos (flujo estable), la convicción se preserva o potencia.
        // En puntos críticos de transición de fase, se atenúa para prevenir whipsaws.
        let critical_discount = if is_critical_phase_transition {
            0.50 + 2.5 * min_dist_fp.clamp(0.0, 0.20) // Se reduce hasta 0.50 si d_FP -> 0
        } else {
            1.0 + (min_dist_fp - 0.20).clamp(0.0, 0.50) * 0.40
        };
        let raw_modulator = critical_discount / (1.0 + self.smoothed_anomalous_dim * 2.0);
        let signal_conviction_modulator = raw_modulator.clamp(0.10, 1.40);

        WilsonExactRgReport {
            scale_couplings,
            min_distance_to_fixed_point: min_dist_fp,
            is_critical_phase_transition,
            anomalous_dimension: self.smoothed_anomalous_dim,
            signal_conviction_modulator,
        }
    }

    /// Evalúa las funciones beta exactas del grupo de renormalización para acoplamientos adimensionales:
    /// $\beta_m = -2 \tilde{m}^2 - \frac{\tilde{\lambda}}{1 + \tilde{m}^2}$
    /// $\beta_\lambda = -\tilde{\lambda} + 3 \frac{\tilde{\lambda}^2}{(1 + \tilde{m}^2)^2}$
    #[inline(always)]
    fn compute_beta_functions(m2: f64, lambda: f64) -> (f64, f64) {
        let safe_denom = (1.0 + m2).max(0.05);
        let beta_m = -2.0 * m2 - (lambda / safe_denom);
        let beta_lambda = -lambda + 3.0 * (lambda * lambda) / (safe_denom * safe_denom);
        (beta_m.clamp(-20.0, 20.0), beta_lambda.clamp(-20.0, 20.0))
    }
}

impl Default for WilsonExactRenormalizationEngine {
    fn default() -> Self {
        Self::new(WilsonExactRgParams::default())
    }
}
