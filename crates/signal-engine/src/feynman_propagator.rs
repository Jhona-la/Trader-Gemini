use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::{QuantumStrategy, TradeHorizon};
use crate::voto_espectral::{VotoEspectral, ESCALAS_VOTO};

/// ⚛️ ALGORITMO #101: COLECTOR DE INTEGRAL DE CAMINO DE FEYNMAN Y PROPAGADOR CUÁNTICO (FEYNMAN PROPAGATOR ENGINE)
///
/// Modela la propagación del precio como una suma coherente sobre todas las trayectorias posibles en el espacio de Hilbert:
///
///   K(x_b, t_b; x_a, t_a) = ∫ D[x(t)] exp( (i / ħ_fin) S[x(t)] )
///
/// Principios Físicos Cuánticos:
/// 1. **Acción Lagrangiana Discreta Inter-Escala S(k) (REV-9)**:
///    Para cada escala diádica k ∈ [0, 31] de Hilbert (τ_k = 4^k μs), la dinámica discreta se evalúa
///    sobre el espacio de fases canónico (z_k, vel_k) donde vel_k = z_k - z_{k-1} es el gradiente
///    de momentum entre escalas diádicas contiguas:
///      L(z, vel) = 1/2 m vel² - (1/2 k_spring z² + 1/4 λ z⁴)
///    La fase cuántica se compone del ángulo canónico θ_k = atan2(vel · m, z · √k_spring)
///    y una modulación lagrangiana normalizada sobre la superficie de energía total (1 + z² + vel²).
/// 2. **Amplitud Compleja de Probabilidad y Fase Cuántica**:
///    Cada escala temporal contribuye una amplitud compleja de probabilidad:
///      ψ_k = A_k · exp(i θ_k) = A_k · (cos(θ_k) + i sin(θ_k))
///    donde A_k = tanh(|z_k| · 0.5).
/// 3. **Interferencia Constructiva vs Destructiva (Coherencia Cuántica C_coh)**:
///    - Suma coherente multiescala: Ψ_total = ∑_{k=0}^{31} ψ_k
///    - Densidad coherente: |Ψ_total|² = (∑ Re(ψ_k))² + (∑ Im(ψ_k))²
///    - Densidad incoherente: I_incoh = ∑_{k=0}^{31} |ψ_k|²
///    - Coeficiente de Coherencia Cuántica normalizado por escalas activas (REV-7):
///        C_coh = (|Ψ_total|² / (I_incoh + ε)) / active_count ∈ [0.0, 1.0]
///      - Si C_coh > 0.40: Interferencia fuertemente constructiva (escalas alineadas
///        en fase hacia una misma dirección ⇒ momentum cuántico de alta convicción).
///      - Si C_coh ≤ 0.20: Interferencia destructiva / decoherencia térmica cuántica
///        (fases desalineadas cancelan la señal ⇒ abstención estricta).
/// 4. **Garantías de Alto Rendimiento**:
///    - O(1) determinista en CPU (< 25 ns en hot path), #[inline(always)], zero heap allocations.
///    - Inmunidad total fail-closed ante NaN, ±Inf o parámetros no finitos.
/// ⚛️ OLA Ω122: Parámetros desacoplados y configurables para el propagador cuántico de Feynman
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeynmanPropagatorParams {
    /// Constante de Planck efectiva del mercado ħ_eff (default: 1.0)
    pub hbar_effective: f64,
    /// Masa inercial efectiva modal m (default: 1.0)
    pub mass: f64,
    /// Constante armónica de resorte cuántico k_spring (default: 1.0)
    pub k_spring: f64,
    /// Coeficiente de no-linealidad cuántica cuártica λ_quartic (default: 0.10)
    pub lambda_quartic: f64,
    /// Factor de acoplamiento de la acción cuántica lagrangiana en la fase (default: 0.10)
    pub action_coupling: f64,
    /// Factor de escala del argumento tanh en la amplitud modal (default: 0.50)
    pub amplitude_scale: f64,
    /// Umbral mínimo de coherencia cuántica constructiva C_coh (default: 0.20)
    pub coherence_threshold: f64,
}

impl Default for FeynmanPropagatorParams {
    fn default() -> Self {
        Self {
            hbar_effective: FeynmanPropagatorEngine::DEFAULT_HBAR_EFF,
            mass: FeynmanPropagatorEngine::DEFAULT_MASS,
            k_spring: FeynmanPropagatorEngine::DEFAULT_K_SPRING,
            lambda_quartic: FeynmanPropagatorEngine::DEFAULT_LAMBDA_QUARTIC,
            action_coupling: 0.10,
            amplitude_scale: 0.50,
            coherence_threshold: 0.20,
        }
    }
}

impl FeynmanPropagatorParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.hbar_effective.is_finite() || !(0.01..=100.0).contains(&self.hbar_effective) {
            return Err(format!("Invalid hbar_effective: {}", self.hbar_effective));
        }
        if !self.mass.is_finite() || !(0.01..=100.0).contains(&self.mass) {
            return Err(format!("Invalid mass: {}", self.mass));
        }
        if !self.k_spring.is_finite() || !(0.0..=100.0).contains(&self.k_spring) {
            return Err(format!("Invalid k_spring: {}", self.k_spring));
        }
        if !self.lambda_quartic.is_finite() || !(0.0..=10.0).contains(&self.lambda_quartic) {
            return Err(format!("Invalid lambda_quartic: {}", self.lambda_quartic));
        }
        if !self.action_coupling.is_finite() || !(0.0..=2.0).contains(&self.action_coupling) {
            return Err(format!("Invalid action_coupling: {}", self.action_coupling));
        }
        if !self.amplitude_scale.is_finite() || !(0.05..=5.0).contains(&self.amplitude_scale) {
            return Err(format!("Invalid amplitude_scale: {}", self.amplitude_scale));
        }
        if !self.coherence_threshold.is_finite() || !(0.01..=0.80).contains(&self.coherence_threshold) {
            return Err(format!("Invalid coherence_threshold: {}", self.coherence_threshold));
        }
        Ok(())
    }
}

#[derive(Clone)]
#[repr(C, align(64))]
pub struct FeynmanPropagatorEngine {
    registry: Option<Arc<OmniscientRegistry>>,
    pub params: FeynmanPropagatorParams,
}

impl Default for FeynmanPropagatorEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FeynmanPropagatorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeynmanPropagatorEngine")
            .field("params", &self.params)
            .finish()
    }
}

impl FeynmanPropagatorEngine {
    pub const DEFAULT_HBAR_EFF: f64 = 1.0;
    pub const DEFAULT_MASS: f64 = 1.0;
    pub const DEFAULT_K_SPRING: f64 = 1.0;
    pub const DEFAULT_LAMBDA_QUARTIC: f64 = 0.10;

    pub fn new() -> Self {
        Self {
            registry: None,
            params: FeynmanPropagatorParams::default(),
        }
    }

    /// OLA Ω122: Constructor con parámetros desacoplados específicos
    pub fn with_params(mut self, params: FeynmanPropagatorParams) -> Self {
        if params.validate().is_ok() {
            self.params = params;
        }
        self
    }

    /// OLA Ω122: Actualización dinámica de parámetros
    pub fn update_params(&mut self, params: FeynmanPropagatorParams) {
        if params.validate().is_ok() {
            self.params = params;
        }
    }

    /// Calcula la función lagrangiana L(z, v) = T - V:
    ///   T = 1/2 m v²
    ///   V = 1/2 k z² + 1/4 λ z⁴
    #[inline(always)]
    pub fn compute_lagrangian(
        momentum_z: f64,
        velocity_dz: f64,
        mass: f64,
        k_spring: f64,
        lambda_quartic: f64,
    ) -> f64 {
        let safe_z = if momentum_z.is_finite() {
            momentum_z.clamp(-10.0, 10.0)
        } else {
            0.0
        };
        let safe_v = if velocity_dz.is_finite() {
            velocity_dz.clamp(-10.0, 10.0)
        } else {
            0.0
        };
        let safe_m = if mass.is_finite() && mass > 0.0 {
            mass.clamp(0.01, 100.0)
        } else {
            Self::DEFAULT_MASS
        };
        let safe_k = if k_spring.is_finite() && k_spring >= 0.0 {
            k_spring.clamp(0.0, 100.0)
        } else {
            Self::DEFAULT_K_SPRING
        };
        let safe_l = if lambda_quartic.is_finite() && lambda_quartic >= 0.0 {
            lambda_quartic.clamp(0.0, 10.0)
        } else {
            Self::DEFAULT_LAMBDA_QUARTIC
        };

        let kinetic = 0.5 * safe_m * safe_v * safe_v;
        let z2 = safe_z * safe_z;
        let potential = 0.5 * safe_k * z2 + 0.25 * safe_l * z2 * z2;
        let lagrangian = kinetic - potential;

        if lagrangian.is_finite() {
            lagrangian.clamp(-1000.0, 1000.0)
        } else {
            0.0
        }
    }

    /// Calcula la fase cuántica exp(i θ) = (cos(θ), sin(θ)) a partir de la acción S / ħ_eff.
    #[inline(always)]
    pub fn compute_quantum_phase(action: f64, hbar_effective: f64) -> (f64, f64) {
        let safe_s = if action.is_finite() {
            action
        } else {
            0.0
        };
        let safe_hbar = if hbar_effective.is_finite() && hbar_effective > 0.0 {
            hbar_effective.clamp(0.01, 100.0)
        } else {
            Self::DEFAULT_HBAR_EFF
        };

        let theta = safe_s / safe_hbar;
        let c = theta.cos();
        let s = theta.sin();
        (
            if c.is_finite() { c } else { 1.0 },
            if s.is_finite() { s } else { 0.0 },
        )
    }

    /// Evalúa el factor de coherencia cuántica C_coh = |∑ ψ_k|² / (∑ |ψ_k|² + ε)
    /// Retorna un escalar en [0.0, 32.0] donde 32.0 representa coherencia de fase perfecta en las 32 escalas.
    #[inline(always)]
    pub fn compute_coherence_factor(
        phases: &[(f64, f64); ESCALAS_VOTO],
        amplitudes: &[f64; ESCALAS_VOTO],
    ) -> f64 {
        let mut re_sum = 0.0f64;
        let mut im_sum = 0.0f64;
        let mut incoh_sum = 0.0f64;

        for k in 0..ESCALAS_VOTO {
            let (cos_th, sin_th) = phases[k];
            let a = amplitudes[k];
            if a.is_finite() && a > 0.0 {
                let safe_a = a.clamp(0.0, 1.0);
                re_sum += safe_a * cos_th;
                im_sum += safe_a * sin_th;
                incoh_sum += safe_a * safe_a;
            }
        }

        if incoh_sum <= 1e-12 {
            return 0.0;
        }

        let coherent_norm = re_sum * re_sum + im_sum * im_sum;
        let ratio = coherent_norm / (incoh_sum + 1e-12);
        if ratio.is_finite() {
            ratio.clamp(0.0, ESCALAS_VOTO as f64)
        } else {
            0.0
        }
    }

    /// Genera el voto espectral de la integral de camino y retorna simultáneamente (VotoEspectral, coherencia_normalizada).
    ///
    /// Modula la señal por la coherencia cuántica multiescala: si las escalas interfieren destructivamente,
    /// el voto colapsa a 0.0 protegiendo contra fases estocásticas desalineadas.
    /// OLA Ω122: Genera el voto espectral de la integral de camino con parámetros dinámicos configurables.
    #[inline(always)]
    pub fn voto_espectral_con_coherencia_with_params(
        desplazamientos: &[f64; ESCALAS_VOTO],
        params: &FeynmanPropagatorParams,
    ) -> (VotoEspectral, f64) {
        let mut phases = [(1.0f64, 0.0f64); ESCALAS_VOTO];
        let mut amplitudes = [0.0f64; ESCALAS_VOTO];

        let safe_m = if params.mass.is_finite() && params.mass > 0.0 { params.mass } else { Self::DEFAULT_MASS };
        let safe_k = if params.k_spring.is_finite() && params.k_spring >= 0.0 { params.k_spring } else { Self::DEFAULT_K_SPRING };
        let safe_lambda = if params.lambda_quartic.is_finite() && params.lambda_quartic >= 0.0 { params.lambda_quartic } else { Self::DEFAULT_LAMBDA_QUARTIC };
        let safe_hbar = if params.hbar_effective.is_finite() && params.hbar_effective > 0.0 { params.hbar_effective } else { Self::DEFAULT_HBAR_EFF };
        let action_coupling = if params.action_coupling.is_finite() { params.action_coupling } else { 0.10 };
        let amp_scale = if params.amplitude_scale.is_finite() && params.amplitude_scale > 0.0 { params.amplitude_scale } else { 0.50 };

        // 1. Calcular lagrangiano, acción y fase canónica en cada escala
        for k in 0..ESCALAS_VOTO {
            let z = desplazamientos[k];
            if !z.is_finite() {
                continue;
            }
            let safe_z = z.clamp(-10.0, 10.0);
            // REV-8: Velocidad efectiva dz entre escalas diádicas contiguas, filtrando NaN en vecino
            let prev_z = if k > 0 && desplazamientos[k - 1].is_finite() {
                desplazamientos[k - 1].clamp(-10.0, 10.0)
            } else {
                safe_z
            };
            let vel = safe_z - prev_z;

            // En el espacio de fases de Hilbert (z, vel), la fase cuántica modal se define por el ángulo de fase canónico:
            //   θ_k = atan2(vel · mass, safe_z · √k_spring)
            // Cuando todas las escalas están alineadas en la misma dirección, θ_k ≈ constante (en fase coherente pura).
            let p_momentum = vel * safe_m;
            let q_coord = safe_z * safe_k.max(1e-4).sqrt();
            let phase_angle = p_momentum.atan2(q_coord);

            // REV-6: Modulación por acción cuántica lagrangiana S / ħ acotada sobre la superficie de energía
            let lagrangian = Self::compute_lagrangian(safe_z, vel, safe_m, safe_k, safe_lambda);
            let action_phase = (lagrangian / (safe_hbar * (1.0 + safe_z * safe_z + vel * vel))).tanh();
            let total_theta = phase_angle + action_coupling * action_phase;

            phases[k] = (total_theta.cos(), total_theta.sin());
            amplitudes[k] = (safe_z.abs() * amp_scale).tanh().clamp(0.0, 1.0);
        }

        // 2. REV-7: Coherencia cuántica global normalizada por el número de escalas activas (no static 32)
        let active_count = amplitudes.iter().filter(|&&a| a > 0.0).count().max(1);
        let raw_coherence = Self::compute_coherence_factor(&phases, &amplitudes);
        let normalized_coherence = (raw_coherence / (active_count as f64)).clamp(0.0, 1.0);

        // 3. Modulación de señal por escala ponderada por coherencia de fase
        let mut por_escala = [0.0f64; ESCALAS_VOTO];
        for k in 0..ESCALAS_VOTO {
            let z = desplazamientos[k];
            if !z.is_finite() {
                continue;
            }
            let safe_z = z.clamp(-10.0, 10.0);
            let dir = safe_z.signum();
            let amp = amplitudes[k];
            // La señal sólo florece si hay coherencia cuántica constructiva
            let signal = dir * amp * normalized_coherence;
            por_escala[k] = signal.clamp(-1.0, 1.0);
        }

        (VotoEspectral::desde_arr(&por_escala), normalized_coherence)
    }

    /// OLA Ω122: Voto espectral con parámetros completos
    #[inline(always)]
    pub fn voto_espectral_with_params(
        desplazamientos: &[f64; ESCALAS_VOTO],
        params: &FeynmanPropagatorParams,
    ) -> VotoEspectral {
        Self::voto_espectral_con_coherencia_with_params(desplazamientos, params).0
    }

    /// Genera el voto espectral de la integral de camino y retorna simultáneamente (VotoEspectral, coherencia_normalizada).
    /// Delegado a `voto_espectral_con_coherencia_with_params` para retrocompatibilidad total.
    #[inline(always)]
    pub fn voto_espectral_con_coherencia(
        desplazamientos: &[f64; ESCALAS_VOTO],
        hbar_eff: f64,
        mass: f64,
        k_spring: f64,
    ) -> (VotoEspectral, f64) {
        let mut params = FeynmanPropagatorParams::default();
        if hbar_eff.is_finite() && hbar_eff > 0.0 { params.hbar_effective = hbar_eff; }
        if mass.is_finite() && mass > 0.0 { params.mass = mass; }
        if k_spring.is_finite() && k_spring >= 0.0 { params.k_spring = k_spring; }
        Self::voto_espectral_con_coherencia_with_params(desplazamientos, &params)
    }

    /// Genera el voto espectral de la integral de camino VotoEspectral en O(1) puro sobre la malla de 32 escalas.
    #[inline(always)]
    pub fn voto_espectral(
        desplazamientos: &[f64; ESCALAS_VOTO],
        hbar_eff: f64,
        mass: f64,
        k_spring: f64,
    ) -> VotoEspectral {
        Self::voto_espectral_con_coherencia(desplazamientos, hbar_eff, mass, k_spring).0
    }
}

impl QuantumStrategy for FeynmanPropagatorEngine {
    fn name(&self) -> &str {
        "FeynmanPropagatorEngine"
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

        let coherence = match self.registry.as_ref() {
            Some(r) => r.get_scoped_val_or(sym_opt, cid_opt, "feynman_coherence", 0.50),
            None => 0.50,
        };

        let dominant_v = match self.registry.as_ref() {
            Some(r) => r.get_scoped_val_or(sym_opt, cid_opt, "consenso_espectral_dominante", 0.0),
            None => 0.0,
        };

        // #652/H2: con la escala dominante por debajo de la banda operable el
        // orquestador no consume el veredicto espectral y su consenso escalar
        // debe quedar bit a bit. Reinyectarlo aquí lo saltaría (CL-59).
        let tau_dom = match self.registry.as_ref() {
            Some(r) => r.get_scoped_val_or(sym_opt, cid_opt, "consenso_espectral_tau", 0.0),
            None => 0.0,
        };
        if !(tau_dom >= quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS) {
            return 0.0;
        }

        if !coherence.is_finite() || !dominant_v.is_finite() {
            return 0.0;
        }

        let safe_coh = coherence.clamp(0.0, 1.0);
        let safe_v = dominant_v.clamp(-1.0, 1.0);

        // OLA Ω122: Transmisión C¹ suave via smoothstep cúbico Hermite adaptativa
        let thresh = if self.params.coherence_threshold.is_finite() {
            self.params.coherence_threshold.clamp(0.01, 0.90)
        } else {
            0.20
        };
        let range = (1.0 - thresh).max(0.01);
        if safe_coh > thresh {
            let u = ((safe_coh - thresh) / range).clamp(0.0, 1.0);
            let transmission = u * u * (3.0 - 2.0 * u);
            (safe_v * transmission).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn horizon(&self) -> TradeHorizon {
        TradeHorizon::Continuous
    }

    fn is_operable_for_coin(&self, coin_id: usize, symbol: &str) -> bool {
        // LG8-145-06 (OLA Ω148): En el régimen de fallback escalar (`tau_dom < TAU_ANCHOR_FAST_MS`
        // o sin `consenso_espectral_dominante` activo), el adaptador escalar de Feynman se auto-silencia
        // por diseño (#652/H2, CL-59) para no reinyectar una escala inoperable. Por tanto, en ese régimen
        // es estructuralmente inoperable y no debe inflar `n_fuentes` en el denominador del ensamble escalar.
        // Si existe dominante espectral en banda operable (`tau_dom >= TAU_ANCHOR_FAST_MS` y `|dominant_v| > 1e-9`),
        // sí es operable (y si `coherence <= threshold` devuelve 0.0, cuenta como abstención genuina por decoherencia).
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return false,
        };
        let sym_opt = if symbol.is_empty() { None } else { Some(symbol) };
        let cid_opt = if symbol.is_empty() { None } else { Some(coin_id) };
        let tau_dom = r.get_scoped_val_or(sym_opt, cid_opt, "consenso_espectral_tau", 0.0);
        let dominant_v = r.get_scoped_val_or(sym_opt, cid_opt, "consenso_espectral_dominante", 0.0);
        tau_dom.is_finite()
            && tau_dom >= quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS
            && dominant_v.is_finite()
            && dominant_v.abs() > 1e-9
    }
}
