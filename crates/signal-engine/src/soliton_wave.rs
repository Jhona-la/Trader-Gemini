use strategy_core::QuantumStrategy;
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;

/// 🌊 ALGORITMO #86: MOTOR DE ONDAS SOLITÓN NO-LINEALES DE SCHRÖDINGER (SOLITON WAVE ENGINE)
/// Modela los impulsos de precio como solitones no-dispersivos de la ecuación NLS (i \psi_t + 1/2 \psi_{xx} + |\psi|^2 \psi = 0),
/// identificando paquetes de liquidez que atraviesan la profundidad del libro de órdenes sin perder amplitud.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct SolitonWaveEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for SolitonWaveEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolitonWaveEngine").finish()
    }
}

impl SolitonWaveEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula la amplitud del perfil solitónico sech(x) en O(1) con protección numérica absoluta
    #[inline(always)]
    pub fn compute_soliton_amplitude(
        amplitude: f64,
        velocity: f64,
        x_pos: f64,
        t_time: f64,
    ) -> f64 {
        // FIX #645: Sanitizar los parámetros entrantes
        let safe_amp = if amplitude.is_finite() { amplitude.abs() } else { 0.0 };
        let safe_vel = if velocity.is_finite() { velocity } else { 0.0 };
        let safe_x = if x_pos.is_finite() { x_pos } else { 0.0 };
        let safe_t = if t_time.is_finite() { t_time } else { 0.0 };

        let raw_phase = safe_amp * (safe_x - safe_vel * safe_t);
        // Clamping a [-50.0, 50.0] para evitar desbordamiento a +inf en cosh()
        let phase = raw_phase.clamp(-50.0, 50.0);
        let cosh_val = phase.cosh();
        if cosh_val == 0.0 || cosh_val.is_infinite() || cosh_val.is_nan() {
            return 0.0;
        }
        let res = safe_amp / cosh_val;
        if res.is_finite() { res } else { 0.0 }
    }

    /// Calcula la potencia de envolvente energética del paquete solitónico (Punto #265)
    #[inline(always)]
    pub fn compute_soliton_envelope_power(amplitude: f64) -> f64 {
        if !amplitude.is_finite() || amplitude <= 0.0 {
            0.0
        } else {
            (2.0 * amplitude).clamp(0.0, 100.0)
        }
    }
}

impl QuantumStrategy for SolitonWaveEngine {
    fn name(&self) -> &str {
        "SolitonWaveEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        let registry = match self.registry.as_ref() {
            Some(r) => r,
            None => return 0.0,
        };
        let amp = registry.get("soliton_amplitude", "SolitonWaveEngine")
            .or_else(|| registry.get("order_flow_imbalance", "SolitonWaveEngine"))
            .or_else(|| registry.get("vol_delta", "SolitonWaveEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let vel = registry.get("soliton_velocity", "SolitonWaveEngine")
            .or_else(|| registry.get("price_velocity", "SolitonWaveEngine"))
            .or_else(|| registry.get("order_flow_velocity", "SolitonWaveEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let pos = registry.get("soliton_pos", "SolitonWaveEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        if !amp.is_finite() || !vel.is_finite() || !pos.is_finite() {
            return 0.0;
        }
        if vel.abs() < 1e-6 {
            return 0.0;
        }

        let amp_val = Self::compute_soliton_amplitude(amp, vel, pos, 0.0).clamp(0.0, 1.0);
        vel.signum() * amp_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soliton_amplitude_overflow_immunity() {
        // Fase extrema que desbordaría sin clamping
        let amp_huge = SolitonWaveEngine::compute_soliton_amplitude(10.0, 1.0, 1000.0, 0.0);
        assert!(!amp_huge.is_nan() && !amp_huge.is_infinite());
        assert!(amp_huge >= 0.0);

        // Fase centrada
        let amp_zero = SolitonWaveEngine::compute_soliton_amplitude(2.0, 1.0, 0.0, 0.0);
        assert!((amp_zero - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_soliton_zero_velocity_returns_flat() {
        let engine = SolitonWaveEngine::new();
        // Sin registro o con velocidad 0, debe retornar estrictamente 0.0
        assert_eq!(engine.evaluate(), 0.0);
    }

    #[test]
    fn test_soliton_envelope_power() {
        let p_zero = SolitonWaveEngine::compute_soliton_envelope_power(0.0);
        assert_eq!(p_zero, 0.0);

        let p_normal = SolitonWaveEngine::compute_soliton_envelope_power(2.5);
        assert_eq!(p_normal, 5.0);

        let p_nan = SolitonWaveEngine::compute_soliton_envelope_power(f64::NAN);
        assert_eq!(p_nan, 0.0);
    }

    #[test]
    fn test_soliton_wave_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("soliton_amplitude", 0.8);
        registry.set("soliton_velocity", 1.5);
        registry.set("soliton_pos", 0.0);

        let mut engine = SolitonWaveEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(eval > 0.0, "Velocidad positiva y amplitud centrada deben producir señal positiva");
        assert!(eval <= 1.0);
    }
}
