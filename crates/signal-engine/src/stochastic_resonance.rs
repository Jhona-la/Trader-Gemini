use strategy_core::QuantumStrategy;
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;

/// ⚡ ALGORITMO #98: FILTRO DE RESONANCIA ESTOCÁSTICA CUÁNTICA (STOCHASTIC RESONANCE ENGINE)
/// Utiliza el ruido de fondo de microestructura para amplificar señales sub-umbral débiles de Alpha,
/// mediante el fenómeno de resonancia estocástica no-lineal en pozos de potencial simétricos.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct StochasticResonanceEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for StochasticResonanceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StochasticResonanceEngine").finish()
    }
}

impl StochasticResonanceEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Amplifica una señal sub-umbral combinándola con la intensidad de ruido \sigma en O(1)
    /// Simétrica para señales Long (> 0) y Short (< 0) mediante pozo de potencial bi-estable
    #[inline(always)]
    pub fn amplify_signal_with_noise(weak_signal: f64, noise_variance: f64) -> f64 {
        // FIX #649: Sanitizar parámetros de resonancia estocástica
        let safe_signal = if weak_signal.is_finite() { weak_signal } else { 0.0 };
        let safe_noise = if noise_variance.is_finite() && noise_variance > 0.0 { noise_variance.max(1e-6) } else { 0.0001 };

        let ratio = (-safe_signal.abs() / safe_noise).clamp(-50.0, 50.0);
        let resonance_factor = (1.0 / (1.0 + ratio.exp())).clamp(0.0, 1.0);
        let res = safe_signal * (1.0 + resonance_factor);
        if res.is_finite() { res } else { 0.0 }
    }
}

impl QuantumStrategy for StochasticResonanceEngine {
    fn name(&self) -> &str {
        "StochasticResonanceEngine"
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
        let weak_signal = registry.get("weak_alpha_signal", "StochasticResonanceEngine")
            .or_else(|| registry.get("order_flow_imbalance", "StochasticResonanceEngine"))
            .or_else(|| registry.get("alpha_signal", "StochasticResonanceEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        let noise_variance = registry.get("microstructure_noise_variance", "StochasticResonanceEngine")
            .or_else(|| registry.get("tick_variance", "StochasticResonanceEngine"))
            .or_else(|| registry.get("atr_pct", "StochasticResonanceEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0001);

        if !weak_signal.is_finite() || !noise_variance.is_finite() {
            return 0.0;
        }

        Self::amplify_signal_with_noise(weak_signal, noise_variance).clamp(-1.0, 1.0)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Scalp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_resonance_amplification() {
        let weak = 0.1;
        let amp = StochasticResonanceEngine::amplify_signal_with_noise(weak, 0.05);
        assert!(amp > weak);
        assert!(!amp.is_nan() && !amp.is_infinite());
    }

    #[test]
    fn test_stochastic_resonance_nan_and_negative_symmetry() {
        let weak_short = -0.1;
        let amp_short = StochasticResonanceEngine::amplify_signal_with_noise(weak_short, 0.05);
        assert!(amp_short < weak_short);
        assert!(amp_short.is_finite());

        let nan_res = StochasticResonanceEngine::amplify_signal_with_noise(f64::NAN, 0.05);
        assert!(nan_res.is_finite() && nan_res == 0.0);
    }

    #[test]
    fn test_stochastic_resonance_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("weak_alpha_signal", 0.2);
        registry.set("microstructure_noise_variance", 0.01);

        let mut engine = StochasticResonanceEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(eval > 0.2, "Resonancia estocástica debe amplificar la señal sub-umbral");
        assert!(eval <= 1.0);
    }
}


