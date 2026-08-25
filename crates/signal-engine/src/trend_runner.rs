use std::f64;
use strategy_core::QuantumStrategy;
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;

/// 🚀 ALGORITMO #27: MOTOR DE EXPANSIÓN DE TENDENCIA DE ALTA GANANCIA (HIGH-PAYOFF TREND-RUNNER)
/// Amplifica los objetivos de Take-Profit en Swing (+3.5% a +8.0%) cuando se confirma inercia estocástica.
/// Al lograr un payoff de +500 bps frente a 4.5 bps de fee, el usuario retiene más del 99.1% de la ganancia.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct HighPayoffTrendRunner {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for HighPayoffTrendRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HighPayoffTrendRunner").finish()
    }
}

impl HighPayoffTrendRunner {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula el objetivo de extensión de ganancia alta usando funciones continuas tensoriales (O(1))
    #[inline(always)]
    pub fn calculate_expanded_tp(
        base_tp_magnitude: f64,
        hurst: f64,
        vpin: f64,
        atr_pct: f64,
        upper_bound: f64, // Extracted from SuperGenotype
    ) -> f64 {
        // FIX #641: Sanitizar finitud de parámetros entrantes
        if !base_tp_magnitude.is_finite() || !hurst.is_finite() || !vpin.is_finite() || !atr_pct.is_finite() || !upper_bound.is_finite() {
            return 0.02;
        }

        let safe_base_tp = base_tp_magnitude.clamp(0.001, 0.50);
        let safe_hurst = hurst.clamp(0.0, 1.0);
        let safe_vpin = vpin.clamp(0.0, 1.0);
        let safe_atr = atr_pct.clamp(0.0001, 0.50);

        // Activación continua: Hurst (>0.5 indica persistencia de tendencia)
        let trend_score = ((safe_hurst - 0.5) * 4.0).tanh().max(0.0);
        
        // VPIN (Volume-Synchronized Probability of Toxicity): Ante alta toxicidad institucional (vpin > 0.50),
        // reducimos la dilatación del TP para capturar ganancias rápidamente antes de la reversión por selección adversa.
        let toxicity_penalty = (1.0 - (safe_vpin * 1.5).tanh()).max(0.2);

        // Factor de expansión tensorial calibrado
        let expansion_factor = trend_score * toxicity_penalty;

        // Cálculo del TP expandido, integrando directamente la volatilidad (atr_pct)
        let expanded_tp = safe_base_tp * (1.0 + (expansion_factor * 3.0));
        let vol_boost = safe_atr * expansion_factor * 4.0;

        let final_tp = (expanded_tp + vol_boost).max(safe_base_tp.max(0.0020));

        // Límite superior suave continuo usando tanh() en lugar de un clamp() brusco.
        // Asymptotically approaches upper_bound sin romper derivabilidad.
        let safe_bound = upper_bound.clamp(0.01, 0.50); // Prevent div by zero
        let res = safe_bound * (final_tp / safe_bound).tanh();
        if res.is_finite() { res } else { 0.02 }
    }
}

impl QuantumStrategy for HighPayoffTrendRunner {
    fn name(&self) -> &str {
        "HighPayoffTrendRunner"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Swing
    }

    fn evaluate(&self) -> f64 {
        let hurst = self.registry.as_ref()
            .and_then(|r| {
                r.get("hurst_exponent", "HighPayoffTrendRunner")
                    .or_else(|| r.get("global_hurst", "HighPayoffTrendRunner"))
                    .or_else(|| r.get("hurst", "HighPayoffTrendRunner"))
            })
            .map(|p| p.get_value())
            .unwrap_or(0.50);

        let vpin = self.registry.as_ref()
            .and_then(|r| {
                r.get("cvpin", "HighPayoffTrendRunner")
                    .or_else(|| r.get("order_flow_vpin", "HighPayoffTrendRunner"))
                    .or_else(|| r.get("vpin", "HighPayoffTrendRunner"))
            })
            .map(|p| p.get_value())
            .unwrap_or(0.50);

        let atr_pct = self.registry.as_ref()
            .and_then(|r| r.get("atr_pct", "HighPayoffTrendRunner").or_else(|| r.get("relative_atr_pct", "HighPayoffTrendRunner")))
            .map(|p| p.get_value())
            .unwrap_or(0.01);

        let trend_direction = self.registry.as_ref()
            .and_then(|r| {
                r.get("trend_direction", "HighPayoffTrendRunner")
                    .or_else(|| r.get("ema_trend_swing", "HighPayoffTrendRunner"))
                    .or_else(|| r.get("ema_trend", "HighPayoffTrendRunner"))
                    .or_else(|| r.get("price_velocity", "HighPayoffTrendRunner"))
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        // FIX #684: Sanitizar lecturas de registros
        let safe_hurst = if hurst.is_finite() { hurst } else { 0.50 };
        let safe_vpin = if vpin.is_finite() { vpin } else { 0.50 };
        let safe_atr = if atr_pct.is_finite() && atr_pct > 0.0 { atr_pct } else { 0.01 };
        let safe_dir = if trend_direction.is_finite() { trend_direction } else { 0.0 };

        // Si no hay persistencia de tendencia (Hurst <= 0.52) o el flujo es neutro, convicción cero
        if safe_hurst <= 0.52 || safe_dir.abs() < 1e-4 {
            return 0.0;
        }

        let tp = Self::calculate_expanded_tp(0.02, safe_hurst, safe_vpin, safe_atr, 0.08);
        // FIX #406: La señal debe portar el signo de la dirección del flujo de mercado
        safe_dir.signum() * (tp * 10.0).tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expanded_tp_calculation() {
        let tp = HighPayoffTrendRunner::calculate_expanded_tp(0.02, 0.70, 0.5, 0.01, 0.10);
        assert!(tp >= 0.02, "TP expandido debe ser mayor o igual al base");
        assert!(tp <= 0.10, "TP expandido no debe superar upper_bound");
    }

    #[test]
    fn test_trend_runner_nan_immunity() {
        let tp_nan = HighPayoffTrendRunner::calculate_expanded_tp(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        assert!(tp_nan.is_finite() && tp_nan > 0.0);
    }

    #[test]
    fn test_trend_runner_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("hurst_exponent", 0.75);
        registry.set("cvpin", 0.20);
        registry.set("atr_pct", 0.02);
        registry.set("trend_direction", 1.0);

        let mut runner = HighPayoffTrendRunner::new();
        assert!(runner.init(registry).is_ok());

        let eval = runner.evaluate();
        assert!(eval > 0.0, "Hurst alto y dirección alcista deben generar señal positiva");
        assert!(eval <= 1.0);
    }
}


