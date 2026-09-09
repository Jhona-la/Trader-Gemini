use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// 🧠 ALGORITMO #96: MOTOR DE COMPUERTA PERCEPTRÓN HEBBIANA ADAPTATIVA (PERCEPTRON GATE ENGINE)
/// Perceptrón binario ultra-rápido de ciclo único de CPU con regla de actualización Hebbiana adaptativa,
/// regulando el paso atómico de señales de compra/venta en función de los PnL recientes.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct PerceptronGateEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for PerceptronGateEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerceptronGateEngine").finish()
    }
}

impl PerceptronGateEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Inferencia del tensor sin actualizar el peso (para el Engine Core), simétrica para Long y Short
    #[inline(always)]
    pub fn infer(signal_score: f64, weight: f64) -> f64 {
        if !signal_score.is_finite() || !weight.is_finite() || signal_score == 0.0 {
            return 0.0;
        }
        let abs_score = signal_score.abs();
        let activation = abs_score * weight;
        // Exploración mínima (0.15) para evitar bloqueo cognitivo permanente tras pérdidas
        let gate_strength = ((activation - 0.5) * 5.0).tanh().clamp(0.15, 1.0);
        signal_score.signum() * gate_strength
    }

    /// Aprendizaje Hebbiano Adaptativo V2 (Fase 21)
    /// Incorpora la varianza/volatilidad para escalar la tasa de aprendizaje.
    #[inline(always)]
    pub fn update_weight(weight: &mut f64, recent_pnl: f64, std_dev: f64) {
        // FIX #646: Sanitizar entradas para asegurar estabilidad del perceptrón
        let safe_std = if std_dev.is_finite() && std_dev >= 0.0 {
            std_dev
        } else {
            0.01
        };
        let current_w = if weight.is_finite() { *weight } else { 1.0 };

        // Tasa de aprendizaje base (0.05) modulada inversamente por el riesgo (std_dev).
        // A mayor volatilidad/incertidumbre (alto std_dev), menor es el paso de mutación.
        let learning_rate = (0.05 / (1.0 + safe_std * 100.0)).clamp(0.001, 0.05);

        if !weight.is_finite() {
            *weight = 1.0;
        }

        if recent_pnl.is_finite() {
            if recent_pnl > 0.0 {
                *weight = (current_w + learning_rate).clamp(0.25, 2.0);
            } else if recent_pnl < 0.0 {
                *weight = (current_w - learning_rate).clamp(0.25, 2.0);
            } else {
                *weight = current_w.clamp(0.25, 2.0);
            }
        }
    }
}

impl QuantumStrategy for PerceptronGateEngine {
    fn name(&self) -> &str {
        "PerceptronGateEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() {
            None
        } else {
            Some(symbol)
        };
        let cid_opt = if symbol.is_empty() {
            None
        } else {
            Some(coin_id)
        };
        let registry = match self.registry.as_ref() {
            Some(r) => r,
            None => return 0.0,
        };
        let signal_score = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "perceptron_candidate_signal",
                "PerceptronGateEngine",
            )
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "PerceptronGateEngine",
                )
            })
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "alpha_signal",
                    "PerceptronGateEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let weight = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "perceptron_hebbian_weight",
                "PerceptronGateEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(1.0);
        if !signal_score.is_finite() || !weight.is_finite() {
            return 0.0;
        }
        Self::infer(signal_score, weight)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Scalp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron_gate_symmetric_long_and_short() {
        let long_out = PerceptronGateEngine::infer(0.8, 1.0);
        assert!(long_out > 0.0);

        let short_out = PerceptronGateEngine::infer(-0.8, 1.0);
        assert!(short_out < 0.0);
        assert!(
            (long_out + short_out).abs() < 1e-6,
            "Debe ser perfectamente simétrico"
        );
    }

    #[test]
    fn test_perceptron_gate_hebbian_update_and_nan_immunity() {
        let mut weight = 1.0;
        PerceptronGateEngine::update_weight(&mut weight, 50.0, 0.01);
        assert!(weight > 1.0);

        PerceptronGateEngine::update_weight(&mut weight, -50.0, 0.01);
        assert!(weight <= 1.05);

        // NaN immunity
        let mut nan_w = f64::NAN;
        PerceptronGateEngine::update_weight(&mut nan_w, f64::NAN, f64::NAN);
        assert!(nan_w.is_finite());
        assert!((0.1..=2.0).contains(&nan_w));
    }

    #[test]
    fn test_perceptron_gate_evaluate_with_registry() {
        let mut engine = PerceptronGateEngine::new();
        assert_eq!(engine.evaluate(), 0.0);

        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("perceptron_candidate_signal", 0.9);
        registry.set("perceptron_hebbian_weight", 1.2);

        engine.init(registry).expect("init should succeed");
        let signal = engine.evaluate();
        assert!(signal > 0.0);
        assert!((-1.0..=1.0).contains(&signal));
    }
}
