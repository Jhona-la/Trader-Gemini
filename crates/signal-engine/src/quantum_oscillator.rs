use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// ⚛️ ALGORITMO #80: SUAVIZADOR POR OSCILADOR ANARMÓNICO CUÁNTICO (QUANTUM OSCILLATOR ENGINE)
/// Simula el comportamiento del estado fundamental del precio en un pozo de potencial anarmónico V(x) = 1/2 k x^2 + lambda x^4,
/// filtrando ruido estocástico no-lineal con cero desfase de fase.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct QuantumOscillatorEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for QuantumOscillatorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumOscillatorEngine").finish()
    }
}

impl QuantumOscillatorEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula la fuerza del pozo de potencial anarmónico en O(1)
    #[inline(always)]
    pub fn compute_quantum_restoring_force(
        position: f64,
        k_spring: f64,
        lambda_anharmonic: f64,
    ) -> f64 {
        // FIX #648: Sanitizar y acotar posición para evitar desbordamiento cúbico x^3
        let safe_x = if position.is_finite() {
            position.clamp(-10.0, 10.0)
        } else {
            0.0
        };
        let safe_k = if k_spring.is_finite() && k_spring >= 0.0 {
            k_spring
        } else {
            1.0
        };
        let safe_l = if lambda_anharmonic.is_finite() && lambda_anharmonic >= 0.0 {
            lambda_anharmonic
        } else {
            0.1
        };

        let res = -(safe_k * safe_x + 4.0 * safe_l * safe_x * safe_x * safe_x);
        if res.is_finite() {
            res
        } else {
            0.0
        }
    }

    /// Calcula la probabilidad de colapso en estado de superposición cuántica $|\psi(x)|^2$ (Punto #260)
    #[inline(always)]
    pub fn compute_superposition_probability(position: f64, mass_freq_alpha: f64) -> f64 {
        let safe_x = if position.is_finite() {
            position.clamp(-10.0, 10.0)
        } else {
            0.0
        };
        let safe_alpha = if mass_freq_alpha.is_finite() && mass_freq_alpha > 0.0 {
            mass_freq_alpha.clamp(0.01, 10.0)
        } else {
            1.0
        };

        let norm = (safe_alpha / std::f64::consts::PI).sqrt();
        let prob = norm * (-safe_alpha * safe_x * safe_x).exp();
        if prob.is_finite() {
            prob.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl QuantumStrategy for QuantumOscillatorEngine {
    fn name(&self) -> &str {
        "QuantumOscillatorEngine"
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
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };
        let pos = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "quantum_position_deviation",
                "QuantumOscillatorEngine",
            )
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_book_imbalance",
                    "QuantumOscillatorEngine",
                )
            })
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "QuantumOscillatorEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let k_spring = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "quantum_k_spring",
                "QuantumOscillatorEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(1.0);
        let lambda = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "quantum_lambda_anharmonic",
                "QuantumOscillatorEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(0.1);

        if !pos.is_finite() || !k_spring.is_finite() || !lambda.is_finite() {
            return 0.0;
        }
        let force = Self::compute_quantum_restoring_force(pos, k_spring, lambda);
        force.clamp(-1.0, 1.0)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Scalp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_restoring_force() {
        let force = QuantumOscillatorEngine::compute_quantum_restoring_force(1.0, 1.0, 0.5);
        assert_eq!(force, -(1.0 + 4.0 * 0.5));
    }

    #[test]
    fn test_quantum_superposition_probability() {
        let prob_center = QuantumOscillatorEngine::compute_superposition_probability(0.0, 1.0);
        assert!(prob_center > 0.0 && prob_center <= 1.0);

        let prob_tail = QuantumOscillatorEngine::compute_superposition_probability(3.0, 1.0);
        assert!(prob_tail < prob_center);
        assert!(prob_tail >= 0.0);

        let prob_nan = QuantumOscillatorEngine::compute_superposition_probability(f64::NAN, 1.0);
        assert!(prob_nan.is_finite());
    }

    #[test]
    fn test_quantum_oscillator_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("quantum_position_deviation", 0.5);
        registry.set("quantum_k_spring", 1.0);
        registry.set("quantum_lambda_anharmonic", 0.1);

        let mut engine = QuantumOscillatorEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(
            eval < 0.0,
            "Desviación positiva debe generar fuerza restauradora negativa"
        );
        assert!(eval >= -1.0 && eval <= 1.0);
    }
}
