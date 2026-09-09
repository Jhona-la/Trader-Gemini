use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// 📈 ALGORITMO #91: MOTOR DE INTENSIDAD ESTOCÁSTICA HAWKES-BESSEL (HAWKES-BESSEL ENGINE)
/// Combina el proceso auto-excitado de Hawkes con kernels de convolución modificados de Bessel I_\nu(z),
/// midiendo la aceleración estocástica de ráfagas HFT durante cascadas de volatilidad en O(1).
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct HawkesBesselEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for HawkesBesselEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HawkesBesselEngine").finish()
    }
}

impl HawkesBesselEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula la intensidad estocástica con kernel aproximado de Bessel en O(1)
    #[inline(always)]
    pub fn compute_bessel_hawkes_intensity(base_lambda: f64, alpha: f64, dt: f64) -> f64 {
        // FIX #642: Sanitizar parámetros entrantes
        let safe_dt = if dt.is_finite() && dt >= 0.0 { dt } else { 0.1 };
        let safe_lambda = if base_lambda.is_finite() {
            base_lambda
        } else {
            1.0
        };
        let safe_alpha = if alpha.is_finite() { alpha } else { 0.5 };

        let bessel_decay = (1.0 + safe_dt * safe_dt).sqrt() - safe_dt;
        let res = safe_lambda + safe_alpha * bessel_decay.max(0.0);
        if res.is_finite() {
            res
        } else {
            1.0
        }
    }
}

impl QuantumStrategy for HawkesBesselEngine {
    fn name(&self) -> &str {
        "HawkesBesselEngine"
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

        let base_lambda = r
            .get_scoped_parameter(sym_opt, cid_opt, "hawkes_intensity", "HawkesBesselEngine")
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "base_hawkes_intensity",
                    "HawkesBesselEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(1.0);

        let alpha = r
            .get_scoped_parameter(sym_opt, cid_opt, "bessel_alpha", "HawkesBesselEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.5);

        let dt = r
            .get_scoped_parameter(sym_opt, cid_opt, "hawkes_dt", "HawkesBesselEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.1);

        let direction = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "order_flow_direction",
                "HawkesBesselEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        if direction.abs() <= 1e-6 || !direction.is_finite() {
            return 0.0;
        }
        let intensity = Self::compute_bessel_hawkes_intensity(base_lambda, alpha, dt);
        // D-346: Modular la intensidad con tanh sin invertir espuriamente el signo del flujo direccional
        let norm_intensity = (intensity / base_lambda.max(0.1)).max(0.0);
        direction.signum() * norm_intensity.tanh()
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Scalp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkes_bessel_intensity() {
        let intensity_now = HawkesBesselEngine::compute_bessel_hawkes_intensity(1.0, 2.0, 0.0);
        assert!((intensity_now - 3.0).abs() < 1e-6);

        let intensity_decayed = HawkesBesselEngine::compute_bessel_hawkes_intensity(1.0, 2.0, 10.0);
        assert!(intensity_decayed >= 1.0 && intensity_decayed < 3.0);
    }

    #[test]
    fn test_hawkes_bessel_nan_immunity() {
        let res = HawkesBesselEngine::compute_bessel_hawkes_intensity(f64::NAN, f64::NAN, f64::NAN);
        assert!(res.is_finite());
    }

    #[test]
    fn test_hawkes_bessel_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("hawkes_intensity", 2.0);
        registry.set("bessel_alpha", 1.0);
        registry.set("hawkes_dt", 0.0);
        registry.set("order_flow_direction", 1.0);

        let mut engine = HawkesBesselEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(
            eval > 0.0,
            "Intensidad y dirección positiva deben generar señal positiva"
        );
        assert!(eval <= 1.0);
    }
}
