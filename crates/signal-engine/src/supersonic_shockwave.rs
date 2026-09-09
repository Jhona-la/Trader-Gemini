use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// ⚡ ALGORITMO #82: MOTOR DE ONDA DE CHOQUE SUPERSÓNICA Y NÚMERO DE MACH (SUPERSONIC SHOCKWAVE ENGINE)
/// Modela el flujo de órdenes como una onda de presión supersónica, calculando el número de Mach M = v_flujo / v_sonido,
/// identificando la ignición explosiva de rupturas de volatilidad cuando M > 1.0.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct SupersonicShockwaveEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for SupersonicShockwaveEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupersonicShockwaveEngine").finish()
    }
}

impl SupersonicShockwaveEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula el número de Mach de mercado M en O(1) con protección contra división por cero
    #[inline(always)]
    pub fn compute_mach_number(order_flow_speed: f64, spread_speed_of_sound: f64) -> f64 {
        // FIX #644: Sanitizar parámetros de Mach
        let safe_speed = if order_flow_speed.is_finite() {
            order_flow_speed.abs()
        } else {
            0.0
        };
        let safe_sound = if spread_speed_of_sound.is_finite() && spread_speed_of_sound > 0.0 {
            spread_speed_of_sound.max(1e-6)
        } else {
            0.001
        };
        safe_speed / safe_sound
    }

    /// Evalúa la compresión de salto de Rankine-Hugoniot en el frente de onda
    #[inline(always)]
    pub fn compute_shockwave_jump(mach: f64) -> f64 {
        if mach.is_finite() && mach > 1.0 {
            ((mach * mach - 1.0) / (mach * mach + 1.0))
                .tanh()
                .clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl QuantumStrategy for SupersonicShockwaveEngine {
    fn name(&self) -> &str {
        "SupersonicShockwaveEngine"
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
        let speed = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "order_flow_speed",
                "SupersonicShockwaveEngine",
            )
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_velocity",
                    "SupersonicShockwaveEngine",
                )
            })
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "price_velocity",
                    "SupersonicShockwaveEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let sound = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "spread_speed_of_sound",
                "SupersonicShockwaveEngine",
            )
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "atr_pct",
                    "SupersonicShockwaveEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.001);

        if !speed.is_finite() || !sound.is_finite() {
            return 0.0;
        }

        let mid_price = registry
            .get_scoped_parameter(sym_opt, cid_opt, "mid_price", "SupersonicShockwaveEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        // D-349: Homogeneizar dimensionalmente velocidad y velocidad del sonido respecto al precio nominal
        let speed_norm = if mid_price > 1.0 && speed.abs() > 1.0 {
            speed / mid_price
        } else {
            speed
        };
        let sound_norm = if mid_price > 1.0 && sound > 1.0 {
            sound / mid_price
        } else {
            sound
        };

        let mach = Self::compute_mach_number(speed_norm.abs(), sound_norm);
        let jump = Self::compute_shockwave_jump(mach);
        speed.signum() * jump
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Scalp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mach_number_and_shockwave_jump() {
        let mach = SupersonicShockwaveEngine::compute_mach_number(2.0, 1.0);
        assert_eq!(mach, 2.0);

        let jump = SupersonicShockwaveEngine::compute_shockwave_jump(mach);
        assert!(jump > 0.0);

        let sub_mach = SupersonicShockwaveEngine::compute_mach_number(0.5, 1.0);
        let sub_jump = SupersonicShockwaveEngine::compute_shockwave_jump(sub_mach);
        assert_eq!(sub_jump, 0.0);
    }

    #[test]
    fn test_mach_number_nan_and_infinite_immunity() {
        let mach_nan = SupersonicShockwaveEngine::compute_mach_number(f64::NAN, 1.0);
        assert!(mach_nan.is_finite() && mach_nan == 0.0);

        let jump_nan = SupersonicShockwaveEngine::compute_shockwave_jump(f64::NAN);
        assert_eq!(jump_nan, 0.0);

        let mach_zero_sound = SupersonicShockwaveEngine::compute_mach_number(10.0, 0.0);
        assert!(mach_zero_sound.is_finite() && mach_zero_sound > 0.0);
    }

    #[test]
    fn test_supersonic_shockwave_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("order_flow_speed", 5.0);
        registry.set("spread_speed_of_sound", 1.0);

        let mut engine = SupersonicShockwaveEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(
            eval > 0.0,
            "Velocidad supersónica positiva debe generar señal de choque positiva"
        );
        assert!(eval <= 1.0);
    }
}
