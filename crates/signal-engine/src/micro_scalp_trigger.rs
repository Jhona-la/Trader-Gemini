use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// ⚡ ALGORITMO #66: DISPARADOR ADAPTATIVO DE MICRO-SCALPING HAWKES/OBI (MICRO SCALP TRIGGER ENGINE)
/// Dispara entradas de micro-scalp de alta probabilidad calibrando dinámicamente el umbral a partir del
/// Ratio Hawkes de Auto-Excitación y Orderbook Imbalance (OBI) Z-Score en O(1).
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct MicroScalpTriggerEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for MicroScalpTriggerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicroScalpTriggerEngine").finish()
    }
}

impl MicroScalpTriggerEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Evalúa si las condiciones de micro-scalping de alta precisión están activas
    #[inline(always)]
    pub fn should_trigger_micro_scalp(
        arena: &quantum_arena::GlobalArena,
        hawkes_ratio: f64,
        obi_zscore: f64,
        ml_prob: f64,
        is_long: bool,
    ) -> bool {
        if !hawkes_ratio.is_finite() || !obi_zscore.is_finite() || !ml_prob.is_finite() {
            return false;
        }

        use std::sync::atomic::Ordering;
        let hawkes_thresh = arena.config.hawkes_scalp_threshold.load(Ordering::Relaxed);
        let obi_thresh = arena.config.obi_zscore_threshold.load(Ordering::Relaxed);
        let ml_long_thresh = arena.config.ml_threshold_long.load(Ordering::Relaxed);
        let ml_short_thresh = arena.config.ml_threshold_short.load(Ordering::Relaxed);

        let hawkes_ok = hawkes_ratio >= hawkes_thresh;
        let obi_ok = if is_long {
            obi_zscore >= obi_thresh
        } else {
            obi_zscore <= -obi_thresh
        };
        let ml_ok = if is_long {
            ml_prob >= ml_long_thresh
        } else {
            ml_prob <= ml_short_thresh
        };
        hawkes_ok && obi_ok && ml_ok
    }

    /// Evalúa el disparador de micro-scalping exigiendo cobertura conformal de incertidumbre (Punto #095)
    #[inline(always)]
    pub fn should_trigger_micro_scalp_conformal(
        arena: &quantum_arena::GlobalArena,
        hawkes_ratio: f64,
        obi_zscore: f64,
        ml_prob: f64,
        conformal_p_value: f64,
        is_long: bool,
    ) -> bool {
        if !conformal_p_value.is_finite() || conformal_p_value < 0.70 {
            return false;
        }
        Self::should_trigger_micro_scalp(arena, hawkes_ratio, obi_zscore, ml_prob, is_long)
    }
}

impl QuantumStrategy for MicroScalpTriggerEngine {
    fn name(&self) -> &str {
        "MicroScalpTriggerEngine"
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

        let hawkes = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "hawkes_intensity",
                "MicroScalpTriggerEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(1.0);
        let obi = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "order_book_imbalance",
                "MicroScalpTriggerEngine",
            )
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "orderbook_imbalance",
                    "MicroScalpTriggerEngine",
                )
            })
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "MicroScalpTriggerEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let ml_prob = r
            .get_scoped_parameter(sym_opt, cid_opt, "ml_prob_scalp", "MicroScalpTriggerEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.5);

        // FIX #643: Guarda de finitud estricta en indicadores de entrada
        if !hawkes.is_finite() || !obi.is_finite() || !ml_prob.is_finite() {
            return 0.0;
        }

        if hawkes >= 1.2 && obi.abs() >= 0.2 {
            let is_long = obi > 0.0 && ml_prob >= 0.5;
            let is_short = obi < 0.0 && ml_prob <= 0.5;

            if is_long {
                (obi * (ml_prob - 0.5) * 4.0 * (hawkes / 2.0).min(2.0)).clamp(0.0, 1.0)
            } else if is_short {
                (obi * (0.5 - ml_prob) * 4.0 * (hawkes / 2.0).min(2.0)).clamp(-1.0, 0.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_scalp_trigger() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Condición Long completa: hawkes alto, obi positivo alto, ml_prob alcista
        let should_trigger =
            MicroScalpTriggerEngine::should_trigger_micro_scalp(&arena, 3.0, 2.5, 0.85, true);
        assert!(should_trigger);
    }

    #[test]
    fn test_micro_scalp_trigger_conformal() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Conformal p-value suficiente (0.85 >= 0.70)
        let ok = MicroScalpTriggerEngine::should_trigger_micro_scalp_conformal(
            &arena, 3.0, 2.5, 0.85, 0.85, true,
        );
        assert!(ok);

        // Conformal p-value insuficiente o NaN -> rechazar
        let rejected = MicroScalpTriggerEngine::should_trigger_micro_scalp_conformal(
            &arena, 3.0, 2.5, 0.85, 0.50, true,
        );
        assert!(!rejected);

        let nan_rejected = MicroScalpTriggerEngine::should_trigger_micro_scalp_conformal(
            &arena,
            3.0,
            2.5,
            0.85,
            f64::NAN,
            true,
        );
        assert!(!nan_rejected);
    }

    #[test]
    fn test_micro_scalp_trigger_short_symmetry() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Short con OBI negativo
        let should_trigger_short =
            MicroScalpTriggerEngine::should_trigger_micro_scalp(&arena, 3.0, -2.5, 0.15, false);
        assert!(should_trigger_short);
    }

    #[test]
    fn test_micro_scalp_trigger_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("hawkes_intensity", 2.0);
        registry.set("order_book_imbalance", 0.5);
        registry.set("ml_prob_scalp", 0.85);

        let mut engine = MicroScalpTriggerEngine::new();
        assert!(engine.init(registry).is_ok());
        assert_eq!(engine.name(), "MicroScalpTriggerEngine");
        assert_eq!(engine.horizon(), strategy_core::TradeHorizon::Continuous);

        let val = engine.evaluate();
        assert!(val.is_finite());
        assert!(val > 0.0);
    }
}
