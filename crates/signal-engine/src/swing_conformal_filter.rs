use strategy_core::QuantumStrategy;
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;

/// 🌊 ALGORITMO #67: FILTRO CONFORMAL DE SWING Y CONFLUENCIA DE TENDENCIA (SWING CONFORMAL FILTER ENGINE)
/// Filtra las entradas de Swing exigiendo la confluencia entre la reversión VECM y la alineación
/// de tendencia macro EMA, elevando el Win Rate de Swing hacia el objetivo superior del 75%-85%.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct SwingConformalFilterEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for SwingConformalFilterEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwingConformalFilterEngine").finish()
    }
}

impl SwingConformalFilterEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Evalúa si la oportunidad de Swing posee confluencia multimarco verificada
    #[inline(always)]
    pub fn is_swing_confluence_valid(
        arena: &quantum_arena::GlobalArena,
        vecm_zscore: f64,
        ema_trend: f64,
        conformal_p_value: f64,
        is_long: bool,
    ) -> bool {
        // FIX #650: Sanitizar finitud de entradas
        if !vecm_zscore.is_finite() || !ema_trend.is_finite() || !conformal_p_value.is_finite() {
            return false;
        }

        use std::sync::atomic::Ordering;
        let conformal_alpha = arena.config.conformal_alpha.load(Ordering::Relaxed);
        let vecm_threshold = arena.config.vecm_beta_hedge.load(Ordering::Relaxed);

        // P-Value must be high enough (1 - alpha approx)
        let conformal_ok = conformal_p_value >= (1.0 - conformal_alpha);
        let trend_ok = if is_long {
            ema_trend > 0.0
        } else {
            ema_trend < 0.0
        };
        // Umbral crítico normalizado para reversión a la media VECM (Z-score en [1.50, 3.00] desviaciones estándar)
        let vecm_z_threshold = (1.50 + vecm_threshold * 0.50).clamp(1.5, 3.0);
        let vecm_ok = if is_long {
            vecm_zscore <= -vecm_z_threshold
        } else {
            vecm_zscore >= vecm_z_threshold
        };
        conformal_ok && trend_ok && vecm_ok
    }
}

impl QuantumStrategy for SwingConformalFilterEngine {
    fn name(&self) -> &str {
        "SwingConformalFilterEngine"
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
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };
        let vecm_zscore = r
            .get_scoped_parameter(sym_opt, cid_opt, "vecm_zscore", "SwingConformalFilterEngine")
            .or_else(|| r.get_scoped_parameter(sym_opt, cid_opt, "cointegration_zscore", "SwingConformalFilterEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let ema_trend = r
            .get_scoped_parameter(sym_opt, cid_opt, "ema_trend_swing", "SwingConformalFilterEngine")
            .or_else(|| r.get_scoped_parameter(sym_opt, cid_opt, "trend_direction", "SwingConformalFilterEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let conformal_p = r
            .get_scoped_parameter(sym_opt, cid_opt, "conformal_p_value", "SwingConformalFilterEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.95);
        let conformal_alpha = r
            .get_scoped_parameter(sym_opt, cid_opt, "conformal_alpha", "SwingConformalFilterEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.10);
        let safe_p = if conformal_p.is_finite() { conformal_p } else { 0.95 };
        let safe_alpha = if conformal_alpha.is_finite() { conformal_alpha } else { 0.10 };
        let min_conformal_p = (1.0 - safe_alpha).clamp(0.50, 0.99);

        if !vecm_zscore.is_finite() || !ema_trend.is_finite() {
            return 0.0;
        }

        // FIX #789: Umbral conformal evolutivo guiado por conformal_alpha
        if safe_p >= min_conformal_p && vecm_zscore.abs() >= 1.5 {
            if vecm_zscore <= -1.5 && ema_trend >= 0.0 {
                // Reversión alcista confluente con tendencia
                ((-vecm_zscore - 1.0) * conformal_p * 0.5).clamp(0.0, 1.0)
            } else if vecm_zscore >= 1.5 && ema_trend <= 0.0 {
                // Reversión bajista confluente con tendencia
                (-(vecm_zscore - 1.0) * conformal_p * 0.5).clamp(-1.0, 0.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Swing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swing_conformal_filter() {
        let arena = quantum_arena::GlobalArena::new(13.0);
        // Confluencia Long: vecm_zscore bajo (-2.0), ema_trend > 0, conformal_p_value alto
        let is_valid = SwingConformalFilterEngine::is_swing_confluence_valid(
            &arena, -2.0, 1.0, 0.99, true
        );
        assert!(is_valid);
    }

    #[test]
    fn test_swing_conformal_filter_short_and_rejection_cases() {
        let arena = quantum_arena::GlobalArena::new(13.0);
        // Confluencia Short: vecm_zscore alto (2.0), ema_trend < 0, conformal_p_value alto
        let is_valid_short = SwingConformalFilterEngine::is_swing_confluence_valid(
            &arena, 2.0, -1.0, 0.99, false
        );
        assert!(is_valid_short);

        // Rechazo por tendencia opuesta
        let is_rejected = SwingConformalFilterEngine::is_swing_confluence_valid(
            &arena, -2.0, -1.0, 0.99, true
        );
        assert!(!is_rejected);
    }

    #[test]
    fn test_swing_conformal_filter_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("vecm_zscore", -2.5);
        registry.set("ema_trend_swing", 1.0);
        registry.set("conformal_p_value", 0.95);
        registry.set("conformal_alpha", 0.10);

        let mut engine = SwingConformalFilterEngine::new();
        assert!(engine.init(registry).is_ok());
        assert_eq!(engine.name(), "SwingConformalFilterEngine");
        assert_eq!(engine.horizon(), strategy_core::TradeHorizon::Swing);

        let score = engine.evaluate();
        assert!(score.is_finite());
        assert!(score > 0.0);
    }
}
