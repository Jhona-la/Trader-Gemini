/// 🌊 ALGORITMO #190: OPTIMIZADOR MACRO-REGIME PARA SWING TRADING (MACRO-REGIME SWING OPTIMIZER)
/// Escanea las características Macro del mercado (Tendencia vs Rango) derivadas de Hurst Exponent, Volatilidad (ATR)
/// y Cumulative Volume Delta (CVD) para ajustar dinámicamente los parámetros de Swing Trade.

#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct MacroRegimeSwingOptimizer;

pub struct SwingRegimeParams {
    pub cooldown_ticks: usize,
    pub is_active: bool,
    pub leverage_modifier: f64,
}

impl MacroRegimeSwingOptimizer {
    /// Evalúa el régimen del mercado y devuelve los parámetros ajustados para Swing Trading.
    ///
    /// * `hurst_exponent`: Exponente de Hurst (0.5 = random, >0.5 = tendencia, <0.5 = reversión)
    /// * `atr_pct`: Rango Verdadero Promedio en porcentaje
    /// * `cvd_imbalance`: Desequilibrio del Flujo de Volumen Acumulado (-1.0 a 1.0)
    pub fn evaluate_swing_regime(
        hurst_exponent: f64,
        atr_pct: f64,
        cvd_imbalance: f64,
        regime_duration_ms: f64,
        arena: &quantum_arena::GlobalArena,
    ) -> SwingRegimeParams {
        // FIX #655: Sanitizar tensores macro entrantes
        let safe_hurst = if hurst_exponent.is_finite() { hurst_exponent } else { 0.5 };
        let safe_atr = if atr_pct.is_finite() && atr_pct >= 0.0 { atr_pct } else { 0.01 };
        let safe_cvd = if cvd_imbalance.is_finite() { cvd_imbalance } else { 0.0 };
        let safe_regime_ms = if regime_duration_ms.is_finite() && regime_duration_ms >= 0.0 { regime_duration_ms } else { 60_000.0 };

        let base_cooldown_f64 = (safe_regime_ms / 60_000.0 * 15.0).clamp(20.0, 300.0);
        
        let c_offset = arena.config.macro_hurst_confidence_offset.load(std::sync::atomic::Ordering::Relaxed);
        let c_scale = arena.config.macro_hurst_confidence_scale.load(std::sync::atomic::Ordering::Relaxed);
        let v_scale = arena.config.macro_vol_confidence_scale.load(std::sync::atomic::Ordering::Relaxed);
        let min_cool_r = arena.config.macro_min_cooldown_ratio.load(std::sync::atomic::Ordering::Relaxed);
        let max_cool_r = arena.config.macro_max_cooldown_ratio.load(std::sync::atomic::Ordering::Relaxed);
        let cool_red_f = arena.config.macro_cooldown_reduction_factor.load(std::sync::atomic::Ordering::Relaxed);
        let lev_mom_s = arena.config.macro_leverage_momentum_scale.load(std::sync::atomic::Ordering::Relaxed);
        
        let hurst_confidence = ((safe_hurst - c_offset) * c_scale).tanh().clamp(0.0, 1.0);
        let momentum_confidence = safe_cvd.abs().clamp(0.0, 1.0);
        
        let vol_confidence = (safe_atr * v_scale).clamp(0.0, 1.0);
        
        let combined_confidence = hurst_confidence * vol_confidence;
        
        let min_cooldown = (base_cooldown_f64 * min_cool_r).max(10.0);
        let max_cooldown = base_cooldown_f64 * max_cool_r;
        
        let cooldown_reduction = combined_confidence * momentum_confidence * (base_cooldown_f64 * cool_red_f);
        let base_cooldown_dynamic = base_cooldown_f64 + (1.0 - combined_confidence) * max_cooldown;
        // FIX #598: Acotar cooldown final para evitar desbordamientos o subdesbordamientos
        let raw_cooldown = base_cooldown_dynamic - cooldown_reduction;
        let final_cooldown = if raw_cooldown.is_finite() { raw_cooldown.clamp(10.0, 1000.0) as usize } else { 60 };

        let is_active = combined_confidence > 0.05;
        // FIX #617: Si el régimen no está activo, el multiplicador de leverage Swing debe ser estrictamente 0.0
        let leverage_modifier = if is_active {
            let raw_mod = combined_confidence * (1.0 + momentum_confidence * lev_mom_s);
            if raw_mod.is_finite() { raw_mod.clamp(0.1, 5.0) } else { 0.0 }
        } else {
            0.0
        };
        
        SwingRegimeParams {
            cooldown_ticks: final_cooldown.max(min_cooldown.clamp(5.0, 500.0) as usize),
            is_active,
            leverage_modifier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_regime_swing_optimizer_trending_active() {
        let arena = quantum_arena::GlobalArena::new(13.0);
        let params = MacroRegimeSwingOptimizer::evaluate_swing_regime(0.75, 0.03, 0.5, 120_000.0, &arena);

        assert!(params.is_active);
        assert!(params.leverage_modifier > 0.0);
        assert!(params.cooldown_ticks >= 10);
    }

    #[test]
    fn test_macro_regime_swing_optimizer_random_walk_inactive() {
        let arena = quantum_arena::GlobalArena::new(13.0);
        // Hurst = 0.5 (random walk), ATR = 0.0001 (sin volatilidad)
        let params = MacroRegimeSwingOptimizer::evaluate_swing_regime(0.50, 0.0001, 0.0, 60_000.0, &arena);

        assert!(!params.is_active);
        assert_eq!(params.leverage_modifier, 0.0);
    }

    #[test]
    fn test_macro_regime_swing_optimizer_nan_immunity() {
        let arena = quantum_arena::GlobalArena::new(13.0);
        let params = MacroRegimeSwingOptimizer::evaluate_swing_regime(f64::NAN, f64::NAN, f64::NAN, f64::NAN, &arena);

        assert!(params.leverage_modifier.is_finite());
        assert!(params.cooldown_ticks >= 10);
    }
}

