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
        let base_cooldown_f64 = regime_duration_ms / 10.0;
        
        let c_offset = arena.config.macro_hurst_confidence_offset.load(std::sync::atomic::Ordering::Relaxed);
        let c_scale = arena.config.macro_hurst_confidence_scale.load(std::sync::atomic::Ordering::Relaxed);
        let v_scale = arena.config.macro_vol_confidence_scale.load(std::sync::atomic::Ordering::Relaxed);
        let min_cool_r = arena.config.macro_min_cooldown_ratio.load(std::sync::atomic::Ordering::Relaxed);
        let max_cool_r = arena.config.macro_max_cooldown_ratio.load(std::sync::atomic::Ordering::Relaxed);
        let cool_red_f = arena.config.macro_cooldown_reduction_factor.load(std::sync::atomic::Ordering::Relaxed);
        let lev_mom_s = arena.config.macro_leverage_momentum_scale.load(std::sync::atomic::Ordering::Relaxed);
        
        let hurst_confidence = ((hurst_exponent - c_offset) * c_scale).tanh().clamp(0.0, 1.0);
        let momentum_confidence = cvd_imbalance.abs().clamp(0.0, 1.0);
        
        let vol_confidence = (atr_pct * v_scale).clamp(0.0, 1.0);
        
        let combined_confidence = hurst_confidence * vol_confidence;
        
        let min_cooldown = (base_cooldown_f64 * min_cool_r).max(10.0);
        let max_cooldown = base_cooldown_f64 * max_cool_r;
        
        let cooldown_reduction = combined_confidence * momentum_confidence * (base_cooldown_f64 * cool_red_f);
        let base_cooldown_dynamic = base_cooldown_f64 + (1.0 - combined_confidence) * max_cooldown;
        let final_cooldown = (base_cooldown_dynamic - cooldown_reduction) as usize;

        let leverage_modifier = combined_confidence * (1.0 + momentum_confidence * lev_mom_s);
        
        SwingRegimeParams {
            cooldown_ticks: final_cooldown.max(min_cooldown as usize),
            is_active: true,
            leverage_modifier,
        }
    }
}
