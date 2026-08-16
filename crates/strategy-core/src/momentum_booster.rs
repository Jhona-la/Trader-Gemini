use std::f64;

/// 🌊 MOTOR DE EXTENSIÓN DINÁMICA DE MOMENTO (VOLATILE MOMENTUM BOOSTER)
/// Expande dinámicamente el Take Profit cuando el impulso institucional continúa acelerando.
/// Maximiza los recorridos ganadores (+1.20% a +3.50%) en lugar de salidas prematuras.
#[derive(Debug, Clone, Copy, Default)]
pub struct VolatileMomentumBooster;

impl VolatileMomentumBooster {
    /// Infiere el multiplicador de extensión de Take Profit en nanosegundos (O(1) Continuous Math)
    /// Elimina ramificaciones de fuerza bruta (if/else, clamp) mediante activaciones tensoriales suaves (tanh, max).
    #[inline(always)]
    pub fn calculate_tp_extension(
        position_direction: f64, // +1.0 (Long) o -1.0 (Short)
        raw_pnl_pct: f64,
        hawkes_ratio: f64,
        atr_pct: f64,
        arena: &quantum_arena::GlobalArena,
    ) -> f64 {
        use std::sync::atomic::Ordering;
        // Filtrado suave de PnL negativo: (pnl + |pnl|) / 2 = 0 si es negativo, pnl si positivo. (Max es equivalente O(1) en FPU)
        let positive_pnl = raw_pnl_pct.max(0.0);
        
        // Alineación de momentum continuo: (+) si el flujo va a favor de la posición, (-) si va en contra
        let momentum_alignment = hawkes_ratio * position_direction;
        
        // Función de activación sigmoidea escalar para determinar qué tan alineado está el mercado
        let alignment_threshold = arena.config.dynamic_ofi_threshold.load(Ordering::Relaxed);
        let alignment_activation = ((momentum_alignment - alignment_threshold) * 10.0).tanh().max(0.0);
        
        // Cálculo del peso base de la extensión
        let hw = arena.config.tensor_poly_a.load(Ordering::Relaxed);
        let pw = arena.config.tensor_poly_b.load(Ordering::Relaxed);
        let base_weight = hawkes_ratio.abs() * hw + (positive_pnl / atr_pct.max(0.001)) * pw;
        
        // Expansión suave de Take Profit hasta max_boost (asintótico)
        let max_boost = arena.config.explosive_leverage_multiplier.load(Ordering::Relaxed);
        let boost_factor = 1.0 + (alignment_activation * base_weight).tanh() * (max_boost - 1.0);
        
        boost_factor
    }
}
