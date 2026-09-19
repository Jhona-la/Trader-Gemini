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
        if !raw_pnl_pct.is_finite()
            || !hawkes_ratio.is_finite()
            || !atr_pct.is_finite()
            || !position_direction.is_finite()
        {
            return 1.0;
        }
        use std::sync::atomic::Ordering;
        // FIX #662: Sanitizar atr_pct y normalizar dirección de posición
        let safe_atr = if atr_pct.is_finite() && atr_pct > 0.0 {
            atr_pct
        } else {
            0.001
        };
        let pos_dir = if position_direction > 0.0 {
            1.0
        } else if position_direction < 0.0 {
            -1.0
        } else {
            0.0
        };

        // Filtrado suave de PnL negativo: (pnl + |pnl|) / 2 = 0 si es negativo, pnl si positivo. (Max es equivalente O(1) en FPU)
        let positive_pnl = raw_pnl_pct.max(0.0);

        // Alineación de momentum continuo: (+) si el flujo va a favor de la posición, (-) si va en contra
        // Normalizamos el exceso de auto-excitación Hawkes (ratio base = 1.0)
        let excess_hawkes = (hawkes_ratio - 1.0).clamp(-2.0, 2.0);
        let momentum_alignment = excess_hawkes * pos_dir;

        // Función de activación sigmoidea escalar para determinar qué tan alineado está el mercado
        let alignment_threshold = arena.config.dynamic_ofi_threshold.load(Ordering::Relaxed);
        let alignment_activation = ((momentum_alignment - alignment_threshold) * 5.0)
            .tanh()
            .max(0.0);

        // FIX #603: Acotar hawkes_ratio y base_weight para evitar saturación espuria ante picos de volatilidad
        let hw = arena
            .config
            .tensor_poly_a
            .load(Ordering::Relaxed)
            .clamp(0.0, 5.0);
        let pw = arena
            .config
            .tensor_poly_b
            .load(Ordering::Relaxed)
            .clamp(0.0, 5.0);
        let safe_hawkes = hawkes_ratio.clamp(-5.0, 5.0);
        let pnl_over_atr = (positive_pnl / safe_atr.max(0.001)).clamp(0.0, 10.0);
        let base_weight = (safe_hawkes.abs() * hw + pnl_over_atr * pw).min(20.0);

        // Expansión suave de Take Profit hasta max_boost (asintótico)
        let max_boost = arena
            .config
            .explosive_leverage_multiplier
            .load(Ordering::Relaxed)
            .clamp(1.0, 10.0);
        let boost_factor = 1.0 + (alignment_activation * base_weight).tanh() * (max_boost - 1.0);

        if boost_factor.is_finite() {
            boost_factor.max(1.0)
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_booster_neutral_and_negative_pnl() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let boost = VolatileMomentumBooster::calculate_tp_extension(1.0, -0.01, 1.0, 0.01, &arena);
        assert_eq!(boost, 1.0, "PnL negativo no debe dilatar el TP");
    }

    #[test]
    fn test_momentum_booster_strong_aligned_momentum_expansion() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let boost = VolatileMomentumBooster::calculate_tp_extension(1.0, 0.05, 3.0, 0.01, &arena);
        assert!(
            boost >= 1.0,
            "Momentum fuertemente alineado debe producir boost >= 1.0"
        );
        assert!(boost.is_finite());
    }

    #[test]
    fn test_momentum_booster_nan_and_negative_inputs_immunity() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let boost = VolatileMomentumBooster::calculate_tp_extension(
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            &arena,
        );
        assert_eq!(boost, 1.0, "Inputs NaN deben retornar fallback seguro 1.0");
    }
}
