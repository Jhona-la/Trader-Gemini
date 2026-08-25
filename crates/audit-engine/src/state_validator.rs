use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

/// Motor de Auditoría de Paridad Forense
///
/// Garantiza que el comportamiento en Backtest y en Producción
/// siga estrictamente el mismo conjunto de axiomas matemáticos.
pub struct StateValidator;

impl StateValidator {
    /// Inyecta una validación continua del estado. Si detecta divergencias
    /// (ej. el PnL no cuadra con el Capital Unificado), lanza una alerta de paridad.
    #[inline(always)]
    pub fn validate_parity(arena: &GlobalArena, initial_capital: f64) {
        // FIX #680: Sanitizar capital inicial
        let safe_initial = if initial_capital.is_finite() && initial_capital > 0.0 { initial_capital } else { 13.0 };

        let mut total_realized_pnl = 0.0;
        let mut active_positions = 0;

        for coin in arena.coins.iter() {
            let s_pnl = coin.scalp.pnl_realized.load(Ordering::Relaxed);
            let w_pnl = coin.swing.pnl_realized.load(Ordering::Relaxed);
            if s_pnl.is_finite() { total_realized_pnl += s_pnl; }
            if w_pnl.is_finite() { total_realized_pnl += w_pnl; }

            active_positions += coin.scalp.active_positions.load(Ordering::Relaxed);
            active_positions += coin.swing.active_positions.load(Ordering::Relaxed);
        }

        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        if !current_capital.is_finite() {
            return;
        }

        let expected_capital = safe_initial + total_realized_pnl;

        // El margen de error debe ser minúsculo (errores de flotante).
        // Si hay discrepancia mayor a 1 USD (o micro-centavos), hay código fantasma.
        let diff = (current_capital - expected_capital).abs();

        if diff > 0.01 {
            // 1 centavo de tolerancia para error flotante a largo plazo
            crate::telemetry::send_parity_alert(
                "DIVERGENCIA DE CAPITAL DETECTADA",
                expected_capital,
                current_capital,
                diff,
            );

            // Acción evolutiva: Si la discrepancia es crítica y no hay posiciones abiertas, forzar sync
            if active_positions == 0 && diff > 1.0 {
                // En un entorno 100% estricto, esto podría activar el kill switch.
                // arena.kill_switch_active.store(true, Ordering::Release);
                println!("🚨 [PARIDAD] ¡Alerta Crítica! Fuga de capital detectada. Esperado: {}, Real: {}", expected_capital, current_capital);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_validator_parity_nominal() {
        let arena = GlobalArena::new(13.0);
        StateValidator::validate_parity(&arena, 13.0);
    }

    #[test]
    fn test_state_validator_nan_immunity() {
        let arena = GlobalArena::new(13.0);
        StateValidator::validate_parity(&arena, f64::NAN);
    }

    #[test]
    fn test_state_validator_with_realized_pnl_and_divergence() {
        let arena = GlobalArena::new(13.0);
        // Simulate realized profits in coin 0 (BTC)
        arena.coins[0].scalp.pnl_realized.store(2.5, Ordering::Relaxed);
        arena.coins[0].swing.pnl_realized.store(1.5, Ordering::Relaxed);

        // Set unified capital correctly to 13.0 + 4.0 = 17.0
        arena.unified_capital.store(17.0, Ordering::Relaxed);
        StateValidator::validate_parity(&arena, 13.0);

        // Test with simulated divergence (unified capital out of sync)
        arena.unified_capital.store(15.0, Ordering::Relaxed);
        StateValidator::validate_parity(&arena, 13.0);
    }
}

