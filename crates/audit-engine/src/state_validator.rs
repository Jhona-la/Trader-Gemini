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
        let safe_initial = if initial_capital.is_finite() && initial_capital > 0.0 {
            initial_capital
        } else {
            13.0
        };

        let mut total_realized_pnl = 0.0;
        let mut active_positions = 0;
        let mut open_entry_fees = 0.0;

        for coin in arena.coins.iter() {
            let m_pnl = coin.metrics.pnl_realized.load(Ordering::Relaxed);
            if m_pnl.is_finite() {
                total_realized_pnl += m_pnl;
            }

            for pos in coin.positions.slots() {
                if pos.is_open() {
                    active_positions += 1;
                    let fee = pos.entry_fee.load(Ordering::Relaxed);
                    if fee.is_finite() && fee > 0.0 {
                        open_entry_fees += fee;
                    }
                }
            }
        }

        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        if !current_capital.is_finite() {
            return;
        }

        // D-225: Deducir open_entry_fees del capital esperado puesto que las comisiones de apertura
        // se descuentan inmediatamente de unified_capital pero no entran a pnl_realized hasta el cierre
        let expected_capital = safe_initial + total_realized_pnl - open_entry_fees;

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
        // D-714: pila suficiente para construir el arena.
        let arena = GlobalArena::build_in_own_stack(13.0);
        StateValidator::validate_parity(&arena, 13.0);
    }

    #[test]
    fn test_state_validator_nan_immunity() {
        // D-714: pila suficiente para construir el arena.
        let arena = GlobalArena::build_in_own_stack(13.0);
        StateValidator::validate_parity(&arena, f64::NAN);
    }

    #[test]
    fn test_state_validator_with_realized_pnl_and_divergence() {
        // D-714: pila suficiente para construir el arena.
        let arena = GlobalArena::build_in_own_stack(13.0);
        // Simulate realized profits in coin 0 (BTC) — U-1: métrica unificada.
        arena.coins[0]
            .metrics
            .pnl_realized
            .store(4.0, Ordering::Relaxed);

        // Set unified capital correctly to 13.0 + 4.0 = 17.0
        arena.unified_capital.store(17.0, Ordering::Relaxed);
        StateValidator::validate_parity(&arena, 13.0);

        // Test with simulated divergence (unified capital out of sync)
        arena.unified_capital.store(15.0, Ordering::Relaxed);
        StateValidator::validate_parity(&arena, 13.0);
    }
}
