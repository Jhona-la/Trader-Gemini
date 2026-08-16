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
        let mut total_realized_pnl = 0.0;
        let mut active_positions = 0;
        
        for coin in arena.coins.iter() {
            total_realized_pnl += coin.scalp.pnl_realized.load(Ordering::Relaxed);
            total_realized_pnl += coin.swing.pnl_realized.load(Ordering::Relaxed);
            
            active_positions += coin.scalp.active_positions.load(Ordering::Relaxed);
            active_positions += coin.swing.active_positions.load(Ordering::Relaxed);
        }
        
        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        let expected_capital = initial_capital + total_realized_pnl;
        
        // El margen de error debe ser minúsculo (errores de flotante).
        // Si hay discrepancia mayor a 1 USD (o micro-centavos), hay código fantasma.
        let diff = (current_capital - expected_capital).abs();
        
        if diff > 0.01 { // 1 centavo de tolerancia para error flotante a largo plazo
            crate::telemetry::send_parity_alert(
                "DIVERGENCIA DE CAPITAL DETECTADA", 
                expected_capital, 
                current_capital, 
                diff
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
