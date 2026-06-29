pub mod guard;
pub mod kelly;

use signal_engine::{SignalIntent, SignalType};
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct ValidatedOrder {
    pub signal: SignalType,
    pub volume_usd: f64,
    pub leverage: f64,
}

impl ValidatedOrder {
    pub fn rejected() -> Self {
        Self {
            signal: SignalType::Flat,
            volume_usd: 0.0,
            leverage: 1.0,
        }
    }
}

pub struct RiskEngine {
    pub peak_capital: f64, // Memoria histórica del capital más alto
}

impl RiskEngine {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            peak_capital: initial_capital,
        }
    }

    /// Evalúa la intención de señal contra las matemáticas del Kelly y el estado de la Arena.
    pub fn evaluate(
        &mut self,
        intent: SignalIntent,
        arena: &GlobalArena,
        is_scalp: bool,
    ) -> ValidatedOrder {
        if intent.signal == SignalType::Flat {
            return ValidatedOrder::rejected();
        }

        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        
        // 1. Actualizar pico de capital
        if current_capital > self.peak_capital {
            self.peak_capital = current_capital;
        }

        // 2. Comprobar cortafuegos (Drawdown)
        let max_dd = arena.config.global_max_drawdown.load(Ordering::Relaxed);
        if !guard::check_drawdown_limit(current_capital, self.peak_capital, max_dd) {
            // Drawdown excedido, bloquear operaciones
            return ValidatedOrder::rejected();
        }

        // 3. Obtener métricas históricas de la estrategia (Scalp o Swing)
        let (win_rate, profit_factor) = if is_scalp {
            (
                arena.scalp.win_rate.load(Ordering::Relaxed),
                arena.scalp.profit_factor.load(Ordering::Relaxed),
            )
        } else {
            (
                arena.swing.win_rate.load(Ordering::Relaxed),
                arena.swing.profit_factor.load(Ordering::Relaxed),
            )
        };

        // 4. Calcular fracción de Kelly
        let kelly_pct = kelly::calculate_kelly_fraction(win_rate, profit_factor);
        if kelly_pct == 0.0 {
            // Sistema en pérdida matemática, rechazar
            return ValidatedOrder::rejected();
        }

        let intended_volume = current_capital * kelly_pct;
        let global_leverage = arena.config.global_leverage.load(Ordering::Relaxed);
        let min_notional = arena.config.min_notional.load(Ordering::Relaxed);

        // 5. Validar Notional de Binance
        let (is_valid, final_volume) = guard::enforce_minimum_notional(intended_volume, min_notional, global_leverage);

        if !is_valid {
            return ValidatedOrder::rejected();
        }

        ValidatedOrder {
            signal: intent.signal,
            volume_usd: final_volume,
            leverage: global_leverage,
        }
    }
}
