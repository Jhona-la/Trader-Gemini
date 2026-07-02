pub mod guard;
pub mod kelly;
pub mod orchestrator;
pub mod regime;

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

pub fn get_symbol_constraints(symbol: &str) -> (f64, f64, f64) {
    // Returns (minQty, stepSize, minNotional)
    match symbol.to_uppercase().as_str() {
        "BTCUSDT" => (0.001, 0.001, 5.05),
        "ETHUSDT" => (0.001, 0.001, 5.05),
        "SOLUSDT" => (1.0, 1.0, 5.05),
        "ADAUSDT" => (1.0, 1.0, 5.05),
        "DOGEUSDT" => (1.0, 1.0, 5.05),
        "XRPUSDT" => (1.0, 1.0, 5.05),
        "BNBUSDT" => (0.01, 0.01, 5.05),
        "AVAXUSDT" => (0.1, 0.1, 5.05),
        "DOTUSDT" => (0.1, 0.1, 5.05),
        "LINKUSDT" => (0.1, 0.1, 5.05),
        _ => (1.0, 1.0, 5.05),
    }
}

pub fn calculate_dynamic_position_size(
    symbol: &str,
    current_price: f64,
    leverage: f64,
    capital: f64,
    current_atr_pct: f64,
    win_rate: f64,
    profit_factor: f64,
) -> Option<f64> {
    let (min_qty, step_size, min_notional) = get_symbol_constraints(symbol);
    
    // 1. Get Base Kelly Fraction
    let mut safe_kelly = kelly::calculate_kelly_fraction(win_rate, profit_factor, capital);
    
    // Fallback if no history or flat kelly: 15% risk for <$50 rapid scaling
    if safe_kelly <= 0.0 {
        safe_kelly = if capital < 50.0 { 0.15 } else { 0.05 };
    }
    
    let max_loss_capital = capital * safe_kelly;
    let avg_sl_pct = current_atr_pct.max(0.001);
    let notional_target = max_loss_capital / avg_sl_pct;
    
    let max_affordable_notional = capital * leverage * 0.95; 
    let mut effective_notional = notional_target.min(max_affordable_notional);
    effective_notional = effective_notional.max(min_notional);
    
    let mut raw_qty = effective_notional / current_price;
    if raw_qty < min_qty {
        raw_qty = min_qty;
    }
    
    let step_multiplier = 1.0 / step_size;
    let final_qty = (raw_qty * step_multiplier).floor() / step_multiplier;
    
    let required_margin = (final_qty * current_price) / leverage;
    if required_margin > capital {
        return None; 
    }
    
    Some(final_qty)
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

    /// Evalúa la intención de señal combinada de Scalp y Swing y retorna la Exposición Neta (Net Delta).
    pub fn evaluate_order(
        &mut self,
        coin_id: usize,
        scalp_intent: SignalIntent,
        swing_intent: SignalIntent,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        
        // 1. Actualizar pico de capital
        if current_capital > self.peak_capital {
            self.peak_capital = current_capital;
        }

        // 2. Comprobar cortafuegos (Drawdown)
        let max_dd = arena.config.global_max_drawdown.load(Ordering::Relaxed);
        let current_drawdown = if self.peak_capital > 0.0 {
            (self.peak_capital - current_capital) / self.peak_capital
        } else {
            0.0
        };
        
        // HARD STOP KILL SWITCH: Dinámico según capital
        // Si el capital es < $50, toleramos hasta 50% de DD para permitir volatilidad inicial con alto apalancamiento
        let hard_stop_limit = if self.peak_capital < 50.0 { 0.50 } else { 0.20 };
        
        if current_drawdown >= hard_stop_limit {
            // Se bloquean silenciosamente para no inundar el log en simulaciones HFT
            return ValidatedOrder::rejected();
        }

        if !guard::check_drawdown_limit(current_capital, self.peak_capital, max_dd) {
            return ValidatedOrder::rejected(); // Drawdown normal excedido
        }

        // 3. Obtener métricas históricas de la moneda actual
        let coin = &arena.coins[coin_id];
        let scalp_wr = coin.scalp.win_rate.load(Ordering::Relaxed);
        let scalp_pf = coin.scalp.profit_factor.load(Ordering::Relaxed);
        let swing_wr = coin.swing.win_rate.load(Ordering::Relaxed);
        let swing_pf = coin.swing.profit_factor.load(Ordering::Relaxed);

        // 4. Calcular fracciones de Kelly independientes
        let mut scalp_kelly = kelly::calculate_kelly_fraction(scalp_wr, scalp_pf, current_capital);
        let mut swing_kelly = kelly::calculate_kelly_fraction(swing_wr, swing_pf, current_capital);
        
        // Fallback básico para arrancar si no hay suficientes trades para Kelly
        if scalp_kelly <= 0.0 { scalp_kelly = 0.1; }
        if swing_kelly <= 0.0 { swing_kelly = 0.1; }

        // Calcular exposiciones direccionales (Long = positivo, Short = negativo, Flat = 0)
        let scalp_dir = match scalp_intent.signal {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            SignalType::Flat => 0.0,
        };
        
        let swing_dir = match swing_intent.signal {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            SignalType::Flat => 0.0,
        };

        // Multiplicar dirección por Confidence * Kelly * Capital
        let scalp_exposure = scalp_dir * scalp_intent.confidence * scalp_kelly * current_capital;
        let swing_exposure = swing_dir * swing_intent.confidence * swing_kelly * current_capital;

        // 5. Matriz de Sinergia y Delta Neto
        let mut net_exposure = scalp_exposure + swing_exposure;
        
        // Boost si están en la misma dirección y ambos tienen señal
        if scalp_dir == swing_dir && scalp_dir != 0.0 {
            net_exposure *= 1.5; // Sinergia (Axioma V)
        }

        if net_exposure == 0.0 {
            return ValidatedOrder::rejected();
        }

        let global_leverage = arena.config.global_leverage.load(Ordering::Relaxed);
        let min_notional = arena.config.min_notional.load(Ordering::Relaxed);

        // Limitar la exposición bruta al MARGEN disponible (capital)
        let max_margin_exposure = current_capital;
        let bounded_exposure = net_exposure.clamp(-max_margin_exposure, max_margin_exposure);
        
        let mut final_margin = bounded_exposure.abs();
        
        // Force minimum notional for micro-accounts ($13)
        let required_margin_for_min_notional = min_notional / global_leverage;
        if final_margin < required_margin_for_min_notional {
            final_margin = required_margin_for_min_notional;
        }
        
        // Si aun forzando el mínimo no nos alcanza el capital, entonces sí rechazamos
        if final_margin > current_capital * 0.95 { // 95% para dejar buffer de fees
            return ValidatedOrder::rejected();
        }

        let final_signal = if bounded_exposure > 0.0 {
            SignalType::Long
        } else {
            SignalType::Short
        };

        // 6. Check con el Orquestador del Portafolio (Capa 3)
        let orchestrator = orchestrator::PortfolioOrchestrator::new(arena);
        
        let raw_regime = arena.market_regime.load(Ordering::Relaxed);
        let regime = match raw_regime {
            1 => crate::regime::MarketRegime::BullRun,
            2 => crate::regime::MarketRegime::Crash,
            3 => crate::regime::MarketRegime::Chaotic,
            _ => crate::regime::MarketRegime::Range,
        };

        if !orchestrator.allow_trade(bounded_exposure > 0.0, final_margin * global_leverage, regime) {
            return ValidatedOrder::rejected();
        }

        ValidatedOrder {
            signal: final_signal,
            volume_usd: final_margin,
            leverage: global_leverage,
        }
    }
}
