use std::sync::atomic::Ordering;
use quantum_arena::GlobalArena;

/// FASE 8: Portfolio Orchestrator (Capa 3)
/// Responsable de analizar la exposición cruzada (correlación direccional) en todo el portafolio
/// y bloquear posiciones (Flash Crash Protection) si el riesgo se vuelve asimétrico o sistémico.
pub struct PortfolioOrchestrator<'a> {
    arena: &'a GlobalArena,
}

impl<'a> PortfolioOrchestrator<'a> {
    pub fn new(arena: &'a GlobalArena) -> Self {
        Self { arena }
    }

    /// Calcula la asignación dinámica de capital (Fase 8: Redistribución basada en rendimiento)
    #[inline(always)]
    pub fn calculate_dynamic_allocation(&self, coin_id: usize, base_leverage: f64) -> f64 {
        let coin = &self.arena.coins[coin_id];
        
        let win_rate = coin.scalp.win_rate.load(Ordering::Relaxed);
        let profit_factor = coin.scalp.profit_factor.load(Ordering::Relaxed);
        
        // Pseudo-Sharpe Ratio (Rendimiento ajustado al riesgo)
        // Si el win_rate es alto y el PF es alto, multiplicamos la confianza
        let performance_multiplier = if win_rate > 0.55 && profit_factor > 1.2 {
            1.5 // Sinergia positiva, asignar más capital
        } else if win_rate < 0.45 || profit_factor < 0.9 {
            0.5 // Degradación, reducir capital
        } else {
            1.0
        };

        // Drawdown concurrente (Fase 8)
        // Calculamos el PnL no realizado global del portafolio
        let mut global_unrealized: f64 = 0.0;
        for c in self.arena.coins.iter() {
            global_unrealized += c.scalp.pnl_unrealized.load(Ordering::Relaxed);
            global_unrealized += c.swing.pnl_unrealized.load(Ordering::Relaxed);
        }
        
        let capital = self.arena.unified_capital.load(Ordering::Relaxed);
        let drawdown_penalty = if capital > 0.0 && global_unrealized < 0.0 {
            let dd_pct = (global_unrealized.abs() / capital).clamp(0.0, 1.0);
            if dd_pct > 0.05 { // Si el portafolio entero está en -5% DD
                0.2 // Cortamos severamente la nueva exposición
            } else if dd_pct > 0.02 {
                0.5 // Reducción conservadora
            } else {
                1.0
            }
        } else {
            1.0
        };

        base_leverage * performance_multiplier * drawdown_penalty
    }

    /// Evalúa si el portafolio permite la apertura de una nueva posición direccional
    #[inline(always)]
    pub fn allow_trade(&self, intent_is_long: bool, required_margin: f64, regime: crate::regime::MarketRegime) -> bool {
        // Regime Orchestration (Fase 13: Kill-Switch macro)
        if regime == crate::regime::MarketRegime::Crash && intent_is_long {
            return false; // Bloqueo absoluto de compras en caída libre sistémica.
        }
        if regime == crate::regime::MarketRegime::BullRun && !intent_is_long {
            return false; // Bloqueo absoluto de cortos en pleno Bull Run.
        }

        let mut total_long_margin = 0.0;
        let mut total_short_margin = 0.0;
        
        // O(1) lock-free iteration over 30 coins to calculate net delta and exposure
        for coin in self.arena.coins.iter() {
            let scalp_pos = &coin.positions.scalp_position;
            if scalp_pos.is_open() {
                let margin = scalp_pos.margin_used.load(Ordering::Relaxed);
                if scalp_pos.is_long.load(Ordering::Relaxed) {
                    total_long_margin += margin;
                } else {
                    total_short_margin += margin;
                }
            }
            
            let swing_pos = &coin.positions.swing_position;
            if swing_pos.is_open() {
                let margin = swing_pos.margin_used.load(Ordering::Relaxed);
                if swing_pos.is_long.load(Ordering::Relaxed) {
                    total_long_margin += margin;
                } else {
                    total_short_margin += margin;
                }
            }
        }

        let capital = self.arena.unified_capital.load(Ordering::Relaxed);
        if capital <= 0.0 {
            return false;
        }

        let total_exposure = total_long_margin + total_short_margin + required_margin;
        
        // 1. Max Gross Exposure limit
        let exposure_limit = if regime == crate::regime::MarketRegime::BullRun {
            0.95 // En Bull Run permitimos desplegar hasta el 95% del capital
        } else if regime == crate::regime::MarketRegime::Crash {
            0.40 // En Crash somos conservadores
        } else {
            0.80
        };

        if total_exposure > capital * exposure_limit {
            return false;
        }

        // 2. Net Delta / Correlation Limit
        if intent_is_long {
            let delta_limit = if regime == crate::regime::MarketRegime::BullRun { 0.90 } else { 0.50 };
            if (total_long_margin + required_margin) > capital * delta_limit {
                return false; 
            }
        } else {
            let delta_limit = if regime == crate::regime::MarketRegime::Crash { 0.80 } else { 0.50 };
            if (total_short_margin + required_margin) > capital * delta_limit {
                return false; 
            }
        }

        true
    }
}
