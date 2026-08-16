use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

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

        // CONTINUOUS Performance Multiplier (sigmoid-based, no step functions)
        // Maps WR×PF product into a smooth [0.3, 2.0] range via generalized logistic
        // Center at WR=0.50, PF=1.0 (breakeven point)
        let performance_score = win_rate * profit_factor;
        let portfolio_perf_mult_steepness = self
            .arena
            .config
            .portfolio_perf_mult_steepness
            .load(Ordering::Relaxed);
        let portfolio_perf_mult_min = self
            .arena
            .config
            .portfolio_perf_mult_min
            .load(Ordering::Relaxed);
        let portfolio_perf_mult_max = self
            .arena
            .config
            .portfolio_perf_mult_max
            .load(Ordering::Relaxed);
        let portfolio_perf_mult_center = self
            .arena
            .config
            .portfolio_perf_mult_center
            .load(Ordering::Relaxed);

        let range = portfolio_perf_mult_max - portfolio_perf_mult_min;
        let performance_multiplier = portfolio_perf_mult_min
            + range
                / (1.0
                    + (-portfolio_perf_mult_steepness
                        * (performance_score - portfolio_perf_mult_center))
                        .exp());

        // CONTINUOUS Drawdown Penalty (exponential decay, no step functions)
        let mut global_unrealized: f64 = 0.0;
        for c in self.arena.coins.iter() {
            global_unrealized += c.scalp.pnl_unrealized.load(Ordering::Relaxed);
            global_unrealized += c.swing.pnl_unrealized.load(Ordering::Relaxed);
        }

        let capital = self.arena.unified_capital.load(Ordering::Relaxed);
        let portfolio_dd_penalty_decay = self
            .arena
            .config
            .portfolio_dd_penalty_decay
            .load(Ordering::Relaxed);
        let drawdown_penalty = if capital > 0.0 && global_unrealized < 0.0 {
            let dd_pct = (global_unrealized.abs() / capital).clamp(0.0, 1.0);
            // Smooth exponential decay: at 0% DD = 1.0, at 5% DD ≈ 0.47, at 10% DD ≈ 0.22
            (-dd_pct * portfolio_dd_penalty_decay).exp()
        } else {
            1.0
        };

        base_leverage * performance_multiplier * drawdown_penalty
    }

    /// Evalúa si el portafolio permite la apertura de una nueva posición direccional
    #[inline(always)]
    pub fn allow_trade(
        &self,
        intent_is_long: bool,
        required_margin: f64,
        regime: crate::regime::MarketRegime,
    ) -> bool {
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

        // Max Gross Exposure limit: Read from arena config (genome-evolvable)
        // Defaults to 1.0 (100% capital efficiency) but can be tightened by the genome
        // Prevent complete freezing by asserting a minimal theoretical bounds
        let exposure_limit = self
            .arena
            .config
            .global_max_drawdown
            .load(Ordering::Relaxed)
            .max(0.1);

        if total_exposure > capital * exposure_limit {
            return false;
        }

        // Net Delta / Directional Limit: Smooth continuous check
        // Allow full directional exposure but respect capital limits
        if intent_is_long {
            if (total_long_margin + required_margin) > capital * exposure_limit {
                return false;
            }
        } else {
            if (total_short_margin + required_margin) > capital * exposure_limit {
                return false;
            }
        }

        true
    }
}
