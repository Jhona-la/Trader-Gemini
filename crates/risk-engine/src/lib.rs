pub mod guard;
pub mod kelly;
pub mod leverage_matrix;
pub mod orchestrator;
pub mod regime;

use quantum_arena::GlobalArena;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct ValidatedOrder {
    pub signal: SignalType,
    pub volume_usd: f64,
    pub leverage: f64,
    pub maker_only: bool,
    pub tp_target: f64,
    pub sl_target: f64,
    pub fee_buffer_multiplier: f64,
}

impl ValidatedOrder {
    pub fn rejected() -> Self {
        Self {
            signal: SignalType::Flat,
            volume_usd: 0.0,
            leverage: 1.0,
            maker_only: false,
            tp_target: 0.0,
            sl_target: 0.0,
            fee_buffer_multiplier: 1.01,
        }
    }
}

// Symbol constraints and size limits have been removed to allow purely dynamic and infinite asset discovery.
// The engine now strictly relies on mathematical limits derived from Kelly and margin constraints.

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
    ) -> (ValidatedOrder, ValidatedOrder) {
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

        let base_capital = arena.config.base_capital.load(Ordering::Relaxed);
        let capital_ratio = self.peak_capital / base_capital.max(1.0);
        let hard_stop_base = arena.config.hard_stop_base_limit.load(Ordering::Relaxed);
        let hard_stop_decay = arena.config.hard_stop_decay_factor.load(Ordering::Relaxed);
        let hard_stop_limit =
            hard_stop_base / (1.0 + capital_ratio.ln().max(0.0) * hard_stop_decay).clamp(1.0, 2.5);

        if current_drawdown >= hard_stop_limit {
            return (ValidatedOrder::rejected(), ValidatedOrder::rejected());
        }

        let guard_dd_sigmoid_steepness = arena
            .config
            .guard_dd_sigmoid_steepness
            .load(Ordering::Relaxed);
        let guard_dd_sigmoid_center = arena.config.guard_dd_sigmoid_center.load(Ordering::Relaxed);
        if !guard::check_drawdown_limit(
            current_capital,
            self.peak_capital,
            max_dd,
            base_capital,
            guard_dd_sigmoid_steepness,
            guard_dd_sigmoid_center,
        ) {
            return (ValidatedOrder::rejected(), ValidatedOrder::rejected());
        }

        let coin = &arena.coins[coin_id];
        let scalp_wr = coin.scalp.win_rate.load(Ordering::Relaxed);
        let scalp_pf = coin.scalp.profit_factor.load(Ordering::Relaxed);
        let swing_wr = coin.swing.win_rate.load(Ordering::Relaxed);
        let swing_pf = coin.swing.profit_factor.load(Ordering::Relaxed);

        let kelly_survival_cap_ratio = arena
            .config
            .kelly_survival_cap_ratio
            .load(Ordering::Relaxed);
        let kelly_expansion_mult = arena.config.kelly_expansion_mult.load(Ordering::Relaxed);

        let split = arena
            .config
            .capital_split_scalp
            .load(Ordering::Relaxed)
            .clamp(0.1, 0.9);
        let scalp_capital = current_capital * split;
        let swing_capital = current_capital * (1.0 - split);

        let mut scalp_kelly = kelly::calculate_kelly_fraction(
            scalp_wr,
            scalp_pf,
            scalp_capital,
            base_capital * split,
            kelly_survival_cap_ratio,
            kelly_expansion_mult,
        );
        let mut swing_kelly = kelly::calculate_kelly_fraction(
            swing_wr,
            swing_pf,
            swing_capital,
            base_capital * (1.0 - split),
            kelly_survival_cap_ratio,
            kelly_expansion_mult,
        );

        let current_ratio = current_capital / base_capital.max(1.0);
        let kelly_cold = arena.config.kelly_bootstrap_cold.load(Ordering::Relaxed);
        let kelly_bootstrap_ratio_threshold = arena
            .config
            .kelly_bootstrap_ratio_threshold
            .load(Ordering::Relaxed);
        let kelly_bootstrap_min_exposure = arena
            .config
            .kelly_bootstrap_min_exposure
            .load(Ordering::Relaxed);

        if current_ratio < kelly_bootstrap_ratio_threshold {
            scalp_kelly = kelly_cold;
            swing_kelly = kelly_cold;
        } else {
            let spec = quantum_arena::symbol_registry::spec(coin_id);
            let dynamic_min_notional = spec
                .min_notional
                .max(spec.min_qty * coin.current_price.load(Ordering::Relaxed).max(1e-8));

            let safe_bootstrap = (dynamic_min_notional / current_capital.max(1.0))
                .clamp(kelly_bootstrap_min_exposure, 0.5);
            if scalp_kelly <= 0.0 {
                scalp_kelly = safe_bootstrap;
            }
            if swing_kelly <= 0.0 {
                swing_kelly = safe_bootstrap;
            }
        }

        let scalp_order = self.evaluate_single_intent(
            coin_id,
            &scalp_intent,
            scalp_kelly,
            scalp_capital,
            base_capital * split,
            scalp_pf,
            true,
            arena,
        );

        let swing_order = self.evaluate_single_intent(
            coin_id,
            &swing_intent,
            swing_kelly,
            swing_capital,
            base_capital * (1.0 - split),
            swing_pf,
            false,
            arena,
        );

        (scalp_order, swing_order)
    }

    fn evaluate_single_intent(
        &self,
        coin_id: usize,
        intent: &SignalIntent,
        kelly_fraction: f64,
        allocated_capital: f64,
        base_allocated: f64,
        profit_factor: f64,
        is_scalp: bool,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        if intent.signal == SignalType::Flat || allocated_capital <= 0.0 {
            return ValidatedOrder::rejected();
        }

        let dir = match intent.signal {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            _ => 0.0,
        };

        let raw_exposure = dir * intent.confidence * kelly_fraction * allocated_capital;
        if raw_exposure == 0.0 {
            return ValidatedOrder::rejected();
        }

        let coin = &arena.coins[coin_id];
        let spec = quantum_arena::symbol_registry::spec(coin_id);
        let max_exchange_leverage = spec.max_leverage as f64;

        let current_atr = coin.current_atr.load(Ordering::Relaxed);
        let current_price = coin.current_price.load(Ordering::Relaxed).max(1e-8);
        let atr_pct = current_atr / current_price;

        let hurst_exponent = coin.hurst_exponent.load(Ordering::Relaxed);
        let vol_mult = if coin_id == 0 {
            arena
                .config
                .btc_volatility_multiplier
                .load(Ordering::Relaxed)
        } else {
            arena
                .config
                .eth_volatility_multiplier
                .load(Ordering::Relaxed)
        };
        let genome_max_leverage = arena
            .config
            .global_leverage
            .load(Ordering::Relaxed)
            .min(max_exchange_leverage);

        let mut dynamic_leverage =
            leverage_matrix::QuantumLeverageMatrix::calculate_dynamic_leverage(
                intent,
                is_scalp,
                allocated_capital,
                base_allocated,
                atr_pct,
                vol_mult,
                hurst_exponent,
                profit_factor,
                genome_max_leverage,
                arena,
            );

        let maker_fee = arena.config.live_maker_fee.load(Ordering::Relaxed);
        let taker_fee = arena.config.live_taker_fee.load(Ordering::Relaxed);
        let roundtrip_fee = maker_fee + taker_fee;

        let target_volatility = atr_pct.max(0.001);
        let confidence = intent.confidence.max(0.51);
        let r_ratio = arena
            .config
            .tp_rr_ratio_btc
            .load(Ordering::Relaxed)
            .max(1.0);
        let expected_value_pct =
            (confidence * r_ratio * target_volatility) - ((1.0 - confidence) * target_volatility);

        let ev_fee_multiplier = arena
            .config
            .ev_fee_multiplier
            .load(Ordering::Relaxed)
            .max(1.05);
        if expected_value_pct <= (roundtrip_fee * ev_fee_multiplier) {
            return ValidatedOrder::rejected();
        }

        let max_acceptable_fee_pct = arena.config.max_fee_pct.load(Ordering::Relaxed);
        let max_safe_leverage = if roundtrip_fee > 0.0 {
            max_acceptable_fee_pct / roundtrip_fee
        } else {
            100.0
        };
        dynamic_leverage =
            dynamic_leverage.clamp(1.0, max_exchange_leverage.min(max_safe_leverage));

        let dynamic_min_notional = spec.min_notional.max(spec.min_qty * current_price);

        let bounded_exposure = raw_exposure.clamp(-allocated_capital, allocated_capital);
        let mut final_margin = bounded_exposure.abs();

        if allocated_capital > 0.0 && allocated_capital * dynamic_leverage < dynamic_min_notional {
            let candidate_leverage = (dynamic_min_notional / allocated_capital) * 1.10;
            let fee_impact_pct = roundtrip_fee * candidate_leverage;
            if fee_impact_pct > max_acceptable_fee_pct {
                return ValidatedOrder::rejected();
            }
            dynamic_leverage = candidate_leverage
                .min(genome_max_leverage)
                .min(max_safe_leverage);
        }

        let required_margin_for_min_notional = dynamic_min_notional / dynamic_leverage;
        if final_margin < required_margin_for_min_notional {
            final_margin = required_margin_for_min_notional;
        }

        let margin_cushion_pct = arena.config.margin_cushion_pct.load(Ordering::Relaxed);
        if final_margin > allocated_capital * margin_cushion_pct {
            return ValidatedOrder::rejected();
        }

        let orchestrator = orchestrator::PortfolioOrchestrator::new(arena);
        let raw_regime = arena.market_regime.load(Ordering::Relaxed);
        let regime = crate::regime::MarketRegime::from(raw_regime);

        if !orchestrator.allow_trade(bounded_exposure > 0.0, final_margin, regime) {
            return ValidatedOrder::rejected();
        }

        let maker_capital_threshold = arena
            .config
            .maker_only_capital_threshold
            .load(Ordering::Relaxed);
        let maker_only = allocated_capital >= maker_capital_threshold;

        ValidatedOrder {
            signal: intent.signal,
            volume_usd: final_margin,
            leverage: dynamic_leverage,
            maker_only,
            tp_target: intent.tp_price_target,
            sl_target: intent.sl_price_target,
            fee_buffer_multiplier: ev_fee_multiplier,
        }
    }
}
