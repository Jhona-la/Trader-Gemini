pub mod capital_compounder;
pub mod correlation_guard;
pub mod epigenetic_capital_alloc;
pub mod epigenetic_fitness_landscape;
pub mod guard;
pub mod kelly;
pub mod kelly_envelope;
pub mod leverage_matrix;
pub mod macro_regime_swing_optimizer;
pub mod orchestrator;
pub mod regime;

pub use kelly_envelope::{EdgePosterior, RiskEnvelope, SURVIVAL_FLOOR, TRADE_HORIZON};

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
    pub scalp_peak_capital: f64,
    pub swing_peak_capital: f64,
    /// R1.3 — split de capital suavizado (Robbins-Monro). El objetivo
    /// bayesiano se recalcula por evaluación, pero el split EFECTIVO
    /// converge con paso 1/√n para no desestabilizar picos de drawdown,
    /// hard-stops y capital_ratio que dependen de él.
    pub smoothed_split: Option<f64>,
}

impl RiskEngine {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            scalp_peak_capital: initial_capital * 0.5,
            swing_peak_capital: initial_capital * 0.5,
            smoothed_split: None,
        }
    }

    pub fn reset(&mut self, initial_capital: f64) {
        self.scalp_peak_capital = initial_capital * 0.5;
        self.swing_peak_capital = initial_capital * 0.5;
        self.smoothed_split = None;
    }

    /// Evalúa la intención de señal combinada de Scalp y Swing y retorna la Exposición Neta (Net Delta).
    pub fn evaluate_order(
        &mut self,
        coin_id: usize,
        scalp_intent: SignalIntent,
        swing_intent: SignalIntent,
        arena: &GlobalArena,
    ) -> (ValidatedOrder, ValidatedOrder) {
        if coin_id >= arena.coins.len() {
            return (ValidatedOrder::rejected(), ValidatedOrder::rejected());
        }

        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        if !current_capital.is_finite() || current_capital <= 0.0 {
            return (ValidatedOrder::rejected(), ValidatedOrder::rejected());
        }

        // 0. Partición de capital por horizonte — R1.3: posterior bayesiano
        // con crecimiento de evidencia en √n (la información estadística
        // sobre "qué bucket tiene edge" crece con la raíz del número de
        // trades, como todo estadístico). El gen capital_split_scalp actúa
        // como prior de peso 1 (una pseudo-observación): manda hasta que la
        // evidencia lo desplace. A diferencia de la versión lineal en n
        // (que saturaba al clamp con n≥10 y mataba el otro horizonte al
        // piso 0.1), con √n el límite asintótico es el COCIENTE de edges
        // — un bucket sin edge colapsa suyo, no por acumulación de muestras.
        // El split efectivo se suaviza además con paso 1/√n (Robbins-Monro)
        // para que hard-stops y picos de drawdown no respiren con cada tick.
        let genome_split = arena
            .config
            .capital_split_scalp
            .load(Ordering::Relaxed)
            .clamp(0.1, 0.9);
        let coin = &arena.coins[coin_id];
        let scalp_edge = (coin.scalp.win_rate.load(Ordering::Relaxed)
            * coin.scalp.kelly_fraction.load(Ordering::Relaxed))
        .max(0.0);
        let swing_edge = (coin.swing.win_rate.load(Ordering::Relaxed)
            * coin.swing.kelly_fraction.load(Ordering::Relaxed))
        .max(0.0);
        let scalp_n = coin.scalp.trade_count.load(Ordering::Relaxed) as f64;
        let swing_n = coin.swing.trade_count.load(Ordering::Relaxed) as f64;
        let posterior_scalp = scalp_edge * scalp_n.sqrt() + genome_split;
        let posterior_swing = swing_edge * swing_n.sqrt() + (1.0 - genome_split);
        let target_split = if posterior_scalp + posterior_swing > 1e-12 {
            (posterior_scalp / (posterior_scalp + posterior_swing)).clamp(0.1, 0.9)
        } else {
            genome_split
        };
        // Suavizado con tasa decreciente 1/√(n_total): cambio grande con
        // poca evidencia, refinamiento fino con mucha — sin constantes.
        let n_total = scalp_n + swing_n;
        let alpha = 1.0 / (n_total + 1.0).sqrt();
        let split = match self.smoothed_split {
            Some(prev) => prev + (target_split - prev) * alpha,
            None => target_split,
        };
        self.smoothed_split = Some(split);
        let scalp_capital = current_capital * split;
        let swing_capital = current_capital * (1.0 - split);

        // 1. Actualizar picos de capital independientes
        if scalp_capital > self.scalp_peak_capital {
            self.scalp_peak_capital = scalp_capital;
        }
        if swing_capital > self.swing_peak_capital {
            self.swing_peak_capital = swing_capital;
        }

        // 2. Comprobar cortafuegos (Drawdown) aislados por TradeHorizon
        let max_dd = arena.config.global_max_drawdown.load(Ordering::Relaxed);
        
        let scalp_drawdown = if self.scalp_peak_capital > 0.0 {
            (self.scalp_peak_capital - scalp_capital) / self.scalp_peak_capital
        } else {
            0.0
        };
        let swing_drawdown = if self.swing_peak_capital > 0.0 {
            (self.swing_peak_capital - swing_capital) / self.swing_peak_capital
        } else {
            0.0
        };

        let base_capital = arena.config.base_capital.load(Ordering::Relaxed);
        let hard_stop_base = arena.config.hard_stop_base_limit.load(Ordering::Relaxed);
        let hard_stop_decay = arena.config.hard_stop_decay_factor.load(Ordering::Relaxed);
        let scalp_hard_stop_limit = {
            let capital_ratio = (self.scalp_peak_capital / (base_capital * split).max(1.0)).max(1.0);
            (hard_stop_base / (1.0 + capital_ratio.ln() * hard_stop_decay)).clamp(0.20, 0.95)
        };

        let swing_hard_stop_limit = {
            let capital_ratio = (self.swing_peak_capital / (base_capital * (1.0 - split)).max(1.0)).max(1.0);
            (hard_stop_base / (1.0 + capital_ratio.ln() * hard_stop_decay)).clamp(0.20, 0.95)
        };

        // FIX ZOMBIE LOCK: Si el drawdown supera el límite, decaer suavemente el pico histórico
        // en vez de congelar la cuenta para siempre. Permite crecimiento compuesto infinito.
        let scalp_valid = true;
        if scalp_drawdown >= scalp_hard_stop_limit {
            self.scalp_peak_capital = scalp_capital * 1.05;
        }

        let swing_valid = true;
        if swing_drawdown >= swing_hard_stop_limit {
            self.swing_peak_capital = swing_capital * 1.05;
        }

        let guard_dd_sigmoid_steepness = arena
            .config
            .guard_dd_sigmoid_steepness
            .load(Ordering::Relaxed);
        let guard_dd_sigmoid_center = arena.config.guard_dd_sigmoid_center.load(Ordering::Relaxed);
        
        if !guard::check_drawdown_limit(
            scalp_capital,
            self.scalp_peak_capital,
            max_dd,
            base_capital * split,
            guard_dd_sigmoid_steepness,
            guard_dd_sigmoid_center,
        ) {
            self.scalp_peak_capital = scalp_capital * 1.05;
        }
        
        if !guard::check_drawdown_limit(
            swing_capital,
            self.swing_peak_capital,
            max_dd,
            base_capital * (1.0 - split),
            guard_dd_sigmoid_steepness,
            guard_dd_sigmoid_center,
        ) {
            self.swing_peak_capital = swing_capital * 1.05;
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



        let clamp_min = arena.config.kelly_clamp_min.load(Ordering::Relaxed);
        let clamp_max = arena.config.kelly_clamp_max.load(Ordering::Relaxed);
        let scalp_base_frac = arena.config.scalp_kelly_fraction.load(Ordering::Relaxed);
        let swing_base_frac = arena.config.swing_kelly_fraction.load(Ordering::Relaxed);

        let mut scalp_kelly = kelly::calculate_kelly_fraction(
            scalp_wr,
            scalp_pf,
            scalp_capital,
            base_capital * split,
            kelly_survival_cap_ratio,
            kelly_expansion_mult,
            clamp_min,
            clamp_max,
            scalp_base_frac,
        );
        let mut swing_kelly = kelly::calculate_kelly_fraction(
            swing_wr,
            swing_pf,
            swing_capital,
            base_capital * (1.0 - split),
            kelly_survival_cap_ratio,
            kelly_expansion_mult,
            clamp_min,
            clamp_max,
            swing_base_frac,
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
            scalp_kelly = (kelly_cold * split.max(0.1)).clamp(0.01, 0.50);
            swing_kelly = (kelly_cold * (1.0 - split).max(0.1)).clamp(0.01, 0.50);
        } else {
            let spec = match quantum_arena::symbol_registry::try_spec(coin_id) {
                Some(s) => s,
                None => return (ValidatedOrder::rejected(), ValidatedOrder::rejected()),
            };
            let dynamic_min_notional = spec.min_notional.max(5.0);

            let safe_bootstrap_scalp = (dynamic_min_notional / scalp_capital.max(1.0))
                .clamp(kelly_bootstrap_min_exposure * 0.5, 0.4);
            let safe_bootstrap_swing = (dynamic_min_notional / swing_capital.max(1.0))
                .clamp(kelly_bootstrap_min_exposure, 0.5);
            if scalp_kelly <= 0.0 {
                scalp_kelly = safe_bootstrap_scalp * split.max(0.1);
            }
            if swing_kelly <= 0.0 {
                swing_kelly = safe_bootstrap_swing * (1.0 - split).max(0.1);
            }
        }

        let scalp_order = if scalp_valid {
            self.evaluate_single_intent(
                coin_id,
                &scalp_intent,
                scalp_kelly,
                scalp_capital,
                base_capital * split,
                scalp_pf,
                true,
                arena,
            )
        } else {
            ValidatedOrder::rejected()
        };

        let swing_order = if swing_valid {
            self.evaluate_single_intent(
                coin_id,
                &swing_intent,
                swing_kelly,
                swing_capital,
                base_capital * (1.0 - split),
                swing_pf,
                false,
                arena,
            )
        } else {
            ValidatedOrder::rejected()
        };

        (scalp_order, swing_order)
    }

    /// Evalúa la intención cuántica diferenciada por horizonte (Scalping vs Swing)
    pub fn evaluate_quantum_order_by_horizon(
        &mut self,
        coin_id: usize,
        intent: &SignalIntent,
        is_scalp: bool,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        if coin_id >= arena.coins.len() || intent.signal == SignalType::Flat {
            return ValidatedOrder::rejected();
        }

        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        if !current_capital.is_finite() || current_capital <= 0.0 {
            return ValidatedOrder::rejected();
        }

        if is_scalp {
            if current_capital > self.scalp_peak_capital {
                self.scalp_peak_capital = current_capital;
            }
        } else {
            if current_capital > self.swing_peak_capital {
                self.swing_peak_capital = current_capital;
            }
        }

        let base_capital = arena.config.base_capital.load(Ordering::Relaxed);
        let pf = if is_scalp {
            arena.coins[coin_id].metrics.profit_factor.load(Ordering::Relaxed)
        } else {
            arena.coins[coin_id].swing.profit_factor.load(Ordering::Relaxed)
        };
        let clamp_min = arena.config.kelly_clamp_min.load(Ordering::Relaxed).max(0.0);
        let clamp_max = arena.config.kelly_clamp_max.load(Ordering::Relaxed).clamp(clamp_min, 1.0);
        let raw_kelly = if is_scalp {
            arena.coins[coin_id].metrics.kelly_fraction.load(Ordering::Relaxed)
        } else {
            arena.coins[coin_id].swing.kelly_fraction.load(Ordering::Relaxed)
        };
        // N-03: Si Kelly es <= 0.0 (esperanza no positiva o ruinosa), NO forzar piso artificial del 5%
        let kelly_frac = if raw_kelly <= 0.0 {
            0.0
        } else {
            raw_kelly.clamp(clamp_min, clamp_max)
        };

        self.evaluate_single_intent(
            coin_id,
            intent,
            kelly_frac,
            current_capital,
            base_capital,
            pf,
            is_scalp,
            arena,
        )
    }

    /// Evalúa la intención unificada de señal cuántica continua sobre el 100% del capital disponible.
    pub fn evaluate_quantum_order(
        &mut self,
        coin_id: usize,
        intent: &SignalIntent,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        self.evaluate_quantum_order_by_horizon(coin_id, intent, true, arena)
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

        // FASE 16 & BUG-578: Correlation Guard by Horizon
        let is_long = intent.signal == SignalType::Long;
        let mut same_dir_count = 0;
        for c in arena.coins.iter() {
            let pos = if is_scalp {
                if c.positions.scalp_position.is_open() {
                    &c.positions.scalp_position
                } else {
                    &c.positions.position
                }
            } else {
                if c.positions.swing_position.is_open() {
                    &c.positions.swing_position
                } else {
                    &c.positions.position
                }
            };
            if pos.is_open() && (pos.is_long.load(Ordering::Relaxed) == is_long) {
                same_dir_count += 1;
            }
        }
        let corr_thresh = arena.config.global_correlation_threshold.load(Ordering::Relaxed);
        let max_allowed_cluster = (corr_thresh * 5.0).round() as usize;
        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
        if correlation_guard::CorrelationGuardEngine::is_correlation_vetoed_by_horizon(
            is_scalp,
            same_dir_count,
            current_cap,
            max_allowed_cluster.max(2),
        ) {
            return ValidatedOrder::rejected();
        }

        let coin = &arena.coins[coin_id];
        let spec = match quantum_arena::symbol_registry::try_spec(coin_id) {
            Some(s) => s,
            None => return ValidatedOrder::rejected(),
        };
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
        } else if coin_id == 1 {
            arena
                .config
                .eth_volatility_multiplier
                .load(Ordering::Relaxed)
        } else {
            let eth_mult = arena
                .config
                .eth_volatility_multiplier
                .load(Ordering::Relaxed);
            let btc_price = arena.coins[0].current_price.load(Ordering::Relaxed).max(1e-8);
            let btc_atr = arena.coins[0].current_atr.load(Ordering::Relaxed);
            let btc_atr_pct = if btc_atr > 0.0 && btc_price > 1e-6 {
                (btc_atr / btc_price).clamp(0.0005, 0.50)
            } else {
                0.005 // 50 bps baseline default
            };
            let effective_atr_pct = atr_pct.clamp(0.0001, 1.0);
            let relative_vol = (effective_atr_pct / btc_atr_pct.max(0.0005)).clamp(0.5, 3.0);
            eth_mult * relative_vol
        };
        let genome_max_leverage = arena
            .config
            .global_leverage
            .load(Ordering::Relaxed)
            .min(max_exchange_leverage);

        let real_win_rate = if is_scalp {
            coin.scalp.win_rate.load(Ordering::Relaxed)
        } else {
            coin.swing.win_rate.load(Ordering::Relaxed)
        };

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
                real_win_rate,
                genome_max_leverage,
                arena,
            );

        let maker_fee = arena.config.live_maker_fee.load(Ordering::Relaxed);
        let taker_fee = arena.config.live_taker_fee.load(Ordering::Relaxed);
        let roundtrip_fee = maker_fee + taker_fee;

        let expected_win = if is_scalp {
            arena.config.scalp_tp_base.load(Ordering::Relaxed).max(0.0010).max(atr_pct * 1.5)
        } else {
            arena.config.swing_tp_base.load(Ordering::Relaxed).max(0.0050).max(atr_pct * 3.0)
        };

        let expected_loss = if is_scalp {
            arena.config.scalp_sl_base.load(Ordering::Relaxed).max(0.0005).max(atr_pct * 0.8)
        } else {
            arena.config.swing_sl_base.load(Ordering::Relaxed).max(0.0020).max(atr_pct * 1.5)
        };

        let confidence = intent.confidence.max(0.51);
        let expected_value_pct =
            (confidence * expected_win) - ((1.0 - confidence) * expected_loss);

        let min_ev_mult = if allocated_capital <= 15.0 {
            1.005 // Micro-cuenta ($13 USD bootstrap): buffer 0.5% sobre comisiones para desbloquear oportunidades legítimas
        } else {
            1.02
        };
        let ev_fee_multiplier = arena
            .config
            .ev_fee_multiplier
            .load(Ordering::Relaxed)
            .clamp(min_ev_mult, 1.30);
        if expected_value_pct <= (roundtrip_fee * ev_fee_multiplier) {
            return ValidatedOrder::rejected();
        }

        let max_acceptable_fee_pct = arena.config.max_fee_pct.load(Ordering::Relaxed);
        let _max_safe_leverage = if roundtrip_fee > 0.0 {
            max_acceptable_fee_pct / roundtrip_fee
        } else {
            100.0
        };
        let dynamic_min_notional = spec.min_notional.max(5.0);

        let bounded_exposure = raw_exposure.clamp(-allocated_capital, allocated_capital);
        let mut final_margin = bounded_exposure.abs();
        let margin_cushion_pct = arena.config.margin_cushion_pct.load(Ordering::Relaxed);
        let safe_cushion = if allocated_capital <= 15.0 {
            0.98 // Permitir hasta 98% en bootstrap micro-capital ($13 USD) para cumplir con el piso notional de Binance
        } else if margin_cushion_pct.is_finite() && margin_cushion_pct > 0.0 {
            margin_cushion_pct
        } else {
            0.80
        };

        if allocated_capital > 0.0 && allocated_capital * dynamic_leverage * safe_cushion < dynamic_min_notional {
            let candidate_leverage = (dynamic_min_notional / (allocated_capital * safe_cushion)) * 1.02;
            let max_fee_limit = if allocated_capital <= 15.0 {
                0.035 // Permitir hasta 3.5% fee impact para bootstrap micro-cuentas ($13 USD) para cumplir con el lote mínimo de Binance
            } else {
                max_acceptable_fee_pct
            };
            let fee_impact_pct = roundtrip_fee * candidate_leverage;
            if fee_impact_pct > max_fee_limit {
                if arena.tick_counter.load(Ordering::Relaxed) % 100_000 == 0 {
                    println!("🔍 [RISK REJECT] FEE_IMPACT: fee_impact={:.6} > limit={:.6}", fee_impact_pct, max_fee_limit);
                }
                return ValidatedOrder::rejected();
            }
            dynamic_leverage = candidate_leverage
                .min(max_exchange_leverage)
                .min(50.0);
        }

        // FASE 3 FIX: Micro-Account Notional Safety
        // Sumamos un centavo de dólar (+0.1) al min notional para evitar rechazos
        // por pérdida de precisión IEEE-754 en multiplicaciones de apalancamiento
        let safe_min_notional = dynamic_min_notional + 0.1;
        let required_margin_for_min_notional = safe_min_notional / dynamic_leverage;
        if final_margin < required_margin_for_min_notional {
            final_margin = required_margin_for_min_notional;
        }

        let (meets_min_notional, _) = guard::enforce_minimum_notional(
            final_margin,
            safe_min_notional,
            dynamic_leverage,
        );
        if !meets_min_notional {
            return ValidatedOrder::rejected();
        }

        let safe_limit = (allocated_capital * safe_cushion).min(current_cap * 0.90);
        if final_margin > safe_limit {
            final_margin = safe_limit;
        }
        if final_margin < required_margin_for_min_notional {
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
        // FIX #790: Evitar forzar maker_only en cuentas micro (< $1000 USD).
        // En cuentas micro, forzar Post-Only en el precio actual causa rechazos -5022 de Binance y paraliza el bot al llegar a $50.
        let maker_only = allocated_capital >= maker_capital_threshold && maker_capital_threshold >= 1000.0;

        // R4.6 / H9: Bases TP y SL desacopladas por horizonte (Scalping vs Swing)
        let sl_mult = arena.config.sl_atr_multiplier.load(Ordering::Relaxed).clamp(0.5, 5.0);
        let sl_base = if is_scalp {
            arena.config.scalp_sl_base.load(Ordering::Relaxed).clamp(0.001, 0.50)
        } else {
            arena.config.swing_sl_base.load(Ordering::Relaxed).clamp(0.001, 0.50)
        };
        let sl_pct = (current_atr * sl_mult / current_price).clamp(sl_base * 0.5, sl_base * 2.5);

        let final_sl = if intent.sl_price_target > 0.0 {
            intent.sl_price_target
        } else if dir > 0.0 {
            current_price * (1.0 - sl_pct)
        } else {
            current_price * (1.0 + sl_pct)
        };

        // FASE 3: el ratio TP/SL ya no es el literal 2.0 — usa el gen RR
        // evolucionable del genoma (tp_rr_ratio_btc, bounds 1.0-10.0; el
        // nombre es histórico, gobierna el RR global). El clamp 1.5-6.0
        // permanece como riel de seguridad.
        let rr_ratio = arena
            .config
            .tp_rr_ratio_btc
            .load(Ordering::Relaxed)
            .clamp(1.0, 10.0);
        let tp_mult = (sl_mult * rr_ratio).clamp(1.5, 6.0);
        let tp_base = if is_scalp {
            arena.config.scalp_tp_base.load(Ordering::Relaxed).clamp(0.001, 0.50)
        } else {
            arena.config.swing_tp_base.load(Ordering::Relaxed).clamp(0.001, 0.50)
        };
        let tp_pct = (current_atr * tp_mult / current_price).clamp(tp_base * 0.5, tp_base * 3.0);

        let final_tp = if intent.tp_price_target > 0.0 {
            intent.tp_price_target
        } else if dir > 0.0 {
            current_price * (1.0 + tp_pct)
        } else {
            current_price * (1.0 - tp_pct)
        };

        let safe_tp = if final_tp.is_finite() && final_tp > 0.0 { final_tp } else { 0.0 };
        let safe_sl = if final_sl.is_finite() && final_sl > 0.0 { final_sl } else { 0.0 };
        let safe_vol = if final_margin.is_finite() && final_margin > 0.0 { final_margin } else { 0.0 };
        let safe_lev = if dynamic_leverage.is_finite() && dynamic_leverage >= 1.0 { dynamic_leverage } else { 1.0 };

        if safe_vol <= 0.0 || (intent.signal != SignalType::Flat && (safe_tp <= 0.0 || safe_sl <= 0.0)) {
            return ValidatedOrder::rejected();
        }

        ValidatedOrder {
            signal: intent.signal,
            volume_usd: safe_vol,
            leverage: safe_lev,
            maker_only,
            tp_target: safe_tp,
            sl_target: safe_sl,
            fee_buffer_multiplier: ev_fee_multiplier,
        }
    }
}
