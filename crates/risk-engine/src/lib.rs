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

use signal_engine::{SignalIntent, SignalType, TradeHorizon};
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

/// DIAGNÓSTICO SIGNAL-PATH: contadores de rechazo por compuerta de
/// evaluate_single_intent. Índices:
/// 0=flat/coin 1=exposure0 2=correlación 3=spec 4=EV 5=fee_impact
/// 6=min_notional 7=margen_insuf 8=orchestrator 9=otros
use std::sync::atomic::AtomicU64;
pub static REJECT_COUNTERS: [std::sync::atomic::AtomicU64; 10] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn rej(i: usize) -> ValidatedOrder {
    REJECT_COUNTERS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ValidatedOrder::rejected()
}

pub fn reject_report() -> String {
    let names = [
        "flat/coin",
        "exposure0",
        "correlacion",
        "spec",
        "EV",
        "fee_impact",
        "min_notional",
        "margen_insuf",
        "orchestrator",
        "otros",
    ];
    let v: Vec<String> = REJECT_COUNTERS
        .iter()
        .enumerate()
        .filter(|(_i, c)| c.load(std::sync::atomic::Ordering::Relaxed) > 0)
        .map(|(i, c)| {
            format!(
                "{}={}",
                names[i],
                c.load(std::sync::atomic::Ordering::Relaxed)
            )
        })
        .collect();
    if v.is_empty() {
        "sin rechazos".into()
    } else {
        v.join(" ")
    }
}

pub struct RiskEngine {
    pub peak_capital: f64,
    pub scalp_peak_capital: f64,
    pub swing_peak_capital: f64,
    /// R1.3 — split de capital suavizado (Robbins-Monro).
    pub smoothed_split: Option<f64>,
}

impl RiskEngine {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            peak_capital: initial_capital,
            scalp_peak_capital: initial_capital,
            swing_peak_capital: initial_capital,
            smoothed_split: None,
        }
    }

    pub fn reset(&mut self, initial_capital: f64) {
        self.peak_capital = initial_capital;
        self.scalp_peak_capital = initial_capital;
        self.swing_peak_capital = initial_capital;
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
            let capital_ratio =
                (self.scalp_peak_capital / (base_capital * split).max(1.0)).max(1.0);
            (hard_stop_base / (1.0 + capital_ratio.ln() * hard_stop_decay)).clamp(0.20, 0.95)
        };

        let swing_hard_stop_limit = {
            let capital_ratio =
                (self.swing_peak_capital / (base_capital * (1.0 - split)).max(1.0)).max(1.0);
            (hard_stop_base / (1.0 + capital_ratio.ln() * hard_stop_decay)).clamp(0.20, 0.95)
        };

        // D-126: Seguimiento real del peak capital y validación de drawdown
        self.scalp_peak_capital = self.scalp_peak_capital.max(scalp_capital);
        self.swing_peak_capital = self.swing_peak_capital.max(swing_capital);

        let mut scalp_valid = scalp_drawdown < scalp_hard_stop_limit;
        let mut swing_valid = swing_drawdown < swing_hard_stop_limit;

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
            scalp_valid = false;
        }

        if !guard::check_drawdown_limit(
            swing_capital,
            self.swing_peak_capital,
            max_dd,
            base_capital * (1.0 - split),
            guard_dd_sigmoid_steepness,
            guard_dd_sigmoid_center,
        ) {
            swing_valid = false;
        }

        let coin = &arena.coins[coin_id];
        let scalp_wr = coin.metrics.win_rate.load(Ordering::Relaxed);
        let scalp_pf = coin.metrics.profit_factor.load(Ordering::Relaxed);
        let swing_wr = coin.metrics.win_rate.load(Ordering::Relaxed);
        let swing_pf = coin.metrics.profit_factor.load(Ordering::Relaxed);

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
            scalp_kelly = (kelly_cold * split.max(0.1)).clamp(0.05, 0.35);
            swing_kelly = (kelly_cold * (1.0 - split).max(0.1)).clamp(0.05, 0.35);
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

    /// Evalúa la intención unificada de señal cuántica continua sobre el 100% del capital disponible.
    pub fn evaluate_quantum_order(
        &mut self,
        coin_id: usize,
        intent: &SignalIntent,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        if coin_id >= arena.coins.len() || intent.signal == SignalType::Flat {
            return rej(0);
        }

        let current_capital = arena.unified_capital.load(Ordering::Relaxed);
        if !current_capital.is_finite() || current_capital <= 0.0 {
            return ValidatedOrder::rejected();
        }

        if current_capital > self.peak_capital {
            self.peak_capital = current_capital;
        }
        if current_capital > self.scalp_peak_capital {
            self.scalp_peak_capital = current_capital;
        }
        if current_capital > self.swing_peak_capital {
            self.swing_peak_capital = current_capital;
        }

        // O-04 — CIRCUIT BREAKER DE DRAWDOWN en el path cuántico: el gen
        // global_max_drawdown era funcionalmente MUERTO (se convertía en cap
        // de margen constante). Ahora es un VETO REAL: si el drawdown desde
        // el pico supera el gen, no se abre nueva posición hasta recuperación.
        let max_dd = arena.config.global_max_drawdown.load(Ordering::Relaxed);
        if self.peak_capital > 0.0 && max_dd > 0.0 && max_dd < 1.0 {
            let dd = (self.peak_capital - current_capital) / self.peak_capital;
            if dd >= max_dd {
                return rej(2); // correlación bucket para no crear índice nuevo
            }
        }

        let base_capital = arena.config.base_capital.load(Ordering::Relaxed);
        let pf = arena.coins[coin_id]
            .metrics
            .profit_factor
            .load(Ordering::Relaxed);
        let clamp_min = arena
            .config
            .kelly_clamp_min
            .load(Ordering::Relaxed)
            .max(0.0);
        let clamp_max = arena
            .config
            .kelly_clamp_max
            .load(Ordering::Relaxed)
            .clamp(clamp_min, 1.0);
        let raw_kelly = arena.coins[coin_id]
            .metrics
            .kelly_fraction
            .load(Ordering::Relaxed);

        let kelly_cold = arena
            .config
            .kelly_bootstrap_cold
            .load(Ordering::Relaxed)
            .clamp(0.05, 0.35);
        let kelly_frac = if raw_kelly <= 0.0 {
            kelly_cold
        } else {
            raw_kelly.clamp(clamp_min, clamp_max)
        };

        let temporal_scale = arena
            .config
            .temporal_scale
            .load(Ordering::Relaxed)
            .clamp(0.0, 1.0);
        let is_scalp = match intent.horizon {
            TradeHorizon::Scalp => true,
            TradeHorizon::Swing => false,
            TradeHorizon::Continuous => temporal_scale < 0.5,
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

    /// Evalúa la intención cuántica continua (alias para compatibilidad)
    pub fn evaluate_quantum_order_by_horizon(
        &mut self,
        coin_id: usize,
        intent: &SignalIntent,
        _is_scalp: bool,
        arena: &GlobalArena,
    ) -> ValidatedOrder {
        self.evaluate_quantum_order(coin_id, intent, arena)
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

        // O-03 — KELLY ESCALADO POR RIESGO DEL STOP: la fracción Kelly es
        // asintóticamente proporcional al edge/riesgo; si el stop es K× más
        // ancho que el extremo corto, la MISMA fracción arriesga K× más
        // capital por trade. Escalamos por temporal_scale para que el RIESGO
        // POR TRADE sea constante en el continuo (sin el salto invisible
        // de 4-8x en riesgo entre extremos).
        let scalp_sl_ref = arena.config.scalp_sl_base.load(Ordering::Relaxed).max(1e-6);
        let swing_sl_ref = arena.config.swing_sl_base.load(Ordering::Relaxed).max(1e-6);
        let ts = arena
            .config
            .temporal_scale
            .load(Ordering::Relaxed)
            .clamp(0.05, 0.95);
        let sl_interp_est = scalp_sl_ref * (1.0 - ts) + swing_sl_ref * ts;
        let risk_normalizer = (scalp_sl_ref / sl_interp_est).clamp(0.15, 1.0);
        let kelly_adjusted = kelly_fraction * risk_normalizer;

        let raw_exposure = dir * intent.confidence * kelly_adjusted * allocated_capital;
        if raw_exposure == 0.0 {
            return rej(1);
        }

        // FASE 16 & BUG-578: Correlation Guard (Continuous Universal)
        let is_long = intent.signal == SignalType::Long;
        let mut same_dir_count = 0;
        for c in arena.coins.iter() {
            let pos = &c.positions.position;
            if pos.is_open() && (pos.is_long.load(Ordering::Relaxed) == is_long) {
                same_dir_count += 1;
            }
        }
        let corr_thresh = arena
            .config
            .global_correlation_threshold
            .load(Ordering::Relaxed);
        let max_allowed_cluster = (corr_thresh * 5.0).round() as usize;
        let current_cap = arena.unified_capital.load(Ordering::Relaxed);
        if correlation_guard::CorrelationGuardEngine::is_correlation_vetoed_by_horizon(
            is_scalp,
            same_dir_count,
            current_cap,
            max_allowed_cluster.max(2),
        ) {
            return rej(2);
        }

        let coin = &arena.coins[coin_id];
        let spec = match quantum_arena::symbol_registry::try_spec(coin_id) {
            Some(s) => s,
            None => return rej(3),
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
            let btc_price = arena.coins[0]
                .current_price
                .load(Ordering::Relaxed)
                .max(1e-8);
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

        // U-C — FIX ZOMBIE: coin.scalp/swing ya nadie los escribe (motor
        // unificado -> coin.metrics). Antes el leverage leía un win-rate
        // congelado en el valor inicial para SIEMPRE: el sizing dinámico
        // jamás aprendía de resultados (never-start). Ahora lee la fuente
        // viva única.
        let real_win_rate = coin.metrics.win_rate.load(Ordering::Relaxed);

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
        // D-01 — FRICCIÓN REAL EN EL EV GATE: antes el gate comparaba contra
        // solo maker+taker (~7bps), pero el motor EJECUTA 2×(taker + slippage)
        // por roundtrip bajo HyperRealistic. El gate subestimaba la fricción
        // 40-70% y certificaba como EV-positivos trades que la física del
        // propio motor volvía negativos. Ahora incluimos la fricción de
        // física que el fill path aplica (entrada taker + salida mayormente
        // taker, cada una con max(impacto_cuadrático+latency, floor)).
        let slip_floor = arena
            .config
            .base_slippage_floor
            .load(Ordering::Relaxed)
            .max(0.00001);
        let lat_ms = arena
            .config
            .latency_penalty_ms
            .load(Ordering::Relaxed)
            .max(0.0);
        let latency_slip = atr_pct * (lat_ms / 150.0);
        let per_side_slip = (slip_floor + latency_slip).clamp(0.0, 0.05); // H-9: alineado con física (antes 0.01 < 0.05)
        let roundtrip_fee = (maker_fee + taker_fee) + 2.0 * per_side_slip;

        let temp_scale = arena
            .config
            .temporal_scale
            .load(Ordering::Relaxed)
            .clamp(0.0, 1.0);
        let s_eval = match intent.horizon {
            TradeHorizon::Scalp => 0.0,
            TradeHorizon::Swing => 1.0,
            TradeHorizon::Continuous => temp_scale,
        };
        let scalp_win = arena
            .config
            .scalp_tp_base
            .load(Ordering::Relaxed)
            .max(0.0010)
            .max(atr_pct * 1.5);
        let swing_win = arena
            .config
            .swing_tp_base
            .load(Ordering::Relaxed)
            .max(0.0050)
            .max(atr_pct * 3.0);
        let expected_win = scalp_win * (1.0 - s_eval) + swing_win * s_eval;

        let scalp_loss = arena
            .config
            .scalp_sl_base
            .load(Ordering::Relaxed)
            .max(0.0005)
            .max(atr_pct * 0.8);
        let swing_loss = arena
            .config
            .swing_sl_base
            .load(Ordering::Relaxed)
            .max(0.0020)
            .max(atr_pct * 1.5);
        let expected_loss = scalp_loss * (1.0 - s_eval) + swing_loss * s_eval;

        let confidence = intent.confidence.max(0.51);
        let min_required_confidence = if allocated_capital <= 15.0 {
            0.78 // Sniper threshold para micro-cuenta ($13 USD bootstrap): elimina ruido y preserva capital
        } else {
            0.62
        };
        if confidence < min_required_confidence {
            return rej(4);
        }
        let expected_value_pct = (confidence * expected_win) - ((1.0 - confidence) * expected_loss);

        let min_ev_mult = if allocated_capital <= 15.0 {
            1.25 // Micro-cuenta: requiere EV al menos 25% por encima de las comisiones reales
        } else {
            1.05
        };
        let ev_fee_multiplier = arena
            .config
            .ev_fee_multiplier
            .load(Ordering::Relaxed)
            .clamp(min_ev_mult, 1.80);
        if expected_value_pct <= (roundtrip_fee * ev_fee_multiplier) {
            return rej(4);
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

        // D-130: Evaluar si el notional real de la orden (final_margin * dynamic_leverage) cumple con el mínimo
        if final_margin > 0.0 && final_margin * dynamic_leverage < dynamic_min_notional {
            // FIX min_notional (diag R4): el leverage necesario para alcanzar
            // el notional mínimo se calcula sobre el MARGEN DEL TRADE, no
            // sobre el capital total. La fórmula anterior
            // (min_notional/(allocated*cushion)) producía leverage < 1 en
            // micro-cuenta y garantizaba el rechazo posterior: era la causa
            // de 0 trades en la certificación 30d (min_notional ~500-640
            // rechazos/día con señales sanas de conf 0.7+). El fee_impact
            // check de abajo sigue limitando el costo.
            let candidate_leverage = (dynamic_min_notional / final_margin.max(0.01)) * 1.02;
            let max_fee_limit = if allocated_capital <= 15.0 {
                0.035 // Permitir hasta 3.5% fee impact para bootstrap micro-cuentas ($13 USD) para cumplir con el lote mínimo de Binance
            } else {
                max_acceptable_fee_pct
            };
            let fee_impact_pct = roundtrip_fee * candidate_leverage;
            if fee_impact_pct > max_fee_limit {
                if arena.tick_counter.load(Ordering::Relaxed) % 100_000 == 0 {
                    println!(
                        "🔍 [RISK REJECT] FEE_IMPACT: fee_impact={:.6} > limit={:.6}",
                        fee_impact_pct, max_fee_limit
                    );
                }
                return rej(5);
            }
            dynamic_leverage = candidate_leverage.min(max_exchange_leverage).min(50.0);
        }

        // FASE 3 FIX: Micro-Account Notional Safety
        // Sumamos un centavo de dólar (+0.1) al min notional para evitar rechazos
        // por pérdida de precisión IEEE-754 en multiplicaciones de apalancamiento
        let safe_min_notional = dynamic_min_notional + 0.1;
        let required_margin_for_min_notional = safe_min_notional / dynamic_leverage;
        if final_margin < required_margin_for_min_notional {
            // FIX FP (diag R4): lev x (min_notional/lev) puede dar
            // 5.0999... < 5.1 en punto flotante y rechazar la orden en el
            // borde EXACTO — la causa terminal de los 0 trades. Épsilon
            // relativo de 5 bps de margen cierra la frontera.
            final_margin = required_margin_for_min_notional * 1.0005;
        }

        let (meets_min_notional, _) =
            guard::enforce_minimum_notional(final_margin, safe_min_notional, dynamic_leverage);
        if !meets_min_notional {
            if REJECT_COUNTERS[6].load(std::sync::atomic::Ordering::Relaxed) % 200 == 0 {
                println!(
                    "🔍 [REJ6] margin={:.4} lev={:.4} min_notional={:.4} allocated={:.4} kelly={:.4}",
                    final_margin, dynamic_leverage, safe_min_notional, allocated_capital, kelly_fraction
                );
            }
            return rej(6);
        }

        let safe_limit = if allocated_capital <= 15.0 {
            // Micro-cuenta ($13 USD): Permitir hasta 20-25% de margen ($1.50 a $2.60)
            // para que con apalancamiento prudente (2x - 5x) cubra los $5.10 de Binance sin rej(7).
            (allocated_capital * 0.25).clamp(1.20, 2.60)
        } else {
            (allocated_capital * safe_cushion).min(current_cap * 0.90)
        };
        if final_margin > safe_limit {
            final_margin = safe_limit;
            if final_margin > 0.0 && final_margin * dynamic_leverage < safe_min_notional {
                let re_lev = (safe_min_notional / final_margin) * 1.01;
                let fee_impact = roundtrip_fee * re_lev;
                let max_fee_lim = if allocated_capital <= 15.0 {
                    0.035
                } else {
                    max_acceptable_fee_pct
                };
                if fee_impact <= max_fee_lim {
                    dynamic_leverage = re_lev.min(max_exchange_leverage).min(50.0);
                }
            }
        }
        let required_margin_for_min_notional = safe_min_notional / dynamic_leverage;
        if final_margin < required_margin_for_min_notional {
            return rej(7);
        }

        let orchestrator = orchestrator::PortfolioOrchestrator::new(arena);
        let raw_regime = arena.market_regime.load(Ordering::Relaxed);
        let regime = crate::regime::MarketRegime::from(raw_regime);

        if !orchestrator.allow_trade(bounded_exposure > 0.0, final_margin, regime) {
            return rej(8);
        }

        let maker_capital_threshold = arena
            .config
            .maker_only_capital_threshold
            .load(Ordering::Relaxed);
        // FIX #790: Evitar forzar maker_only en cuentas micro (< $1000 USD).
        // En cuentas micro, forzar Post-Only en el precio actual causa rechazos -5022 de Binance y paraliza el bot al llegar a $50.
        let maker_only =
            allocated_capital >= maker_capital_threshold && maker_capital_threshold >= 1000.0;

        // R4.6 / H9: Bases TP y SL desacopladas por horizonte (Scalping vs Swing vs Continuous)
        let sl_mult = arena
            .config
            .sl_atr_multiplier
            .load(Ordering::Relaxed)
            .clamp(0.5, 5.0);
        let temporal_s_eval = arena
            .config
            .temporal_scale
            .load(Ordering::Relaxed)
            .clamp(0.05, 0.95);

        let (sl_base, tp_base) = match intent.horizon {
            TradeHorizon::Scalp => {
                let sl = arena
                    .config
                    .scalp_sl_base
                    .load(Ordering::Relaxed)
                    .clamp(0.0005, 0.0100);
                let tp = arena
                    .config
                    .scalp_tp_base
                    .load(Ordering::Relaxed)
                    .clamp(0.0010, 0.0300);
                (sl, tp)
            }
            TradeHorizon::Swing => {
                let sl = arena
                    .config
                    .swing_sl_base
                    .load(Ordering::Relaxed)
                    .clamp(0.0020, 0.0500);
                let tp = arena
                    .config
                    .swing_tp_base
                    .load(Ordering::Relaxed)
                    .clamp(0.0050, 0.1000);
                (sl, tp)
            }
            TradeHorizon::Continuous => {
                // D-249: Interpolación lineal continua pura entre scalp y swing según temporal_s_eval
                let scalp_sl = arena
                    .config
                    .scalp_sl_base
                    .load(Ordering::Relaxed)
                    .clamp(0.0005, 0.0100);
                let swing_sl = arena
                    .config
                    .swing_sl_base
                    .load(Ordering::Relaxed)
                    .clamp(0.0020, 0.0500);
                let scalp_tp = arena
                    .config
                    .scalp_tp_base
                    .load(Ordering::Relaxed)
                    .clamp(0.0010, 0.0300);
                let swing_tp = arena
                    .config
                    .swing_tp_base
                    .load(Ordering::Relaxed)
                    .clamp(0.0050, 0.1000);
                (
                    scalp_sl * (1.0 - temporal_s_eval) + swing_sl * temporal_s_eval,
                    scalp_tp * (1.0 - temporal_s_eval) + swing_tp * temporal_s_eval,
                )
            }
        };

        // Inmunidad contra ruido browniano: el stop loss nunca debe descender por debajo de 35 bps
        // en crypto para evitar ser detenido por el micro-spread y oscilaciones top-of-book.
        let min_safe_sl = match intent.horizon {
            TradeHorizon::Scalp => 0.0035,
            TradeHorizon::Swing => 0.0060,
            TradeHorizon::Continuous => 0.0035 * (1.0 - temporal_s_eval) + 0.0060 * temporal_s_eval,
        };
        let sl_pct =
            (current_atr * sl_mult / current_price).clamp(sl_base.max(min_safe_sl), sl_base * 2.5);

        let final_sl = if intent.sl_price_target > 0.0 {
            intent.sl_price_target
        } else if dir > 0.0 {
            current_price * (1.0 - sl_pct)
        } else {
            current_price * (1.0 + sl_pct)
        };

        // FASE 3: el ratio TP/SL usa el gen RR evolucionable del genoma
        let rr_ratio = arena
            .config
            .tp_rr_ratio_btc
            .load(Ordering::Relaxed)
            .clamp(1.0, 10.0);
        let tp_mult = (sl_mult * rr_ratio).clamp(1.5, 6.0);
        let tp_pct = (current_atr * tp_mult / current_price).clamp(tp_base * 0.5, tp_base * 3.0);

        let final_tp = if intent.tp_price_target > 0.0 {
            intent.tp_price_target
        } else if dir > 0.0 {
            current_price * (1.0 + tp_pct)
        } else {
            current_price * (1.0 - tp_pct)
        };

        let safe_tp = if final_tp.is_finite() && final_tp > 0.0 {
            final_tp
        } else {
            0.0
        };
        let safe_sl = if final_sl.is_finite() && final_sl > 0.0 {
            final_sl
        } else {
            0.0
        };
        let safe_vol = if final_margin.is_finite() && final_margin > 0.0 {
            final_margin
        } else {
            0.0
        };
        let safe_lev = if dynamic_leverage.is_finite() && dynamic_leverage >= 1.0 {
            dynamic_leverage
        } else {
            1.0
        };

        if safe_vol <= 0.0
            || (intent.signal != SignalType::Flat && (safe_tp <= 0.0 || safe_sl <= 0.0))
        {
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
