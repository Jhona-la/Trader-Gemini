use crate::GlobalArena;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;

/// Axioma X: SuperGenotype (El ADN de Trader Gemini)
/// Integra todos los 50+ parámetros del God Engine en un solo vector.
fn default_temporal_scale() -> f64 {
    0.5
}
fn default_swing_obi_threshold() -> f64 {
    0.5
}
fn default_swing_accel_min_samples() -> f64 {
    30.0
}
/// Anclas por defecto para genomas legacy serializados sin curvas (serde):
/// pasan por los puntos históricos de las dos bandas — carga vieja = mismo
/// comportamiento, sin ruptura.
fn default_tp_curve() -> crate::temporal_spectrum::HorizonCurve {
    crate::temporal_spectrum::HorizonCurve::through_two_points(
        crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
        0.012,
        crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
        0.06,
    )
}
fn default_sl_curve() -> crate::temporal_spectrum::HorizonCurve {
    crate::temporal_spectrum::HorizonCurve::through_two_points(
        crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
        0.006,
        crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
        0.025,
    )
}
fn default_kelly_curve() -> crate::temporal_spectrum::HorizonCurve {
    crate::temporal_spectrum::HorizonCurve::through_two_points(
        crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
        0.20,
        crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
        0.15,
    )
}
fn default_trail_mult_curve() -> crate::temporal_spectrum::HorizonCurve {
    crate::temporal_spectrum::HorizonCurve::through_two_points(
        crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
        2.5,
        crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
        3.5,
    )
}
fn default_trail_act_curve() -> crate::temporal_spectrum::HorizonCurve {
    crate::temporal_spectrum::HorizonCurve::through_two_points(
        crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
        2.0,
        crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
        3.0,
    )
}
fn default_trail_step_curve() -> crate::temporal_spectrum::HorizonCurve {
    crate::temporal_spectrum::HorizonCurve::through_two_points(
        crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
        1.2,
        crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
        1.8,
    )
}
fn default_obi_curve() -> crate::temporal_spectrum::HorizonCurve {
    crate::temporal_spectrum::HorizonCurve::through_two_points(
        crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
        0.25,
        crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
        0.40,
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperGenotype {
    pub global_max_drawdown: f64,
    pub global_leverage: f64,
    pub btc_volatility_multiplier: f64,
    pub eth_volatility_multiplier: f64,
    pub min_trades_per_day: f64,
    pub survival_capital_threshold: f64,
    // 4. Capa RÉGIMEN DE MERCADO (Trend, Range, Volatile)
    pub global_correlation_threshold: f64,
    pub funding_rate_sensitivity: f64,
    pub trend_threshold: f64,
    pub range_threshold: f64,
    pub scalp_kelly_fraction: f64,
    pub swing_kelly_fraction: f64,
    pub scalp_obi_threshold: f64,
    pub scalp_tp_base: f64,
    pub scalp_sl_base: f64,
    pub swing_tp_base: f64,
    pub swing_sl_base: f64,
    /// F8 — TP y SL como curvas continuas del horizonte τ (ms). Los campos
    /// scalp_*/swing_* quedan como ANCLAS legacy (vistas de la curva en las
    /// bandas históricas); los nuevos genes del GA son (a, b) por curva.
    #[serde(default = "default_tp_curve")]
    pub tp_horizon_curve: crate::temporal_spectrum::HorizonCurve,
    #[serde(default = "default_sl_curve")]
    pub sl_horizon_curve: crate::temporal_spectrum::HorizonCurve,
    #[serde(default = "default_kelly_curve")]
    pub kelly_horizon_curve: crate::temporal_spectrum::HorizonCurve,
    #[serde(default = "default_trail_mult_curve")]
    pub trail_mult_horizon_curve: crate::temporal_spectrum::HorizonCurve,
    #[serde(default = "default_trail_act_curve")]
    pub trail_act_horizon_curve: crate::temporal_spectrum::HorizonCurve,
    #[serde(default = "default_trail_step_curve")]
    pub trail_step_horizon_curve: crate::temporal_spectrum::HorizonCurve,
    #[serde(default = "default_obi_curve")]
    pub obi_horizon_curve: crate::temporal_spectrum::HorizonCurve,
    pub sl_atr_mult_btc: f64,
    pub tp_rr_ratio_btc: f64,
    pub min_confidence_btc: f64,
    pub veto_threshold_btc: f64,
    pub tech_threshold: f64,
    pub ml_threshold_long: f64,
    pub ml_threshold_short: f64,
    pub maker_spread_pct: f64,
    pub maker_obi_threshold: f64,
    pub target_volatility: f64,
    pub dynamic_atr_min: f64,
    pub dynamic_obi_threshold: f64,
    pub dynamic_ema_trend: f64,
    pub dynamic_ofi_threshold: f64,
    pub capital_split_scalp: f64,
    pub kelly_clamp_min: f64,
    pub kelly_clamp_max: f64,
    pub explosive_leverage_multiplier: f64,
    pub quantum_mutation_rate: f64,
    pub temporal_memory_decay: f64,
    pub leverage_cap: f64,
    pub explosive_confidence_threshold: f64,
    pub weight_obi: f64,
    pub weight_ofi: f64,
    pub weight_vpin: f64,
    pub regime_duration_ms: f64,
    pub regime_atr_multiplier: f64,
    pub scalp_trail_act_atr: f64,
    pub scalp_trail_step_atr: f64,
    pub scalp_trail_max_atr: f64,
    pub scalp_trail_min_pnl: f64,
    pub scalp_trail_atr_mult_base: f64,
    pub swing_trail_act_atr: f64,
    pub swing_trail_step_atr: f64,
    pub swing_trail_max_atr: f64,
    pub swing_trail_min_pnl: f64,
    pub swing_trail_atr_mult_base: f64,
    pub zombie_timeout_ms: f64,
    pub hurst_trend_threshold: f64,
    pub cvd_veto_threshold: f64,
    pub wall_veto_threshold: f64,
    pub flash_crash_jump_pct: f64,
    pub latency_ms_panic_threshold: f64,
    // --- Adaptive Feature Engineering (Phase 80) ---
    pub ema_fast_period: f64,
    pub ema_slow_period: f64,
    pub hurst_scalp_threshold: f64, // Below this = Scalping regime
    pub hurst_swing_threshold: f64, // Above this = Swing regime
    // --- Risk Engine Adaptivo (Phase 81) ---
    pub synergy_exposure_boost: f64,
    pub synergy_leverage_boost: f64,
    pub max_fee_pct: f64,
    pub kelly_bootstrap_cold: f64,
    // --- Fine-tuning Physics & Spot (Phase 82) ---
    pub latency_penalty_ms: f64,
    pub base_slippage_floor: f64,
    pub spot_spread_threshold: f64,
    pub spot_bias_value: f64,
    pub obi_confidence_fallback: f64,
    // --- Cognitive Integrity & ML Hardcode Eradication (Phase 9) ---
    pub ml_clip_lower: f64,
    pub ml_clip_upper: f64,
    pub tensor_poly_a: f64,
    pub tensor_poly_b: f64,
    // --- Meta-Council & Fractional Dynamics (Phase 10) ---
    pub fractional_alpha_order: f64,
    pub fractional_clip_max: f64,
    pub bft_consensus_tolerance: f64,
    // --- Signal Engine & Tensor Math (Phase 11) ---
    pub turbo_coherence_threshold: f64,
    pub turbo_z_score_stdev: f64,
    pub sl_atr_multiplier: f64,
    pub coaxial_squeeze_threshold: f64,
    pub tensor_op_add_bias: f64,
    pub tensor_op_mul_weight: f64,
    // --- Phase 13: Topological Evolution (NEAT/Quantum Tensors) ---
    pub topo_layer_1_activation: f64,
    pub topo_layer_2_activation: f64,
    pub tensor_dropout_rate: f64,
    pub quantum_entropy_seed: f64,
    // --- Phase 25: PPO Engine Adaptive Weights & Learning ---
    pub ppo_weight_ofi: f64,
    pub ppo_weight_obi: f64,
    pub ppo_weight_hawkes: f64,
    pub ppo_weight_leadlag: f64,
    pub ppo_weight_regime: f64,
    pub global_learning_rate: f64,
    pub global_momentum: f64,
    pub vecm_alpha_speed: f64,
    pub vecm_beta_hedge: f64,
    pub conformal_alpha: f64,
    // --- Phase 99: Mathematical Hardcode Eradication ---
    pub ppo_clip_eps: f64,
    pub ppo_weight_min_clip: f64,
    pub hard_stop_decay_factor: f64,
    pub hard_stop_base_limit: f64,
    pub kelly_bootstrap_ratio_threshold: f64,
    pub kelly_bootstrap_min_exposure: f64,
    pub ev_fee_multiplier: f64,
    pub margin_cushion_pct: f64,
    pub maker_only_capital_threshold: f64,
    pub hawkes_scalp_threshold: f64,
    pub obi_zscore_threshold: f64,
    // --- FASE 14: Erradicación Matemática de Capital Fijo (13 USD vs 10k USD) ---
    pub hawkes_volume_norm: f64,
    pub base_duration_ms: f64,

    // FASE 3: Kelly, Guard & Orchestrator Evolutive Parameters
    pub kelly_survival_cap_ratio: f64,
    pub kelly_expansion_mult: f64,
    pub guard_dd_sigmoid_steepness: f64,
    pub guard_dd_sigmoid_center: f64,
    pub portfolio_perf_mult_steepness: f64,
    pub portfolio_dd_penalty_decay: f64,
    pub portfolio_perf_mult_min: f64,
    pub portfolio_perf_mult_max: f64,
    pub portfolio_perf_mult_center: f64,

    // --- FASE 15: Optimizador Macro-Regime Genómico ---
    pub macro_hurst_confidence_offset: f64,
    pub macro_hurst_confidence_scale: f64,
    pub macro_vol_confidence_scale: f64,
    pub macro_min_cooldown_ratio: f64,
    pub macro_max_cooldown_ratio: f64,
    pub macro_cooldown_reduction_factor: f64,
    pub macro_leverage_momentum_scale: f64,

    // --- FASE 4: Quantum De-hardcoding (Leverage, Scalp, Executor) ---
    pub lev_matrix_vol_clamp_min: f64,
    pub lev_matrix_growth_scalar: f64,
    pub lev_matrix_log_cap_divisor: f64,
    pub scalp_accel_min_samples: f64,
    pub executor_max_orders_10s: f64,
    pub executor_max_weight_1m: f64,

    // --- FASE 6: Iceberg & IOC Limits ---
    pub iceberg_volume_threshold: f64,
    pub iceberg_slice_count: f64,
    #[serde(default = "default_swing_obi_threshold")]
    pub swing_obi_threshold: f64,
    #[serde(default = "default_swing_accel_min_samples")]
    pub swing_accel_min_samples: f64,
    /// F3-2 — EJE TEMPORAL CONTINUO: s ∈ [0,1] posiciona el ciclo de vida
    /// de la posición en el continuo temporal: s=0 extremo corto (scalp
    /// puro), s=1 extremo largo (swing puro). Los parámetros del extremo
    /// largo se derivan del corto por span exponencial: span = 10^(2s) —
    /// s=0.5 da ×10 (el ratio histórico scalp↔swing). UN dial evolutivo
    /// reemplaza la dicotomía de genes duplicados por horizonte.
    #[serde(default = "default_temporal_scale")]
    pub temporal_scale: f64,
}

impl Default for SuperGenotype {
    fn default() -> Self {
        Self::new_baseline(0.0002, 0.0005)
    }
}

impl SuperGenotype {
    pub fn current_from_arena(arena: &GlobalArena) -> Self {
        Self {
            global_max_drawdown: arena.config.global_max_drawdown.load(Ordering::Relaxed),
            global_leverage: arena.config.global_leverage.load(Ordering::Relaxed),
            btc_volatility_multiplier: arena
                .config
                .btc_volatility_multiplier
                .load(Ordering::Relaxed),
            eth_volatility_multiplier: arena
                .config
                .eth_volatility_multiplier
                .load(Ordering::Relaxed),
            min_trades_per_day: arena.config.min_trades_per_day.load(Ordering::Relaxed),
            survival_capital_threshold: arena
                .config
                .survival_capital_threshold
                .load(Ordering::Relaxed),
            funding_rate_sensitivity: arena
                .config
                .funding_rate_sensitivity
                .load(Ordering::Relaxed),
            global_correlation_threshold: arena
                .config
                .global_correlation_threshold
                .load(Ordering::Relaxed),
            trend_threshold: arena.config.trend_threshold.load(Ordering::Relaxed),
            range_threshold: arena.config.range_threshold.load(Ordering::Relaxed),
            scalp_kelly_fraction: arena.config.scalp_kelly_fraction.load(Ordering::Relaxed),
            swing_kelly_fraction: arena.config.swing_kelly_fraction.load(Ordering::Relaxed),
            scalp_obi_threshold: arena.config.scalp_obi_threshold.load(Ordering::Relaxed),
            scalp_tp_base: arena.config.scalp_tp_base.load(Ordering::Relaxed),
            scalp_sl_base: arena.config.scalp_sl_base.load(Ordering::Relaxed),
            swing_tp_base: arena.config.swing_tp_base.load(Ordering::Relaxed),
            swing_sl_base: arena.config.swing_sl_base.load(Ordering::Relaxed),
            tp_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                arena.config.scalp_tp_base.load(Ordering::Relaxed),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                arena.config.swing_tp_base.load(Ordering::Relaxed),
            ),
            sl_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                arena.config.scalp_sl_base.load(Ordering::Relaxed),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                arena.config.swing_sl_base.load(Ordering::Relaxed),
            ),
            kelly_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                arena.config.scalp_kelly_fraction.load(Ordering::Relaxed),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                arena.config.swing_kelly_fraction.load(Ordering::Relaxed),
            ),
            trail_mult_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                arena
                    .config
                    .scalp_trail_atr_mult_base
                    .load(Ordering::Relaxed),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                arena
                    .config
                    .swing_trail_atr_mult_base
                    .load(Ordering::Relaxed),
            ),
            trail_act_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                arena.config.scalp_trail_act_atr.load(Ordering::Relaxed),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                arena.config.swing_trail_act_atr.load(Ordering::Relaxed),
            ),
            trail_step_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                arena.config.scalp_trail_step_atr.load(Ordering::Relaxed),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                arena.config.swing_trail_step_atr.load(Ordering::Relaxed),
            ),
            obi_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                arena.config.scalp_obi_threshold.load(Ordering::Relaxed),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                arena.config.swing_obi_threshold.load(Ordering::Relaxed),
            ),
            sl_atr_mult_btc: arena.config.sl_atr_mult_btc.load(Ordering::Relaxed),
            tp_rr_ratio_btc: arena.config.tp_rr_ratio_btc.load(Ordering::Relaxed),
            min_confidence_btc: arena.config.min_confidence_btc.load(Ordering::Relaxed),
            veto_threshold_btc: arena.config.veto_threshold_btc.load(Ordering::Relaxed),
            tech_threshold: arena.config.tech_threshold.load(Ordering::Relaxed),
            ml_threshold_long: arena.config.ml_threshold_long.load(Ordering::Relaxed),
            ml_threshold_short: arena.config.ml_threshold_short.load(Ordering::Relaxed),
            maker_spread_pct: arena.config.maker_spread_pct.load(Ordering::Relaxed),
            maker_obi_threshold: arena.config.maker_obi_threshold.load(Ordering::Relaxed),
            target_volatility: arena.config.target_volatility.load(Ordering::Relaxed),
            dynamic_atr_min: arena.config.dynamic_atr_min.load(Ordering::Relaxed),
            dynamic_obi_threshold: arena.config.dynamic_obi_threshold.load(Ordering::Relaxed),
            dynamic_ema_trend: arena.config.dynamic_ema_trend.load(Ordering::Relaxed),
            dynamic_ofi_threshold: arena.config.dynamic_ofi_threshold.load(Ordering::Relaxed),
            capital_split_scalp: arena.config.capital_split_scalp.load(Ordering::Relaxed),
            kelly_clamp_min: arena.config.kelly_clamp_min.load(Ordering::Relaxed),
            kelly_clamp_max: arena.config.kelly_clamp_max.load(Ordering::Relaxed),
            explosive_leverage_multiplier: arena
                .config
                .explosive_leverage_multiplier
                .load(Ordering::Relaxed),
            quantum_mutation_rate: arena.config.quantum_mutation_rate.load(Ordering::Relaxed),
            temporal_memory_decay: arena.config.temporal_memory_decay.load(Ordering::Relaxed),
            leverage_cap: arena.config.leverage_cap.load(Ordering::Relaxed),
            explosive_confidence_threshold: arena
                .config
                .explosive_confidence_threshold
                .load(Ordering::Relaxed),
            weight_obi: arena.config.weight_obi.load(Ordering::Relaxed),
            weight_ofi: arena.config.weight_ofi.load(Ordering::Relaxed),
            weight_vpin: arena.config.weight_vpin.load(Ordering::Relaxed),
            regime_duration_ms: arena.config.regime_duration_ms.load(Ordering::Relaxed),
            regime_atr_multiplier: arena.config.regime_atr_multiplier.load(Ordering::Relaxed),
            scalp_trail_act_atr: arena.config.scalp_trail_act_atr.load(Ordering::Relaxed),
            scalp_trail_step_atr: arena.config.scalp_trail_step_atr.load(Ordering::Relaxed),
            scalp_trail_max_atr: arena.config.scalp_trail_max_atr.load(Ordering::Relaxed),
            scalp_trail_min_pnl: arena.config.scalp_trail_min_pnl.load(Ordering::Relaxed),
            scalp_trail_atr_mult_base: arena
                .config
                .scalp_trail_atr_mult_base
                .load(Ordering::Relaxed),
            swing_trail_act_atr: arena.config.swing_trail_act_atr.load(Ordering::Relaxed),
            swing_trail_step_atr: arena.config.swing_trail_step_atr.load(Ordering::Relaxed),
            swing_trail_max_atr: arena.config.swing_trail_max_atr.load(Ordering::Relaxed),
            swing_trail_min_pnl: arena.config.swing_trail_min_pnl.load(Ordering::Relaxed),
            swing_trail_atr_mult_base: arena
                .config
                .swing_trail_atr_mult_base
                .load(Ordering::Relaxed),
            zombie_timeout_ms: arena.config.zombie_timeout_ms.load(Ordering::Relaxed),
            hurst_trend_threshold: arena.config.hurst_trend_threshold.load(Ordering::Relaxed),
            cvd_veto_threshold: arena.config.cvd_veto_threshold.load(Ordering::Relaxed),
            wall_veto_threshold: arena.config.wall_veto_threshold.load(Ordering::Relaxed),
            flash_crash_jump_pct: arena.config.flash_crash_jump_pct.load(Ordering::Relaxed),
            latency_ms_panic_threshold: arena
                .config
                .latency_ms_panic_threshold
                .load(Ordering::Relaxed),
            ema_fast_period: arena.config.ema_fast_period.load(Ordering::Relaxed),
            ema_slow_period: arena.config.ema_slow_period.load(Ordering::Relaxed),
            hurst_scalp_threshold: arena.config.hurst_scalp_threshold.load(Ordering::Relaxed),
            hurst_swing_threshold: arena.config.hurst_swing_threshold.load(Ordering::Relaxed),
            synergy_exposure_boost: arena.config.synergy_exposure_boost.load(Ordering::Relaxed),
            synergy_leverage_boost: arena.config.synergy_leverage_boost.load(Ordering::Relaxed),
            max_fee_pct: arena.config.max_fee_pct.load(Ordering::Relaxed),
            kelly_bootstrap_cold: arena.config.kelly_bootstrap_cold.load(Ordering::Relaxed),
            latency_penalty_ms: arena.config.latency_penalty_ms.load(Ordering::Relaxed),
            base_slippage_floor: arena.config.base_slippage_floor.load(Ordering::Relaxed),
            spot_spread_threshold: arena.config.spot_spread_threshold.load(Ordering::Relaxed),
            spot_bias_value: arena.config.spot_bias_value.load(Ordering::Relaxed),
            obi_confidence_fallback: arena.config.obi_confidence_fallback.load(Ordering::Relaxed),
            ml_clip_lower: arena.config.ml_clip_lower.load(Ordering::Relaxed),
            ml_clip_upper: arena.config.ml_clip_upper.load(Ordering::Relaxed),
            tensor_poly_a: arena.config.tensor_poly_a.load(Ordering::Relaxed),
            tensor_poly_b: arena.config.tensor_poly_b.load(Ordering::Relaxed),
            fractional_alpha_order: arena.config.fractional_alpha_order.load(Ordering::Relaxed),
            fractional_clip_max: arena.config.fractional_clip_max.load(Ordering::Relaxed),
            bft_consensus_tolerance: arena.config.bft_consensus_tolerance.load(Ordering::Relaxed),
            turbo_coherence_threshold: arena
                .config
                .turbo_coherence_threshold
                .load(Ordering::Relaxed),
            turbo_z_score_stdev: arena.config.turbo_z_score_stdev.load(Ordering::Relaxed),
            sl_atr_multiplier: arena.config.sl_atr_multiplier.load(Ordering::Relaxed),
            coaxial_squeeze_threshold: arena
                .config
                .coaxial_squeeze_threshold
                .load(Ordering::Relaxed),
            tensor_op_add_bias: arena.config.tensor_op_add_bias.load(Ordering::Relaxed),
            tensor_op_mul_weight: arena.config.tensor_op_mul_weight.load(Ordering::Relaxed),
            topo_layer_1_activation: arena.config.topo_layer_1_activation.load(Ordering::Relaxed),
            topo_layer_2_activation: arena.config.topo_layer_2_activation.load(Ordering::Relaxed),
            tensor_dropout_rate: arena.config.tensor_dropout_rate.load(Ordering::Relaxed),
            quantum_entropy_seed: arena.config.quantum_entropy_seed.load(Ordering::Relaxed),
            ppo_weight_ofi: arena.config.ppo_weight_ofi.load(Ordering::Relaxed),
            ppo_weight_obi: arena.config.ppo_weight_obi.load(Ordering::Relaxed),
            ppo_weight_hawkes: arena.config.ppo_weight_hawkes.load(Ordering::Relaxed),
            ppo_weight_leadlag: arena.config.ppo_weight_leadlag.load(Ordering::Relaxed),
            ppo_weight_regime: arena.config.ppo_weight_regime.load(Ordering::Relaxed),
            global_learning_rate: arena.config.global_learning_rate.load(Ordering::Relaxed),
            global_momentum: arena.config.global_momentum.load(Ordering::Relaxed),
            hawkes_volume_norm: arena.config.hawkes_volume_norm.load(Ordering::Relaxed),
            base_duration_ms: arena.config.base_duration_ms.load(Ordering::Relaxed),
            vecm_alpha_speed: arena.config.vecm_alpha_speed.load(Ordering::Relaxed),
            vecm_beta_hedge: arena.config.vecm_beta_hedge.load(Ordering::Relaxed),
            conformal_alpha: arena.config.conformal_alpha.load(Ordering::Relaxed),
            ppo_clip_eps: arena.config.ppo_clip_eps.load(Ordering::Relaxed),
            ppo_weight_min_clip: arena.config.ppo_weight_min_clip.load(Ordering::Relaxed),
            hard_stop_decay_factor: arena.config.hard_stop_decay_factor.load(Ordering::Relaxed),
            hard_stop_base_limit: arena.config.hard_stop_base_limit.load(Ordering::Relaxed),
            kelly_bootstrap_ratio_threshold: arena
                .config
                .kelly_bootstrap_ratio_threshold
                .load(Ordering::Relaxed),
            kelly_bootstrap_min_exposure: arena
                .config
                .kelly_bootstrap_min_exposure
                .load(Ordering::Relaxed),
            ev_fee_multiplier: arena.config.ev_fee_multiplier.load(Ordering::Relaxed),
            margin_cushion_pct: arena.config.margin_cushion_pct.load(Ordering::Relaxed),
            maker_only_capital_threshold: arena
                .config
                .maker_only_capital_threshold
                .load(Ordering::Relaxed),

            kelly_survival_cap_ratio: arena
                .config
                .kelly_survival_cap_ratio
                .load(Ordering::Relaxed),
            kelly_expansion_mult: arena.config.kelly_expansion_mult.load(Ordering::Relaxed),
            guard_dd_sigmoid_steepness: arena
                .config
                .guard_dd_sigmoid_steepness
                .load(Ordering::Relaxed),
            guard_dd_sigmoid_center: arena.config.guard_dd_sigmoid_center.load(Ordering::Relaxed),
            portfolio_perf_mult_steepness: arena
                .config
                .portfolio_perf_mult_steepness
                .load(Ordering::Relaxed),
            portfolio_dd_penalty_decay: arena
                .config
                .portfolio_dd_penalty_decay
                .load(Ordering::Relaxed),
            portfolio_perf_mult_min: arena.config.portfolio_perf_mult_min.load(Ordering::Relaxed),
            portfolio_perf_mult_max: arena.config.portfolio_perf_mult_max.load(Ordering::Relaxed),
            portfolio_perf_mult_center: arena
                .config
                .portfolio_perf_mult_center
                .load(Ordering::Relaxed),

            macro_hurst_confidence_offset: arena
                .config
                .macro_hurst_confidence_offset
                .load(Ordering::Relaxed),
            macro_hurst_confidence_scale: arena
                .config
                .macro_hurst_confidence_scale
                .load(Ordering::Relaxed),
            macro_vol_confidence_scale: arena
                .config
                .macro_vol_confidence_scale
                .load(Ordering::Relaxed),
            macro_min_cooldown_ratio: arena
                .config
                .macro_min_cooldown_ratio
                .load(Ordering::Relaxed),
            macro_max_cooldown_ratio: arena
                .config
                .macro_max_cooldown_ratio
                .load(Ordering::Relaxed),
            macro_cooldown_reduction_factor: arena
                .config
                .macro_cooldown_reduction_factor
                .load(Ordering::Relaxed),
            macro_leverage_momentum_scale: arena
                .config
                .macro_leverage_momentum_scale
                .load(Ordering::Relaxed),
            hawkes_scalp_threshold: arena.config.hawkes_scalp_threshold.load(Ordering::Relaxed),
            obi_zscore_threshold: arena.config.obi_zscore_threshold.load(Ordering::Relaxed),
            lev_matrix_vol_clamp_min: arena
                .config
                .lev_matrix_vol_clamp_min
                .load(Ordering::Relaxed),
            lev_matrix_growth_scalar: arena
                .config
                .lev_matrix_growth_scalar
                .load(Ordering::Relaxed),
            lev_matrix_log_cap_divisor: arena
                .config
                .lev_matrix_log_cap_divisor
                .load(Ordering::Relaxed),
            scalp_accel_min_samples: arena.config.scalp_accel_min_samples.load(Ordering::Relaxed),
            executor_max_orders_10s: arena.config.executor_max_orders_10s.load(Ordering::Relaxed),
            executor_max_weight_1m: arena.config.executor_max_weight_1m.load(Ordering::Relaxed),
            iceberg_volume_threshold: arena
                .config
                .iceberg_volume_threshold
                .load(Ordering::Relaxed),
            iceberg_slice_count: arena.config.iceberg_slice_count.load(Ordering::Relaxed),
            swing_obi_threshold: arena.config.swing_obi_threshold.load(Ordering::Relaxed),
            swing_accel_min_samples: arena.config.swing_accel_min_samples.load(Ordering::Relaxed),
            temporal_scale: arena.config.temporal_scale.load(Ordering::Relaxed),
        }
    }

    pub fn load_or_default() -> Self {
        Self::load_or_baseline(0.0002, 0.0005)
    }

    pub fn load_or_baseline(maker_base: f64, taker_base: f64) -> Self {
        // F4.3: envelope versionado primero (fuente de verdad con linaje);
        // la ruta legacy queda como fallback de compatibilidad.
        if let Some(envelope) = crate::genome_store::GenomeEnvelope::load_active() {
            telemetry_engine::telemetry!(
                "🧬 [GENOMA] Envelope activo: generación {} (fuente: {}, padre: {}) — {}",
                envelope.generation,
                envelope.source,
                envelope.parent_generation,
                envelope.promotion_reason
            );
            return envelope.genome;
        }
        // D-507 / CERT-F-14: fallback de emergencia al ESPEJO DEL PROPIO
        // ENTORNO si el envelope no resolvió. Antes leía la ruta compartida
        // `active_genome.json`: un boot de prod con prod/active.json
        // ausente cargaba silenciosamente lo que demo escribió en el
        // espejo — herencia cross-env que D-651 prohibió para el linaje
        // principal. quantum_champion.json (artefacto deliberado del
        // config_compiler) sigue siendo válido como última instancia.
        let legacy_data = crate::genome_store::legacy_mirror()
            .and_then(|mirror| std::fs::read_to_string(mirror).ok())
            .or_else(|| std::fs::read_to_string("config_dir/genotypes/quantum_champion.json").ok());
        if let Some(data) = legacy_data {
            if let Ok(genome) = serde_json::from_str::<Self>(&data) {
                telemetry_engine::telemetry!("🧬 [GENOMA] Loaded evolved SuperGenotype from config_dir/genotypes/active_genome.json (Emergency Fallback)");
                return genome;
            }
        }
        if let Ok(data) = std::fs::read_to_string(crate::paths::data_join("genesis_genome.json")) {
            if let Ok(genome) = serde_json::from_str::<Self>(&data) {
                telemetry_engine::telemetry!(
                    "🧬 [GENOMA] Loaded surviving SuperGenotype from genesis_genome.json"
                );
                return genome;
            }
        }
        // telemetry_engine::telemetry!("🧬 [GENOMA] No valid SuperGenotype found, initializing mathematical baseline with live fees (Maker: {:.4}%, Taker: {:.4}%).", maker_base * 100.0, taker_base * 100.0);
        Self::new_baseline(maker_base, taker_base)
    }

    pub fn save(&self) {
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(crate::paths::data_join("genesis_genome.json"), data);
        }
    }

    pub fn new_baseline(maker_base: f64, taker_base: f64) -> Self {
        // DERIVACIÓN MATEMÁTICA Y ESTOCÁSTICA DE LÍMITES BASE (Cero Hardcoding Heurístico)
        let w_base = 0.55_f64;
        let pi = std::f64::consts::PI;
        let e_const = std::f64::consts::E;
        let golden_ratio = 1.618033988749895_f64;

        // D-636 (DÉCIMA OLA) — EL BASELINE DERIVA SU RR DE LA FRICCIÓN.
        // Antes: `sl = tp / 2.0` fijaba RR = 2,0 por un literal, sin relación
        // con el coste real. Como 2,0 quedaba por debajo del mínimo correcto
        // en la banda corta, el propio genoma de arranque era EV-negativo.
        //
        // Ahora: el stop parte del mínimo viable impuesto por la fricción
        // (nunca por debajo: ahí ninguna RR alcanzable supera el coste) y el
        // objetivo se deriva del RR que ese stop exige.
        let roundtrip = 2.0 * taker_base; // D-645: la física aplica 2xtaker
        let scalp_sl_math = (taker_base * 3.0).max(Self::min_viable_sl(roundtrip) * 1.15);
        let swing_sl_math = (taker_base * 15.0).max(scalp_sl_math * 2.0);

        // Holgura del 15 % sobre el mínimo: el baseline no debe nacer pegado
        // al límite del gate, o la primera mutación lo saca de bounds.
        let r_scalp = Self::min_rr_for(w_base, roundtrip, scalp_sl_math) * 1.15;
        let r_swing = Self::min_rr_for(w_base, roundtrip, swing_sl_math) * 1.15;

        let scalp_tp_math = scalp_sl_math * r_scalp;
        let swing_tp_math = swing_sl_math * r_swing;

        let k_scalp = w_base - ((1.0 - w_base) / r_scalp);
        let k_swing = w_base - ((1.0 - w_base) / r_swing);

        let mut g = Self {
            global_max_drawdown: 1.0 - (taker_base * 100.0).clamp(0.01, 0.10), // Derivado del costo del mercado
            global_leverage: 30.0, // Apalancamiento para cuentas pequeñas futures
            btc_volatility_multiplier: 1.0,
            eth_volatility_multiplier: e_const / 2.0,
            min_trades_per_day: golden_ratio * 3.0,
            survival_capital_threshold: golden_ratio / 2.0, // ~0.809
            funding_rate_sensitivity: w_base,
            global_correlation_threshold: golden_ratio - 1.0, // 0.618
            trend_threshold: 0.55,
            range_threshold: e_const / 6.0, // ~0.453
            scalp_kelly_fraction: k_scalp,
            swing_kelly_fraction: k_swing,
            scalp_obi_threshold: taker_base * 50.0,
            scalp_tp_base: scalp_tp_math,
            scalp_sl_base: scalp_sl_math,
            swing_tp_base: swing_tp_math,
            swing_sl_base: swing_sl_math,
            // F8 — CURVAS DE HORIZONTE CONTINUO: TP(τ) y SL(τ) como funciones
            // log-lineales que pasan EXACTAMENTE por los valores históricos
            // de las bandas fast/slow (migración sin cambio de comportamiento
            // en los anchos legacy). El GA evoluciona (a,b) — la PENDIENTE
            // define cómo escala el parámetro con el horizonte: una decisión
            // continua sobre TODO el espectro, no dos buckets sueltos.
            tp_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                scalp_tp_math,
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                swing_tp_math,
            ),
            sl_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                scalp_sl_math,
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                swing_sl_math,
            ),
            kelly_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                k_scalp,
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                k_swing,
            ),
            trail_mult_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                2.5,
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                3.5,
            ),
            trail_act_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                2.0,
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                3.0,
            ),
            trail_step_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                1.2,
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                1.8,
            ),
            obi_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                taker_base * 50.0,
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                taker_base * 100.0,
            ),
            sl_atr_mult_btc: 1.0,
            tp_rr_ratio_btc: r_scalp,
            min_confidence_btc: w_base * 1.018, // Ligeramente mayor que base
            veto_threshold_btc: w_base * 1.09,
            tech_threshold: 0.24, // D-625: dentro de los bounds
            ml_threshold_long: w_base * 1.036,
            ml_threshold_short: 1.0 - (w_base * 1.036),
            maker_spread_pct: maker_base,
            maker_obi_threshold: w_base,
            target_volatility: taker_base * 40.0,
            dynamic_atr_min: 0.000001, // Scaled down to match tick-level ATR ~ 0.000002
            dynamic_obi_threshold: taker_base * 300.0,
            dynamic_ema_trend: 0.0012, // Calibrado a 12 bps para filtrar micro-ruido de velas 1m
            dynamic_ofi_threshold: taker_base * 200.0,
            capital_split_scalp: 0.5,
            kelly_clamp_min: k_scalp / 10.0,
            kelly_clamp_max: w_base,
            explosive_leverage_multiplier: pi / 2.0,
            quantum_mutation_rate: taker_base * 100.0,
            temporal_memory_decay: 1.0 - maker_base,
            leverage_cap: 1.0 / (taker_base * 20.0), // Escala inversamente al costo
            explosive_confidence_threshold: 1.0 - taker_base,
            weight_obi: w_base * 0.7,
            weight_ofi: w_base * 0.7,
            weight_vpin: w_base * 0.35,
            regime_duration_ms: (1.0 / taker_base) * 60.0,
            regime_atr_multiplier: golden_ratio,
            scalp_trail_act_atr: taker_base * 10.0,
            scalp_trail_step_atr: pi,
            scalp_trail_max_atr: e_const * 2.0,
            scalp_trail_min_pnl: taker_base * 10000.0, // Dinámico según taker
            scalp_trail_atr_mult_base: pi,
            swing_trail_act_atr: taker_base * 20.0,
            swing_trail_step_atr: pi * golden_ratio,
            swing_trail_max_atr: e_const * 2.5,
            swing_trail_min_pnl: taker_base * 15000.0,
            swing_trail_atr_mult_base: pi * 1.5,
            zombie_timeout_ms: 14_400_000.0, // D-649b: suelo de la banda (4 h)
            hurst_trend_threshold: golden_ratio - 1.0, // ~0.618
            cvd_veto_threshold: w_base * golden_ratio,
            wall_veto_threshold: pi * 5.0,
            flash_crash_jump_pct: taker_base * 300.0,
            latency_ms_panic_threshold: 1000.0 * pi, // ~3141ms
            ema_fast_period: pi * 4.0,               // ~12.5
            ema_slow_period: pi * 8.0,               // ~25.1
            hurst_scalp_threshold: (golden_ratio - 1.0) * 0.7, // ~0.43
            hurst_swing_threshold: (golden_ratio - 1.0) * 0.9, // ~0.55
            synergy_exposure_boost: golden_ratio,
            synergy_leverage_boost: e_const / 2.0,
            max_fee_pct: taker_base * 20.0,
            kelly_bootstrap_cold: 0.5,
            latency_penalty_ms: e_const * 5.0,
            base_slippage_floor: taker_base / 5.0,
            spot_spread_threshold: golden_ratio,
            spot_bias_value: taker_base * 300.0,
            obi_confidence_fallback: 1.0 - taker_base * 300.0,
            ml_clip_lower: taker_base * 100.0,
            ml_clip_upper: 1.0 - (taker_base * 100.0),
            tensor_poly_a: w_base / 10.0,
            tensor_poly_b: taker_base * 20.0,
            fractional_alpha_order: 0.5,
            fractional_clip_max: pi * 3.0,
            bft_consensus_tolerance: w_base * 0.3,
            turbo_coherence_threshold: pi / 10.0,
            turbo_z_score_stdev: 2.5,
            sl_atr_multiplier: golden_ratio / 2.0,
            coaxial_squeeze_threshold: e_const / 8.0,
            tensor_op_add_bias: taker_base * 200.0,
            tensor_op_mul_weight: golden_ratio,
            topo_layer_1_activation: 0.5,
            topo_layer_2_activation: 0.5,
            tensor_dropout_rate: taker_base * 200.0,
            quantum_entropy_seed: pi * e_const * golden_ratio, // ~13.8
            ppo_weight_ofi: w_base * 0.35,
            ppo_weight_obi: w_base * 0.35,
            ppo_weight_hawkes: w_base * 0.35,
            ppo_weight_leadlag: w_base * 0.35,
            ppo_weight_regime: w_base * 0.35,
            global_learning_rate: taker_base * 2.0,
            global_momentum: 1.0 - (taker_base * 200.0),
            vecm_alpha_speed: taker_base * 20.0,
            vecm_beta_hedge: taker_base * 20.0,
            conformal_alpha: taker_base * 100.0,
            ppo_clip_eps: taker_base * 400.0,
            ppo_weight_min_clip: taker_base * 100.0,
            hard_stop_decay_factor: 0.5,
            hard_stop_base_limit: 0.5,
            kelly_bootstrap_ratio_threshold: taker_base * 200.0,
            kelly_bootstrap_min_exposure: taker_base * 200.0,
            ev_fee_multiplier: 1.1,
            margin_cushion_pct: 0.98, // D-687: el efectivo de `1 + taker·100` tras el acotado
            maker_only_capital_threshold: taker_base * 100000.0,

            kelly_survival_cap_ratio: golden_ratio,
            kelly_expansion_mult: golden_ratio,
            guard_dd_sigmoid_steepness: pi,
            guard_dd_sigmoid_center: pi,
            portfolio_perf_mult_steepness: e_const * 3.0,
            portfolio_dd_penalty_decay: pi * 5.0,
            portfolio_perf_mult_min: w_base * 0.5,
            portfolio_perf_mult_max: w_base * 3.0,
            portfolio_perf_mult_center: w_base,

            macro_hurst_confidence_offset: 0.45,
            macro_hurst_confidence_scale: 10.0,
            macro_vol_confidence_scale: 200.0,
            macro_min_cooldown_ratio: 0.2,
            macro_max_cooldown_ratio: 100.0,
            macro_cooldown_reduction_factor: 0.8,
            macro_leverage_momentum_scale: 0.5,
            hawkes_scalp_threshold: 0.55,
            obi_zscore_threshold: 1.0,

            hawkes_volume_norm: taker_base * 100000.0, // Default to a derived ratio
            base_duration_ms: (1.0 / taker_base) * 60.0,

            lev_matrix_vol_clamp_min: 0.3,
            lev_matrix_growth_scalar: 0.3,
            lev_matrix_log_cap_divisor: 7.0,
            scalp_accel_min_samples: 30.0,
            executor_max_orders_10s: 280.0,
            executor_max_weight_1m: 2200.0,
            iceberg_volume_threshold: 500.0,
            iceberg_slice_count: 5.0,
            swing_obi_threshold: 0.5,
            swing_accel_min_samples: 30.0,
            temporal_scale: 0.5,
        };
        // D-636/D-608: el baseline se somete al MISMO invariante que cualquier
        // mutante — nace dentro de la banda operable y con RR suficiente.
        g.enforce_curve_rr();
        g.derive_anchors_from_curves();
        g.sync_continuous_curves();
        g
    }

    pub fn new_random() -> Self {
        let mut g = Self {
            global_max_drawdown: rand::rng().random_range(0.5..0.99),
            // D-644: misma banda que la mutación y los bounds.
            global_leverage: rand::rng().random_range(1.0..50.0),
            btc_volatility_multiplier: rand::rng().random_range(0.5..3.0),
            eth_volatility_multiplier: rand::rng().random_range(0.5..3.0),
            min_trades_per_day: rand::rng().random_range(1.0..50.0),
            survival_capital_threshold: rand::rng().random_range(0.5..0.95),
            funding_rate_sensitivity: rand::rng().random_range(0.1..2.0),
            global_correlation_threshold: rand::rng().random_range(0.2..0.9),
            trend_threshold: rand::rng().random_range(0.52..0.85),
            range_threshold: rand::rng().random_range(0.1..0.9),
            scalp_kelly_fraction: rand::rng().random_range(0.1..2.0),
            swing_kelly_fraction: rand::rng().random_range(0.01..1.0),
            scalp_obi_threshold: rand::rng().random_range(0.05..1.0),
            scalp_tp_base: rand::rng().random_range(0.0060..0.0250),
            scalp_sl_base: rand::rng().random_range(0.0025..0.0080),
            swing_tp_base: rand::rng().random_range(0.0300..0.1200),
            swing_sl_base: rand::rng().random_range(0.0100..0.0350),
            tp_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                rand::rng().random_range(0.0060..0.0350),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                rand::rng().random_range(0.03..0.15),
            ),
            sl_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                rand::rng().random_range(0.0025..0.0120),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                rand::rng().random_range(0.01..0.045),
            ),
            kelly_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                rand::rng().random_range(0.1..2.0),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                rand::rng().random_range(0.01..1.0),
            ),
            trail_mult_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                rand::rng().random_range(0.5..3.0),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                rand::rng().random_range(1.0..5.0),
            ),
            trail_act_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                rand::rng().random_range(0.5..2.5),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                rand::rng().random_range(1.0..4.0),
            ),
            trail_step_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                rand::rng().random_range(0.2..1.5),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                rand::rng().random_range(0.5..2.5),
            ),
            obi_horizon_curve: crate::temporal_spectrum::HorizonCurve::through_two_points(
                crate::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                rand::rng().random_range(0.1..0.8),
                crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                rand::rng().random_range(0.3..0.9),
            ),
            sl_atr_mult_btc: rand::rng().random_range(0.5..5.0),
            tp_rr_ratio_btc: rand::rng().random_range(1.0..10.0),
            min_confidence_btc: rand::rng().random_range(0.5..0.95),
            veto_threshold_btc: rand::rng().random_range(0.5..0.99),
            tech_threshold: rand::rng().random_range(0.24..0.30),
            ml_threshold_long: rand::rng().random_range(0.5..0.95),
            ml_threshold_short: rand::rng().random_range(0.05..0.49),
            maker_spread_pct: rand::rng().random_range(0.0001..0.01),
            maker_obi_threshold: rand::rng().random_range(0.1..0.95),
            target_volatility: rand::rng().random_range(0.005..0.1),
            dynamic_atr_min: rand::rng().random_range(0.0000001..0.0000050),
            dynamic_obi_threshold: rand::rng().random_range(0.05..0.95),
            dynamic_ema_trend: rand::rng().random_range(0.0008..0.0035),
            dynamic_ofi_threshold: rand::rng().random_range(0.05..0.95),
            capital_split_scalp: rand::rng().random_range(0.1..1.0),
            kelly_clamp_min: rand::rng().random_range(0.001..0.1),
            kelly_clamp_max: rand::rng().random_range(0.1..1.0),
            explosive_leverage_multiplier: rand::rng().random_range(1.0..10.0),
            quantum_mutation_rate: rand::rng().random_range(0.01..0.5),
            temporal_memory_decay: rand::rng().random_range(0.8..0.9999),
            leverage_cap: rand::rng().random_range(10.0..125.0),
            explosive_confidence_threshold: rand::rng().random_range(0.7..0.99),
            weight_obi: rand::rng().random_range(0.1..1.0),
            weight_ofi: rand::rng().random_range(0.1..1.0),
            weight_vpin: rand::rng().random_range(0.1..1.0),
            regime_duration_ms: rand::rng().random_range(30000.0..600000.0),
            regime_atr_multiplier: rand::rng().random_range(0.5..5.0),
            scalp_trail_act_atr: rand::rng().random_range(0.001..20.0),
            scalp_trail_step_atr: rand::rng().random_range(0.001..20.0),
            scalp_trail_max_atr: rand::rng().random_range(0.001..20.0),
            scalp_trail_min_pnl: rand::rng().random_range(0.001..20.0),
            scalp_trail_atr_mult_base: rand::rng().random_range(0.001..20.0),
            swing_trail_act_atr: rand::rng().random_range(0.001..20.0),
            swing_trail_step_atr: rand::rng().random_range(0.001..20.0),
            swing_trail_max_atr: rand::rng().random_range(0.001..20.0),
            swing_trail_min_pnl: rand::rng().random_range(0.001..20.0),
            swing_trail_atr_mult_base: rand::rng().random_range(0.001..20.0),
            zombie_timeout_ms: rand::rng().random_range(14_400_000.0..28_800_000.0),
            hurst_trend_threshold: rand::rng().random_range(0.4..0.8),
            cvd_veto_threshold: rand::rng().random_range(0.1..1.5),
            wall_veto_threshold: rand::rng().random_range(0.5..5.0),
            flash_crash_jump_pct: rand::rng().random_range(0.05..0.5),
            latency_ms_panic_threshold: rand::rng().random_range(500.0..10000.0),
            ema_fast_period: rand::rng().random_range(5.0..30.0),
            ema_slow_period: rand::rng().random_range(15.0..60.0),
            hurst_scalp_threshold: rand::rng().random_range(0.30..0.50),
            hurst_swing_threshold: rand::rng().random_range(0.50..0.70),
            synergy_exposure_boost: rand::rng().random_range(1.0..3.0),
            synergy_leverage_boost: rand::rng().random_range(1.0..2.0),
            max_fee_pct: rand::rng().random_range(0.005..0.05),
            kelly_bootstrap_cold: rand::rng().random_range(0.1..1.0),
            latency_penalty_ms: rand::rng().random_range(5.0..100.0),
            base_slippage_floor: rand::rng().random_range(0.00005..0.0005),
            spot_spread_threshold: rand::rng().random_range(0.5..3.0),
            spot_bias_value: rand::rng().random_range(0.01..0.5),
            obi_confidence_fallback: rand::rng().random_range(0.5..0.99),
            ml_clip_lower: rand::rng().random_range(0.01..0.20),
            ml_clip_upper: rand::rng().random_range(0.80..0.99),
            tensor_poly_a: rand::rng().random_range(0.01..0.20),
            tensor_poly_b: rand::rng().random_range(0.001..0.05),
            fractional_alpha_order: rand::rng().random_range(0.1..0.9),
            fractional_clip_max: rand::rng().random_range(5.0..20.0),
            bft_consensus_tolerance: rand::rng().random_range(0.05..0.30),
            turbo_coherence_threshold: rand::rng().random_range(0.1..0.6),
            turbo_z_score_stdev: rand::rng().random_range(0.1..1.5),
            sl_atr_multiplier: rand::rng().random_range(0.2..2.0),
            coaxial_squeeze_threshold: rand::rng().random_range(0.1..0.8),
            tensor_op_add_bias: rand::rng().random_range(0.01..0.5),
            tensor_op_mul_weight: rand::rng().random_range(0.5..3.0),
            topo_layer_1_activation: rand::rng().random_range(0.1..0.9),
            topo_layer_2_activation: rand::rng().random_range(0.1..0.9),
            tensor_dropout_rate: rand::rng().random_range(0.01..0.5),
            quantum_entropy_seed: rand::rng().random_range(1.0..1000.0),
            ppo_weight_ofi: rand::rng().random_range(0.01..0.5),
            ppo_weight_obi: rand::rng().random_range(0.01..0.5),
            ppo_weight_hawkes: rand::rng().random_range(0.01..0.5),
            ppo_weight_leadlag: rand::rng().random_range(0.01..0.5),
            ppo_weight_regime: rand::rng().random_range(0.01..0.5),
            global_learning_rate: rand::rng().random_range(0.0001..0.05),
            global_momentum: rand::rng().random_range(0.5..0.99),
            vecm_alpha_speed: rand::rng().random_range(0.001..0.1),
            vecm_beta_hedge: rand::rng().random_range(0.001..0.1),
            conformal_alpha: rand::rng().random_range(0.01..0.2),
            ppo_clip_eps: rand::rng().random_range(0.05..0.40),
            ppo_weight_min_clip: rand::rng().random_range(0.01..0.10),
            hard_stop_decay_factor: rand::rng().random_range(0.1..1.0),
            hard_stop_base_limit: rand::rng().random_range(0.10..0.80),
            kelly_bootstrap_ratio_threshold: rand::rng().random_range(0.05..0.30),
            kelly_bootstrap_min_exposure: rand::rng().random_range(0.01..0.5),
            ev_fee_multiplier: rand::rng().random_range(1.05..2.0),
            margin_cushion_pct: rand::rng().random_range(0.50..0.98),
            maker_only_capital_threshold: rand::rng().random_range(10.0..200.0),

            kelly_survival_cap_ratio: rand::rng().random_range(1.0..3.0),
            kelly_expansion_mult: rand::rng().random_range(1.0..3.0),
            guard_dd_sigmoid_steepness: rand::rng().random_range(1.0..5.0),
            guard_dd_sigmoid_center: rand::rng().random_range(1.0..5.0),
            portfolio_perf_mult_steepness: rand::rng().random_range(3.0..15.0),
            portfolio_dd_penalty_decay: rand::rng().random_range(5.0..25.0),
            portfolio_perf_mult_min: rand::rng().random_range(0.1..0.5),
            portfolio_perf_mult_max: rand::rng().random_range(1.5..3.0),
            portfolio_perf_mult_center: rand::rng().random_range(0.3..0.8),

            macro_hurst_confidence_offset: rand::rng().random_range(0.35..0.55),
            macro_hurst_confidence_scale: rand::rng().random_range(1.0..20.0),
            macro_vol_confidence_scale: rand::rng().random_range(10.0..500.0),
            macro_min_cooldown_ratio: rand::rng().random_range(0.05..0.5),
            macro_max_cooldown_ratio: rand::rng().random_range(10.0..500.0),
            macro_cooldown_reduction_factor: rand::rng().random_range(0.1..2.0),
            macro_leverage_momentum_scale: rand::rng().random_range(0.1..2.0),
            hawkes_scalp_threshold: rand::rng().random_range(0.50..0.95),
            obi_zscore_threshold: rand::rng().random_range(0.1..3.0),

            hawkes_volume_norm: rand::rng().random_range(100.0..10_000_000.0),
            base_duration_ms: rand::rng().random_range(10_000.0..120_000.0),

            lev_matrix_vol_clamp_min: rand::rng().random_range(0.1..0.8),
            lev_matrix_growth_scalar: rand::rng().random_range(0.1..1.0),
            lev_matrix_log_cap_divisor: rand::rng().random_range(3.0..10.0),
            scalp_accel_min_samples: rand::rng().random_range(10.0..50.0),
            executor_max_orders_10s: rand::rng().random_range(100.0..500.0),
            executor_max_weight_1m: rand::rng().random_range(1000.0..5000.0),
            iceberg_volume_threshold: rand::rng().random_range(100.0..2000.0),
            iceberg_slice_count: rand::rng().random_range(3.0..10.0),
            swing_obi_threshold: rand::rng().random_range(0.05..1.0),
            swing_accel_min_samples: rand::rng().random_range(10.0..50.0),
            temporal_scale: rand::rng().random_range(0.05..0.95),
        };
        g.enforce_curve_rr();
        g.derive_anchors_from_curves();
        g.sync_continuous_curves();
        g
    }

    /// Applica el genoma completo directamente al Arena lock-free
    pub fn apply_to_arena(&self, arena: &GlobalArena) {
        // F8 — EL CONTINUO MANDA: TP/SL de las bandas legacy se derivan de las
        // CURVAS de horizonte (la pendiente evolucionada define cómo escala el
        // parámetro con τ). Los campos scalp_*/swing_* del genoma quedan como
        // anclas de compatibilidad; TODO lector del arena (core, OCO, sizing)
        // recibe valores del continuo sin un solo cambio — evolucionar la
        // pendiente mueve el espectro entero coherentemente.
        let fast_tp = self
            .tp_horizon_curve
            .eval(crate::temporal_spectrum::TAU_ANCHOR_FAST_MS);
        let slow_tp = self
            .tp_horizon_curve
            .eval(crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS);
        let fast_sl = self
            .sl_horizon_curve
            .eval(crate::temporal_spectrum::TAU_ANCHOR_FAST_MS);
        let slow_sl = self
            .sl_horizon_curve
            .eval(crate::temporal_spectrum::TAU_ANCHOR_SLOW_MS);
        arena.config.scalp_tp_base.store(fast_tp, Ordering::Relaxed);
        arena.config.swing_tp_base.store(slow_tp, Ordering::Relaxed);
        arena.config.scalp_sl_base.store(fast_sl, Ordering::Relaxed);
        arena.config.swing_sl_base.store(slow_sl, Ordering::Relaxed);

        // Continuo Universal: almacenamiento atómico de todas las curvas de horizonte
        arena
            .config
            .tp_curve_a
            .store(self.tp_horizon_curve.a, Ordering::Relaxed);
        arena
            .config
            .tp_curve_b
            .store(self.tp_horizon_curve.b, Ordering::Relaxed);
        arena
            .config
            .sl_curve_a
            .store(self.sl_horizon_curve.a, Ordering::Relaxed);
        arena
            .config
            .sl_curve_b
            .store(self.sl_horizon_curve.b, Ordering::Relaxed);
        // D-683 (DÉCIMA OLA): las curvas de Kelly, trailing y OBI NO están en el
        // vector de genes; `from_vector`, `mutate` y los constructores las
        // derivan de los genes scalp/swing con `sync_continuous_curves`. Pero el
        // genoma que se carga desde disco pasa por serde, que rellena las curvas
        // ausentes con LITERALES (OBI 0,25–0,40, Kelly 0,20–0,15, trailing
        // 2,5–3,5 ATR): producción y el backtest forense ejecutaban parámetros
        // que el ciclo evolutivo jamás evaluó —en el genoma de producción, un
        // umbral de OBI de 0,25 donde sus genes dicen 0,80—. Se derivan aquí,
        // en la entrada al arena, igual que en la evolución.
        let curvas = self.with_synced_continuous_curves();
        arena
            .config
            .kelly_curve_a
            .store(curvas.kelly_horizon_curve.a, Ordering::Relaxed);
        arena
            .config
            .kelly_curve_b
            .store(curvas.kelly_horizon_curve.b, Ordering::Relaxed);
        arena
            .config
            .trail_mult_curve_a
            .store(curvas.trail_mult_horizon_curve.a, Ordering::Relaxed);
        arena
            .config
            .trail_mult_curve_b
            .store(curvas.trail_mult_horizon_curve.b, Ordering::Relaxed);
        arena
            .config
            .trail_act_curve_a
            .store(curvas.trail_act_horizon_curve.a, Ordering::Relaxed);
        arena
            .config
            .trail_act_curve_b
            .store(curvas.trail_act_horizon_curve.b, Ordering::Relaxed);
        arena
            .config
            .trail_step_curve_a
            .store(curvas.trail_step_horizon_curve.a, Ordering::Relaxed);
        arena
            .config
            .trail_step_curve_b
            .store(curvas.trail_step_horizon_curve.b, Ordering::Relaxed);
        arena
            .config
            .obi_curve_a
            .store(curvas.obi_horizon_curve.a, Ordering::Relaxed);
        arena
            .config
            .obi_curve_b
            .store(curvas.obi_horizon_curve.b, Ordering::Relaxed);

        arena
            .config
            .global_max_drawdown
            .store(self.global_max_drawdown, Ordering::Relaxed);
        arena
            .config
            .global_leverage
            .store(self.global_leverage, Ordering::Relaxed);
        arena
            .config
            .btc_volatility_multiplier
            .store(self.btc_volatility_multiplier, Ordering::Relaxed);
        arena
            .config
            .eth_volatility_multiplier
            .store(self.eth_volatility_multiplier, Ordering::Relaxed);
        arena
            .config
            .funding_rate_sensitivity
            .store(self.funding_rate_sensitivity, Ordering::Relaxed);
        arena
            .config
            .global_correlation_threshold
            .store(self.global_correlation_threshold, Ordering::Relaxed);
        arena.config.trend_threshold.store(
            Self::clamp_slot(self.trend_threshold, Self::SLOT_TREND_THRESHOLD),
            Ordering::Relaxed,
        );
        arena
            .config
            .range_threshold
            .store(self.range_threshold, Ordering::Relaxed);
        arena
            .config
            .scalp_kelly_fraction
            .store(self.scalp_kelly_fraction, Ordering::Relaxed);
        arena
            .config
            .swing_kelly_fraction
            .store(self.swing_kelly_fraction, Ordering::Relaxed);
        arena
            .config
            .scalp_obi_threshold
            .store(self.scalp_obi_threshold, Ordering::Relaxed);
        // X-003 (REHAB-1): los stores de anclas scalp/swing tp/sl fueron
        // ELIMINADOS — pisaban los valores derivados de las CURVAS escritos
        // al inicio de este método (última escritura ganaba: el continuo era
        // decorativo). Las curvas son ahora la única fuente que llega al arena;
        // los campos ancla del struct quedan como vistas de compatibilidad
        // (se re-derivan de las curvas en mutación/carga/serialización).
        arena
            .config
            .sl_atr_mult_btc
            .store(self.sl_atr_mult_btc, Ordering::Relaxed);
        arena
            .config
            .tp_rr_ratio_btc
            .store(self.tp_rr_ratio_btc, Ordering::Relaxed);
        arena.config.min_confidence_btc.store(
            Self::clamp_slot(self.min_confidence_btc, Self::SLOT_MIN_CONFIDENCE),
            Ordering::Relaxed,
        );
        arena
            .config
            .veto_threshold_btc
            .store(self.veto_threshold_btc, Ordering::Relaxed);
        arena.config.tech_threshold.store(
            Self::clamp_slot(self.tech_threshold, Self::SLOT_TECH_THRESHOLD),
            Ordering::Relaxed,
        );
        arena
            .config
            .ml_threshold_long
            .store(self.ml_threshold_long, Ordering::Relaxed);
        arena
            .config
            .ml_threshold_short
            .store(self.ml_threshold_short, Ordering::Relaxed);
        arena
            .config
            .maker_spread_pct
            .store(self.maker_spread_pct, Ordering::Relaxed);
        arena
            .config
            .maker_obi_threshold
            .store(self.maker_obi_threshold, Ordering::Relaxed);
        arena
            .config
            .target_volatility
            .store(self.target_volatility, Ordering::Relaxed);
        arena
            .config
            .dynamic_atr_min
            .store(self.dynamic_atr_min, Ordering::Relaxed);
        arena
            .config
            .dynamic_obi_threshold
            .store(self.dynamic_obi_threshold, Ordering::Relaxed);
        arena
            .config
            .dynamic_ema_trend
            .store(self.dynamic_ema_trend, Ordering::Relaxed);
        arena
            .config
            .dynamic_ofi_threshold
            .store(self.dynamic_ofi_threshold, Ordering::Relaxed);
        arena
            .config
            .capital_split_scalp
            .store(self.capital_split_scalp, Ordering::Relaxed);
        arena
            .config
            .kelly_clamp_min
            .store(self.kelly_clamp_min, Ordering::Relaxed);
        arena
            .config
            .kelly_clamp_max
            .store(self.kelly_clamp_max, Ordering::Relaxed);
        arena
            .config
            .explosive_leverage_multiplier
            .store(self.explosive_leverage_multiplier, Ordering::Relaxed);
        arena
            .config
            .quantum_mutation_rate
            .store(self.quantum_mutation_rate, Ordering::Relaxed);
        arena
            .config
            .temporal_memory_decay
            .store(self.temporal_memory_decay, Ordering::Relaxed);
        arena
            .config
            .leverage_cap
            .store(self.leverage_cap, Ordering::Relaxed);
        arena
            .config
            .explosive_confidence_threshold
            .store(self.explosive_confidence_threshold, Ordering::Relaxed);
        arena
            .config
            .hawkes_volume_norm
            .store(self.hawkes_volume_norm, Ordering::Relaxed);
        arena
            .config
            .base_duration_ms
            .store(self.base_duration_ms, Ordering::Relaxed);
        arena
            .config
            .weight_obi
            .store(self.weight_obi, Ordering::Relaxed);
        arena
            .config
            .weight_ofi
            .store(self.weight_ofi, Ordering::Relaxed);
        arena
            .config
            .weight_vpin
            .store(self.weight_vpin, Ordering::Relaxed);
        arena
            .config
            .regime_duration_ms
            .store(self.regime_duration_ms, Ordering::Relaxed);
        arena
            .config
            .regime_atr_multiplier
            .store(self.regime_atr_multiplier, Ordering::Relaxed);

        arena
            .config
            .scalp_trail_act_atr
            .store(self.scalp_trail_act_atr, Ordering::Relaxed);
        arena
            .config
            .scalp_trail_step_atr
            .store(self.scalp_trail_step_atr, Ordering::Relaxed);
        arena
            .config
            .scalp_trail_max_atr
            .store(self.scalp_trail_max_atr, Ordering::Relaxed);
        arena
            .config
            .scalp_trail_min_pnl
            .store(self.scalp_trail_min_pnl, Ordering::Relaxed);
        arena
            .config
            .scalp_trail_atr_mult_base
            .store(self.scalp_trail_atr_mult_base, Ordering::Relaxed);

        arena
            .config
            .swing_trail_act_atr
            .store(self.swing_trail_act_atr, Ordering::Relaxed);
        arena
            .config
            .swing_trail_step_atr
            .store(self.swing_trail_step_atr, Ordering::Relaxed);
        arena
            .config
            .swing_trail_max_atr
            .store(self.swing_trail_max_atr, Ordering::Relaxed);
        arena
            .config
            .swing_trail_min_pnl
            .store(self.swing_trail_min_pnl, Ordering::Relaxed);
        arena
            .config
            .swing_trail_atr_mult_base
            .store(self.swing_trail_atr_mult_base, Ordering::Relaxed);

        arena.config.zombie_timeout_ms.store(
            Self::clamp_slot(self.zombie_timeout_ms, Self::SLOT_ZOMBIE_TIMEOUT),
            Ordering::Relaxed,
        );
        arena
            .config
            .hurst_trend_threshold
            .store(self.hurst_trend_threshold, Ordering::Relaxed);
        arena
            .config
            .cvd_veto_threshold
            .store(self.cvd_veto_threshold, Ordering::Relaxed);
        arena
            .config
            .wall_veto_threshold
            .store(self.wall_veto_threshold, Ordering::Relaxed);
        arena
            .config
            .flash_crash_jump_pct
            .store(self.flash_crash_jump_pct, Ordering::Relaxed);
        arena
            .config
            .latency_ms_panic_threshold
            .store(self.latency_ms_panic_threshold, Ordering::Relaxed);
        arena
            .config
            .ema_fast_period
            .store(self.ema_fast_period, Ordering::Relaxed);
        arena
            .config
            .ema_slow_period
            .store(self.ema_slow_period, Ordering::Relaxed);
        arena
            .config
            .hurst_scalp_threshold
            .store(self.hurst_scalp_threshold, Ordering::Relaxed);
        arena
            .config
            .hurst_swing_threshold
            .store(self.hurst_swing_threshold, Ordering::Relaxed);
        arena
            .config
            .synergy_exposure_boost
            .store(self.synergy_exposure_boost, Ordering::Relaxed);
        arena
            .config
            .synergy_leverage_boost
            .store(self.synergy_leverage_boost, Ordering::Relaxed);
        arena
            .config
            .max_fee_pct
            .store(self.max_fee_pct, Ordering::Relaxed);
        arena
            .config
            .kelly_bootstrap_cold
            .store(self.kelly_bootstrap_cold, Ordering::Relaxed);
        arena
            .config
            .latency_penalty_ms
            .store(self.latency_penalty_ms, Ordering::Relaxed);
        arena
            .config
            .base_slippage_floor
            .store(self.base_slippage_floor, Ordering::Relaxed);
        arena
            .config
            .spot_spread_threshold
            .store(self.spot_spread_threshold, Ordering::Relaxed);
        arena
            .config
            .spot_bias_value
            .store(self.spot_bias_value, Ordering::Relaxed);
        arena
            .config
            .obi_confidence_fallback
            .store(self.obi_confidence_fallback, Ordering::Relaxed);
        arena
            .config
            .ml_clip_lower
            .store(self.ml_clip_lower, Ordering::Relaxed);
        arena
            .config
            .ml_clip_upper
            .store(self.ml_clip_upper, Ordering::Relaxed);
        arena
            .config
            .tensor_poly_a
            .store(self.tensor_poly_a, Ordering::Relaxed);
        arena
            .config
            .tensor_poly_b
            .store(self.tensor_poly_b, Ordering::Relaxed);
        arena
            .config
            .fractional_alpha_order
            .store(self.fractional_alpha_order, Ordering::Relaxed);
        arena
            .config
            .fractional_clip_max
            .store(self.fractional_clip_max, Ordering::Relaxed);
        arena
            .config
            .bft_consensus_tolerance
            .store(self.bft_consensus_tolerance, Ordering::Relaxed);
        arena
            .config
            .turbo_coherence_threshold
            .store(self.turbo_coherence_threshold, Ordering::Relaxed);
        arena
            .config
            .turbo_z_score_stdev
            .store(self.turbo_z_score_stdev, Ordering::Relaxed);
        arena
            .config
            .sl_atr_multiplier
            .store(self.sl_atr_multiplier, Ordering::Relaxed);
        arena
            .config
            .coaxial_squeeze_threshold
            .store(self.coaxial_squeeze_threshold, Ordering::Relaxed);
        arena
            .config
            .tensor_op_add_bias
            .store(self.tensor_op_add_bias, Ordering::Relaxed);
        arena
            .config
            .tensor_op_mul_weight
            .store(self.tensor_op_mul_weight, Ordering::Relaxed);
        arena
            .config
            .topo_layer_1_activation
            .store(self.topo_layer_1_activation, Ordering::Relaxed);
        arena
            .config
            .topo_layer_2_activation
            .store(self.topo_layer_2_activation, Ordering::Relaxed);
        arena
            .config
            .tensor_dropout_rate
            .store(self.tensor_dropout_rate, Ordering::Relaxed);
        arena
            .config
            .quantum_entropy_seed
            .store(self.quantum_entropy_seed, Ordering::Relaxed);

        arena
            .config
            .ppo_clip_eps
            .store(self.ppo_clip_eps, Ordering::Relaxed);
        arena
            .config
            .ppo_weight_min_clip
            .store(self.ppo_weight_min_clip, Ordering::Relaxed);
        arena
            .config
            .hard_stop_decay_factor
            .store(self.hard_stop_decay_factor, Ordering::Relaxed);
        arena
            .config
            .hard_stop_base_limit
            .store(self.hard_stop_base_limit, Ordering::Relaxed);
        arena
            .config
            .kelly_bootstrap_ratio_threshold
            .store(self.kelly_bootstrap_ratio_threshold, Ordering::Relaxed);
        arena
            .config
            .kelly_bootstrap_min_exposure
            .store(self.kelly_bootstrap_min_exposure, Ordering::Relaxed);
        arena
            .config
            .ev_fee_multiplier
            .store(self.ev_fee_multiplier, Ordering::Relaxed);
        arena.config.margin_cushion_pct.store(
            Self::clamp_slot(self.margin_cushion_pct, Self::SLOT_MARGIN_CUSHION),
            Ordering::Relaxed,
        );
        arena
            .config
            .maker_only_capital_threshold
            .store(self.maker_only_capital_threshold, Ordering::Relaxed);
        arena
            .config
            .hawkes_scalp_threshold
            .store(self.hawkes_scalp_threshold, Ordering::Relaxed);
        arena
            .config
            .obi_zscore_threshold
            .store(self.obi_zscore_threshold, Ordering::Relaxed);

        // CORRECCIÓN CRÍTICA: Estos 9 campos estaban definidos, se mutaban,
        // pero NUNCA se propagaban al Arena. Eran nodos silenciosos (FALLO TIPO 1).
        // El evolver gastaba CPU evolucionando valores que el sistema nunca leía.
        arena
            .config
            .kelly_survival_cap_ratio
            .store(self.kelly_survival_cap_ratio, Ordering::Relaxed);
        arena
            .config
            .kelly_expansion_mult
            .store(self.kelly_expansion_mult, Ordering::Relaxed);
        arena
            .config
            .guard_dd_sigmoid_steepness
            .store(self.guard_dd_sigmoid_steepness, Ordering::Relaxed);
        arena
            .config
            .guard_dd_sigmoid_center
            .store(self.guard_dd_sigmoid_center, Ordering::Relaxed);
        arena
            .config
            .portfolio_perf_mult_steepness
            .store(self.portfolio_perf_mult_steepness, Ordering::Relaxed);
        arena
            .config
            .portfolio_dd_penalty_decay
            .store(self.portfolio_dd_penalty_decay, Ordering::Relaxed);
        arena
            .config
            .portfolio_perf_mult_min
            .store(self.portfolio_perf_mult_min, Ordering::Relaxed);
        arena
            .config
            .portfolio_perf_mult_max
            .store(self.portfolio_perf_mult_max, Ordering::Relaxed);
        arena
            .config
            .portfolio_perf_mult_center
            .store(self.portfolio_perf_mult_center, Ordering::Relaxed);

        arena
            .config
            .macro_hurst_confidence_offset
            .store(self.macro_hurst_confidence_offset, Ordering::Relaxed);
        arena
            .config
            .macro_hurst_confidence_scale
            .store(self.macro_hurst_confidence_scale, Ordering::Relaxed);
        arena
            .config
            .macro_vol_confidence_scale
            .store(self.macro_vol_confidence_scale, Ordering::Relaxed);
        arena
            .config
            .macro_min_cooldown_ratio
            .store(self.macro_min_cooldown_ratio, Ordering::Relaxed);
        arena
            .config
            .macro_max_cooldown_ratio
            .store(self.macro_max_cooldown_ratio, Ordering::Relaxed);
        arena
            .config
            .macro_cooldown_reduction_factor
            .store(self.macro_cooldown_reduction_factor, Ordering::Relaxed);
        arena
            .config
            .macro_leverage_momentum_scale
            .store(self.macro_leverage_momentum_scale, Ordering::Relaxed);

        // FASE 4: De-hardcoding
        arena
            .config
            .lev_matrix_vol_clamp_min
            .store(self.lev_matrix_vol_clamp_min, Ordering::Relaxed);
        arena
            .config
            .lev_matrix_growth_scalar
            .store(self.lev_matrix_growth_scalar, Ordering::Relaxed);
        arena
            .config
            .lev_matrix_log_cap_divisor
            .store(self.lev_matrix_log_cap_divisor, Ordering::Relaxed);
        arena
            .config
            .scalp_accel_min_samples
            .store(self.scalp_accel_min_samples, Ordering::Relaxed);
        arena
            .config
            .swing_obi_threshold
            .store(self.swing_obi_threshold, Ordering::Relaxed);
        arena
            .config
            .swing_accel_min_samples
            .store(self.swing_accel_min_samples, Ordering::Relaxed);
        arena
            .config
            .temporal_scale
            .store(self.temporal_scale, Ordering::Relaxed);
        arena
            .config
            .executor_max_orders_10s
            .store(self.executor_max_orders_10s, Ordering::Relaxed);
        arena
            .config
            .executor_max_weight_1m
            .store(self.executor_max_weight_1m, Ordering::Relaxed);
        arena
            .config
            .iceberg_volume_threshold
            .store(self.iceberg_volume_threshold, Ordering::Relaxed);
        arena
            .config
            .iceberg_slice_count
            .store(self.iceberg_slice_count, Ordering::Relaxed);

        // D-650 (DÉCIMA OLA) — LOS 12 GENES QUE EL HOT-SWAP NUNCA REFRESCABA.
        //
        // `from_genome()` inicializaba 140 genes y `apply_to_arena()` sólo
        // refrescaba 128. Tras cualquier promoción evolutiva en un proceso
        // vivo, el arena quedaba con 128 genes del genoma NUEVO y 12 del de
        // ARRANQUE: un organismo quimérico que ninguna función de aptitud
        // había evaluado jamás, porque nunca existió en la población.
        //
        // Dos de ellos son decisorios y activos: `conformal_alpha` gobierna el
        // umbral del filtro conformal y `vecm_beta_hedge` el del z-score VECM.
        // La evolución no podía modificarlos en un proceso vivo pero SÍ en el
        // backtest (que arranca en frío por `from_genome`), de modo que un
        // genoma evaluado con alpha=0,05 se desplegaba operando con el alpha
        // del genoma de arranque. Cuarta contribución verificada a la
        // divergencia backtest↔producción.
        //
        // El test `t2_simetria_from_genome_vs_apply_to_arena` fija la
        // exhaustividad como contrato: no volverá a reintroducirse por olvido.
        arena
            .config
            .conformal_alpha
            .store(self.conformal_alpha, Ordering::Relaxed);
        arena
            .config
            .global_learning_rate
            .store(self.global_learning_rate, Ordering::Relaxed);
        arena
            .config
            .global_momentum
            .store(self.global_momentum, Ordering::Relaxed);
        arena
            .config
            .min_trades_per_day
            .store(self.min_trades_per_day, Ordering::Relaxed);
        arena
            .config
            .survival_capital_threshold
            .store(self.survival_capital_threshold, Ordering::Relaxed);
        arena
            .config
            .vecm_alpha_speed
            .store(self.vecm_alpha_speed, Ordering::Relaxed);
        arena
            .config
            .vecm_beta_hedge
            .store(self.vecm_beta_hedge, Ordering::Relaxed);
        arena
            .config
            .ppo_weight_ofi
            .store(self.ppo_weight_ofi, Ordering::Relaxed);
        arena
            .config
            .ppo_weight_obi
            .store(self.ppo_weight_obi, Ordering::Relaxed);
        arena
            .config
            .ppo_weight_hawkes
            .store(self.ppo_weight_hawkes, Ordering::Relaxed);
        arena
            .config
            .ppo_weight_leadlag
            .store(self.ppo_weight_leadlag, Ordering::Relaxed);
        arena
            .config
            .ppo_weight_regime
            .store(self.ppo_weight_regime, Ordering::Relaxed);
    }

    pub fn mutate_cmaes(&self, rate: f64) -> Self {
        let mut rng = rand::rng();
        self.mutate_with_rng(rate, &mut rng)
    }

    /// DETERMINISMO MULTI-DÍA: variante sembrada — la certificación de
    /// backtests multi-día requiere que el enjambre mutante produzca la
    /// MISMA población en cada corrida (antes: rand::rng() sin semilla hacía
    /// que cada día de evolución promoviera un genoma distinto por corrida
    /// y el día siguiente heredara la lotería).
    pub fn mutate_cmaes_seeded(&self, rate: f64, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        self.mutate_with_rng(rate, &mut rng)
    }

    /// F8: muta los coeficientes (a, b) de una curva de horizonte dentro de
    /// bandas evolutivas. La pendiente b acotada a [b_min, b_max]: TP/SL
    /// CRECEN con el horizonte (física del costo de oportunidad) pero sin
    /// explotar (el clamp evita curvas absurdas en los extremos del espectro).
    fn mutate_curve<R: rand::Rng>(
        &self,
        curve: crate::temporal_spectrum::HorizonCurve,
        a_min: f64,
        a_max: f64,
        b_min: f64,
        b_max: f64,
        rng: &mut R,
        rate: f64,
    ) -> crate::temporal_spectrum::HorizonCurve {
        let mut a = curve.a;
        let mut b = curve.b;
        let span = a_max - a_min;
        a = (a + span * rate * rng.random_range(-0.5..0.5)).clamp(a_min, a_max);
        let span_b = b_max - b_min;
        b = (b + span_b * rate * rng.random_range(-0.5..0.5)).clamp(b_min, b_max);
        crate::temporal_spectrum::HorizonCurve { a, b }
    }

    fn mutate_with_rng<R: rand::Rng>(&self, rate: f64, rng: &mut R) -> Self {
        // F8: las curvas se mutan ANTES del literal (el closure mutate_val
        // captura rng por referencia única — no puede compartirse inline).
        // D-658 (DÉCIMA OLA): las bandas eran literales [−9; −2] y [−10; −3],
        // más estrechas que los `*_BOUNDS` que se declaran fuente única: toda
        // mutación de un genoma con `a` en [−9,5; −9) o [−10,5; −10) lo empujaba
        // hacia arriba aunque la tasa fuese cero. Ahora se leen las constantes.
        let mutated_tp_curve = self.mutate_curve(
            self.tp_horizon_curve,
            Self::TP_A_BOUNDS.0,
            Self::TP_A_BOUNDS.1,
            Self::TP_B_BOUNDS.0,
            Self::TP_B_BOUNDS.1,
            rng,
            rate,
        );
        let mutated_sl_curve = self.mutate_curve(
            self.sl_horizon_curve,
            Self::SL_A_BOUNDS.0,
            Self::SL_A_BOUNDS.1,
            Self::SL_B_BOUNDS.0,
            Self::SL_B_BOUNDS.1,
            rng,
            rate,
        );
        let mut mutate_val = |base: f64, min_val: f64, max_val: f64| -> f64 {
            let range = max_val - min_val;
            let change = range * rate * rng.random_range(-0.5..0.5);
            (base + change).clamp(min_val, max_val)
        };

        let mut mutated = Self {
            global_max_drawdown: mutate_val(self.global_max_drawdown, 0.5, 0.99),
            // D-644 (DÉCIMA OLA): la banda era [25, 35] en inicialización,
            // mutación y validación. La evolución NO PODÍA bajar de 25×: un
            // genoma prudente con 5× era empujado a 25× en su primera mutación,
            // una decisión de diseño disfrazada de resultado evolutivo. La banda
            // pasa a [1, 50], la misma que aplica el motor al leer el gen, de
            // modo que la prudencia vuelve a ser explorable y seleccionable.
            global_leverage: mutate_val(self.global_leverage, 1.0, 50.0),
            btc_volatility_multiplier: mutate_val(self.btc_volatility_multiplier, 0.5, 3.0),
            eth_volatility_multiplier: mutate_val(self.eth_volatility_multiplier, 0.5, 3.0),
            min_trades_per_day: mutate_val(self.min_trades_per_day, 1.0, 50.0),
            survival_capital_threshold: mutate_val(self.survival_capital_threshold, 0.5, 0.95),
            funding_rate_sensitivity: mutate_val(self.funding_rate_sensitivity, 0.1, 2.0),
            global_correlation_threshold: mutate_val(self.global_correlation_threshold, 0.2, 0.9),
            trend_threshold: mutate_val(self.trend_threshold, 0.52, 0.85),
            range_threshold: mutate_val(self.range_threshold, 0.1, 0.9),
            scalp_kelly_fraction: mutate_val(self.scalp_kelly_fraction, 0.1, 2.0),
            swing_kelly_fraction: mutate_val(self.swing_kelly_fraction, 0.01, 1.0),
            scalp_obi_threshold: mutate_val(self.scalp_obi_threshold, 0.05, 1.0),
            scalp_tp_base: self.scalp_tp_base, // X-005: vista — re-derivada de curvas post-literal
            scalp_sl_base: self.scalp_sl_base, // X-005: vista
            // F8: el GA evoluciona los COEFICIENTES de las curvas (a,b) — la
            // pendiente b controla cómo escala el parámetro con el horizonte.
            tp_horizon_curve: mutated_tp_curve,
            sl_horizon_curve: mutated_sl_curve,
            kelly_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            trail_mult_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            trail_act_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            trail_step_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            obi_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            swing_tp_base: self.swing_tp_base, // X-005: vista
            swing_sl_base: self.swing_sl_base, // X-005: vista
            sl_atr_mult_btc: mutate_val(self.sl_atr_mult_btc, 0.5, 5.0),
            tp_rr_ratio_btc: mutate_val(self.tp_rr_ratio_btc, 1.0, 10.0),
            min_confidence_btc: mutate_val(self.min_confidence_btc, 0.5, 0.95),
            veto_threshold_btc: mutate_val(self.veto_threshold_btc, 0.5, 0.99),
            // D-625: banda de mutación = bounds [0,24; 0,30]. Antes (0,05; 0,35): el
            // piso lo imponía el motor al LEER y el techo excedía los bounds.
            tech_threshold: mutate_val(self.tech_threshold, 0.24, 0.30),
            ml_threshold_long: mutate_val(self.ml_threshold_long, 0.5, 0.95),
            ml_threshold_short: mutate_val(self.ml_threshold_short, 0.05, 0.49),
            maker_spread_pct: mutate_val(self.maker_spread_pct, 0.0001, 0.01),
            maker_obi_threshold: mutate_val(self.maker_obi_threshold, 0.1, 0.95),
            target_volatility: mutate_val(self.target_volatility, 0.005, 0.1),
            dynamic_atr_min: mutate_val(self.dynamic_atr_min, 0.0001, 0.01),
            dynamic_obi_threshold: mutate_val(self.dynamic_obi_threshold, 0.05, 0.95),
            dynamic_ema_trend: mutate_val(self.dynamic_ema_trend, 0.0001, 0.01),
            dynamic_ofi_threshold: mutate_val(self.dynamic_ofi_threshold, 0.05, 0.95),
            capital_split_scalp: mutate_val(self.capital_split_scalp, 0.1, 1.0),
            kelly_clamp_min: mutate_val(self.kelly_clamp_min, 0.001, 0.1),
            kelly_clamp_max: mutate_val(self.kelly_clamp_max, 0.1, 1.0),
            explosive_leverage_multiplier: mutate_val(
                self.explosive_leverage_multiplier,
                1.0,
                10.0,
            ),
            quantum_mutation_rate: mutate_val(self.quantum_mutation_rate, 0.01, 0.5),
            temporal_memory_decay: mutate_val(self.temporal_memory_decay, 0.8, 0.9999),
            leverage_cap: mutate_val(self.leverage_cap, 10.0, 125.0),
            explosive_confidence_threshold: mutate_val(
                self.explosive_confidence_threshold,
                0.7,
                0.99,
            ),
            weight_obi: mutate_val(self.weight_obi, 0.1, 1.0),
            weight_ofi: mutate_val(self.weight_ofi, 0.1, 1.0),
            weight_vpin: mutate_val(self.weight_vpin, 0.1, 1.0),
            regime_duration_ms: mutate_val(self.regime_duration_ms, 30000.0, 600000.0),
            regime_atr_multiplier: mutate_val(self.regime_atr_multiplier, 0.5, 5.0),
            scalp_trail_act_atr: mutate_val(self.scalp_trail_act_atr, 0.001, 20.0),
            scalp_trail_step_atr: mutate_val(self.scalp_trail_step_atr, 0.001, 20.0),
            scalp_trail_max_atr: mutate_val(self.scalp_trail_max_atr, 0.001, 20.0),
            scalp_trail_min_pnl: mutate_val(self.scalp_trail_min_pnl, 0.001, 20.0),
            scalp_trail_atr_mult_base: mutate_val(self.scalp_trail_atr_mult_base, 0.001, 20.0),
            swing_trail_act_atr: mutate_val(self.swing_trail_act_atr, 0.001, 20.0),
            swing_trail_step_atr: mutate_val(self.swing_trail_step_atr, 0.001, 20.0),
            swing_trail_max_atr: mutate_val(self.swing_trail_max_atr, 0.001, 20.0),
            swing_trail_min_pnl: mutate_val(self.swing_trail_min_pnl, 0.001, 20.0),
            swing_trail_atr_mult_base: mutate_val(self.swing_trail_atr_mult_base, 0.001, 20.0),
            zombie_timeout_ms: mutate_val(self.zombie_timeout_ms, 14_400_000.0, 28_800_000.0),
            hurst_trend_threshold: mutate_val(self.hurst_trend_threshold, 0.4, 0.8),
            cvd_veto_threshold: mutate_val(self.cvd_veto_threshold, 0.1, 2.0),
            wall_veto_threshold: mutate_val(self.wall_veto_threshold, 5.0, 50.0),
            flash_crash_jump_pct: mutate_val(self.flash_crash_jump_pct, 0.05, 0.50),
            latency_ms_panic_threshold: mutate_val(self.latency_ms_panic_threshold, 500.0, 10000.0),
            ema_fast_period: mutate_val(self.ema_fast_period, 5.0, 30.0),
            ema_slow_period: mutate_val(self.ema_slow_period, 15.0, 60.0),
            hurst_scalp_threshold: mutate_val(self.hurst_scalp_threshold, 0.30, 0.50),
            hurst_swing_threshold: mutate_val(self.hurst_swing_threshold, 0.50, 0.70),
            synergy_exposure_boost: mutate_val(self.synergy_exposure_boost, 1.0, 3.0),
            synergy_leverage_boost: mutate_val(self.synergy_leverage_boost, 1.0, 2.0),
            max_fee_pct: mutate_val(self.max_fee_pct, 0.005, 0.05),
            kelly_bootstrap_cold: mutate_val(self.kelly_bootstrap_cold, 0.1, 1.0),
            latency_penalty_ms: mutate_val(self.latency_penalty_ms, 5.0, 100.0),
            base_slippage_floor: mutate_val(self.base_slippage_floor, 0.00005, 0.0005),
            spot_spread_threshold: mutate_val(self.spot_spread_threshold, 0.5, 3.0),
            spot_bias_value: mutate_val(self.spot_bias_value, 0.05, 0.30),
            obi_confidence_fallback: mutate_val(self.obi_confidence_fallback, 0.70, 0.95),
            ml_clip_lower: mutate_val(self.ml_clip_lower, 0.01, 0.20),
            ml_clip_upper: mutate_val(self.ml_clip_upper, 0.80, 0.99),
            tensor_poly_a: mutate_val(self.tensor_poly_a, 0.01, 0.20),
            tensor_poly_b: mutate_val(self.tensor_poly_b, 0.001, 0.05),
            fractional_alpha_order: mutate_val(self.fractional_alpha_order, 0.1, 0.9),
            fractional_clip_max: mutate_val(self.fractional_clip_max, 5.0, 20.0),
            bft_consensus_tolerance: mutate_val(self.bft_consensus_tolerance, 0.05, 0.30),
            turbo_coherence_threshold: mutate_val(self.turbo_coherence_threshold, 0.1, 0.6),
            turbo_z_score_stdev: mutate_val(self.turbo_z_score_stdev, 0.1, 1.5),
            sl_atr_multiplier: mutate_val(self.sl_atr_multiplier, 0.2, 2.0),
            coaxial_squeeze_threshold: mutate_val(self.coaxial_squeeze_threshold, 0.1, 0.8),
            tensor_op_add_bias: mutate_val(self.tensor_op_add_bias, 0.01, 0.5),
            tensor_op_mul_weight: mutate_val(self.tensor_op_mul_weight, 0.5, 3.0),
            topo_layer_1_activation: mutate_val(self.topo_layer_1_activation, 0.1, 0.9),
            topo_layer_2_activation: mutate_val(self.topo_layer_2_activation, 0.1, 0.9),
            tensor_dropout_rate: mutate_val(self.tensor_dropout_rate, 0.01, 0.5),
            quantum_entropy_seed: mutate_val(self.quantum_entropy_seed, 0.0, 1000.0),
            ppo_weight_ofi: mutate_val(self.ppo_weight_ofi, 0.01, 0.5),
            ppo_weight_obi: mutate_val(self.ppo_weight_obi, 0.01, 0.5),
            ppo_weight_hawkes: mutate_val(self.ppo_weight_hawkes, 0.01, 0.5),
            ppo_weight_leadlag: mutate_val(self.ppo_weight_leadlag, 0.01, 0.5),
            ppo_weight_regime: mutate_val(self.ppo_weight_regime, 0.01, 0.5),
            global_learning_rate: mutate_val(self.global_learning_rate, 0.0001, 0.05),
            global_momentum: mutate_val(self.global_momentum, 0.5, 0.99),
            vecm_alpha_speed: mutate_val(self.vecm_alpha_speed, 0.001, 0.1),
            vecm_beta_hedge: mutate_val(self.vecm_beta_hedge, 0.001, 0.1),
            conformal_alpha: mutate_val(self.conformal_alpha, 0.01, 0.2),
            ppo_clip_eps: mutate_val(self.ppo_clip_eps, 0.05, 0.40),
            ppo_weight_min_clip: mutate_val(self.ppo_weight_min_clip, 0.01, 0.10),
            hard_stop_decay_factor: mutate_val(self.hard_stop_decay_factor, 0.1, 1.0),
            hard_stop_base_limit: mutate_val(self.hard_stop_base_limit, 0.10, 0.80),
            kelly_bootstrap_ratio_threshold: mutate_val(
                self.kelly_bootstrap_ratio_threshold,
                0.05,
                0.30,
            ),
            kelly_bootstrap_min_exposure: mutate_val(self.kelly_bootstrap_min_exposure, 0.01, 0.5),
            ev_fee_multiplier: mutate_val(self.ev_fee_multiplier, 1.0, 5.0),
            margin_cushion_pct: mutate_val(self.margin_cushion_pct, 0.50, 0.98),
            maker_only_capital_threshold: 50.0,
            hawkes_scalp_threshold: mutate_val(self.hawkes_scalp_threshold, 0.50, 0.95),
            obi_zscore_threshold: mutate_val(self.obi_zscore_threshold, 0.1, 3.0),

            hawkes_volume_norm: mutate_val(self.hawkes_volume_norm, 100.0, 10_000_000.0),
            base_duration_ms: mutate_val(self.base_duration_ms, 10_000.0, 120_000.0),

            kelly_survival_cap_ratio: mutate_val(self.kelly_survival_cap_ratio, 1.0, 3.0),
            kelly_expansion_mult: mutate_val(self.kelly_expansion_mult, 1.0, 3.0),
            guard_dd_sigmoid_steepness: mutate_val(self.guard_dd_sigmoid_steepness, 1.0, 5.0),
            guard_dd_sigmoid_center: mutate_val(self.guard_dd_sigmoid_center, 1.0, 5.0),
            portfolio_perf_mult_steepness: mutate_val(
                self.portfolio_perf_mult_steepness,
                3.0,
                15.0,
            ),
            portfolio_dd_penalty_decay: mutate_val(self.portfolio_dd_penalty_decay, 5.0, 25.0),
            portfolio_perf_mult_min: mutate_val(self.portfolio_perf_mult_min, 0.1, 0.5),
            portfolio_perf_mult_max: mutate_val(self.portfolio_perf_mult_max, 1.1, 3.0),
            portfolio_perf_mult_center: mutate_val(self.portfolio_perf_mult_center, 0.4, 0.6),

            macro_hurst_confidence_offset: mutate_val(
                self.macro_hurst_confidence_offset,
                0.35,
                0.55,
            ),
            macro_hurst_confidence_scale: mutate_val(self.macro_hurst_confidence_scale, 1.0, 20.0),
            macro_vol_confidence_scale: mutate_val(self.macro_vol_confidence_scale, 10.0, 500.0),
            macro_min_cooldown_ratio: mutate_val(self.macro_min_cooldown_ratio, 0.05, 0.5),
            macro_max_cooldown_ratio: mutate_val(self.macro_max_cooldown_ratio, 10.0, 500.0),
            macro_cooldown_reduction_factor: mutate_val(
                self.macro_cooldown_reduction_factor,
                0.1,
                2.0,
            ),
            macro_leverage_momentum_scale: mutate_val(self.macro_leverage_momentum_scale, 0.1, 2.0),

            lev_matrix_vol_clamp_min: mutate_val(self.lev_matrix_vol_clamp_min, 0.1, 0.8),
            lev_matrix_growth_scalar: mutate_val(self.lev_matrix_growth_scalar, 0.1, 1.0),
            lev_matrix_log_cap_divisor: mutate_val(self.lev_matrix_log_cap_divisor, 3.0, 10.0),
            scalp_accel_min_samples: mutate_val(self.scalp_accel_min_samples, 10.0, 50.0),
            executor_max_orders_10s: mutate_val(self.executor_max_orders_10s, 100.0, 500.0),
            executor_max_weight_1m: mutate_val(self.executor_max_weight_1m, 1000.0, 5000.0),
            iceberg_volume_threshold: mutate_val(
                self.iceberg_volume_threshold,
                10_000.0,
                100_000.0,
            ),
            iceberg_slice_count: mutate_val(self.iceberg_slice_count, 2.0, 10.0),
            swing_obi_threshold: mutate_val(self.swing_obi_threshold, 0.05, 1.0),
            swing_accel_min_samples: mutate_val(self.swing_accel_min_samples, 10.0, 50.0),
            temporal_scale: mutate_val(self.temporal_scale, 0.05, 0.95),
        };

        // X-005 (REHAB-1) — FUENTE ÚNICA: el reparo RR vive SOBRE LAS CURVAS
        // (en todo el espectro, no en dos puntos) y las anclas se derivan
        // después. La versión anterior reparaba las anclas del literal — que
        // ya no son genes (son vistas) — dejando a las curvas libres de
        // violar TP(τ)>SL(τ) fuera de los dos puntos de anclaje.
        mutated.enforce_curve_rr();
        mutated.derive_anchors_from_curves();
        mutated.sync_continuous_curves();

        mutated
    }

    // Generated extensions
    /// X-004: 140 anclas legacy + 4 coeficientes de curvas (tp_a, tp_b, sl_a, sl_b).
    pub const DIMENSION: usize = 144;

    /// X-004/X-005 (REHAB-1): las anclas legacy se RE-DERIVAN de las curvas —
    /// vistas de compatibilidad, jamás fuente independiente. Todo camino que
    /// construya o mute un genoma debe terminar llamando a esto.
    pub fn derive_anchors_from_curves(&mut self) {
        use crate::temporal_spectrum::{TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};
        self.scalp_tp_base = self.tp_horizon_curve.eval(TAU_ANCHOR_FAST_MS);
        self.swing_tp_base = self.tp_horizon_curve.eval(TAU_ANCHOR_SLOW_MS);
        self.scalp_sl_base = self.sl_horizon_curve.eval(TAU_ANCHOR_FAST_MS);
        self.swing_sl_base = self.sl_horizon_curve.eval(TAU_ANCHOR_SLOW_MS);
    }

    /// Sincroniza las curvas continuas de horizonte para Kelly, Trailing y OBI
    /// a partir de los parámetros del genoma en los puntos de anclaje (fast/slow).
    /// D-683 (DÉCIMA OLA): copia con las curvas continuas derivadas de los
    /// genes. Es lo que el arena debe recibir, venga el genoma de donde venga.
    pub fn with_synced_continuous_curves(&self) -> Self {
        let mut g = self.clone();
        g.sync_continuous_curves();
        g
    }

    pub fn sync_continuous_curves(&mut self) {
        use crate::temporal_spectrum::{HorizonCurve, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};
        self.kelly_horizon_curve = HorizonCurve::through_two_points(
            TAU_ANCHOR_FAST_MS,
            self.scalp_kelly_fraction.max(0.001),
            TAU_ANCHOR_SLOW_MS,
            self.swing_kelly_fraction.max(0.001),
        );
        self.trail_mult_horizon_curve = HorizonCurve::through_two_points(
            TAU_ANCHOR_FAST_MS,
            self.scalp_trail_atr_mult_base.max(0.001),
            TAU_ANCHOR_SLOW_MS,
            self.swing_trail_atr_mult_base.max(0.001),
        );
        self.trail_act_horizon_curve = HorizonCurve::through_two_points(
            TAU_ANCHOR_FAST_MS,
            self.scalp_trail_act_atr.max(0.001),
            TAU_ANCHOR_SLOW_MS,
            self.swing_trail_act_atr.max(0.001),
        );
        self.trail_step_horizon_curve = HorizonCurve::through_two_points(
            TAU_ANCHOR_FAST_MS,
            self.scalp_trail_step_atr.max(0.001),
            TAU_ANCHOR_SLOW_MS,
            self.swing_trail_step_atr.max(0.001),
        );
        self.obi_horizon_curve = HorizonCurve::through_two_points(
            TAU_ANCHOR_FAST_MS,
            self.scalp_obi_threshold.max(0.001),
            TAU_ANCHOR_SLOW_MS,
            self.swing_obi_threshold.max(0.001),
        );
    }

    /// Bandas evolutivas de los coeficientes de curva — FUENTE ÚNICA de las
    /// cotas (los bounds del vector las leen; el reparo RR clampa contra ellas).
    pub const TP_A_BOUNDS: (f64, f64) = (-9.5, -2.0);
    pub const TP_B_BOUNDS: (f64, f64) = (-0.2, 0.35);
    pub const SL_A_BOUNDS: (f64, f64) = (-10.5, -3.0);
    pub const SL_B_BOUNDS: (f64, f64) = (-0.2, 0.35);

    /// C-05 (INFORME 14, FASE 0): τ de DECISIÓN acotada a la banda operativa
    /// [TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS] (30s–12h). Con la τ degenerada
    /// de la fusión espectral (escala 31 ≈ 146 años), la extrapolación
    /// exponencial de `HorizonCurve::eval` fuera de banda EXPLOTA (SL/TP de
    /// más del 100% del precio: los brackets +65%/−32% del informe). El
    /// espectro puede VER más allá de la banda; NINGÚN lector del genoma
    /// evalúa las curvas fuera de ella.
    #[inline]
    fn tau_in_operating_band(tau_ms: f64) -> f64 {
        use crate::temporal_spectrum::{TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};
        if tau_ms.is_finite() && tau_ms > 0.0 {
            tau_ms.clamp(TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS)
        } else {
            TAU_ANCHOR_FAST_MS
        }
    }

    #[inline]
    pub fn tp_at_tau(&self, tau_ms: f64) -> f64 {
        self.tp_horizon_curve
            .eval(Self::tau_in_operating_band(tau_ms))
    }

    #[inline]
    pub fn sl_at_tau(&self, tau_ms: f64) -> f64 {
        self.sl_horizon_curve
            .eval(Self::tau_in_operating_band(tau_ms))
    }

    #[inline]
    pub fn kelly_at_tau(&self, tau_ms: f64) -> f64 {
        self.kelly_horizon_curve
            .eval(Self::tau_in_operating_band(tau_ms))
            .clamp(0.01, 3.0)
    }

    #[inline]
    pub fn trail_params_at_tau(&self, tau_ms: f64) -> (f64, f64, f64) {
        // C-05: las tres curvas se evalúan en la MISMA τ de banda — sin el
        // clamp, un τ degenerado separaba mult/act/step a decades de distancia.
        let tau = Self::tau_in_operating_band(tau_ms);
        let mult = self.trail_mult_horizon_curve.eval(tau).clamp(0.01, 20.0);
        let act = self.trail_act_horizon_curve.eval(tau).clamp(0.01, 20.0);
        let step = self.trail_step_horizon_curve.eval(tau).clamp(0.01, 20.0);
        (mult, act, step)
    }

    #[inline]
    pub fn obi_threshold_at_tau(&self, tau_ms: f64) -> f64 {
        // MOD3/5-005 (INFORME 14, C-09): el techo del lector es 0.95, igual
        // que la banda evolutiva del gen (mutate 0.05..1.0). Un techo de 0.60
        // hacía invisible el 25% superior de la banda — el campeón con OBI
        // 0.797 se leía como 0.60.
        self.obi_horizon_curve
            .eval(Self::tau_in_operating_band(tau_ms))
            .clamp(0.05, 0.95)
    }

    /// X-005 (REHAB-1): invariante RR SOBRE CURVAS — TP(τ) ≥ SL(τ)·MIN_RR_MUTATION
    /// en ambas anclas del espectro. Estrategia: (1) deprimir a_sl (escala
    /// uniforme de SL sin tocar pendiente); (2) re-clamp a banda evolutiva;
    /// (3) si la pendiente evolucionada hace la violación irrepurable dentro
    /// de bandas, fallback determinista: SL paralelo a TP con RR=REPAIR en
    /// TODO el espectro (matemáticamente seguro por construcción).
    pub fn enforce_curve_rr(&mut self) {
        let fee = Self::REFERENCE_ROUNDTRIP_FEE;
        let scales = crate::temporal_spectrum::SPECTRUM_SCALES_MS;
        let hi_spec = scales[scales.len() - 1];
        let sl_floor = Self::min_viable_sl(fee);

        // PASO 0 — GARANTIZAR BANDA OPERABLE. Si la curva de SL nunca alcanza
        // el mínimo viable, el genoma no puede operar en ninguna escala:
        // se eleva el intercepto de SL hasta que el extremo largo del
        // espectro sí lo alcance (D-636b).
        if self.tradeable_band_ms(fee).is_none() {
            let sl_hi = self.sl_horizon_curve.eval(hi_spec).max(1e-12);
            let target = sl_floor * 1.10;
            self.sl_horizon_curve.a = (self.sl_horizon_curve.a + (target / sl_hi).ln())
                .clamp(Self::SL_A_BOUNDS.0, Self::SL_A_BOUNDS.1);
        }

        // RR exigido a cada tau (D-636): crece cuando el stop se estrecha,
        // porque la fricción pesa más sobre un riesgo menor. Se añade un 10 %
        // de holgura para que la mutación no quede pegada al límite del gate.
        let required_at = |g: &Self, tau: f64| -> f64 {
            let sl = g.sl_horizon_curve.eval(tau);
            Self::min_rr_for(Self::WORST_TOLERATED_WR, fee, sl) * 1.10
        };
        let band_taus = |g: &Self| -> [f64; 2] {
            match g.tradeable_band_ms(fee) {
                Some((lo, hi)) => [lo, hi],
                None => [hi_spec, hi_spec],
            }
        };
        let violated = |g: &Self| {
            band_taus(g).iter().any(|&tau| {
                let tp = g.tp_horizon_curve.eval(tau);
                let sl = g.sl_horizon_curve.eval(tau);
                !tp.is_finite() || !sl.is_finite() || tp < sl * required_at(g, tau)
            })
        };
        if !violated(self) {
            return;
        }

        // PASO 1 — ELEVAR EL TP, NO DEPRIMIR EL SL.
        //
        // En el borde inferior de la banda el SL está PINNED al mínimo viable
        // por definición: deprimirlo solo desplaza el borde a la derecha y
        // deja el mismo conflicto en el borde nuevo (punto fijo que nunca
        // converge — el bug de la versión anterior de esta reparación).
        // La lectura económica correcta es la contraria: si la fricción exige
        // más recompensa por el mismo riesgo, lo que sube es la recompensa.
        // Escala uniforme de TP(τ) en TODO el espectro sin tocar pendiente.
        for _ in 0..6 {
            let mut worst = 1.0f64;
            for &tau in band_taus(self).iter() {
                let tp = self.tp_horizon_curve.eval(tau);
                let sl = self.sl_horizon_curve.eval(tau);
                let req = required_at(self, tau);
                if tp > 0.0 && sl > 0.0 && tp < sl * req {
                    worst = worst.max(sl * req / tp);
                }
            }
            if worst <= 1.0 {
                break;
            }
            self.tp_horizon_curve.a = (self.tp_horizon_curve.a + worst.ln())
                .clamp(Self::TP_A_BOUNDS.0, Self::TP_A_BOUNDS.1);
            if !violated(self) {
                break;
            }
        }

        // PASO 2 — SI EL TP TOPÓ CON SU BANDA EVOLUTIVA, ESTRECHAR EL RIESGO.
        // Con a_tp en su techo la única salida es reducir SL; eso encoge la
        // banda operable por abajo (menos escalas operables), que es el
        // resultado honesto: ese genoma solo puede operar horizontes largos.
        for _ in 0..6 {
            if !violated(self) {
                break;
            }
            let mut worst = 1.0f64;
            for &tau in band_taus(self).iter() {
                let tp = self.tp_horizon_curve.eval(tau);
                let sl = self.sl_horizon_curve.eval(tau);
                let req = required_at(self, tau);
                if tp > 0.0 && sl > 0.0 && tp < sl * req {
                    worst = worst.max(sl * req / tp);
                }
            }
            if worst <= 1.0 {
                break;
            }
            self.sl_horizon_curve.a = (self.sl_horizon_curve.a - worst.ln())
                .clamp(Self::SL_A_BOUNDS.0, Self::SL_A_BOUNDS.1);
        }

        // PASO 3 — FALLBACK DETERMINISTA. Curvas paralelas con RR constante
        // igual al peor exigido en la banda: segura por construcción en todo
        // el espectro, sea cual sea la pendiente evolucionada.
        if violated(self) {
            let req = band_taus(self)
                .iter()
                .map(|&t| required_at(self, t))
                .fold(Self::MIN_RR_REPAIR, f64::max);
            self.sl_horizon_curve.b = self.tp_horizon_curve.b;
            self.sl_horizon_curve.a = (self.tp_horizon_curve.a - req.ln())
                .clamp(Self::SL_A_BOUNDS.0, Self::SL_A_BOUNDS.1);
            // Re-garantizar la banda tras el paralelizado.
            if self.tradeable_band_ms(fee).is_none() {
                let sl_hi = self.sl_horizon_curve.eval(hi_spec).max(1e-12);
                let target = sl_floor * 1.10;
                self.sl_horizon_curve.a = (self.sl_horizon_curve.a + (target / sl_hi).ln())
                    .clamp(Self::SL_A_BOUNDS.0, Self::SL_A_BOUNDS.1);
                self.tp_horizon_curve.a = (self.sl_horizon_curve.a + req.ln())
                    .clamp(Self::TP_A_BOUNDS.0, Self::TP_A_BOUNDS.1);
            }
        }
    }

    /// D-636 (DÉCIMA OLA) — RR MÍNIMO CORRECTAMENTE DERIVADO.
    ///
    /// LA DERIVACIÓN ANTERIOR ERA DIMENSIONALMENTE ERRÓNEA. Escribía el
    /// término de comisión como el FACTOR multiplicativo (1+f)/(1−f), que
    /// compara `f` con la UNIDAD (el nocional completo). El término correcto
    /// es ADITIVO y compara `f` con `SL`, que es la magnitud de la pérdida.
    /// Con SL ≈ 0,005 el peso real de la comisión es f/SL ≈ 0,16, mientras
    /// que la fórmula vieja lo estimaba en f/1 ≈ 0,0008: un factor 200×.
    ///
    /// DERIVACIÓN CORRECTA. Con TP y SL movimientos fraccionales de precio y
    /// `f` la comisión de ida y vuelta como fracción del nocional:
    ///
    /// ```text
    /// EV = w·(TP − f) − (1 − w)·(SL + f) ≥ 0
    /// w·TP ≥ (1 − w)·SL + f
    /// RR = TP/SL ≥ (1 − w)/w + f/(w·SL)
    ///              └─ equilibrio   └─ término de FRICCIÓN, el que faltaba
    ///                 sin fees
    /// ```
    ///
    /// El valor viejo 1,5 es EXACTAMENTE (1−w)/w con w=0,40: el equilibrio
    /// SIN comisiones. Evaluando el EV en ese límite se obtiene el resultado
    /// analítico `EV = −f` — independiente del nivel de SL. Es decir: un
    /// genoma que pasaba el gate en el límite pagaba la comisión íntegra en
    /// cada operación y no capturaba ningún edge (−8 bps por trade con
    /// f = 0,0008).
    ///
    /// Al ser función de SL, el mínimo ya NO puede ser una constante:
    /// `min_rr_for()` lo calcula por nivel de stop.
    #[inline]
    pub fn min_rr_for(win_rate: f64, roundtrip_fee: f64, sl: f64) -> f64 {
        // Saneamiento: fuera de rango se cae al peor caso tolerado, nunca a
        // un valor permisivo.
        let w = if win_rate.is_finite() {
            win_rate.clamp(0.05, 0.95)
        } else {
            Self::WORST_TOLERATED_WR
        };
        let f = if roundtrip_fee.is_finite() && roundtrip_fee >= 0.0 {
            roundtrip_fee
        } else {
            Self::REFERENCE_ROUNDTRIP_FEE
        };
        let sl_eff = if sl.is_finite() && sl > 1e-9 {
            sl
        } else {
            Self::REFERENCE_SL
        };
        (1.0 - w) / w + f / (w * sl_eff)
    }

    /// B3.2 — PISOS DE VIABILIDAD POR FRICCIÓN para cualquier geometría de
    /// bracket (fallback de entrada, watchdog, restore, adopción, gestión
    /// temporal). La regla de `risk_engine::tp_sl::compute_tp_sl`
    /// (D-636/D-681) promulgada como INVARIANTE sobre fracciones ya
    /// calculadas: el stop nunca baja del mínimo viable frente a la fricción
    /// y el objetivo nunca baja del stop por el RR que la fricción exige al
    /// win rate de diseño. Con esto NINGÚN bracket nace con un TP incapaz
    /// de pagar sus propias comisiones (arrastre medido: fees −108,7 % del
    /// PnL bruto, 10 trades, testnet 2026-09-15).
    ///
    /// Devuelve `(sl_frac, tp_frac)` — fracciones del precio, ambas > 0 y
    /// con EV ≥ 0 por construcción a `WORST_TOLERATED_WR`.
    pub fn friction_floors(roundtrip_fee: f64, sl_frac: f64, tp_frac: f64) -> (f64, f64) {
        let fee = if roundtrip_fee.is_finite() && roundtrip_fee > 0.0 {
            roundtrip_fee
        } else {
            Self::REFERENCE_ROUNDTRIP_FEE
        };
        let sl_floor = Self::min_viable_sl(fee);
        let mut sl = if sl_frac.is_finite() && sl_frac > 0.0 {
            sl_frac.max(sl_floor)
        } else {
            sl_floor
        };
        let tp_floor = sl * Self::min_rr_for(Self::WORST_TOLERATED_WR, fee, sl).max(1.0);
        let mut tp = if tp_frac.is_finite() && tp_frac > 0.0 {
            tp_frac.max(tp_floor)
        } else {
            tp_floor
        };
        // B3.24 — CAP DE ASIMETRÍA REALIZADA (RR geométrico ≥ 2 SIEMPRE).
        // Medido en OOS jun+sep: stops difusivos escalaban a 3-6% con el
        // ATR mientras el trailing tomaba los wins a 0.6-2% — la asimetría
        // win/loss REALIZADA invertía el RR de diseño (BNB WR 57% con neto
        // negativo). El suelo de viabilidad acota el MÍNIMO del SL; esto
        // acota el MÁXIMO: jamás arriesgar más de la mitad del objetivo.
        // Orden: primero se encoge el SL al cap, después el TP se re-asegura
        // ≥ SL·RR_min(fee) — converge en una pasada porque el SL sólo baja
        // (y al bajar, su RR_min baja con él).
        if sl > tp * 0.5 {
            sl = tp * 0.5;
            let rr_min = Self::min_rr_for(Self::WORST_TOLERATED_WR, fee, sl).max(1.0);
            if tp < sl * rr_min {
                tp = sl * rr_min;
            }
        }
        (sl, tp)
    }

    /// Peor win rate que el sistema tolera — el punto de diseño conservador
    /// desde el que se dimensiona el gate.
    pub const WORST_TOLERATED_WR: f64 = 0.40;

    /// Win rate realista de diseño en régimen espectral continuo (55%).
    pub const DESIGN_WIN_RATE: f64 = 0.55;

    /// D-680 (DÉCIMA OLA): peso del prior del win rate, en operaciones
    /// equivalentes. Es `z²` con `z = 1,959964` (95 %): la misma corrección que
    /// centra el intervalo de Wilson / Agresti–Coull, aquí hacia el win rate de
    /// diseño en lugar de hacia ½. Una sola operación ya no puede llevar la
    /// estimación a 0 ni a 1.
    pub const WR_PRIOR_PSEUDO_TRADES: f64 = 1.959_963_984_540_054 * 1.959_963_984_540_054;

    /// Media posterior del win rate tras una operación más. `current` es la
    /// media posterior tras `n_prev` operaciones (el prior de diseño cuando
    /// `n_prev` = 0). Equivale a `(aciertos + k·w₀) / (n + k)`.
    #[inline]
    pub fn posterior_win_rate(current: f64, n_prev: f64, is_win: bool) -> f64 {
        let k = Self::WR_PRIOR_PSEUDO_TRADES;
        let cur = if current.is_finite() {
            current.clamp(0.0, 1.0)
        } else {
            Self::WORST_TOLERATED_WR
        };
        let n = if n_prev.is_finite() && n_prev >= 0.0 {
            n_prev
        } else {
            0.0
        };
        let x = if is_win { 1.0 } else { 0.0 };
        (cur * (n + k) + x) / (n + 1.0 + k)
    }
    /// Fricción de referencia de ida y vuelta. D-645: la física del motor
    /// aplica 2×taker + deslizamiento, NO maker+taker. Con taker = 5 bps por
    /// pierna: 2 × 0,0005 = 0,0010.
    pub const REFERENCE_ROUNDTRIP_FEE: f64 = 0.0010;
    /// SL de referencia para las cotas estáticas (el centro de la banda
    /// operativa real del sistema, 40–60 bps → 50 bps).
    pub const REFERENCE_SL: f64 = 0.0050;

    /// D-636b (DÉCIMA OLA) — BANDA OPERABLE DERIVADA DE LA FRICCIÓN.
    ///
    /// Al corregir la derivación del RR emergió un hecho físico que el
    /// sistema nunca había representado: `RR_req(SL) = (1−w)/w + f/(w·SL)`
    /// DIVERGE cuando SL → 0. A un stop de 1,5 bps con una fricción de ida y
    /// vuelta de 10 bps se exigiría RR ≈ 17,8 — un recorrido que el mercado
    /// no entrega antes de tocar el stop. **No es que el sistema no deba
    /// operar a 1 ms: es que a 1 ms no existe operación rentable posible.**
    ///
    /// Esto NO reintroduce un bucket arbitrario. La frontera se DERIVA: una
    /// operación es viable mientras la fricción no domine el riesgo asumido.
    /// Expresado como presupuesto de fricción sobre el riesgo:
    ///
    /// ```text
    /// f ≤ MAX_FRICTION_SHARE_OF_RISK · SL
    /// ```
    ///
    /// Si la fricción supera el riesgo, la posición deja de ser una tesis de
    /// mercado y pasa a ser una máquina de generar comisiones: su resultado
    /// esperado lo fija el coste, no el análisis. La mitad es la línea
    /// conservadora natural — el punto en que el coste iguala a la mitad de
    /// lo que se arriesga conscientemente.
    ///
    /// El espectro sigue OBSERVÁNDOSE completo (las 19 escalas alimentan el
    /// score espectral); lo que la banda acota es dónde se ABRE posición.
    pub const MAX_FRICTION_SHARE_OF_RISK: f64 = 0.65;

    /// SL mínimo con el que una operación puede ser rentable dada la fricción.
    /// Derivado, no elegido: `SL_min = f / MAX_FRICTION_SHARE_OF_RISK`.
    #[inline]
    pub fn min_viable_sl(roundtrip_fee: f64) -> f64 {
        let f = if roundtrip_fee.is_finite() && roundtrip_fee > 0.0 {
            roundtrip_fee
        } else {
            Self::REFERENCE_ROUNDTRIP_FEE
        };
        f / Self::MAX_FRICTION_SHARE_OF_RISK
    }

    /// Horizonte mínimo OPERABLE de ESTE genoma: la τ a la que su propia
    /// curva de SL alcanza el mínimo viable. Como `SL(τ) = exp(a + b·ln τ)`
    /// es monótona, se despeja en forma cerrada:
    ///
    /// ```text
    /// ln τ_min = (ln SL_min − a) / b        (b > 0)
    /// ```
    ///
    /// Con `b ≤ 0` (SL que no crece con el horizonte) la curva es plana o
    /// decreciente: o bien todo el espectro es operable, o ninguno lo es.
    pub fn min_tradeable_tau_ms(&self, roundtrip_fee: f64) -> f64 {
        let scales = crate::temporal_spectrum::SPECTRUM_SCALES_MS;
        let (lo, hi) = (scales[0], scales[scales.len() - 1]);
        let sl_min = Self::min_viable_sl(roundtrip_fee);
        let b = self.sl_horizon_curve.b;
        if b.abs() < 1e-12 {
            return if self.sl_horizon_curve.eval(lo) >= sl_min {
                lo
            } else {
                f64::INFINITY
            };
        }
        let ln_tau = (sl_min.ln() - self.sl_horizon_curve.a) / b;
        if !ln_tau.is_finite() {
            return f64::INFINITY;
        }
        let tau = ln_tau.exp();
        if b > 0.0 {
            tau.clamp(lo, f64::INFINITY)
        } else {
            // SL decrece con tau: lo operable está por DEBAJO de tau; si ni
            // siquiera la escala mas corta es viable, nada lo es.
            if self.sl_horizon_curve.eval(lo) >= sl_min {
                lo
            } else {
                f64::INFINITY
            }
        }
        .min(if b > 0.0 { f64::INFINITY } else { hi })
    }

    /// Extremos de la banda operable sobre los que se verifica la invariante
    /// RR. Devuelve `None` si el genoma no tiene NINGUNA escala operable —
    /// condición que el gate de promoción rechaza explícitamente.
    pub fn tradeable_band_ms(&self, roundtrip_fee: f64) -> Option<(f64, f64)> {
        let scales = crate::temporal_spectrum::SPECTRUM_SCALES_MS;
        let hi = scales[scales.len() - 1];
        let lo = self.min_tradeable_tau_ms(roundtrip_fee);
        if !lo.is_finite() || lo > hi {
            return None;
        }
        Some((lo.max(scales[0]), hi))
    }

    /// RR mínimo que el GATE de promoción exige, evaluado en el punto de
    /// referencia. Antes 1,5 (equilibrio sin fees); ahora ≈ 2,00.
    pub const MIN_RR_GATE: f64 = (1.0 - Self::WORST_TOLERATED_WR) / Self::WORST_TOLERATED_WR
        + Self::REFERENCE_ROUNDTRIP_FEE / (Self::WORST_TOLERATED_WR * Self::REFERENCE_SL);
    /// RR mínimo que la MUTACIÓN exige: el gate + 10 % de holgura por spread
    /// y deslizamiento no modelados, para que el operador de mutación no
    /// produzca sistemáticamente genomas que el gate rechazará.
    pub const MIN_RR_MUTATION: f64 = Self::MIN_RR_GATE * 1.10;
    /// Objetivo de reparación al violar la invariante: mutación + 10 % más,
    /// para no reparar pegado al límite y volver a violarlo al mutar.
    pub const MIN_RR_REPAIR: f64 = Self::MIN_RR_MUTATION * 1.10;

    pub fn to_vector(&self) -> Vec<f64> {
        let mut vec = Vec::with_capacity(Self::DIMENSION);
        vec.push(self.global_max_drawdown);
        vec.push(self.global_leverage);
        vec.push(self.btc_volatility_multiplier);
        vec.push(self.eth_volatility_multiplier);
        vec.push(self.min_trades_per_day);
        vec.push(self.survival_capital_threshold);
        vec.push(self.funding_rate_sensitivity);
        vec.push(self.global_correlation_threshold);
        vec.push(self.trend_threshold);
        vec.push(self.range_threshold);
        vec.push(self.scalp_kelly_fraction);
        vec.push(self.swing_kelly_fraction);
        vec.push(self.scalp_obi_threshold);
        vec.push(self.scalp_tp_base);
        vec.push(self.scalp_sl_base);
        vec.push(self.swing_tp_base);
        vec.push(self.swing_sl_base);
        vec.push(self.sl_atr_mult_btc);
        vec.push(self.tp_rr_ratio_btc);
        vec.push(self.min_confidence_btc);
        vec.push(self.veto_threshold_btc);
        vec.push(self.tech_threshold);
        vec.push(self.ml_threshold_long);
        vec.push(self.ml_threshold_short);
        vec.push(self.maker_spread_pct);
        vec.push(self.maker_obi_threshold);
        vec.push(self.target_volatility);
        vec.push(self.dynamic_atr_min);
        vec.push(self.dynamic_obi_threshold);
        vec.push(self.dynamic_ema_trend);
        vec.push(self.dynamic_ofi_threshold);
        vec.push(self.capital_split_scalp);
        vec.push(self.kelly_clamp_min);
        vec.push(self.kelly_clamp_max);
        vec.push(self.explosive_leverage_multiplier);
        vec.push(self.quantum_mutation_rate);
        vec.push(self.temporal_memory_decay);
        vec.push(self.leverage_cap);
        vec.push(self.explosive_confidence_threshold);
        vec.push(self.weight_obi);
        vec.push(self.weight_ofi);
        vec.push(self.weight_vpin);
        vec.push(self.regime_duration_ms);
        vec.push(self.regime_atr_multiplier);
        vec.push(self.scalp_trail_act_atr);
        vec.push(self.scalp_trail_step_atr);
        vec.push(self.scalp_trail_max_atr);
        vec.push(self.scalp_trail_min_pnl);
        vec.push(self.scalp_trail_atr_mult_base);
        vec.push(self.swing_trail_act_atr);
        vec.push(self.swing_trail_step_atr);
        vec.push(self.swing_trail_max_atr);
        vec.push(self.swing_trail_min_pnl);
        vec.push(self.swing_trail_atr_mult_base);
        vec.push(self.zombie_timeout_ms);
        vec.push(self.hurst_trend_threshold);
        vec.push(self.cvd_veto_threshold);
        vec.push(self.wall_veto_threshold);
        vec.push(self.flash_crash_jump_pct);
        vec.push(self.latency_ms_panic_threshold);
        vec.push(self.ema_fast_period);
        vec.push(self.ema_slow_period);
        vec.push(self.hurst_scalp_threshold);
        vec.push(self.hurst_swing_threshold);
        vec.push(self.synergy_exposure_boost);
        vec.push(self.synergy_leverage_boost);
        vec.push(self.max_fee_pct);
        vec.push(self.kelly_bootstrap_cold);
        vec.push(self.latency_penalty_ms);
        vec.push(self.base_slippage_floor);
        vec.push(self.spot_spread_threshold);
        vec.push(self.spot_bias_value);
        vec.push(self.obi_confidence_fallback);
        vec.push(self.ml_clip_lower);
        vec.push(self.ml_clip_upper);
        vec.push(self.tensor_poly_a);
        vec.push(self.tensor_poly_b);
        vec.push(self.fractional_alpha_order);
        vec.push(self.fractional_clip_max);
        vec.push(self.bft_consensus_tolerance);
        vec.push(self.turbo_coherence_threshold);
        vec.push(self.turbo_z_score_stdev);
        vec.push(self.sl_atr_multiplier);
        vec.push(self.coaxial_squeeze_threshold);
        vec.push(self.tensor_op_add_bias);
        vec.push(self.tensor_op_mul_weight);
        vec.push(self.topo_layer_1_activation);
        vec.push(self.topo_layer_2_activation);
        vec.push(self.tensor_dropout_rate);
        vec.push(self.quantum_entropy_seed);
        vec.push(self.ppo_weight_ofi);
        vec.push(self.ppo_weight_obi);
        vec.push(self.ppo_weight_hawkes);
        vec.push(self.ppo_weight_leadlag);
        vec.push(self.ppo_weight_regime);
        vec.push(self.global_learning_rate);
        vec.push(self.global_momentum);
        vec.push(self.vecm_alpha_speed);
        vec.push(self.vecm_beta_hedge);
        vec.push(self.conformal_alpha);
        vec.push(self.ppo_clip_eps);
        vec.push(self.ppo_weight_min_clip);
        vec.push(self.hard_stop_decay_factor);
        vec.push(self.hard_stop_base_limit);
        vec.push(self.kelly_bootstrap_ratio_threshold);
        vec.push(self.kelly_bootstrap_min_exposure);
        vec.push(self.ev_fee_multiplier);
        vec.push(self.margin_cushion_pct);
        vec.push(self.maker_only_capital_threshold);
        vec.push(self.hawkes_scalp_threshold);
        vec.push(self.obi_zscore_threshold);
        vec.push(self.hawkes_volume_norm);
        vec.push(self.base_duration_ms);
        vec.push(self.kelly_survival_cap_ratio);
        vec.push(self.kelly_expansion_mult);
        vec.push(self.guard_dd_sigmoid_steepness);
        vec.push(self.guard_dd_sigmoid_center);
        vec.push(self.portfolio_perf_mult_steepness);
        vec.push(self.portfolio_dd_penalty_decay);
        vec.push(self.portfolio_perf_mult_min);
        vec.push(self.portfolio_perf_mult_max);
        vec.push(self.portfolio_perf_mult_center);
        vec.push(self.macro_hurst_confidence_offset);
        vec.push(self.macro_hurst_confidence_scale);
        vec.push(self.macro_vol_confidence_scale);
        vec.push(self.macro_min_cooldown_ratio);
        vec.push(self.macro_max_cooldown_ratio);
        vec.push(self.macro_cooldown_reduction_factor);
        vec.push(self.macro_leverage_momentum_scale);
        vec.push(self.lev_matrix_vol_clamp_min);
        vec.push(self.lev_matrix_growth_scalar);
        vec.push(self.lev_matrix_log_cap_divisor);
        vec.push(self.scalp_accel_min_samples);
        vec.push(self.executor_max_orders_10s);
        vec.push(self.executor_max_weight_1m);
        vec.push(self.iceberg_volume_threshold);
        vec.push(self.iceberg_slice_count);
        vec.push(self.swing_obi_threshold);
        vec.push(self.swing_accel_min_samples);
        vec.push(self.temporal_scale);
        // X-004: coeficientes de curvas como genes de pleno derecho.
        vec.push(self.tp_horizon_curve.a);
        vec.push(self.tp_horizon_curve.b);
        vec.push(self.sl_horizon_curve.a);
        vec.push(self.sl_horizon_curve.b);
        vec
    }

    /// Convierte el genoma a un vector normalizado en el hipercubo canónico unitario [0.0, 1.0]^DIMENSION.
    /// D-405: Erradica el desfase de 9 órdenes de magnitud en optimizadores gaussianos / CMA-ES.
    pub fn to_normalized_vector(&self) -> Vec<f64> {
        let raw = self.to_vector();
        let lo = Self::get_lower_bounds();
        let hi = Self::get_upper_bounds();
        let mut norm = Vec::with_capacity(raw.len());
        for i in 0..raw.len() {
            let span = hi[i] - lo[i];
            let val = if span > 1e-12 {
                ((raw[i] - lo[i]) / span).clamp(0.0, 1.0)
            } else {
                0.5
            };
            norm.push(val);
        }
        norm
    }

    /// Reconstruye el genoma a partir de un vector unitario normalizado [0.0, 1.0]^DIMENSION.
    /// D-405: Aplica mapeo afín exacto x = lo + u * (hi - lo) previo a la reconstrucción física.
    pub fn from_normalized_vector(u: &[f64]) -> Self {
        let lo = Self::get_lower_bounds();
        let hi = Self::get_upper_bounds();
        let mut raw = Vec::with_capacity(u.len());
        for i in 0..u.len() {
            let span = hi[i] - lo[i];
            let u_clamped = if u[i].is_finite() {
                u[i].clamp(0.0, 1.0)
            } else {
                0.5
            };
            raw.push(lo[i] + u_clamped * span);
        }
        Self::from_vector(&raw)
    }

    pub fn from_vector(vec: &[f64]) -> Self {
        // R1.1 — FUENTE ÚNICA DE VERDAD: los clamps se leen de los arrays de
        // bounds (las mismas cotas que GenomeStore::validate usa como gate y
        // que la evolución CMA-ES usa como espacio de búsqueda). Antes los
        // literales inline eran DISJUNTOS de los bounds en 4 genes de TP/SL,
        // con lo que todo genoma reconstruido por vector violaba el gate de
        // promoción y la evolución versionada quedaba bloqueada.
        let lo = Self::get_lower_bounds();
        let hi = Self::get_upper_bounds();
        let rebuilt = Self {
            global_max_drawdown: vec[0].clamp(lo[0], hi[0]),
            global_leverage: vec[1].clamp(lo[1], hi[1]),
            btc_volatility_multiplier: vec[2].clamp(lo[2], hi[2]),
            eth_volatility_multiplier: vec[3].clamp(lo[3], hi[3]),
            min_trades_per_day: vec[4].clamp(lo[4], hi[4]),
            survival_capital_threshold: vec[5].clamp(lo[5], hi[5]),
            funding_rate_sensitivity: vec[6].clamp(lo[6], hi[6]),
            global_correlation_threshold: vec[7].clamp(lo[7], hi[7]),
            trend_threshold: vec[8].clamp(lo[8], hi[8]),
            range_threshold: vec[9].clamp(lo[9], hi[9]),
            scalp_kelly_fraction: vec[10].clamp(lo[10], hi[10]),
            swing_kelly_fraction: vec[11].clamp(lo[11], hi[11]),
            scalp_obi_threshold: vec[12].clamp(lo[12], hi[12]),
            scalp_tp_base: vec[13].clamp(lo[13], hi[13]),
            scalp_sl_base: vec[14].clamp(lo[14], hi[14]),
            swing_tp_base: vec[15].clamp(lo[15], hi[15]),
            swing_sl_base: vec[16].clamp(lo[16], hi[16]),
            sl_atr_mult_btc: vec[17].clamp(lo[17], hi[17]),
            tp_rr_ratio_btc: vec[18].clamp(lo[18], hi[18]),
            min_confidence_btc: vec[19].clamp(lo[19], hi[19]),
            veto_threshold_btc: vec[20].clamp(lo[20], hi[20]),
            tech_threshold: vec[21].clamp(lo[21], hi[21]),
            ml_threshold_long: vec[22].clamp(lo[22], hi[22]),
            ml_threshold_short: vec[23].clamp(lo[23], hi[23]),
            maker_spread_pct: vec[24].clamp(lo[24], hi[24]),
            maker_obi_threshold: vec[25].clamp(lo[25], hi[25]),
            target_volatility: vec[26].clamp(lo[26], hi[26]),
            dynamic_atr_min: vec[27].clamp(lo[27], hi[27]),
            dynamic_obi_threshold: vec[28].clamp(lo[28], hi[28]),
            dynamic_ema_trend: vec[29].clamp(lo[29], hi[29]),
            dynamic_ofi_threshold: vec[30].clamp(lo[30], hi[30]),
            capital_split_scalp: vec[31].clamp(lo[31], hi[31]),
            kelly_clamp_min: vec[32].clamp(lo[32], hi[32]),
            kelly_clamp_max: vec[33].clamp(lo[33], hi[33]),
            explosive_leverage_multiplier: vec[34].clamp(lo[34], hi[34]),
            quantum_mutation_rate: vec[35].clamp(lo[35], hi[35]),
            temporal_memory_decay: vec[36].clamp(lo[36], hi[36]),
            leverage_cap: vec[37].clamp(lo[37], hi[37]),
            explosive_confidence_threshold: vec[38].clamp(lo[38], hi[38]),
            weight_obi: vec[39].clamp(lo[39], hi[39]),
            weight_ofi: vec[40].clamp(lo[40], hi[40]),
            weight_vpin: vec[41].clamp(lo[41], hi[41]),
            regime_duration_ms: vec[42].clamp(lo[42], hi[42]),
            regime_atr_multiplier: vec[43].clamp(lo[43], hi[43]),
            scalp_trail_act_atr: vec[44].clamp(lo[44], hi[44]),
            scalp_trail_step_atr: vec[45].clamp(lo[45], hi[45]),
            scalp_trail_max_atr: vec[46].clamp(lo[46], hi[46]),
            scalp_trail_min_pnl: vec[47].clamp(lo[47], hi[47]),
            scalp_trail_atr_mult_base: vec[48].clamp(lo[48], hi[48]),
            swing_trail_act_atr: vec[49].clamp(lo[49], hi[49]),
            swing_trail_step_atr: vec[50].clamp(lo[50], hi[50]),
            swing_trail_max_atr: vec[51].clamp(lo[51], hi[51]),
            swing_trail_min_pnl: vec[52].clamp(lo[52], hi[52]),
            swing_trail_atr_mult_base: vec[53].clamp(lo[53], hi[53]),
            zombie_timeout_ms: vec[54].clamp(lo[54], hi[54]),
            hurst_trend_threshold: vec[55].clamp(lo[55], hi[55]),
            cvd_veto_threshold: vec[56].clamp(lo[56], hi[56]),
            wall_veto_threshold: vec[57].clamp(lo[57], hi[57]),
            flash_crash_jump_pct: vec[58].clamp(lo[58], hi[58]),
            latency_ms_panic_threshold: vec[59].clamp(lo[59], hi[59]),
            ema_fast_period: vec[60].clamp(lo[60], hi[60]),
            ema_slow_period: vec[61].clamp(lo[61], hi[61]),
            hurst_scalp_threshold: vec[62].clamp(lo[62], hi[62]),
            hurst_swing_threshold: vec[63].clamp(lo[63], hi[63]),
            synergy_exposure_boost: vec[64].clamp(lo[64], hi[64]),
            synergy_leverage_boost: vec[65].clamp(lo[65], hi[65]),
            max_fee_pct: vec[66].clamp(lo[66], hi[66]),
            kelly_bootstrap_cold: vec[67].clamp(lo[67], hi[67]),
            latency_penalty_ms: vec[68].clamp(lo[68], hi[68]),
            base_slippage_floor: vec[69].clamp(lo[69], hi[69]),
            spot_spread_threshold: vec[70].clamp(lo[70], hi[70]),
            spot_bias_value: vec[71].clamp(lo[71], hi[71]),
            obi_confidence_fallback: vec[72].clamp(lo[72], hi[72]),
            ml_clip_lower: vec[73].clamp(lo[73], hi[73]),
            ml_clip_upper: vec[74].clamp(lo[74], hi[74]),
            tensor_poly_a: vec[75].clamp(lo[75], hi[75]),
            tensor_poly_b: vec[76].clamp(lo[76], hi[76]),
            fractional_alpha_order: vec[77].clamp(lo[77], hi[77]),
            fractional_clip_max: vec[78].clamp(lo[78], hi[78]),
            bft_consensus_tolerance: vec[79].clamp(lo[79], hi[79]),
            turbo_coherence_threshold: vec[80].clamp(lo[80], hi[80]),
            turbo_z_score_stdev: vec[81].clamp(lo[81], hi[81]),
            sl_atr_multiplier: vec[82].clamp(lo[82], hi[82]),
            coaxial_squeeze_threshold: vec[83].clamp(lo[83], hi[83]),
            tensor_op_add_bias: vec[84].clamp(lo[84], hi[84]),
            tensor_op_mul_weight: vec[85].clamp(lo[85], hi[85]),
            topo_layer_1_activation: vec[86].clamp(lo[86], hi[86]),
            topo_layer_2_activation: vec[87].clamp(lo[87], hi[87]),
            tensor_dropout_rate: vec[88].clamp(lo[88], hi[88]),
            quantum_entropy_seed: vec[89].clamp(lo[89], hi[89]),
            ppo_weight_ofi: vec[90].clamp(lo[90], hi[90]),
            ppo_weight_obi: vec[91].clamp(lo[91], hi[91]),
            ppo_weight_hawkes: vec[92].clamp(lo[92], hi[92]),
            ppo_weight_leadlag: vec[93].clamp(lo[93], hi[93]),
            ppo_weight_regime: vec[94].clamp(lo[94], hi[94]),
            global_learning_rate: vec[95].clamp(lo[95], hi[95]),
            global_momentum: vec[96].clamp(lo[96], hi[96]),
            vecm_alpha_speed: vec[97].clamp(lo[97], hi[97]),
            vecm_beta_hedge: vec[98].clamp(lo[98], hi[98]),
            conformal_alpha: vec[99].clamp(lo[99], hi[99]),
            ppo_clip_eps: vec[100].clamp(lo[100], hi[100]),
            ppo_weight_min_clip: vec[101].clamp(lo[101], hi[101]),
            hard_stop_decay_factor: vec[102].clamp(lo[102], hi[102]),
            hard_stop_base_limit: vec[103].clamp(lo[103], hi[103]),
            kelly_bootstrap_ratio_threshold: vec[104].clamp(lo[104], hi[104]),
            kelly_bootstrap_min_exposure: vec[105].clamp(lo[105], hi[105]),
            ev_fee_multiplier: vec[106].clamp(lo[106], hi[106]),
            margin_cushion_pct: vec[107].clamp(lo[107], hi[107]),
            maker_only_capital_threshold: vec[108].clamp(lo[108], hi[108]),
            hawkes_scalp_threshold: vec[109].clamp(lo[109], hi[109]),
            obi_zscore_threshold: vec[110].clamp(lo[110], hi[110]),
            hawkes_volume_norm: vec[111].clamp(lo[111], hi[111]),
            base_duration_ms: vec[112].clamp(lo[112], hi[112]),
            kelly_survival_cap_ratio: vec[113].clamp(lo[113], hi[113]),
            kelly_expansion_mult: vec[114].clamp(lo[114], hi[114]),
            guard_dd_sigmoid_steepness: vec[115].clamp(lo[115], hi[115]),
            guard_dd_sigmoid_center: vec[116].clamp(lo[116], hi[116]),
            portfolio_perf_mult_steepness: vec[117].clamp(lo[117], hi[117]),
            portfolio_dd_penalty_decay: vec[118].clamp(lo[118], hi[118]),
            portfolio_perf_mult_min: vec[119].clamp(lo[119], hi[119]),
            portfolio_perf_mult_max: vec[120].clamp(lo[120], hi[120]),
            portfolio_perf_mult_center: vec[121].clamp(lo[121], hi[121]),
            macro_hurst_confidence_offset: vec[122].clamp(lo[122], hi[122]),
            macro_hurst_confidence_scale: vec[123].clamp(lo[123], hi[123]),
            macro_vol_confidence_scale: vec[124].clamp(lo[124], hi[124]),
            macro_min_cooldown_ratio: vec[125].clamp(lo[125], hi[125]),
            macro_max_cooldown_ratio: vec[126].clamp(lo[126], hi[126]),
            macro_cooldown_reduction_factor: vec[127].clamp(lo[127], hi[127]),
            macro_leverage_momentum_scale: vec[128].clamp(lo[128], hi[128]),
            lev_matrix_vol_clamp_min: vec[129].clamp(lo[129], hi[129]),
            lev_matrix_growth_scalar: vec[130].clamp(lo[130], hi[130]),
            lev_matrix_log_cap_divisor: vec[131].clamp(lo[131], hi[131]),
            scalp_accel_min_samples: vec[132].clamp(lo[132], hi[132]),
            executor_max_orders_10s: vec[133].clamp(lo[133], hi[133]),
            executor_max_weight_1m: vec[134].clamp(lo[134], hi[134]),
            iceberg_volume_threshold: vec[135].clamp(lo[135], hi[135]),
            iceberg_slice_count: vec[136].clamp(lo[136], hi[136]),
            swing_obi_threshold: vec[137].clamp(lo[137], hi[137]),
            swing_accel_min_samples: vec[138].clamp(lo[138], hi[138]),
            temporal_scale: vec[139].clamp(lo[139], hi[139]),
            // X-004 (REHAB-1): las CURVAS son genes del vector (slots 140-153).
            // Son la FUENTE ÚNICA: las anclas del literal se sobrescriben con
            // vistas derivadas de las curvas abajo (derive_anchors), y el
            // reparo RR opera SOBRE CURVAS — nunca más doble verdad.
            tp_horizon_curve: crate::temporal_spectrum::HorizonCurve {
                a: vec[140].clamp(lo[140], hi[140]),
                b: vec[141].clamp(lo[141], hi[141]),
            },
            sl_horizon_curve: crate::temporal_spectrum::HorizonCurve {
                a: vec[142].clamp(lo[142], hi[142]),
                b: vec[143].clamp(lo[143], hi[143]),
            },
            kelly_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            trail_mult_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            trail_act_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            trail_step_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
            obi_horizon_curve: crate::temporal_spectrum::HorizonCurve::flat(1.0),
        };

        let mut g = rebuilt;
        // N-02 evolucionado a nivel CURVA: si el clamping independiente de
        // (a_tp,b_tp) y (a_sl,b_sl) produce TP(τ) ≤ SL(τ)·MIN_RR en alguna
        // ancla, se deprime el INTERCEPTO de la curva SL (escala uniforme de
        // SL(τ) en todo el espectro: a_sl -= ln(factor)) hasta restablecer el
        // invariante — determinista y sin tocar la pendiente evolucionada.
        g.enforce_curve_rr();
        g.derive_anchors_from_curves();
        g.sync_continuous_curves();
        g
    }

    /// D-649b (DÉCIMA OLA): la escala de caducidad de posiciones.
    ///
    /// Al conectar el gen muerto `zombie_timeout_ms` (D-649) entró en vivo un
    /// valor que la evolución jamás seleccionó —35 min en el genoma de
    /// producción—, que acortaba el debounce de 4–8 h a ~52 min y el
    /// hard-timeout de 12–24 h a ~2,6 h. D-649 además bajó el umbral de pérdida
    /// de la rama de inversión de tendencia de −0,60 % a −0,50 %. En el
    /// backtest forense: 12 de 31 salidas ZOMBIE con PnL ≈ −0,52 %.
    ///
    /// La banda pasa a [4 h, 8 h] y el umbral vuelve a −0,60 %: con el gen en
    /// su suelo, `g·(1+s)` y `3g·(1+s)` coinciden con las fórmulas anteriores
    /// `4h·(1+s)` y `12h·(1+s)`. Se conserva lo que D-649 añadió con razón —la
    /// caducidad absoluta `12g·(1+s)` de posiciones huérfanas— y el gen queda
    /// evolucionable dentro de una banda con sentido.
    pub const SLOT_ZOMBIE_TIMEOUT: usize = 54;
    /// D-643: umbral de confianza, acotado en la entrada en lugar de en la lectura.
    pub const SLOT_MIN_CONFIDENCE: usize = 19;
    /// D-687 (DÉCIMA OLA): fracción del capital comprometible como margen. Sus
    /// bounds eran [1,01; 1,20] mientras el único consumidor lo acotaba a
    /// [0,50; 0,98]: el gen valía siempre 0,98 y la evolución nunca pudo
    /// moverlo. La banda pasa a la del consumidor; los genomas existentes
    /// (> 0,98) entran en 0,98, su valor efectivo de siempre.
    pub const SLOT_MARGIN_CUSHION: usize = 107;

    /// Acota un gen a sus bounds evolutivos.
    ///
    /// D-625/D-643/D-649b: la regla es que un gen se acota donde se muta y
    /// donde entra al sistema —`apply_to_arena` y `QuantumConfig::from_genome`—,
    /// no en cada sitio de lectura. Así el valor en vivo y el que la evolución
    /// explora coinciden, y la aptitud mide lo que realmente se ejecuta.
    pub fn clamp_slot(value: f64, slot: usize) -> f64 {
        let lo = Self::get_lower_bounds();
        let hi = Self::get_upper_bounds();
        if slot >= lo.len() || !value.is_finite() {
            return value;
        }
        value.clamp(lo[slot], hi[slot])
    }

    /// D-625 (DÉCIMA OLA): slots del vector de genes cuyo acotado se aplica al
    /// ENTRAR al arena. Verificados contra `to_vector` por test.
    pub const SLOT_TREND_THRESHOLD: usize = 8;
    pub const SLOT_TECH_THRESHOLD: usize = 21;

    pub fn get_lower_bounds() -> Vec<f64> {
        vec![
            0.5,
            1.0, // D-644: antes 25.0 — la evolución no podía bajar de 25×
            0.5,
            0.5,
            1.0,
            0.5,
            0.1,
            0.2,
            0.52,
            0.1,
            0.1,
            0.01,
            0.01,
            // X-004: anclas tp/sl (13-16) son VISTAS de las curvas — sus bounds
            // cubren el sobre alcanzable por (a,b) evolucionables, no cotas
            // evolutivas por sí mismas (las cotas reales viven en 140-143).
            0.0000005,
            0.0000005,
            0.0000005,
            0.0000005,
            0.5,
            1.0,
            0.5,
            0.5,
            0.24, // D-625: antes 0.05 — el motor imponía .max(0.24) al leer
            0.5,
            0.05,
            0.0001,
            0.1,
            0.005,
            0.0000001,
            0.05,
            0.0000001,
            0.05,
            0.1,
            0.001,
            0.1,
            1.0,
            0.01,
            0.8,
            10.0,
            0.7,
            0.1,
            0.1,
            0.1,
            30000.0,
            0.5,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            14_400_000.0, // D-649b: 4 h — reproduce el debounce previo
            0.4,
            0.1,
            5.0,
            0.05,
            500.0,
            5.0,
            15.0,
            0.30,
            0.50,
            1.0,
            1.0,
            0.005,
            0.1,
            5.0,
            0.00005,
            0.5,
            0.05,
            0.70,
            0.01,
            0.80,
            0.01,
            0.001,
            0.1,
            5.0,
            0.05,
            0.1,
            0.1,
            0.2,
            0.1,
            0.01,
            0.5,
            0.1,
            0.1,
            0.01,
            0.0,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.0001,
            0.5,
            0.001,
            0.001,
            0.01,
            0.05,
            0.01,
            0.1,
            0.10,
            0.05,
            0.01,
            1.0,
            0.50, // D-687: fracción del capital comprometible
            10.0,
            0.50, // #535: hawkes_scalp_threshold (slot 109) — piso = ritmo normal del símbolo
            0.1,
            10.0,
            10000.0,
            1.0,
            1.0,
            1.0,
            1.0,
            3.0,
            5.0,
            0.1,
            1.1,
            0.4,
            0.35,
            1.0,
            10.0,
            0.05,
            10.0,
            0.1,
            0.1,
            0.1,
            0.1,
            3.0,
            10.0,
            100.0,
            1000.0,
            100.0,
            2.0,
            0.05,
            10.0,
            0.05,
            // X-004: bounds de coeficientes de curvas (tp_a, tp_b, sl_a, sl_b)
            // — mismas bandas que mutate_curve (una sola definición a mantener).
            Self::TP_A_BOUNDS.0,
            Self::TP_B_BOUNDS.0,
            Self::SL_A_BOUNDS.0,
            Self::SL_B_BOUNDS.0,
        ]
    }

    pub fn get_upper_bounds() -> Vec<f64> {
        vec![
            0.99,
            50.0, // D-644: antes 35.0 — alineado con el clamp del motor
            3.0,
            3.0,
            50.0,
            0.95,
            2.0,
            0.9,
            0.9,
            0.9,
            2.0,
            1.0,
            1.0,
            // X-004: techos de VISTAS — sobre máximo alcanzable por las curvas
            // en el ancla LENTA (exp(a_hi + b_hi·ln τ_slow): tp≈63.7, sl≈23.3),
            // con holgura. Las cotas evolutivas reales viven en 140-143.
            70.0,
            30.0,
            70.0,
            30.0,
            5.0,
            10.0,
            0.95,
            0.99,
            0.30,
            0.95,
            0.49,
            0.01,
            0.95,
            0.1,
            0.01,
            0.95,
            0.01,
            0.95,
            1.0,
            0.1,
            1.0,
            10.0,
            0.5,
            0.9999,
            125.0,
            0.9999,
            1.0,
            1.0,
            1.0,
            600000.0,
            5.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            28_800_000.0, // D-649b: 8 h
            0.8,
            2.0,
            50.0,
            0.50,
            10000.0,
            30.0,
            60.0,
            0.50,
            0.70,
            3.0,
            2.0,
            0.05,
            1.0,
            100.0,
            0.0005,
            3.0,
            0.30,
            0.95,
            0.20,
            0.99,
            0.20,
            0.05,
            0.9,
            20.0,
            0.30,
            0.6,
            5.0,
            2.0,
            0.8,
            0.5,
            3.0,
            0.9,
            0.9,
            0.5,
            1000.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.05,
            0.99,
            0.1,
            0.1,
            0.2,
            0.40,
            0.10,
            1.0,
            0.80,
            0.30,
            0.5,
            5.0,
            0.98, // D-687: MAX_MARGIN_UTILIZATION
            5000.0,
            0.95, // #535: hawkes_scalp_threshold (slot 109) — techo = +90% de excitación
            3.0,
            10000000.0,
            120000.0,
            3.0,
            3.0,
            5.0,
            5.0,
            15.0,
            25.0,
            0.5,
            3.0,
            0.6,
            0.55,
            20.0,
            500.0,
            0.5,
            500.0,
            2.0,
            2.0,
            0.8,
            1.0,
            10.0,
            50.0,
            500.0,
            5000.0,
            100_000.0,
            10.0,
            1.0,
            50.0,
            0.95,
            // X-004: techo de coeficientes de curvas.
            Self::TP_A_BOUNDS.1,
            Self::TP_B_BOUNDS.1,
            Self::SL_A_BOUNDS.1,
            Self::SL_B_BOUNDS.1,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B3.2: NINGUNA geometría que pase por `friction_floors` puede tener EV
    /// negativo al win rate de diseño — el invariante que cierra el arrastre
    /// de comisiones (−108,7 % del bruto medido en testnet). Cubre fricción
    /// VIP0 (0,0012), VIP-alto y curvas de genoma degeneradas (cero).
    #[test]
    fn b3_2_friction_floors_garantizan_ev_no_negativo() {
        for fee in [0.0008, 0.0010, 0.0012, 0.0020, 0.0050] {
            for sl in [0.0, 0.0005, 0.0010, 0.0030, 0.0100] {
                for tp in [0.0, 0.0005, 0.0010, 0.0020, 0.0100] {
                    let (sl_f, tp_f) = SuperGenotype::friction_floors(fee, sl, tp);
                    assert!(sl_f > 0.0 && tp_f > 0.0);
                    let w = SuperGenotype::WORST_TOLERATED_WR;
                    let ev = w * (tp_f - fee) - (1.0 - w) * (sl_f + fee);
                    assert!(
                        ev >= -1e-12,
                        "EV negativo con fee={fee} sl={sl} tp={tp} → ({sl_f},{tp_f}): ev={ev}"
                    );
                    // El TP resultante siempre paga la fricción con margen.
                    assert!(tp_f > fee, "tp={tp_f} no cubre fee={fee}");
                    // B3.24: RR geométrico ≥ 2 SIEMPRE (cap de asimetría).
                    assert!(
                        sl_f <= tp_f * 0.5 + 1e-15,
                        "SL {sl_f} > TP/2 {tp_f} — cap de asimetría violado (fee={fee}, in sl={sl} tp={tp})"
                    );
                }
            }
        }
    }

    /// B3.2 (auditoría) — PROPIEDAD EN TODA LA FRONTERA CONTINUA, no sólo en
    /// la rejilla: barre la fricción en (0, 0,01] con muestreo log-uniforme y
    /// evalúa el EV en los puntos críticos de cada nivel — el SL exactamente
    /// en el piso (`f/0,50`, el peor caso por construcción) y por encima
    /// (donde el término `f/(w·sl)` decae y el EV sólo puede mejorar) — con
    /// TP adversarial (cero) y TP en su propio piso.
    ///
    /// La cota analítica: en el piso doble (sl = f/0,5 y tp = sl·RR_min) el
    /// EV es EXACTAMENTE 0 — `0,4·(5,5f − f) = 0,6·(2f + f) = 1,8f` — así que
    /// cualquier tolerancia debe ser ε puro de coma flotante, no holgura.
    #[test]
    fn b3_2_ev_no_negativo_en_toda_la_frontera_continua() {
        let w = SuperGenotype::WORST_TOLERATED_WR;
        for k in 0..60 {
            // fee ∈ [1e-5, 0.01] log-espaciado (incluye el 0,0012 VIP0 real).
            let fee = 1e-5 * (1000.0f64).powf(k as f64 / 59.0);
            let floor = SuperGenotype::min_viable_sl(fee);
            assert!((floor - fee / SuperGenotype::MAX_FRICTION_SHARE_OF_RISK).abs() < 1e-15, "piso mal derivado");

            for &sl_mult in &[0.0, 1.0, 1.37, 2.0, 10.0] {
                let sl_in = floor * sl_mult;
                for &tp_in in &[0.0, fee, floor * 0.01, floor * 3.0] {
                    let (sl_f, tp_f) = SuperGenotype::friction_floors(fee, sl_in, tp_in);
                    assert!(sl_f >= floor - 1e-15, "SL bajo el piso");
                    let ev = w * (tp_f - fee) - (1.0 - w) * (sl_f + fee);
                    assert!(
                        ev >= -1e-12,
                        "EV={ev} con fee={fee} sl_in={sl_in} tp_in={tp_in} → ({sl_f},{tp_f})"
                    );
                    // Monotonía: dar MÁS geometría de entrada nunca empeora
                    // el resultado (los pisos son max(), no mezclas).
                    let (sl_g, tp_g) = SuperGenotype::friction_floors(fee, sl_in * 1.5, tp_in * 1.5);
                    assert!(sl_g >= sl_f - 1e-15 && tp_g >= tp_f - 1e-15);
                }
            }
        }
        // Caso degenerado documentado: fee inválido → fricción de referencia.
        let (sl_f, tp_f) = SuperGenotype::friction_floors(0.0, 0.0, 0.0);
        assert!((sl_f - SuperGenotype::REFERENCE_ROUNDTRIP_FEE / SuperGenotype::MAX_FRICTION_SHARE_OF_RISK).abs() < 1e-15);
        assert!(tp_f > sl_f);
    }

    /// B3.2/D-645 — el fee que pasan los llamadores es 2×taker + 2×piso de
    /// slippage; con taker 5 bps y el piso derivado del baseline (taker/5 =
    /// 1 bp) la fricción efectiva es 0,0012 y el piso de SL resultante
    /// (24 bps) queda por ENCIMA del de referencia (20 bps): el fallback
    /// `REFERENCE_ROUNDTRIP_FEE` nunca es más permisivo que la fricción real.
    #[test]
    fn b3_2_friccion_de_llamadores_domina_al_fallback_de_referencia() {
        let slip_floor = 0.0005_f64 / 5.0; // derivación del baseline: taker/5 = 1 bp
        let live_fee = 2.0 * 0.0005 + 2.0 * slip_floor;
        assert!((live_fee - 0.0012).abs() < 1e-9);
        assert!(SuperGenotype::min_viable_sl(live_fee) > SuperGenotype::min_viable_sl(0.0));
        // El RR exigido al SL vivo también es mayor o igual que al de
        // referencia en todo el rango útil de stops (el fee del genoma activo
        // en producción, 2×0,0005 + 2×5,68e-05 ≈ 0,00111, domina igualmente).
        let prod_fee = 2.0 * 0.0005 + 2.0 * 5.683_610_778_189_746e-05;
        for fee in [live_fee, prod_fee] {
            for sl in [0.0024, 0.005, 0.01, 0.05] {
                assert!(
                    SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, fee, sl)
                        >= SuperGenotype::min_rr_for(
                            SuperGenotype::WORST_TOLERATED_WR,
                            SuperGenotype::REFERENCE_ROUNDTRIP_FEE,
                            sl
                        ) - 1e-12
                );
            }
        }
    }

    #[test]
    fn test_genome_vector_symmetry_exact_139d() {
        let genome = SuperGenotype::load_or_baseline(0.0002, 0.0005);
        let vec = genome.to_vector();
        assert_eq!(
            vec.len(),
            SuperGenotype::DIMENSION,
            "to_vector length must be exactly 139"
        );
        assert_eq!(
            SuperGenotype::get_lower_bounds().len(),
            SuperGenotype::DIMENSION
        );
        assert_eq!(
            SuperGenotype::get_upper_bounds().len(),
            SuperGenotype::DIMENSION
        );
        let reconstructed = SuperGenotype::from_vector(&vec);
        let vec2 = reconstructed.to_vector();
        assert_eq!(vec.len(), vec2.len());
    }

    /// D-658: la mutación lee las cotas de las curvas de `*_BOUNDS`. Con tasa
    /// cero, interceptos en la franja que las bandas literales no alcanzaban
    /// deben conservarse (antes se empujaban a −9,0 y −10,0).
    ///
    /// Genoma factible por construcción: curvas paralelas (b = 0,35) con
    /// RR = e^1,3 ≈ 3,67 en todo el espectro. En el extremo inferior de la banda
    /// operable (SL = f/0,5 = 0,002) el RR exigido es 1,1·(1,5 + 1,25) ≈ 3,03,
    /// así que el reparo RR no toca las curvas.
    #[test]
    fn d658_mutacion_de_curvas_respeta_las_cotas_declaradas() {
        let mut g = SuperGenotype::load_or_baseline(0.0002, 0.0005);
        g.tp_horizon_curve = crate::temporal_spectrum::HorizonCurve { a: -9.2, b: 0.35 };
        g.sl_horizon_curve = crate::temporal_spectrum::HorizonCurve { a: -10.5, b: 0.35 };
        g.enforce_curve_rr();
        assert_eq!(
            g.tp_horizon_curve.a, -9.2,
            "precondición: el reparo RR no debe mover TP"
        );
        assert_eq!(
            g.sl_horizon_curve.a, -10.5,
            "precondición: el reparo RR no debe mover SL"
        );
        let m = g.mutate_cmaes_seeded(0.0, 7);
        assert_eq!(m.tp_horizon_curve.a, g.tp_horizon_curve.a);
        assert_eq!(m.sl_horizon_curve.a, g.sl_horizon_curve.a);
        assert_eq!(m.tp_horizon_curve.b, g.tp_horizon_curve.b);
        assert_eq!(m.sl_horizon_curve.b, g.sl_horizon_curve.b);
    }

    /// D-656: los slots 13–16 (anclas TP/SL) son vistas de las curvas. Un
    /// optimizador vectorial que los perturbe no cambia el genoma: debe
    /// excluirlos de su espacio de búsqueda.
    #[test]
    fn d656_anclas_tp_sl_son_vistas_en_el_vector() {
        let g = SuperGenotype::load_or_baseline(0.0002, 0.0005);
        let mut v = g.to_vector();
        let base = SuperGenotype::from_vector(&v).to_vector();
        let lo = SuperGenotype::get_lower_bounds();
        let hi = SuperGenotype::get_upper_bounds();
        for slot in 13..=16 {
            v[slot] = if v[slot] > 0.5 * (lo[slot] + hi[slot]) {
                lo[slot]
            } else {
                hi[slot]
            };
        }
        let altered = SuperGenotype::from_vector(&v).to_vector();
        assert_eq!(base, altered);
    }
}
