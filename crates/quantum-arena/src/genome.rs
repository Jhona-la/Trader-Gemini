use serde::{Serialize, Deserialize};
use crate::GlobalArena;
use std::sync::atomic::Ordering;
use rand::Rng;

/// Axioma X: SuperGenotype (El ADN de Trader Gemini)
/// Integra todos los 50+ parámetros del God Engine en un solo vector.
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
            btc_volatility_multiplier: arena.config.btc_volatility_multiplier.load(Ordering::Relaxed),
            eth_volatility_multiplier: arena.config.eth_volatility_multiplier.load(Ordering::Relaxed),
            min_trades_per_day: arena.config.min_trades_per_day.load(Ordering::Relaxed),
            survival_capital_threshold: arena.config.survival_capital_threshold.load(Ordering::Relaxed),
            funding_rate_sensitivity: arena.config.funding_rate_sensitivity.load(Ordering::Relaxed),
            global_correlation_threshold: arena.config.global_correlation_threshold.load(Ordering::Relaxed),
            trend_threshold: arena.config.trend_threshold.load(Ordering::Relaxed),
            range_threshold: arena.config.range_threshold.load(Ordering::Relaxed),
            scalp_kelly_fraction: arena.config.scalp_kelly_fraction.load(Ordering::Relaxed),
            swing_kelly_fraction: arena.config.swing_kelly_fraction.load(Ordering::Relaxed),
            scalp_obi_threshold: arena.config.scalp_obi_threshold.load(Ordering::Relaxed),
            scalp_tp_base: arena.config.scalp_tp_base.load(Ordering::Relaxed),
            scalp_sl_base: arena.config.scalp_sl_base.load(Ordering::Relaxed),
            swing_tp_base: arena.config.swing_tp_base.load(Ordering::Relaxed),
            swing_sl_base: arena.config.swing_sl_base.load(Ordering::Relaxed),
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
            explosive_leverage_multiplier: arena.config.explosive_leverage_multiplier.load(Ordering::Relaxed),
            quantum_mutation_rate: arena.config.quantum_mutation_rate.load(Ordering::Relaxed),
            temporal_memory_decay: arena.config.temporal_memory_decay.load(Ordering::Relaxed),
            leverage_cap: arena.config.leverage_cap.load(Ordering::Relaxed),
            explosive_confidence_threshold: arena.config.explosive_confidence_threshold.load(Ordering::Relaxed),
            weight_obi: arena.config.weight_obi.load(Ordering::Relaxed),
            weight_ofi: arena.config.weight_ofi.load(Ordering::Relaxed),
            weight_vpin: arena.config.weight_vpin.load(Ordering::Relaxed),
            regime_duration_ms: arena.config.regime_duration_ms.load(Ordering::Relaxed),
            regime_atr_multiplier: arena.config.regime_atr_multiplier.load(Ordering::Relaxed),
            scalp_trail_act_atr: arena.config.scalp_trail_act_atr.load(Ordering::Relaxed),
            scalp_trail_step_atr: arena.config.scalp_trail_step_atr.load(Ordering::Relaxed),
            scalp_trail_max_atr: arena.config.scalp_trail_max_atr.load(Ordering::Relaxed),
            scalp_trail_min_pnl: arena.config.scalp_trail_min_pnl.load(Ordering::Relaxed),
            scalp_trail_atr_mult_base: arena.config.scalp_trail_atr_mult_base.load(Ordering::Relaxed),
            swing_trail_act_atr: arena.config.swing_trail_act_atr.load(Ordering::Relaxed),
            swing_trail_step_atr: arena.config.swing_trail_step_atr.load(Ordering::Relaxed),
            swing_trail_max_atr: arena.config.swing_trail_max_atr.load(Ordering::Relaxed),
            swing_trail_min_pnl: arena.config.swing_trail_min_pnl.load(Ordering::Relaxed),
            swing_trail_atr_mult_base: arena.config.swing_trail_atr_mult_base.load(Ordering::Relaxed),
            zombie_timeout_ms: arena.config.zombie_timeout_ms.load(Ordering::Relaxed),
            hurst_trend_threshold: arena.config.hurst_trend_threshold.load(Ordering::Relaxed),
            cvd_veto_threshold: arena.config.cvd_veto_threshold.load(Ordering::Relaxed),
            wall_veto_threshold: arena.config.wall_veto_threshold.load(Ordering::Relaxed),
            flash_crash_jump_pct: arena.config.flash_crash_jump_pct.load(Ordering::Relaxed),
            latency_ms_panic_threshold: arena.config.latency_ms_panic_threshold.load(Ordering::Relaxed),
            ema_fast_period: arena.config.ema_fast_period.load(Ordering::Relaxed),
            ema_slow_period: arena.config.ema_slow_period.load(Ordering::Relaxed),
            hurst_scalp_threshold: arena.config.hurst_scalp_threshold.load(Ordering::Relaxed),
            hurst_swing_threshold: arena.config.hurst_swing_threshold.load(Ordering::Relaxed),
            synergy_exposure_boost: arena.config.synergy_exposure_boost.load(Ordering::Relaxed),
            synergy_leverage_boost: arena.config.synergy_leverage_boost.load(Ordering::Relaxed),
            max_fee_pct: arena.config.max_fee_pct.load(Ordering::Relaxed),
            kelly_bootstrap_cold: arena.config.kelly_bootstrap_cold.load(Ordering::Relaxed),
            latency_penalty_ms: arena.config.latency_penalty_ms_physics.load(Ordering::Relaxed),
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
            turbo_coherence_threshold: arena.config.turbo_coherence_threshold.load(Ordering::Relaxed),
            turbo_z_score_stdev: arena.config.turbo_z_score_stdev.load(Ordering::Relaxed),
            sl_atr_multiplier: arena.config.sl_atr_multiplier.load(Ordering::Relaxed),
            coaxial_squeeze_threshold: arena.config.coaxial_squeeze_threshold.load(Ordering::Relaxed),
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
            kelly_bootstrap_ratio_threshold: arena.config.kelly_bootstrap_ratio_threshold.load(Ordering::Relaxed),
            kelly_bootstrap_min_exposure: arena.config.kelly_bootstrap_min_exposure.load(Ordering::Relaxed),
            ev_fee_multiplier: arena.config.ev_fee_multiplier.load(Ordering::Relaxed),
            margin_cushion_pct: arena.config.margin_cushion_pct.load(Ordering::Relaxed),
            maker_only_capital_threshold: arena.config.maker_only_capital_threshold.load(Ordering::Relaxed),
            
            kelly_survival_cap_ratio: arena.config.kelly_survival_cap_ratio.load(Ordering::Relaxed),
            kelly_expansion_mult: arena.config.kelly_expansion_mult.load(Ordering::Relaxed),
            guard_dd_sigmoid_steepness: arena.config.guard_dd_sigmoid_steepness.load(Ordering::Relaxed),
            guard_dd_sigmoid_center: arena.config.guard_dd_sigmoid_center.load(Ordering::Relaxed),
            portfolio_perf_mult_steepness: arena.config.portfolio_perf_mult_steepness.load(Ordering::Relaxed),
            portfolio_dd_penalty_decay: arena.config.portfolio_dd_penalty_decay.load(Ordering::Relaxed),
            portfolio_perf_mult_min: arena.config.portfolio_perf_mult_min.load(Ordering::Relaxed),
            portfolio_perf_mult_max: arena.config.portfolio_perf_mult_max.load(Ordering::Relaxed),
            portfolio_perf_mult_center: arena.config.portfolio_perf_mult_center.load(Ordering::Relaxed),
            
            macro_hurst_confidence_offset: arena.config.macro_hurst_confidence_offset.load(Ordering::Relaxed),
            macro_hurst_confidence_scale: arena.config.macro_hurst_confidence_scale.load(Ordering::Relaxed),
            macro_vol_confidence_scale: arena.config.macro_vol_confidence_scale.load(Ordering::Relaxed),
            macro_min_cooldown_ratio: arena.config.macro_min_cooldown_ratio.load(Ordering::Relaxed),
            macro_max_cooldown_ratio: arena.config.macro_max_cooldown_ratio.load(Ordering::Relaxed),
            macro_cooldown_reduction_factor: arena.config.macro_cooldown_reduction_factor.load(Ordering::Relaxed),
            macro_leverage_momentum_scale: arena.config.macro_leverage_momentum_scale.load(Ordering::Relaxed),
            hawkes_scalp_threshold: arena.config.hawkes_scalp_threshold.load(Ordering::Relaxed),
            obi_zscore_threshold: arena.config.obi_zscore_threshold.load(Ordering::Relaxed),
            lev_matrix_vol_clamp_min: arena.config.lev_matrix_vol_clamp_min.load(Ordering::Relaxed),
            lev_matrix_growth_scalar: arena.config.lev_matrix_growth_scalar.load(Ordering::Relaxed),
            lev_matrix_log_cap_divisor: arena.config.lev_matrix_log_cap_divisor.load(Ordering::Relaxed),
            scalp_accel_min_samples: arena.config.scalp_accel_min_samples.load(Ordering::Relaxed),
            executor_max_orders_10s: arena.config.executor_max_orders_10s.load(Ordering::Relaxed),
            executor_max_weight_1m: arena.config.executor_max_weight_1m.load(Ordering::Relaxed),
            iceberg_volume_threshold: arena.config.iceberg_volume_threshold.load(Ordering::Relaxed),
            iceberg_slice_count: arena.config.iceberg_slice_count.load(Ordering::Relaxed),
        }
    }

    pub fn load_or_default() -> Self {
        Self::load_or_baseline(0.0002, 0.0005)
    }

    pub fn load_or_baseline(maker_base: f64, taker_base: f64) -> Self {
        if let Ok(data) = std::fs::read_to_string("config_dir/genotypes/active_genome.json") {
            if let Ok(genome) = serde_json::from_str::<Self>(&data) {
                telemetry_engine::telemetry!("🧬 [GENOMA] Loaded evolved SuperGenotype from config_dir/genotypes/active_genome.json");
                return genome;
            }
        }
        if let Ok(data) = std::fs::read_to_string("data/genesis_genome.json") {
            if let Ok(genome) = serde_json::from_str::<Self>(&data) {
                telemetry_engine::telemetry!("🧬 [GENOMA] Loaded surviving SuperGenotype from genesis_genome.json");
                return genome;
            }
        }
        // telemetry_engine::telemetry!("🧬 [GENOMA] No valid SuperGenotype found, initializing mathematical baseline with live fees (Maker: {:.4}%, Taker: {:.4}%).", maker_base * 100.0, taker_base * 100.0);
        Self::new_baseline(maker_base, taker_base)
    }

    pub fn save(&self) {
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write("data/genesis_genome.json", data);
        }
    }

    pub fn new_baseline(maker_base: f64, taker_base: f64) -> Self {
        // DERIVACIÓN MATEMÁTICA Y ESTOCÁSTICA DE LÍMITES BASE (Cero Hardcoding Heurístico)
        let w_base = 0.55_f64;
        let pi = std::f64::consts::PI;
        let e_const = std::f64::consts::E;
        let golden_ratio = 1.618033988749895_f64;
        
        let scalp_tp_math = taker_base * 6.0; 
        let scalp_sl_math = scalp_tp_math / 2.0; 
        let swing_tp_math = taker_base * 30.0; 
        let swing_sl_math = swing_tp_math / 2.0; 
        
        let r_scalp = scalp_tp_math / scalp_sl_math;
        let r_swing = swing_tp_math / swing_sl_math;
        
        let k_scalp = w_base - ((1.0 - w_base) / r_scalp);
        let k_swing = w_base - ((1.0 - w_base) / r_swing);
        
        Self {
            global_max_drawdown: 1.0 - (taker_base * 100.0).clamp(0.01, 0.10), // Derivado del costo del mercado
            global_leverage: 30.0, // Apalancamiento para cuentas pequeñas futures
            btc_volatility_multiplier: 1.0,
            eth_volatility_multiplier: e_const / 2.0,
            min_trades_per_day: golden_ratio * 3.0,
            survival_capital_threshold: golden_ratio / 2.0, // ~0.809
            funding_rate_sensitivity: w_base,
            global_correlation_threshold: golden_ratio - 1.0, // 0.618
            trend_threshold: pi / 10.0, // ~0.314
            range_threshold: e_const / 6.0, // ~0.453
            scalp_kelly_fraction: k_scalp, 
            swing_kelly_fraction: k_swing, 
            scalp_obi_threshold: taker_base * 50.0, 
            scalp_tp_base: scalp_tp_math,
            scalp_sl_base: scalp_sl_math,
            swing_tp_base: swing_tp_math,
            swing_sl_base: swing_sl_math,
            sl_atr_mult_btc: 1.0,
            tp_rr_ratio_btc: r_scalp,
            min_confidence_btc: w_base * 1.018, // Ligeramente mayor que base
            veto_threshold_btc: w_base * 1.09, 
            tech_threshold: taker_base * 5.0,
            ml_threshold_long: w_base * 1.036,
            ml_threshold_short: w_base * 1.036,
            maker_spread_pct: maker_base,
            maker_obi_threshold: w_base,
            target_volatility: taker_base * 40.0,
            dynamic_atr_min: 0.000001, // Scaled down to match tick-level ATR ~ 0.000002
            dynamic_obi_threshold: taker_base * 300.0,
            dynamic_ema_trend: 0.00001, // Scaled down to match tick-level EMA diff ~ 0.000015
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
            zombie_timeout_ms: (1.0 / taker_base) * 900.0,
            hurst_trend_threshold: golden_ratio - 1.0, // ~0.618
            cvd_veto_threshold: w_base * golden_ratio,
            wall_veto_threshold: pi * 5.0,
            flash_crash_jump_pct: taker_base * 300.0,
            latency_ms_panic_threshold: 1000.0 * pi, // ~3141ms
            ema_fast_period: pi * 4.0, // ~12.5
            ema_slow_period: pi * 8.0, // ~25.1
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
            margin_cushion_pct: 1.0 + (taker_base * 100.0),
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
        }
    }

    pub fn new_random() -> Self {
        Self {
            global_max_drawdown: rand::rng().random_range(0.5..0.99),
            global_leverage: rand::rng().random_range(25.0..35.0),
            btc_volatility_multiplier: rand::rng().random_range(0.5..3.0),
            eth_volatility_multiplier: rand::rng().random_range(0.5..3.0),
            min_trades_per_day: rand::rng().random_range(1.0..50.0),
            survival_capital_threshold: rand::rng().random_range(0.5..0.95),
            funding_rate_sensitivity: rand::rng().random_range(0.1..2.0),
            global_correlation_threshold: rand::rng().random_range(0.2..0.9),
            trend_threshold: rand::rng().random_range(0.1..0.9),
            range_threshold: rand::rng().random_range(0.1..0.9),
            scalp_kelly_fraction: rand::rng().random_range(0.1..2.0),
            swing_kelly_fraction: rand::rng().random_range(0.01..1.0),
            scalp_obi_threshold: rand::rng().random_range(0.05..1.0),
            scalp_tp_base: rand::rng().random_range(0.0005..0.005),
            scalp_sl_base: rand::rng().random_range(0.0002..0.002),
            swing_tp_base: rand::rng().random_range(0.005..0.10),
            swing_sl_base: rand::rng().random_range(0.001..0.05),
            sl_atr_mult_btc: rand::rng().random_range(0.5..5.0),
            tp_rr_ratio_btc: rand::rng().random_range(1.0..10.0),
            min_confidence_btc: rand::rng().random_range(0.5..0.95),
            veto_threshold_btc: rand::rng().random_range(0.5..0.99),
            tech_threshold: rand::rng().random_range(0.001..0.05),
            ml_threshold_long: rand::rng().random_range(0.5..0.95),
            ml_threshold_short: rand::rng().random_range(0.5..0.95),
            maker_spread_pct: rand::rng().random_range(0.0001..0.01),
            maker_obi_threshold: rand::rng().random_range(0.1..0.95),
            target_volatility: rand::rng().random_range(0.005..0.1),
            dynamic_atr_min: rand::rng().random_range(0.0000001..0.0000050),
            dynamic_obi_threshold: rand::rng().random_range(0.05..0.95),
            dynamic_ema_trend: rand::rng().random_range(0.000001..0.000050),
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
            zombie_timeout_ms: rand::rng().random_range(10000.0..3600000.0),
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
            margin_cushion_pct: rand::rng().random_range(1.01..1.20),
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
            hawkes_scalp_threshold: rand::rng().random_range(0.1..1.0),
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
        }
    }

    /// Applica el genoma completo directamente al Arena lock-free
    pub fn apply_to_arena(&self, arena: &GlobalArena) {
        arena.config.global_max_drawdown.store(self.global_max_drawdown, Ordering::Relaxed);
        arena.config.global_leverage.store(self.global_leverage, Ordering::Relaxed);
        arena.config.btc_volatility_multiplier.store(self.btc_volatility_multiplier, Ordering::Relaxed);
        arena.config.eth_volatility_multiplier.store(self.eth_volatility_multiplier, Ordering::Relaxed);
        arena.config.funding_rate_sensitivity.store(self.funding_rate_sensitivity, Ordering::Relaxed);
        arena.config.global_correlation_threshold.store(self.global_correlation_threshold, Ordering::Relaxed);
        arena.config.trend_threshold.store(self.trend_threshold, Ordering::Relaxed);
        arena.config.range_threshold.store(self.range_threshold, Ordering::Relaxed);
        arena.config.scalp_kelly_fraction.store(self.scalp_kelly_fraction, Ordering::Relaxed);
        arena.config.swing_kelly_fraction.store(self.swing_kelly_fraction, Ordering::Relaxed);
        arena.config.scalp_obi_threshold.store(self.scalp_obi_threshold, Ordering::Relaxed);
        arena.config.scalp_tp_base.store(self.scalp_tp_base, Ordering::Relaxed);
        arena.config.scalp_sl_base.store(self.scalp_sl_base, Ordering::Relaxed);
        arena.config.swing_tp_base.store(self.swing_tp_base, Ordering::Relaxed);
        arena.config.swing_sl_base.store(self.swing_sl_base, Ordering::Relaxed);
        arena.config.sl_atr_mult_btc.store(self.sl_atr_mult_btc, Ordering::Relaxed);
        arena.config.tp_rr_ratio_btc.store(self.tp_rr_ratio_btc, Ordering::Relaxed);
        arena.config.min_confidence_btc.store(self.min_confidence_btc, Ordering::Relaxed);
        arena.config.veto_threshold_btc.store(self.veto_threshold_btc, Ordering::Relaxed);
        arena.config.tech_threshold.store(self.tech_threshold, Ordering::Relaxed);
        arena.config.ml_threshold_long.store(self.ml_threshold_long, Ordering::Relaxed);
        arena.config.ml_threshold_short.store(self.ml_threshold_short, Ordering::Relaxed);
        arena.config.maker_spread_pct.store(self.maker_spread_pct, Ordering::Relaxed);
        arena.config.maker_obi_threshold.store(self.maker_obi_threshold, Ordering::Relaxed);
        arena.config.target_volatility.store(self.target_volatility, Ordering::Relaxed);
        arena.config.dynamic_atr_min.store(self.dynamic_atr_min, Ordering::Relaxed);
        arena.config.dynamic_obi_threshold.store(self.dynamic_obi_threshold, Ordering::Relaxed);
        arena.config.dynamic_ema_trend.store(self.dynamic_ema_trend, Ordering::Relaxed);
        arena.config.dynamic_ofi_threshold.store(self.dynamic_ofi_threshold, Ordering::Relaxed);
        arena.config.capital_split_scalp.store(self.capital_split_scalp, Ordering::Relaxed);
        arena.config.kelly_clamp_min.store(self.kelly_clamp_min, Ordering::Relaxed);
        arena.config.kelly_clamp_max.store(self.kelly_clamp_max, Ordering::Relaxed);
        arena.config.explosive_leverage_multiplier.store(self.explosive_leverage_multiplier, Ordering::Relaxed);
        arena.config.quantum_mutation_rate.store(self.quantum_mutation_rate, Ordering::Relaxed);
        arena.config.temporal_memory_decay.store(self.temporal_memory_decay, Ordering::Relaxed);
        arena.config.leverage_cap.store(self.leverage_cap, Ordering::Relaxed);
        arena.config.explosive_confidence_threshold.store(self.explosive_confidence_threshold, Ordering::Relaxed);
        arena.config.hawkes_volume_norm.store(self.hawkes_volume_norm, Ordering::Relaxed);
        arena.config.base_duration_ms.store(self.base_duration_ms, Ordering::Relaxed);
        arena.config.weight_obi.store(self.weight_obi, Ordering::Relaxed);
        arena.config.weight_ofi.store(self.weight_ofi, Ordering::Relaxed);
        arena.config.weight_vpin.store(self.weight_vpin, Ordering::Relaxed);
        arena.config.regime_duration_ms.store(self.regime_duration_ms, Ordering::Relaxed);
        arena.config.regime_atr_multiplier.store(self.regime_atr_multiplier, Ordering::Relaxed);
        
        arena.config.scalp_trail_act_atr.store(self.scalp_trail_act_atr, Ordering::Relaxed);
        arena.config.scalp_trail_step_atr.store(self.scalp_trail_step_atr, Ordering::Relaxed);
        arena.config.scalp_trail_max_atr.store(self.scalp_trail_max_atr, Ordering::Relaxed);
        arena.config.scalp_trail_min_pnl.store(self.scalp_trail_min_pnl, Ordering::Relaxed);
        arena.config.scalp_trail_atr_mult_base.store(self.scalp_trail_atr_mult_base, Ordering::Relaxed);
        
        arena.config.swing_trail_act_atr.store(self.swing_trail_act_atr, Ordering::Relaxed);
        arena.config.swing_trail_step_atr.store(self.swing_trail_step_atr, Ordering::Relaxed);
        arena.config.swing_trail_max_atr.store(self.swing_trail_max_atr, Ordering::Relaxed);
        arena.config.swing_trail_min_pnl.store(self.swing_trail_min_pnl, Ordering::Relaxed);
        arena.config.swing_trail_atr_mult_base.store(self.swing_trail_atr_mult_base, Ordering::Relaxed);
        
        arena.config.zombie_timeout_ms.store(self.zombie_timeout_ms, Ordering::Relaxed);
        arena.config.hurst_trend_threshold.store(self.hurst_trend_threshold, Ordering::Relaxed);
        arena.config.cvd_veto_threshold.store(self.cvd_veto_threshold, Ordering::Relaxed);
        arena.config.wall_veto_threshold.store(self.wall_veto_threshold, Ordering::Relaxed);
        arena.config.flash_crash_jump_pct.store(self.flash_crash_jump_pct, Ordering::Relaxed);
        arena.config.latency_ms_panic_threshold.store(self.latency_ms_panic_threshold, Ordering::Relaxed);
        arena.config.ema_fast_period.store(self.ema_fast_period, Ordering::Relaxed);
        arena.config.ema_slow_period.store(self.ema_slow_period, Ordering::Relaxed);
        arena.config.hurst_scalp_threshold.store(self.hurst_scalp_threshold, Ordering::Relaxed);
        arena.config.hurst_swing_threshold.store(self.hurst_swing_threshold, Ordering::Relaxed);
        arena.config.synergy_exposure_boost.store(self.synergy_exposure_boost, Ordering::Relaxed);
        arena.config.synergy_leverage_boost.store(self.synergy_leverage_boost, Ordering::Relaxed);
        arena.config.max_fee_pct.store(self.max_fee_pct, Ordering::Relaxed);
        arena.config.kelly_bootstrap_cold.store(self.kelly_bootstrap_cold, Ordering::Relaxed);
        arena.config.latency_penalty_ms_physics.store(self.latency_penalty_ms, Ordering::Relaxed);
        arena.config.base_slippage_floor.store(self.base_slippage_floor, Ordering::Relaxed);
        arena.config.spot_spread_threshold.store(self.spot_spread_threshold, Ordering::Relaxed);
        arena.config.spot_bias_value.store(self.spot_bias_value, Ordering::Relaxed);
        arena.config.obi_confidence_fallback.store(self.obi_confidence_fallback, Ordering::Relaxed);
        arena.config.ml_clip_lower.store(self.ml_clip_lower, Ordering::Relaxed);
        arena.config.ml_clip_upper.store(self.ml_clip_upper, Ordering::Relaxed);
        arena.config.tensor_poly_a.store(self.tensor_poly_a, Ordering::Relaxed);
        arena.config.tensor_poly_b.store(self.tensor_poly_b, Ordering::Relaxed);
        arena.config.fractional_alpha_order.store(self.fractional_alpha_order, Ordering::Relaxed);
        arena.config.fractional_clip_max.store(self.fractional_clip_max, Ordering::Relaxed);
        arena.config.bft_consensus_tolerance.store(self.bft_consensus_tolerance, Ordering::Relaxed);
        arena.config.turbo_coherence_threshold.store(self.turbo_coherence_threshold, Ordering::Relaxed);
        arena.config.turbo_z_score_stdev.store(self.turbo_z_score_stdev, Ordering::Relaxed);
        arena.config.sl_atr_multiplier.store(self.sl_atr_multiplier, Ordering::Relaxed);
        arena.config.coaxial_squeeze_threshold.store(self.coaxial_squeeze_threshold, Ordering::Relaxed);
        arena.config.tensor_op_add_bias.store(self.tensor_op_add_bias, Ordering::Relaxed);
        arena.config.tensor_op_mul_weight.store(self.tensor_op_mul_weight, Ordering::Relaxed);
        arena.config.topo_layer_1_activation.store(self.topo_layer_1_activation, Ordering::Relaxed);
        arena.config.topo_layer_2_activation.store(self.topo_layer_2_activation, Ordering::Relaxed);
        arena.config.tensor_dropout_rate.store(self.tensor_dropout_rate, Ordering::Relaxed);
        arena.config.quantum_entropy_seed.store(self.quantum_entropy_seed, Ordering::Relaxed);
        
        arena.config.ppo_clip_eps.store(self.ppo_clip_eps, Ordering::Relaxed);
        arena.config.ppo_weight_min_clip.store(self.ppo_weight_min_clip, Ordering::Relaxed);
        arena.config.hard_stop_decay_factor.store(self.hard_stop_decay_factor, Ordering::Relaxed);
        arena.config.hard_stop_base_limit.store(self.hard_stop_base_limit, Ordering::Relaxed);
        arena.config.kelly_bootstrap_ratio_threshold.store(self.kelly_bootstrap_ratio_threshold, Ordering::Relaxed);
        arena.config.kelly_bootstrap_min_exposure.store(self.kelly_bootstrap_min_exposure, Ordering::Relaxed);
        arena.config.ev_fee_multiplier.store(self.ev_fee_multiplier, Ordering::Relaxed);
        arena.config.margin_cushion_pct.store(self.margin_cushion_pct, Ordering::Relaxed);
        arena.config.maker_only_capital_threshold.store(self.maker_only_capital_threshold, Ordering::Relaxed);
        arena.config.hawkes_scalp_threshold.store(self.hawkes_scalp_threshold, Ordering::Relaxed);
        arena.config.obi_zscore_threshold.store(self.obi_zscore_threshold, Ordering::Relaxed);
        
        // CORRECCIÓN CRÍTICA: Estos 9 campos estaban definidos, se mutaban,
        // pero NUNCA se propagaban al Arena. Eran nodos silenciosos (FALLO TIPO 1).
        // El evolver gastaba CPU evolucionando valores que el sistema nunca leía.
        arena.config.kelly_survival_cap_ratio.store(self.kelly_survival_cap_ratio, Ordering::Relaxed);
        arena.config.kelly_expansion_mult.store(self.kelly_expansion_mult, Ordering::Relaxed);
        arena.config.guard_dd_sigmoid_steepness.store(self.guard_dd_sigmoid_steepness, Ordering::Relaxed);
        arena.config.guard_dd_sigmoid_center.store(self.guard_dd_sigmoid_center, Ordering::Relaxed);
        arena.config.portfolio_perf_mult_steepness.store(self.portfolio_perf_mult_steepness, Ordering::Relaxed);
        arena.config.portfolio_dd_penalty_decay.store(self.portfolio_dd_penalty_decay, Ordering::Relaxed);
        arena.config.portfolio_perf_mult_min.store(self.portfolio_perf_mult_min, Ordering::Relaxed);
        arena.config.portfolio_perf_mult_max.store(self.portfolio_perf_mult_max, Ordering::Relaxed);
        arena.config.portfolio_perf_mult_center.store(self.portfolio_perf_mult_center, Ordering::Relaxed);
        
        arena.config.macro_hurst_confidence_offset.store(self.macro_hurst_confidence_offset, Ordering::Relaxed);
        arena.config.macro_hurst_confidence_scale.store(self.macro_hurst_confidence_scale, Ordering::Relaxed);
        arena.config.macro_vol_confidence_scale.store(self.macro_vol_confidence_scale, Ordering::Relaxed);
        arena.config.macro_min_cooldown_ratio.store(self.macro_min_cooldown_ratio, Ordering::Relaxed);
        arena.config.macro_max_cooldown_ratio.store(self.macro_max_cooldown_ratio, Ordering::Relaxed);
        arena.config.macro_cooldown_reduction_factor.store(self.macro_cooldown_reduction_factor, Ordering::Relaxed);
        arena.config.macro_leverage_momentum_scale.store(self.macro_leverage_momentum_scale, Ordering::Relaxed);
        
        // FASE 4: De-hardcoding
        arena.config.lev_matrix_vol_clamp_min.store(self.lev_matrix_vol_clamp_min, Ordering::Relaxed);
        arena.config.lev_matrix_growth_scalar.store(self.lev_matrix_growth_scalar, Ordering::Relaxed);
        arena.config.lev_matrix_log_cap_divisor.store(self.lev_matrix_log_cap_divisor, Ordering::Relaxed);
        arena.config.scalp_accel_min_samples.store(self.scalp_accel_min_samples, Ordering::Relaxed);
        arena.config.executor_max_orders_10s.store(self.executor_max_orders_10s, Ordering::Relaxed);
        arena.config.executor_max_weight_1m.store(self.executor_max_weight_1m, Ordering::Relaxed);
        arena.config.iceberg_volume_threshold.store(self.iceberg_volume_threshold, Ordering::Relaxed);
        arena.config.iceberg_slice_count.store(self.iceberg_slice_count, Ordering::Relaxed);
    }
    
    pub fn mutate_cmaes(&self, rate: f64) -> Self {
        let mut rng = rand::rng();
        let mut mutate_val = |base: f64, min_val: f64, max_val: f64| -> f64 {
            let change = base * rate * rng.random_range(-1.0..1.0);
            (base + change).clamp(min_val, max_val)
        };
        
        Self {
            global_max_drawdown: mutate_val(self.global_max_drawdown, 0.5, 0.99),
            global_leverage: mutate_val(self.global_leverage, 25.0, 35.0),
            btc_volatility_multiplier: mutate_val(self.btc_volatility_multiplier, 0.5, 3.0),
            eth_volatility_multiplier: mutate_val(self.eth_volatility_multiplier, 0.5, 3.0),
            min_trades_per_day: mutate_val(self.min_trades_per_day, 1.0, 50.0),
            survival_capital_threshold: mutate_val(self.survival_capital_threshold, 0.5, 0.95),
            funding_rate_sensitivity: mutate_val(self.funding_rate_sensitivity, 0.1, 2.0),
            global_correlation_threshold: mutate_val(self.global_correlation_threshold, 0.2, 0.9),
            trend_threshold: mutate_val(self.trend_threshold, 0.1, 0.9),
            range_threshold: mutate_val(self.range_threshold, 0.1, 0.9),
            scalp_kelly_fraction: mutate_val(self.scalp_kelly_fraction, 0.1, 2.0),
            swing_kelly_fraction: mutate_val(self.swing_kelly_fraction, 0.01, 1.0),
            scalp_obi_threshold: mutate_val(self.scalp_obi_threshold, 0.05, 1.0),
            scalp_tp_base: mutate_val(self.scalp_tp_base, 0.0005, 0.005),
            scalp_sl_base: mutate_val(self.scalp_sl_base, 0.0002, 0.002),
            swing_tp_base: mutate_val(self.swing_tp_base, 0.005, 0.10),
            swing_sl_base: mutate_val(self.swing_sl_base, 0.001, 0.05),
            sl_atr_mult_btc: mutate_val(self.sl_atr_mult_btc, 0.5, 5.0),
            tp_rr_ratio_btc: mutate_val(self.tp_rr_ratio_btc, 1.0, 10.0),
            min_confidence_btc: mutate_val(self.min_confidence_btc, 0.5, 0.95),
            veto_threshold_btc: mutate_val(self.veto_threshold_btc, 0.5, 0.99),
            tech_threshold: mutate_val(self.tech_threshold, 0.001, 0.05),
            ml_threshold_long: mutate_val(self.ml_threshold_long, 0.5, 0.95),
            ml_threshold_short: mutate_val(self.ml_threshold_short, 0.5, 0.95),
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
            explosive_leverage_multiplier: mutate_val(self.explosive_leverage_multiplier, 1.0, 10.0),
            quantum_mutation_rate: mutate_val(self.quantum_mutation_rate, 0.01, 0.5),
            temporal_memory_decay: mutate_val(self.temporal_memory_decay, 0.8, 0.9999),
            leverage_cap: mutate_val(self.leverage_cap, 10.0, 125.0),
            explosive_confidence_threshold: mutate_val(self.explosive_confidence_threshold, 0.7, 0.99),
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
            zombie_timeout_ms: mutate_val(self.zombie_timeout_ms, 10000.0, 3600000.0),
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
            kelly_bootstrap_ratio_threshold: mutate_val(self.kelly_bootstrap_ratio_threshold, 0.05, 0.30),
            kelly_bootstrap_min_exposure: mutate_val(self.kelly_bootstrap_min_exposure, 0.01, 0.5),
            ev_fee_multiplier: mutate_val(self.ev_fee_multiplier, 1.0, 5.0),
            margin_cushion_pct: mutate_val(self.margin_cushion_pct, 1.01, 1.20),
            maker_only_capital_threshold: 50.0,
            hawkes_scalp_threshold: mutate_val(self.hawkes_scalp_threshold, 0.1, 1.0),
            obi_zscore_threshold: mutate_val(self.obi_zscore_threshold, 0.1, 3.0),
            
            hawkes_volume_norm: mutate_val(self.hawkes_volume_norm, 100.0, 10_000_000.0),
            base_duration_ms: mutate_val(self.base_duration_ms, 10_000.0, 120_000.0),
            
            kelly_survival_cap_ratio: mutate_val(self.kelly_survival_cap_ratio, 1.0, 3.0),
            kelly_expansion_mult: mutate_val(self.kelly_expansion_mult, 1.0, 3.0),
            guard_dd_sigmoid_steepness: mutate_val(self.guard_dd_sigmoid_steepness, 1.0, 5.0),
            guard_dd_sigmoid_center: mutate_val(self.guard_dd_sigmoid_center, 1.0, 5.0),
            portfolio_perf_mult_steepness: mutate_val(self.portfolio_perf_mult_steepness, 3.0, 15.0),
            portfolio_dd_penalty_decay: mutate_val(self.portfolio_dd_penalty_decay, 5.0, 25.0),
            portfolio_perf_mult_min: mutate_val(self.portfolio_perf_mult_min, 0.1, 0.5),
            portfolio_perf_mult_max: mutate_val(self.portfolio_perf_mult_max, 1.1, 3.0),
            portfolio_perf_mult_center: mutate_val(self.portfolio_perf_mult_center, 0.4, 0.6),
            
            macro_hurst_confidence_offset: mutate_val(self.macro_hurst_confidence_offset, 0.35, 0.55),
            macro_hurst_confidence_scale: mutate_val(self.macro_hurst_confidence_scale, 1.0, 20.0),
            macro_vol_confidence_scale: mutate_val(self.macro_vol_confidence_scale, 10.0, 500.0),
            macro_min_cooldown_ratio: mutate_val(self.macro_min_cooldown_ratio, 0.05, 0.5),
            macro_max_cooldown_ratio: mutate_val(self.macro_max_cooldown_ratio, 10.0, 500.0),
            macro_cooldown_reduction_factor: mutate_val(self.macro_cooldown_reduction_factor, 0.1, 2.0),
            macro_leverage_momentum_scale: mutate_val(self.macro_leverage_momentum_scale, 0.1, 2.0),
            
            lev_matrix_vol_clamp_min: mutate_val(self.lev_matrix_vol_clamp_min, 0.1, 0.8),
            lev_matrix_growth_scalar: mutate_val(self.lev_matrix_growth_scalar, 0.1, 1.0),
            lev_matrix_log_cap_divisor: mutate_val(self.lev_matrix_log_cap_divisor, 3.0, 10.0),
            scalp_accel_min_samples: mutate_val(self.scalp_accel_min_samples, 10.0, 50.0),
            executor_max_orders_10s: mutate_val(self.executor_max_orders_10s, 100.0, 500.0),
            executor_max_weight_1m: mutate_val(self.executor_max_weight_1m, 1000.0, 5000.0),
            iceberg_volume_threshold: mutate_val(self.iceberg_volume_threshold, 10_000.0, 100_000.0),
            iceberg_slice_count: mutate_val(self.iceberg_slice_count, 2.0, 10.0),
        }
    }

// Generated extensions
    pub const DIMENSION: usize = 137;

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
        vec
    }

    pub fn from_vector(vec: &[f64]) -> Self {
        Self {
            global_max_drawdown: vec[0].clamp(0.5, 0.99),
            global_leverage: vec[1].clamp(25.0, 35.0),
            btc_volatility_multiplier: vec[2].clamp(0.5, 3.0),
            eth_volatility_multiplier: vec[3].clamp(0.5, 3.0),
            min_trades_per_day: vec[4].clamp(1.0, 50.0),
            survival_capital_threshold: vec[5].clamp(0.5, 0.95),
            funding_rate_sensitivity: vec[6].clamp(0.1, 2.0),
            global_correlation_threshold: vec[7].clamp(0.2, 0.9),
            trend_threshold: vec[8].clamp(0.1, 0.9),
            range_threshold: vec[9].clamp(0.1, 0.9),
            scalp_kelly_fraction: vec[10].clamp(0.1, 2.0),
            swing_kelly_fraction: vec[11].clamp(0.01, 1.0),
            scalp_obi_threshold: vec[12].clamp(0.05, 1.0),
            scalp_tp_base: vec[13].clamp(0.0005, 0.005),
            scalp_sl_base: vec[14].clamp(0.0002, 0.002),
            swing_tp_base: vec[15].clamp(0.005, 0.10),
            swing_sl_base: vec[16].clamp(0.001, 0.05),
            sl_atr_mult_btc: vec[17].clamp(0.5, 5.0),
            tp_rr_ratio_btc: vec[18].clamp(1.0, 10.0),
            min_confidence_btc: vec[19].clamp(0.5, 0.95),
            veto_threshold_btc: vec[20].clamp(0.5, 0.99),
            tech_threshold: vec[21].clamp(0.001, 0.05),
            ml_threshold_long: vec[22].clamp(0.5, 0.95),
            ml_threshold_short: vec[23].clamp(0.5, 0.95),
            maker_spread_pct: vec[24].clamp(0.0001, 0.01),
            maker_obi_threshold: vec[25].clamp(0.1, 0.95),
            target_volatility: vec[26].clamp(0.005, 0.1),
            dynamic_atr_min: vec[27].clamp(0.0001, 0.01),
            dynamic_obi_threshold: vec[28].clamp(0.05, 0.95),
            dynamic_ema_trend: vec[29].clamp(0.0001, 0.01),
            dynamic_ofi_threshold: vec[30].clamp(0.05, 0.95),
            capital_split_scalp: vec[31].clamp(0.1, 1.0),
            kelly_clamp_min: vec[32].clamp(0.001, 0.1),
            kelly_clamp_max: vec[33].clamp(0.1, 1.0),
            explosive_leverage_multiplier: vec[34].clamp(1.0, 10.0),
            quantum_mutation_rate: vec[35].clamp(0.01, 0.5),
            temporal_memory_decay: vec[36].clamp(0.8, 0.9999),
            leverage_cap: vec[37].clamp(10.0, 125.0),
            explosive_confidence_threshold: vec[38].clamp(0.7, 0.99),
            weight_obi: vec[39].clamp(0.1, 1.0),
            weight_ofi: vec[40].clamp(0.1, 1.0),
            weight_vpin: vec[41].clamp(0.1, 1.0),
            regime_duration_ms: vec[42].clamp(30000.0, 600000.0),
            regime_atr_multiplier: vec[43].clamp(0.5, 5.0),
            scalp_trail_act_atr: vec[44].clamp(0.001, 20.0),
            scalp_trail_step_atr: vec[45].clamp(0.001, 20.0),
            scalp_trail_max_atr: vec[46].clamp(0.001, 20.0),
            scalp_trail_min_pnl: vec[47].clamp(0.001, 20.0),
            scalp_trail_atr_mult_base: vec[48].clamp(0.001, 20.0),
            swing_trail_act_atr: vec[49].clamp(0.001, 20.0),
            swing_trail_step_atr: vec[50].clamp(0.001, 20.0),
            swing_trail_max_atr: vec[51].clamp(0.001, 20.0),
            swing_trail_min_pnl: vec[52].clamp(0.001, 20.0),
            swing_trail_atr_mult_base: vec[53].clamp(0.001, 20.0),
            zombie_timeout_ms: vec[54].clamp(10000.0, 3600000.0),
            hurst_trend_threshold: vec[55].clamp(0.4, 0.8),
            cvd_veto_threshold: vec[56].clamp(0.1, 2.0),
            wall_veto_threshold: vec[57].clamp(5.0, 50.0),
            flash_crash_jump_pct: vec[58].clamp(0.05, 0.50),
            latency_ms_panic_threshold: vec[59].clamp(500.0, 10000.0),
            ema_fast_period: vec[60].clamp(5.0, 30.0),
            ema_slow_period: vec[61].clamp(15.0, 60.0),
            hurst_scalp_threshold: vec[62].clamp(0.30, 0.50),
            hurst_swing_threshold: vec[63].clamp(0.50, 0.70),
            synergy_exposure_boost: vec[64].clamp(1.0, 3.0),
            synergy_leverage_boost: vec[65].clamp(1.0, 2.0),
            max_fee_pct: vec[66].clamp(0.005, 0.05),
            kelly_bootstrap_cold: vec[67].clamp(0.1, 1.0),
            latency_penalty_ms: vec[68].clamp(5.0, 100.0),
            base_slippage_floor: vec[69].clamp(0.00005, 0.0005),
            spot_spread_threshold: vec[70].clamp(0.5, 3.0),
            spot_bias_value: vec[71].clamp(0.05, 0.30),
            obi_confidence_fallback: vec[72].clamp(0.70, 0.95),
            ml_clip_lower: vec[73].clamp(0.01, 0.20),
            ml_clip_upper: vec[74].clamp(0.80, 0.99),
            tensor_poly_a: vec[75].clamp(0.01, 0.20),
            tensor_poly_b: vec[76].clamp(0.001, 0.05),
            fractional_alpha_order: vec[77].clamp(0.1, 0.9),
            fractional_clip_max: vec[78].clamp(5.0, 20.0),
            bft_consensus_tolerance: vec[79].clamp(0.05, 0.30),
            turbo_coherence_threshold: vec[80].clamp(0.1, 0.6),
            turbo_z_score_stdev: vec[81].clamp(0.1, 1.5),
            sl_atr_multiplier: vec[82].clamp(0.2, 2.0),
            coaxial_squeeze_threshold: vec[83].clamp(0.1, 0.8),
            tensor_op_add_bias: vec[84].clamp(0.01, 0.5),
            tensor_op_mul_weight: vec[85].clamp(0.5, 3.0),
            topo_layer_1_activation: vec[86].clamp(0.1, 0.9),
            topo_layer_2_activation: vec[87].clamp(0.1, 0.9),
            tensor_dropout_rate: vec[88].clamp(0.01, 0.5),
            quantum_entropy_seed: vec[89].clamp(0.0, 1000.0),
            ppo_weight_ofi: vec[90].clamp(0.01, 0.5),
            ppo_weight_obi: vec[91].clamp(0.01, 0.5),
            ppo_weight_hawkes: vec[92].clamp(0.01, 0.5),
            ppo_weight_leadlag: vec[93].clamp(0.01, 0.5),
            ppo_weight_regime: vec[94].clamp(0.01, 0.5),
            global_learning_rate: vec[95].clamp(0.0001, 0.05),
            global_momentum: vec[96].clamp(0.5, 0.99),
            vecm_alpha_speed: vec[97].clamp(0.001, 0.1),
            vecm_beta_hedge: vec[98].clamp(0.001, 0.1),
            conformal_alpha: vec[99].clamp(0.01, 0.2),
            ppo_clip_eps: vec[100].clamp(0.05, 0.40),
            ppo_weight_min_clip: vec[101].clamp(0.01, 0.10),
            hard_stop_decay_factor: vec[102].clamp(0.1, 1.0),
            hard_stop_base_limit: vec[103].clamp(0.10, 0.80),
            kelly_bootstrap_ratio_threshold: vec[104].clamp(0.05, 0.30),
            kelly_bootstrap_min_exposure: vec[105].clamp(0.01, 0.5),
            ev_fee_multiplier: vec[106].clamp(1.0, 5.0),
            margin_cushion_pct: vec[107].clamp(1.01, 1.20),
            maker_only_capital_threshold: 50.0,
            hawkes_scalp_threshold: vec[109].clamp(0.1, 1.0),
            obi_zscore_threshold: vec[110].clamp(0.1, 3.0),
            hawkes_volume_norm: vec[111].clamp(100.0, 10000000.0),
            base_duration_ms: vec[112].clamp(10000.0, 120000.0),
            kelly_survival_cap_ratio: vec[113].clamp(1.0, 3.0),
            kelly_expansion_mult: vec[114].clamp(1.0, 3.0),
            guard_dd_sigmoid_steepness: vec[115].clamp(1.0, 5.0),
            guard_dd_sigmoid_center: vec[116].clamp(1.0, 5.0),
            portfolio_perf_mult_steepness: vec[117].clamp(3.0, 15.0),
            portfolio_dd_penalty_decay: vec[118].clamp(5.0, 25.0),
            portfolio_perf_mult_min: vec[119].clamp(0.1, 0.5),
            portfolio_perf_mult_max: vec[120].clamp(1.1, 3.0),
            portfolio_perf_mult_center: vec[121].clamp(0.4, 0.6),
            macro_hurst_confidence_offset: vec[122].clamp(0.35, 0.55),
            macro_hurst_confidence_scale: vec[123].clamp(1.0, 20.0),
            macro_vol_confidence_scale: vec[124].clamp(10.0, 500.0),
            macro_min_cooldown_ratio: vec[125].clamp(0.05, 0.5),
            macro_max_cooldown_ratio: vec[126].clamp(10.0, 500.0),
            macro_cooldown_reduction_factor: vec[127].clamp(0.1, 2.0),
            macro_leverage_momentum_scale: vec[128].clamp(0.1, 2.0),
            lev_matrix_vol_clamp_min: vec[129].clamp(0.1, 0.8),
            lev_matrix_growth_scalar: vec[130].clamp(0.1, 1.0),
            lev_matrix_log_cap_divisor: vec[131].clamp(3.0, 10.0),
            scalp_accel_min_samples: vec[132].clamp(10.0, 50.0),
            executor_max_orders_10s: vec[133].clamp(100.0, 500.0),
            executor_max_weight_1m: vec[134].clamp(1000.0, 5000.0),
            iceberg_volume_threshold: vec[135].clamp(10_000.0, 100_000.0),
            iceberg_slice_count: vec[136].clamp(2.0, 10.0),
        }
    }

    pub fn get_lower_bounds() -> Vec<f64> {
        vec![
            0.5,
            25.0,
            0.5,
            0.5,
            1.0,
            0.5,
            0.1,
            0.2,
            0.1,
            0.1,
            0.1,
            0.01,
            0.05,
            0.0005,
            0.0002,
            0.005,
            0.001,
            0.5,
            1.0,
            0.5,
            0.5,
            0.001,
            0.5,
            0.5,
            0.0001,
            0.1,
            0.005,
            0.0001,
            0.05,
            0.0001,
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
            10000.0,
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
            1.01,
            50.0,
            0.1,
            0.1,
            100.0,
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
            10_000.0,
            2.0,
        ]
    }

    pub fn get_upper_bounds() -> Vec<f64> {
        vec![
            0.99,
            35.0,
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
            0.005,
            0.002,
            0.10,
            0.05,
            5.0,
            10.0,
            0.95,
            0.99,
            0.05,
            0.95,
            0.95,
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
            0.99,
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
            3600000.0,
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
            1.5,
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
            1.20,
            50.0,
            1.0,
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
        ]
    }

}
