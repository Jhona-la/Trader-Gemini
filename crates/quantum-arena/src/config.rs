use crate::atomic_float::AtomicF64;
use crate::fee_schedule::FeeSchedule;
use crate::horizon_policy::{trailing_at_tau, HorizonParameter};
use crate::temporal_spectrum::HorizonCurve;
use arc_swap::ArcSwapOption;
use std::sync::atomic::{AtomicBool, Ordering};

/// F1-009: Límite actuarial canónico de drawdown del Consejo Deliberativo.
/// Derivado de la cota de falsación de Cramér-Lundberg: d* = 1 - alpha^(c / (2 - c))
/// para fracción de Kelly máxima c = 0.25 y nivel alpha = 0.05:
/// 1 - 0.05^(0.25 / 1.75) = 1 - 0.05^(1/7) ≈ 0.6315968501359614.
pub const CANONICAL_COUNCIL_MAX_DRAWDOWN: f64 = 0.6315968501359614;

/// F1-009: Constantes físicas y cuánticas canónicas (adimensionales en unidades de Planck de la red discreta).
pub const CANONICAL_FEYNMAN_HBAR: f64 = 1.0;
pub const CANONICAL_FEYNMAN_MASS: f64 = 1.0;
pub const CANONICAL_FEYNMAN_K_SPRING: f64 = 1.0;

/// Axioma V: El Config Omnisciente (5 Capas Adaptativas)
/// Todo expuesto como atómicos para mutación lock-free (O(1)) desde el Evolver.
/// Alineado a 64-bytes para evitar False Sharing entre núcleos.
#[repr(C, align(64))]
pub struct QuantumConfig {
    pub base_capital: AtomicF64,
    pub global_max_drawdown: AtomicF64,
    pub global_leverage: AtomicF64,
    pub min_notional: AtomicF64,
    pub btc_volatility_multiplier: AtomicF64,
    pub eth_volatility_multiplier: AtomicF64,
    pub min_trades_per_day: AtomicF64,
    pub survival_capital_threshold: AtomicF64,
    pub funding_rate_sensitivity: AtomicF64,
    pub global_correlation_threshold: AtomicF64,
    pub trend_threshold: AtomicF64,
    pub range_threshold: AtomicF64,
    pub scalp_kelly_fraction: AtomicF64,
    pub swing_kelly_fraction: AtomicF64,
    pub scalp_obi_threshold: AtomicF64,
    pub scalp_tp_base: AtomicF64,
    pub scalp_sl_base: AtomicF64,
    pub swing_tp_base: AtomicF64,
    pub swing_sl_base: AtomicF64,
    pub sl_atr_mult_btc: AtomicF64,
    pub tp_rr_ratio_btc: AtomicF64,
    pub min_confidence_btc: AtomicF64,
    pub veto_threshold_btc: AtomicF64,
    pub tech_threshold: AtomicF64,
    pub ml_threshold_long: AtomicF64,
    pub ml_threshold_short: AtomicF64,
    pub maker_spread_pct: AtomicF64,
    pub maker_obi_threshold: AtomicF64,
    pub target_volatility: AtomicF64,
    pub dynamic_atr_min: AtomicF64,
    pub dynamic_obi_threshold: AtomicF64,
    pub dynamic_ema_trend: AtomicF64,
    pub dynamic_ofi_threshold: AtomicF64,
    pub capital_split_scalp: AtomicF64,
    pub kelly_clamp_min: AtomicF64,
    pub kelly_clamp_max: AtomicF64,
    pub explosive_leverage_multiplier: AtomicF64,
    pub quantum_mutation_rate: AtomicF64,
    pub temporal_memory_decay: AtomicF64,
    pub leverage_cap: AtomicF64,
    pub explosive_confidence_threshold: AtomicF64,
    /// Fracción de precio de la comisión maker (ej. 0.0002 = 2 bps = 0.02%).
    pub live_maker_fee: AtomicF64,
    /// Fracción de precio de la comisión taker (ej. 0.0005 = 5 bps = 0.05%).
    pub live_taker_fee: AtomicF64,
    /// Session-owned observed rates, indexed in active-universe order.
    pub live_fee_schedule: ArcSwapOption<FeeSchedule>,
    /// Live admission requires symbol-specific evidence. False permits an
    /// explicit scenario to use the finite legacy scalar rates.
    pub observed_fees_required: AtomicBool,
    /// Serializes local entry publication with OMS snapshot application.
    /// The tick's analytics do not acquire this lock.
    pub entry_publication_gate: std::sync::Mutex<()>,
    pub entry_publication_paused: AtomicBool,
    pub weight_obi: AtomicF64,
    pub weight_ofi: AtomicF64,
    pub weight_vpin: AtomicF64,
    pub regime_duration_ms: AtomicF64,
    pub regime_atr_multiplier: AtomicF64,
    pub scalp_trail_act_atr: AtomicF64,
    pub scalp_trail_step_atr: AtomicF64,
    pub scalp_trail_max_atr: AtomicF64,
    pub scalp_trail_min_pnl: AtomicF64,
    pub scalp_trail_atr_mult_base: AtomicF64,
    pub swing_trail_act_atr: AtomicF64,
    pub swing_trail_step_atr: AtomicF64,
    pub swing_trail_max_atr: AtomicF64,
    pub swing_trail_min_pnl: AtomicF64,
    pub swing_trail_atr_mult_base: AtomicF64,
    pub zombie_timeout_ms: AtomicF64,
    pub hurst_trend_threshold: AtomicF64,
    pub cvd_veto_threshold: AtomicF64,
    pub wall_veto_threshold: AtomicF64,
    pub flash_crash_jump_pct: AtomicF64,
    pub latency_ms_panic_threshold: AtomicF64,
    pub ema_fast_period: AtomicF64,
    pub ema_slow_period: AtomicF64,
    pub hurst_scalp_threshold: AtomicF64,
    pub hurst_swing_threshold: AtomicF64,
    pub synergy_exposure_boost: AtomicF64,
    pub synergy_leverage_boost: AtomicF64,
    pub max_fee_pct: AtomicF64,
    pub kelly_bootstrap_cold: AtomicF64,
    pub latency_penalty_ms: AtomicF64,
    pub base_slippage_floor: AtomicF64,
    pub spot_spread_threshold: AtomicF64,
    pub spot_bias_value: AtomicF64,
    pub obi_confidence_fallback: AtomicF64,
    pub ml_clip_lower: AtomicF64,
    pub ml_clip_upper: AtomicF64,
    pub tensor_poly_a: AtomicF64,
    pub tensor_poly_b: AtomicF64,
    pub fractional_alpha_order: AtomicF64,
    pub fractional_clip_max: AtomicF64,
    pub bft_consensus_tolerance: AtomicF64,
    pub turbo_coherence_threshold: AtomicF64,
    pub turbo_z_score_stdev: AtomicF64,
    pub sl_atr_multiplier: AtomicF64,
    pub coaxial_squeeze_threshold: AtomicF64,
    pub tensor_op_add_bias: AtomicF64,
    pub tensor_op_mul_weight: AtomicF64,
    pub topo_layer_1_activation: AtomicF64,
    pub topo_layer_2_activation: AtomicF64,
    pub tensor_dropout_rate: AtomicF64,
    pub quantum_entropy_seed: AtomicF64,
    pub global_learning_rate: AtomicF64,
    pub global_momentum: AtomicF64,
    pub vecm_alpha_speed: AtomicF64,
    pub vecm_beta_hedge: AtomicF64,
    pub conformal_alpha: AtomicF64,
    pub ppo_weight_ofi: AtomicF64,
    pub ppo_weight_obi: AtomicF64,
    pub ppo_weight_hawkes: AtomicF64,
    pub ppo_weight_leadlag: AtomicF64,
    pub ppo_weight_regime: AtomicF64,
    pub ppo_clip_eps: AtomicF64,
    pub ppo_weight_min_clip: AtomicF64,
    pub hard_stop_decay_factor: AtomicF64,
    pub hard_stop_base_limit: AtomicF64,
    pub kelly_bootstrap_ratio_threshold: AtomicF64,
    pub kelly_bootstrap_min_exposure: AtomicF64,
    pub ev_fee_multiplier: AtomicF64,
    pub margin_cushion_pct: AtomicF64,
    pub maker_only_capital_threshold: AtomicF64,
    pub hawkes_scalp_threshold: AtomicF64,
    pub obi_zscore_threshold: AtomicF64,

    // FASE 14: Erradicación Matemática de Capital Fijo (13 USD vs 10k USD)
    pub hawkes_volume_norm: AtomicF64,
    pub base_duration_ms: AtomicF64,

    // FASE 3: Kelly, Guard & Orchestrator Evolutive Parameters
    pub kelly_survival_cap_ratio: AtomicF64,
    pub kelly_expansion_mult: AtomicF64,
    pub guard_dd_sigmoid_steepness: AtomicF64,
    pub guard_dd_sigmoid_center: AtomicF64,
    pub portfolio_perf_mult_steepness: AtomicF64,
    pub portfolio_dd_penalty_decay: AtomicF64,
    pub portfolio_perf_mult_min: AtomicF64,
    pub portfolio_perf_mult_max: AtomicF64,
    pub portfolio_perf_mult_center: AtomicF64,

    // FASE 15: Optimizador Macro-Regime Genómico
    pub macro_hurst_confidence_offset: AtomicF64,
    pub macro_hurst_confidence_scale: AtomicF64,
    pub macro_vol_confidence_scale: AtomicF64,
    pub macro_min_cooldown_ratio: AtomicF64,
    pub macro_max_cooldown_ratio: AtomicF64,
    pub macro_cooldown_reduction_factor: AtomicF64,
    pub macro_leverage_momentum_scale: AtomicF64,
    pub lev_matrix_vol_clamp_min: AtomicF64,
    pub lev_matrix_growth_scalar: AtomicF64,
    pub lev_matrix_log_cap_divisor: AtomicF64,
    pub scalp_accel_min_samples: AtomicF64,
    pub executor_max_orders_10s: AtomicF64,
    pub executor_max_weight_1m: AtomicF64,

    // --- FASE 6: Iceberg & IOC Limits ---
    pub iceberg_volume_threshold: AtomicF64,
    pub iceberg_slice_count: AtomicF64,

    // --- Swing OBI & Aceleración ---
    pub swing_obi_threshold: AtomicF64,
    pub swing_accel_min_samples: AtomicF64,

    /// F3-2 — eje temporal continuo (ver genome.rs).
    pub temporal_scale: AtomicF64,

    // --- FASE 23: Curvas de Horizonte Continuas (1 ns a 100 años) ---
    pub tp_curve_a: AtomicF64,
    pub tp_curve_b: AtomicF64,
    pub sl_curve_a: AtomicF64,
    pub sl_curve_b: AtomicF64,
    pub kelly_curve_a: AtomicF64,
    pub kelly_curve_b: AtomicF64,
    pub trail_mult_curve_a: AtomicF64,
    pub trail_mult_curve_b: AtomicF64,
    pub trail_act_curve_a: AtomicF64,
    pub trail_act_curve_b: AtomicF64,
    pub trail_step_curve_a: AtomicF64,
    pub trail_step_curve_b: AtomicF64,
    pub obi_curve_a: AtomicF64,
    pub obi_curve_b: AtomicF64,

    /// F1-073: Época del Seqlock para snapshots atómicos de curvas de horizonte.
    /// Par = consistente/inactivo; Impar = escritor actualizando curvas.
    pub genome_epoch: std::sync::atomic::AtomicU64,

    // --- Meta-Consejo Deliberativo (Ola Ω95 / Ω111) ---
    pub council_approval_threshold: AtomicF64,
    pub council_supermajority_override: AtomicF64,
    pub council_override_signal_floor: AtomicF64,
    pub council_override_penalty: AtomicF64,
    pub council_shrinkage_k: AtomicF64,
    pub council_cascade_severity_breaker: AtomicF64,
    pub council_utility_veto_floor: AtomicF64,
    pub council_utility_veto_wr: AtomicF64,
    pub council_fast_slippage_floor: AtomicF64,
    pub council_slow_slippage_span: AtomicF64,
    pub council_max_drawdown_limit: AtomicF64,
    pub council_streak_decay_rate: AtomicF64,
    pub council_causal_veto_threshold: AtomicF64,
    pub council_causal_min_book_imbalance: AtomicF64,
    pub council_causal_aligned_max_risk: AtomicF64,
    pub council_hodge_curl_attenuation: AtomicF64,
    pub council_hodge_laminar_floor: AtomicF64,
    pub council_ym_ref_current: AtomicF64,
    pub council_ym_max_weight: AtomicF64,
    pub council_ml_dead_zone: AtomicF64,
    pub council_ml_full_edge: AtomicF64,
    pub council_meta_dd_penalty_mult: AtomicF64,
    pub council_meta_max_dd_penalty: AtomicF64,
    pub council_meta_majority_factor: AtomicF64,
    pub council_meta_solitary_factor: AtomicF64,
    pub council_whale_z_threshold: AtomicF64,
    pub council_whale_min_floor: AtomicF64,
    pub council_liq_decay_mult: AtomicF64,
    pub council_liq_min_floor: AtomicF64,
    pub council_crowd_imbalance_decay: AtomicF64,
    pub council_crowd_min_floor: AtomicF64,
    pub council_devil_advocate_weight: AtomicF64,
    pub council_discrepancy_veto_threshold: AtomicF64,
    pub council_tracker_weight_floor: AtomicF64,
    pub council_tracker_weight_cap: AtomicF64,
    pub online_learning_rate: AtomicF64,
    pub online_momentum: AtomicF64,
    pub online_q_noise: AtomicF64,
    pub online_r_noise: AtomicF64,
    pub online_l2_decay: AtomicF64,

    // --- Tribunal de Autoengaño y Auditor Interno (Ola Ω120) ---
    pub tribunal_autoengaño_threshold: AtomicF64,
    pub tribunal_culpable_peligroso_threshold: AtomicF64,
    pub tribunal_culpable_recuperable_threshold: AtomicF64,
    pub tribunal_overfitting_recent_min: AtomicF64,
    pub tribunal_overfitting_oos_max: AtomicF64,

    // --- Navier-Stokes e Hidrodinámica (Ola Ω121) ---
    pub navier_critical_laminar: AtomicF64,
    pub navier_critical_turbulent: AtomicF64,
    pub navier_relaxation_time_s: AtomicF64,
    pub ofi_ewma_alpha: AtomicF64,

    // --- Física Cuántica, Gauge y Topología de Flujos (Ola Ω122) ---
    pub feynman_hbar: AtomicF64,
    pub feynman_mass: AtomicF64,
    pub feynman_k_spring: AtomicF64,
    pub feynman_coherence_threshold: AtomicF64,
    pub yang_mills_adaptation_rate: AtomicF64,
    pub hodge_return_sensitivity: AtomicF64,

    // --- Entropía no-extensiva, Oscilador Cuántico y Ondas de Choque (Ola Ω123) ---
    pub tsallis_q: AtomicF64,
    pub renyi_alpha: AtomicF64,
    pub quantum_osc_k: AtomicF64,
    pub quantum_osc_lambda: AtomicF64,
    pub quantum_osc_alpha: AtomicF64,
    pub shockwave_sonic_speed_z: AtomicF64,
    pub shockwave_mach_threshold: AtomicF64,

    // --- Online PPO y Neuro-Plasticidad (Ola Ω125) ---
    pub ppo_alpha_ema: AtomicF64,
    pub ppo_learning_rate: AtomicF64,
    pub plasticity_min_mult: AtomicF64,
    pub plasticity_max_mult: AtomicF64,
    pub plasticity_steepness: AtomicF64,

    // --- Hawkes-Bessel, Perceptrón y Proyección Espectral (Ola Ω126) ---
    pub hawkes_alpha: AtomicF64,
    pub hawkes_beta: AtomicF64,
    pub perceptron_ganancia: AtomicF64,
    pub espectral_masa_minima: AtomicF64,
    pub espectral_masa_plena: AtomicF64,

    // --- Universo Espectral Multiactivo y Régimen Continuo (Ola Ω127) ---
    pub espectral_recencia_x: AtomicF64,
    pub espectral_olvido: AtomicF64,
    pub espectral_muestras_maduras: AtomicF64,
    pub espectral_familia_veto: AtomicF64,
    /// F1-078: Flag 1.0 (activo) / 0.0 (inactivo) para GRAPA adaptive lambda en martingalas de Ville.
    pub espectral_lambda_adaptativa: AtomicF64,
    pub regime_w_accel_fast: AtomicF64,
    pub regime_w_adverse_tide: AtomicF64,
    pub regime_w_coherence_collapse: AtomicF64,
    pub regime_w_micro_anti: AtomicF64,
    pub regime_cadence_regime_ms: AtomicF64,
    pub regime_min_margin_floor: AtomicF64,

    // --- Multifractal Multiescala y Microestructura Continua (Ola Ω128) ---
    pub multifractal_w_micro: AtomicF64,
    pub multifractal_w_meso: AtomicF64,
    pub multifractal_w_macro: AtomicF64,
    pub multifractal_tanh_scale: AtomicF64,
    pub burst_threshold_zscore: AtomicF64,

    // --- VECM Cointegración, Ondas Solitónicas y Resonancia Estocástica (Ola Ω129) ---
    pub vecm_z_entry: AtomicF64,
    pub vecm_z_saturation: AtomicF64,
    pub soliton_sat_momento: AtomicF64,
    pub stochastic_res_peak_gain: AtomicF64,

    // --- Dinámica de Micro-Apalancamiento y Coherencia Espectral (Ola Ω130) ---
    pub micro_lev_base: AtomicF64,
    pub micro_lev_bonus: AtomicF64,
    pub micro_lev_conf_center: AtomicF64,
    pub micro_lev_conf_width: AtomicF64,
    pub macro_lev_cap: AtomicF64,
    pub base_micro_ratio: AtomicF64,
    pub spec_coh_scale: AtomicF64,
    pub spec_ent_scale: AtomicF64,

    // --- Liquidez Oculta y Compuerta Conformal de Microestructura (Ola Ω133) ---
    pub hidden_liquidity_weight: AtomicF64,
    pub conformal_gate_strictness: AtomicF64,

    // --- Gobernanza de Detector LG-22 e-CUSUM (F1-040, Ola Ω203) ---
    pub lg22_arl: AtomicF64,
}

impl QuantumConfig {
    /// Observed symbol rates take precedence in either mode. With required
    /// evidence, a missing book/slot or reordered symbol never falls back to
    /// another symbol's rates or the legacy scenario scalars.
    #[inline]
    pub fn fee_rates_for(&self, coin_id: usize) -> Option<(f64, f64)> {
        if let Some(rates) = self
            .live_fee_schedule
            .load()
            .as_ref()
            .and_then(|schedule| schedule.rates_for(coin_id))
        {
            return Some(rates);
        }
        if self.observed_fees_required.load(Ordering::Acquire) {
            return None;
        }
        let maker = self.live_maker_fee.load(Ordering::Acquire);
        let taker = self.live_taker_fee.load(Ordering::Acquire);
        (maker.is_finite() && taker.is_finite()).then_some((maker, taker))
    }

    pub fn new(initial_capital: f64) -> Self {
        let genome = crate::genome::SuperGenotype::load_or_default();
        Self::from_genome(initial_capital, &genome)
    }

    /// D-650 (DÉCIMA OLA) — CONSTRUCTOR EXPLÍCITO DESDE UN GENOMA.
    ///
    /// Antes el arranque en frío cargaba el genoma de disco DENTRO de `new()`,
    /// de modo que no existía forma de construir una configuración a partir de
    /// un genoma dado. Eso hacía imposible comparar este camino con el del
    /// hot-swap (`apply_to_arena`) y es la razón por la que los 12 genes que
    /// sólo se inicializaban aquí pasaron desapercibidos.
    ///
    /// Con el constructor explícito, el test T-2 puede exigir que ambos
    /// caminos produzcan el MISMO estado gen a gen.
    pub fn from_genome(initial_capital: f64, genome: &crate::genome::SuperGenotype) -> Self {
        // D-683: curvas continuas derivadas de los genes (ver `apply_to_arena`).
        let curvas = genome.with_synced_continuous_curves();
        let genome = genome.clone();
        Self {
            base_capital: AtomicF64::new(initial_capital),
            min_notional: AtomicF64::new(5.0),
            live_maker_fee: AtomicF64::new(0.0002), // Fracción (VIP0 Maker: 2 bps = 0.02%)
            live_taker_fee: AtomicF64::new(0.0005), // Fracción (VIP0 Taker: 5 bps = 0.05%)
            live_fee_schedule: ArcSwapOption::empty(),
            observed_fees_required: AtomicBool::new(false),
            entry_publication_gate: std::sync::Mutex::new(()),
            entry_publication_paused: AtomicBool::new(false),
            global_max_drawdown: AtomicF64::new(genome.global_max_drawdown),
            global_leverage: AtomicF64::new(genome.global_leverage),
            btc_volatility_multiplier: AtomicF64::new(genome.btc_volatility_multiplier),
            eth_volatility_multiplier: AtomicF64::new(genome.eth_volatility_multiplier),
            min_trades_per_day: AtomicF64::new(genome.min_trades_per_day),
            survival_capital_threshold: AtomicF64::new(genome.survival_capital_threshold),
            funding_rate_sensitivity: AtomicF64::new(genome.funding_rate_sensitivity),
            global_correlation_threshold: AtomicF64::new(genome.global_correlation_threshold),
            trend_threshold: AtomicF64::new(crate::genome::SuperGenotype::clamp_slot(
                genome.trend_threshold,
                crate::genome::SuperGenotype::SLOT_TREND_THRESHOLD,
            )),
            range_threshold: AtomicF64::new(genome.range_threshold),
            scalp_kelly_fraction: AtomicF64::new(genome.scalp_kelly_fraction),
            swing_kelly_fraction: AtomicF64::new(genome.swing_kelly_fraction),
            scalp_obi_threshold: AtomicF64::new(genome.scalp_obi_threshold),
            scalp_tp_base: AtomicF64::new(genome.scalp_tp_base),
            scalp_sl_base: AtomicF64::new(genome.scalp_sl_base),
            swing_tp_base: AtomicF64::new(genome.swing_tp_base),
            swing_sl_base: AtomicF64::new(genome.swing_sl_base),
            sl_atr_mult_btc: AtomicF64::new(genome.sl_atr_mult_btc),
            tp_rr_ratio_btc: AtomicF64::new(genome.tp_rr_ratio_btc),
            min_confidence_btc: AtomicF64::new(crate::genome::SuperGenotype::clamp_slot(
                genome.min_confidence_btc,
                crate::genome::SuperGenotype::SLOT_MIN_CONFIDENCE,
            )),
            veto_threshold_btc: AtomicF64::new(genome.veto_threshold_btc),
            tech_threshold: AtomicF64::new(crate::genome::SuperGenotype::clamp_slot(
                genome.tech_threshold,
                crate::genome::SuperGenotype::SLOT_TECH_THRESHOLD,
            )),
            ml_threshold_long: AtomicF64::new(genome.ml_threshold_long),
            ml_threshold_short: AtomicF64::new(genome.ml_threshold_short),
            maker_spread_pct: AtomicF64::new(genome.maker_spread_pct),
            maker_obi_threshold: AtomicF64::new(genome.maker_obi_threshold),
            target_volatility: AtomicF64::new(genome.target_volatility),
            dynamic_atr_min: AtomicF64::new(genome.dynamic_atr_min),
            dynamic_obi_threshold: AtomicF64::new(genome.dynamic_obi_threshold),
            dynamic_ema_trend: AtomicF64::new(genome.dynamic_ema_trend),
            dynamic_ofi_threshold: AtomicF64::new(genome.dynamic_ofi_threshold),
            capital_split_scalp: AtomicF64::new(genome.capital_split_scalp),
            kelly_clamp_min: AtomicF64::new(genome.kelly_clamp_min),
            kelly_clamp_max: AtomicF64::new(genome.kelly_clamp_max),
            explosive_leverage_multiplier: AtomicF64::new(genome.explosive_leverage_multiplier),
            quantum_mutation_rate: AtomicF64::new(genome.quantum_mutation_rate),
            temporal_memory_decay: AtomicF64::new(genome.temporal_memory_decay),
            leverage_cap: AtomicF64::new(genome.leverage_cap),
            explosive_confidence_threshold: AtomicF64::new(genome.explosive_confidence_threshold),
            weight_obi: AtomicF64::new(genome.weight_obi),
            weight_ofi: AtomicF64::new(genome.weight_ofi),
            weight_vpin: AtomicF64::new(genome.weight_vpin),
            regime_duration_ms: AtomicF64::new(genome.regime_duration_ms),
            regime_atr_multiplier: AtomicF64::new(genome.regime_atr_multiplier),
            scalp_trail_act_atr: AtomicF64::new(genome.scalp_trail_act_atr),
            scalp_trail_step_atr: AtomicF64::new(genome.scalp_trail_step_atr),
            scalp_trail_max_atr: AtomicF64::new(genome.scalp_trail_max_atr),
            scalp_trail_min_pnl: AtomicF64::new(genome.scalp_trail_min_pnl),
            scalp_trail_atr_mult_base: AtomicF64::new(genome.scalp_trail_atr_mult_base),
            swing_trail_act_atr: AtomicF64::new(genome.swing_trail_act_atr),
            swing_trail_step_atr: AtomicF64::new(genome.swing_trail_step_atr),
            swing_trail_max_atr: AtomicF64::new(genome.swing_trail_max_atr),
            swing_trail_min_pnl: AtomicF64::new(genome.swing_trail_min_pnl),
            swing_trail_atr_mult_base: AtomicF64::new(genome.swing_trail_atr_mult_base),
            zombie_timeout_ms: AtomicF64::new(crate::genome::SuperGenotype::clamp_slot(
                genome.zombie_timeout_ms,
                crate::genome::SuperGenotype::SLOT_ZOMBIE_TIMEOUT,
            )),
            hurst_trend_threshold: AtomicF64::new(genome.hurst_trend_threshold),
            cvd_veto_threshold: AtomicF64::new(genome.cvd_veto_threshold),
            wall_veto_threshold: AtomicF64::new(genome.wall_veto_threshold),
            flash_crash_jump_pct: AtomicF64::new(genome.flash_crash_jump_pct),
            latency_ms_panic_threshold: AtomicF64::new(genome.latency_ms_panic_threshold),
            ema_fast_period: AtomicF64::new(genome.ema_fast_period),
            ema_slow_period: AtomicF64::new(genome.ema_slow_period),
            hurst_scalp_threshold: AtomicF64::new(genome.hurst_scalp_threshold),
            hurst_swing_threshold: AtomicF64::new(genome.hurst_swing_threshold),
            synergy_exposure_boost: AtomicF64::new(genome.synergy_exposure_boost),
            synergy_leverage_boost: AtomicF64::new(genome.synergy_leverage_boost),
            max_fee_pct: AtomicF64::new(genome.max_fee_pct),
            kelly_bootstrap_cold: AtomicF64::new(genome.kelly_bootstrap_cold),
            latency_penalty_ms: AtomicF64::new(genome.latency_penalty_ms),
            base_slippage_floor: AtomicF64::new(genome.base_slippage_floor),
            spot_spread_threshold: AtomicF64::new(genome.spot_spread_threshold),
            spot_bias_value: AtomicF64::new(genome.spot_bias_value),
            obi_confidence_fallback: AtomicF64::new(genome.obi_confidence_fallback),
            ml_clip_lower: AtomicF64::new(genome.ml_clip_lower),
            ml_clip_upper: AtomicF64::new(genome.ml_clip_upper),
            tensor_poly_a: AtomicF64::new(genome.tensor_poly_a),
            tensor_poly_b: AtomicF64::new(genome.tensor_poly_b),
            fractional_alpha_order: AtomicF64::new(genome.fractional_alpha_order),
            fractional_clip_max: AtomicF64::new(genome.fractional_clip_max),
            bft_consensus_tolerance: AtomicF64::new(genome.bft_consensus_tolerance),
            turbo_coherence_threshold: AtomicF64::new(genome.turbo_coherence_threshold),
            turbo_z_score_stdev: AtomicF64::new(genome.turbo_z_score_stdev),
            sl_atr_multiplier: AtomicF64::new(genome.sl_atr_multiplier),
            coaxial_squeeze_threshold: AtomicF64::new(genome.coaxial_squeeze_threshold),
            tensor_op_add_bias: AtomicF64::new(genome.tensor_op_add_bias),
            tensor_op_mul_weight: AtomicF64::new(genome.tensor_op_mul_weight),
            topo_layer_1_activation: AtomicF64::new(genome.topo_layer_1_activation),
            topo_layer_2_activation: AtomicF64::new(genome.topo_layer_2_activation),
            tensor_dropout_rate: AtomicF64::new(genome.tensor_dropout_rate),
            quantum_entropy_seed: AtomicF64::new(genome.quantum_entropy_seed),
            global_learning_rate: AtomicF64::new(genome.global_learning_rate),
            global_momentum: AtomicF64::new(genome.global_momentum),
            vecm_alpha_speed: AtomicF64::new(genome.vecm_alpha_speed),
            vecm_beta_hedge: AtomicF64::new(genome.vecm_beta_hedge),
            conformal_alpha: AtomicF64::new(genome.conformal_alpha),
            ppo_weight_ofi: AtomicF64::new(genome.ppo_weight_ofi),
            ppo_weight_obi: AtomicF64::new(genome.ppo_weight_obi),
            ppo_weight_hawkes: AtomicF64::new(genome.ppo_weight_hawkes),
            ppo_weight_leadlag: AtomicF64::new(genome.ppo_weight_leadlag),
            ppo_weight_regime: AtomicF64::new(genome.ppo_weight_regime),
            ppo_clip_eps: AtomicF64::new(genome.ppo_clip_eps),
            ppo_weight_min_clip: AtomicF64::new(genome.ppo_weight_min_clip),
            hard_stop_decay_factor: AtomicF64::new(genome.hard_stop_decay_factor),
            hard_stop_base_limit: AtomicF64::new(genome.hard_stop_base_limit),
            kelly_bootstrap_ratio_threshold: AtomicF64::new(genome.kelly_bootstrap_ratio_threshold),
            kelly_bootstrap_min_exposure: AtomicF64::new(genome.kelly_bootstrap_min_exposure),
            ev_fee_multiplier: AtomicF64::new(genome.ev_fee_multiplier),
            margin_cushion_pct: AtomicF64::new(crate::genome::SuperGenotype::clamp_slot(
                genome.margin_cushion_pct,
                crate::genome::SuperGenotype::SLOT_MARGIN_CUSHION,
            )),
            maker_only_capital_threshold: AtomicF64::new(genome.maker_only_capital_threshold),
            hawkes_scalp_threshold: AtomicF64::new(genome.hawkes_scalp_threshold),
            obi_zscore_threshold: AtomicF64::new(genome.obi_zscore_threshold),

            hawkes_volume_norm: AtomicF64::new(genome.hawkes_volume_norm),
            base_duration_ms: AtomicF64::new(genome.base_duration_ms),

            kelly_survival_cap_ratio: AtomicF64::new(genome.kelly_survival_cap_ratio),
            kelly_expansion_mult: AtomicF64::new(genome.kelly_expansion_mult),
            guard_dd_sigmoid_steepness: AtomicF64::new(genome.guard_dd_sigmoid_steepness),
            guard_dd_sigmoid_center: AtomicF64::new(genome.guard_dd_sigmoid_center),
            portfolio_perf_mult_steepness: AtomicF64::new(genome.portfolio_perf_mult_steepness),
            portfolio_dd_penalty_decay: AtomicF64::new(genome.portfolio_dd_penalty_decay),
            portfolio_perf_mult_min: AtomicF64::new(genome.portfolio_perf_mult_min),
            portfolio_perf_mult_max: AtomicF64::new(genome.portfolio_perf_mult_max),
            portfolio_perf_mult_center: AtomicF64::new(genome.portfolio_perf_mult_center),

            macro_hurst_confidence_offset: AtomicF64::new(genome.macro_hurst_confidence_offset),
            macro_hurst_confidence_scale: AtomicF64::new(genome.macro_hurst_confidence_scale),
            macro_vol_confidence_scale: AtomicF64::new(genome.macro_vol_confidence_scale),
            macro_min_cooldown_ratio: AtomicF64::new(genome.macro_min_cooldown_ratio),
            macro_max_cooldown_ratio: AtomicF64::new(genome.macro_max_cooldown_ratio),
            macro_cooldown_reduction_factor: AtomicF64::new(genome.macro_cooldown_reduction_factor),
            macro_leverage_momentum_scale: AtomicF64::new(genome.macro_leverage_momentum_scale),
            lev_matrix_vol_clamp_min: AtomicF64::new(genome.lev_matrix_vol_clamp_min),
            lev_matrix_growth_scalar: AtomicF64::new(genome.lev_matrix_growth_scalar),
            lev_matrix_log_cap_divisor: AtomicF64::new(genome.lev_matrix_log_cap_divisor),
            scalp_accel_min_samples: AtomicF64::new(genome.scalp_accel_min_samples),
            executor_max_orders_10s: AtomicF64::new(genome.executor_max_orders_10s),
            executor_max_weight_1m: AtomicF64::new(genome.executor_max_weight_1m),
            iceberg_volume_threshold: AtomicF64::new(genome.iceberg_volume_threshold),
            iceberg_slice_count: AtomicF64::new(genome.iceberg_slice_count),
            swing_obi_threshold: AtomicF64::new(genome.swing_obi_threshold),
            swing_accel_min_samples: AtomicF64::new(genome.swing_accel_min_samples),
            // D-603 (DÉCIMA OLA) — FUENTE ÚNICA PARA `temporal_scale`.
            //
            // Antes el arranque en frío lo derivaba de `1 − capital_split_scalp`
            // —una fracción de ASIGNACIÓN DE CAPITAL— mientras el hot-swap
            // escribía el gen `temporal_scale`. Dos expresiones distintas para
            // el mismo eje según el camino de código.
            //
            // Peor: `current_from_arena()` lee de vuelta este valor HACIA el
            // gen, de modo que el ciclo Genoma → from_genome → current_from_arena
            // DESTRUÍA irreversiblemente `temporal_scale`, sustituyéndolo por
            // una función de otro gen. La evolución de ese gen era inútil:
            // cada ciclo lo borraba. Y como el backtest construye arenas
            // frescas y producción hace hot-swap, ambos entornos operaban con
            // valores de `s` distintos para el mismo genoma.
            //
            // El acoplamiento con `capital_split_scalp` era además un error de
            // categoría: una fracción de capital no es un horizonte temporal.
            temporal_scale: AtomicF64::new(genome.temporal_scale.clamp(0.05, 0.95)),

            // FASE 23: Curvas de Horizonte Continuas (1 ns a 100 años)
            tp_curve_a: AtomicF64::new(genome.tp_horizon_curve.a),
            tp_curve_b: AtomicF64::new(genome.tp_horizon_curve.b),
            sl_curve_a: AtomicF64::new(genome.sl_horizon_curve.a),
            sl_curve_b: AtomicF64::new(genome.sl_horizon_curve.b),
            kelly_curve_a: AtomicF64::new(curvas.kelly_horizon_curve.a),
            kelly_curve_b: AtomicF64::new(curvas.kelly_horizon_curve.b),
            trail_mult_curve_a: AtomicF64::new(curvas.trail_mult_horizon_curve.a),
            trail_mult_curve_b: AtomicF64::new(curvas.trail_mult_horizon_curve.b),
            trail_act_curve_a: AtomicF64::new(curvas.trail_act_horizon_curve.a),
            trail_act_curve_b: AtomicF64::new(curvas.trail_act_horizon_curve.b),
            trail_step_curve_a: AtomicF64::new(curvas.trail_step_horizon_curve.a),
            trail_step_curve_b: AtomicF64::new(curvas.trail_step_horizon_curve.b),
            obi_curve_a: AtomicF64::new(curvas.obi_horizon_curve.a),
            obi_curve_b: AtomicF64::new(curvas.obi_horizon_curve.b),
            genome_epoch: std::sync::atomic::AtomicU64::new(0),

            // --- Meta-Consejo Deliberativo (Ola Ω95 / Ω111) ---
            council_approval_threshold: AtomicF64::new(0.35),
            council_supermajority_override: AtomicF64::new(0.80),
            council_override_signal_floor: AtomicF64::new(0.28),
            council_override_penalty: AtomicF64::new(0.75),
            council_shrinkage_k: AtomicF64::new(8.0),
            council_cascade_severity_breaker: AtomicF64::new(0.85),
            council_utility_veto_floor: AtomicF64::new(0.02),
            council_utility_veto_wr: AtomicF64::new(0.35),
            council_fast_slippage_floor: AtomicF64::new(35.0),
            council_slow_slippage_span: AtomicF64::new(65.0),
            council_max_drawdown_limit: AtomicF64::new(CANONICAL_COUNCIL_MAX_DRAWDOWN),
            council_streak_decay_rate: AtomicF64::new(0.50),
            council_causal_veto_threshold: AtomicF64::new(0.75),
            council_causal_min_book_imbalance: AtomicF64::new(0.25),
            council_causal_aligned_max_risk: AtomicF64::new(0.88),
            council_hodge_curl_attenuation: AtomicF64::new(0.70),
            council_hodge_laminar_floor: AtomicF64::new(0.30),
            council_ym_ref_current: AtomicF64::new(0.05),
            council_ym_max_weight: AtomicF64::new(0.20),
            council_ml_dead_zone: AtomicF64::new(0.05),
            council_ml_full_edge: AtomicF64::new(0.20),
            council_meta_dd_penalty_mult: AtomicF64::new(5.0),
            council_meta_max_dd_penalty: AtomicF64::new(0.50),
            council_meta_majority_factor: AtomicF64::new(0.70),
            council_meta_solitary_factor: AtomicF64::new(0.40),
            council_whale_z_threshold: AtomicF64::new(4.0),
            council_whale_min_floor: AtomicF64::new(0.60),
            council_liq_decay_mult: AtomicF64::new(1.2),
            council_liq_min_floor: AtomicF64::new(0.40),
            council_crowd_imbalance_decay: AtomicF64::new(0.25),
            council_crowd_min_floor: AtomicF64::new(0.70),
            council_devil_advocate_weight: AtomicF64::new(3.0),
            council_discrepancy_veto_threshold: AtomicF64::new(0.20),
            council_tracker_weight_floor: AtomicF64::new(0.50),
            council_tracker_weight_cap: AtomicF64::new(2.0),
            online_learning_rate: AtomicF64::new(0.001),
            online_momentum: AtomicF64::new(0.90),
            online_q_noise: AtomicF64::new(1e-4),
            online_r_noise: AtomicF64::new(1e-2),
            online_l2_decay: AtomicF64::new(0.99999),
            tribunal_autoengaño_threshold: AtomicF64::new(0.20),
            tribunal_culpable_peligroso_threshold: AtomicF64::new(0.10),
            tribunal_culpable_recuperable_threshold: AtomicF64::new(0.05),
            tribunal_overfitting_recent_min: AtomicF64::new(0.05),
            tribunal_overfitting_oos_max: AtomicF64::new(-0.01),
            navier_critical_laminar: AtomicF64::new(1.0),
            navier_critical_turbulent: AtomicF64::new(5.0),
            navier_relaxation_time_s: AtomicF64::new(0.200),
            ofi_ewma_alpha: AtomicF64::new(0.10),
            feynman_hbar: AtomicF64::new(CANONICAL_FEYNMAN_HBAR),
            feynman_mass: AtomicF64::new(CANONICAL_FEYNMAN_MASS),
            feynman_k_spring: AtomicF64::new(CANONICAL_FEYNMAN_K_SPRING),
            feynman_coherence_threshold: AtomicF64::new(0.20),
            yang_mills_adaptation_rate: AtomicF64::new(0.005),
            hodge_return_sensitivity: AtomicF64::new(100.0),
            tsallis_q: AtomicF64::new(1.5),
            renyi_alpha: AtomicF64::new(2.0),
            quantum_osc_k: AtomicF64::new(1.0),
            quantum_osc_lambda: AtomicF64::new(0.1),
            quantum_osc_alpha: AtomicF64::new(0.5),
            shockwave_sonic_speed_z: AtomicF64::new(1.0),
            shockwave_mach_threshold: AtomicF64::new(1.0),
            ppo_alpha_ema: AtomicF64::new(0.05),
            ppo_learning_rate: AtomicF64::new(0.01),
            plasticity_min_mult: AtomicF64::new(0.20),
            plasticity_max_mult: AtomicF64::new(2.00),
            plasticity_steepness: AtomicF64::new(6.00),
            hawkes_alpha: AtomicF64::new(0.3),
            hawkes_beta: AtomicF64::new(0.5),
            perceptron_ganancia: AtomicF64::new(2.5),
            espectral_masa_minima: AtomicF64::new(0.20),
            espectral_masa_plena: AtomicF64::new(0.35),
            espectral_recencia_x: AtomicF64::new(0.5),
            espectral_olvido: AtomicF64::new(1.0 / 64.0),
            espectral_muestras_maduras: AtomicF64::new(30.0),
            espectral_familia_veto: AtomicF64::new(0.0), // 0.0 = auto-dimensionado honesto según roster vivo C(n,2)·5
            espectral_lambda_adaptativa: AtomicF64::new(1.0), // F1-078: Activo por defecto para acelerar convergencia bajo evidencia
            regime_w_accel_fast: AtomicF64::new(0.35),
            regime_w_adverse_tide: AtomicF64::new(0.30),
            regime_w_coherence_collapse: AtomicF64::new(0.20),
            regime_w_micro_anti: AtomicF64::new(0.15),
            regime_cadence_regime_ms: AtomicF64::new(60_000.0),
            regime_min_margin_floor: AtomicF64::new(0.05),
            multifractal_w_micro: AtomicF64::new(0.40),
            multifractal_w_meso: AtomicF64::new(0.30),
            multifractal_w_macro: AtomicF64::new(0.30),
            multifractal_tanh_scale: AtomicF64::new(0.08),
            burst_threshold_zscore: AtomicF64::new(2.50),
            vecm_z_entry: AtomicF64::new(1.50),
            vecm_z_saturation: AtomicF64::new(3.00),
            soliton_sat_momento: AtomicF64::new(10000.0),
            stochastic_res_peak_gain: AtomicF64::new(2.00),
            micro_lev_base: AtomicF64::new(5.0),
            micro_lev_bonus: AtomicF64::new(1.5),
            micro_lev_conf_center: AtomicF64::new(0.65),
            micro_lev_conf_width: AtomicF64::new(0.10),
            macro_lev_cap: AtomicF64::new(50.0),
            base_micro_ratio: AtomicF64::new(0.66 / 0.62),
            spec_coh_scale: AtomicF64::new(0.10),
            spec_ent_scale: AtomicF64::new(0.30),
            hidden_liquidity_weight: AtomicF64::new(0.30),
            conformal_gate_strictness: AtomicF64::new(1.0),
            lg22_arl: AtomicF64::new(200.0),
        }
    }

    /// F1-073: Inicia una sección crítica de escritura de Seqlock para actualizar curvas.
    /// Incrementa `genome_epoch` a un número impar con barrera Acquire-Release.
    #[inline(always)]
    pub fn begin_write(&self) {
        use std::sync::atomic::Ordering;
        loop {
            let e = self.genome_epoch.load(Ordering::Relaxed);
            if e & 1 != 0 {
                // Otro escritor está activo, girar brevemente
                std::hint::spin_loop();
                continue;
            }
            if self
                .genome_epoch
                .compare_exchange_weak(e, e.wrapping_add(1), Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            std::hint::spin_loop();
        }
    }

    /// F1-073: Finaliza una sección crítica de escritura de Seqlock.
    /// Incrementa `genome_epoch` a un número par con barrera Release.
    #[inline(always)]
    pub fn end_write(&self) {
        use std::sync::atomic::Ordering;
        let e = self.genome_epoch.load(Ordering::Relaxed);
        if e & 1 != 0 {
            self.genome_epoch.store(e.wrapping_add(1), Ordering::Release);
        } else {
            self.genome_epoch.fetch_add(2, Ordering::Release);
        }
    }

    /// F1-073: Snapshot atómico libre de quimeras para la curva de TP.
    #[inline(always)]
    pub fn tp_curve(&self) -> HorizonCurve {
        use std::sync::atomic::Ordering;
        loop {
            let e1 = self.genome_epoch.load(Ordering::Acquire);
            if e1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let a = self.tp_curve_a.load(Ordering::Relaxed);
            let b = self.tp_curve_b.load(Ordering::Relaxed);
            let e2 = self.genome_epoch.load(Ordering::Acquire);
            if e1 == e2 {
                return HorizonCurve { a, b };
            }
            std::hint::spin_loop();
        }
    }

    /// F1-073: Snapshot atómico libre de quimeras para la curva de SL.
    #[inline(always)]
    pub fn sl_curve(&self) -> HorizonCurve {
        use std::sync::atomic::Ordering;
        loop {
            let e1 = self.genome_epoch.load(Ordering::Acquire);
            if e1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let a = self.sl_curve_a.load(Ordering::Relaxed);
            let b = self.sl_curve_b.load(Ordering::Relaxed);
            let e2 = self.genome_epoch.load(Ordering::Acquire);
            if e1 == e2 {
                return HorizonCurve { a, b };
            }
            std::hint::spin_loop();
        }
    }

    /// F1-073: Snapshot atómico conjunto de TP y SL. Garantiza que TP y SL
    /// pertenezcan exactamente al mismo genoma y preserven la invariante RR.
    #[inline(always)]
    pub fn tp_sl_curves(&self) -> (HorizonCurve, HorizonCurve) {
        use std::sync::atomic::Ordering;
        loop {
            let e1 = self.genome_epoch.load(Ordering::Acquire);
            if e1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let tp_a = self.tp_curve_a.load(Ordering::Relaxed);
            let tp_b = self.tp_curve_b.load(Ordering::Relaxed);
            let sl_a = self.sl_curve_a.load(Ordering::Relaxed);
            let sl_b = self.sl_curve_b.load(Ordering::Relaxed);
            let e2 = self.genome_epoch.load(Ordering::Acquire);
            if e1 == e2 {
                return (HorizonCurve { a: tp_a, b: tp_b }, HorizonCurve { a: sl_a, b: sl_b });
            }
            std::hint::spin_loop();
        }
    }

    /// F1-073: Snapshot atómico de la curva continua de Kelly.
    #[inline(always)]
    pub fn kelly_curve(&self) -> HorizonCurve {
        use std::sync::atomic::Ordering;
        loop {
            let e1 = self.genome_epoch.load(Ordering::Acquire);
            if e1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let a = self.kelly_curve_a.load(Ordering::Relaxed);
            let b = self.kelly_curve_b.load(Ordering::Relaxed);
            let e2 = self.genome_epoch.load(Ordering::Acquire);
            if e1 == e2 {
                return HorizonCurve { a, b };
            }
            std::hint::spin_loop();
        }
    }

    /// F1-073: Snapshot atómico conjunto de las curvas de Trailing Stop continuo:
    /// `(mult_curve, act_curve, step_curve)`.
    #[inline(always)]
    pub fn trail_curves(&self) -> (HorizonCurve, HorizonCurve, HorizonCurve) {
        use std::sync::atomic::Ordering;
        loop {
            let e1 = self.genome_epoch.load(Ordering::Acquire);
            if e1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let mult_a = self.trail_mult_curve_a.load(Ordering::Relaxed);
            let mult_b = self.trail_mult_curve_b.load(Ordering::Relaxed);
            let act_a = self.trail_act_curve_a.load(Ordering::Relaxed);
            let act_b = self.trail_act_curve_b.load(Ordering::Relaxed);
            let step_a = self.trail_step_curve_a.load(Ordering::Relaxed);
            let step_b = self.trail_step_curve_b.load(Ordering::Relaxed);
            let e2 = self.genome_epoch.load(Ordering::Acquire);
            if e1 == e2 {
                return (
                    HorizonCurve { a: mult_a, b: mult_b },
                    HorizonCurve { a: act_a, b: act_b },
                    HorizonCurve { a: step_a, b: step_b },
                );
            }
            std::hint::spin_loop();
        }
    }

    /// F1-073: Snapshot atómico de la curva continua de OBI.
    #[inline(always)]
    pub fn obi_curve(&self) -> HorizonCurve {
        use std::sync::atomic::Ordering;
        loop {
            let e1 = self.genome_epoch.load(Ordering::Acquire);
            if e1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let a = self.obi_curve_a.load(Ordering::Relaxed);
            let b = self.obi_curve_b.load(Ordering::Relaxed);
            let e2 = self.genome_epoch.load(Ordering::Acquire);
            if e1 == e2 {
                return HorizonCurve { a, b };
            }
            std::hint::spin_loop();
        }
    }

    /// TP bajo la política compartida runtime-v1 (evaluado con snapshot atómico).
    #[inline(always)]
    pub fn tp_at_tau(&self, tau_ms: f64) -> f64 {
        let curve = self.tp_curve();
        HorizonParameter::TakeProfit.evaluate(curve, tau_ms)
    }

    /// SL bajo runtime-v1 (evaluado con snapshot atómico libre de quimeras).
    #[inline(always)]
    pub fn sl_at_tau(&self, tau_ms: f64) -> f64 {
        let curve = self.sl_curve();
        HorizonParameter::StopLoss.evaluate(curve, tau_ms)
    }

    /// F1-073: Evalúa TP y SL simultáneamente en τ (ms) garantizando paridad RR
    /// del mismo genoma bajo snapshot atómico único.
    #[inline(always)]
    pub fn tp_sl_at_tau(&self, tau_ms: f64) -> (f64, f64) {
        let (tp_curve, sl_curve) = self.tp_sl_curves();
        (
            HorizonParameter::TakeProfit.evaluate(tp_curve, tau_ms),
            HorizonParameter::StopLoss.evaluate(sl_curve, tau_ms),
        )
    }

    /// Evalúa la fracción Kelly continua en cualquier horizonte temporal τ (ms) ∈ [1 ns, 100 años].
    #[inline(always)]
    pub fn kelly_at_tau(&self, tau_ms: f64) -> f64 {
        let curve = self.kelly_curve();
        HorizonParameter::Kelly.evaluate(curve, tau_ms)
    }

    /// Evalúa los parámetros del Trailing Stop continuo en cualquier horizonte temporal τ (ms) ∈ [1 ns, 100 años]:
    /// Retorna `(trail_mult, trail_act_atr, trail_step_atr, trail_max_atr)`.
    #[inline(always)]
    pub fn trail_params_at_tau(&self, tau_ms: f64) -> (f64, f64, f64, f64) {
        let (mult_curve, act_curve, step_curve) = self.trail_curves();
        trailing_at_tau(mult_curve, act_curve, step_curve, tau_ms)
    }

    /// Evalúa el umbral OBI continuo en cualquier horizonte temporal τ (ms) ∈ [1 ns, 100 años].
    #[inline(always)]
    pub fn obi_threshold_at_tau(&self, tau_ms: f64) -> f64 {
        let curve = self.obi_curve();
        HorizonParameter::ObiThreshold.evaluate(curve, tau_ms)
    }

    /// F1-009: Sincronización continua de parámetros adaptativos, deliberativos y físicos
    /// desde el `OmniscientRegistry` hacia los atómicos de `QuantumConfig`.
    /// Permite que los motores de optimización online, de aprendizaje por refuerzo,
    /// análisis cuántico espectral y detección de régimen modulen dinámicamente estos
    /// umbrales en tiempo real sin estados congelados ni literales estáticos.
    pub fn sync_from_registry(&self, reg: &omniscient_registry::OmniscientRegistry) {
        use std::sync::atomic::Ordering::Relaxed;

        // --- 1. Consejo Deliberativo ---
        if let Some(v) = reg.get_value_fast("council_approval_threshold") {
            if (0.0..=1.0).contains(&v) { self.council_approval_threshold.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_supermajority_override") {
            if (0.0..=1.0).contains(&v) { self.council_supermajority_override.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_override_signal_floor") {
            if (0.0..=1.0).contains(&v) { self.council_override_signal_floor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_override_penalty") {
            if (0.0..=1.0).contains(&v) { self.council_override_penalty.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_shrinkage_k") {
            if v.is_finite() && v > 0.0 { self.council_shrinkage_k.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_cascade_severity_breaker") {
            if (0.0..=1.0).contains(&v) { self.council_cascade_severity_breaker.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_utility_veto_floor") {
            if (0.0..=0.50).contains(&v) { self.council_utility_veto_floor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_utility_veto_wr") {
            if (0.0..=1.0).contains(&v) { self.council_utility_veto_wr.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_fast_slippage_floor") {
            if v.is_finite() && v > 0.0 { self.council_fast_slippage_floor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_slow_slippage_span") {
            if v.is_finite() && v > 0.0 { self.council_slow_slippage_span.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_max_drawdown_limit") {
            if (0.0..=1.0).contains(&v) { self.council_max_drawdown_limit.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_streak_decay_rate") {
            if (0.0..=1.0).contains(&v) { self.council_streak_decay_rate.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_causal_veto_threshold") {
            if (0.0..=1.0).contains(&v) { self.council_causal_veto_threshold.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_causal_min_book_imbalance") {
            if (0.0..=1.0).contains(&v) { self.council_causal_min_book_imbalance.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_causal_aligned_max_risk") {
            if (0.0..=1.0).contains(&v) { self.council_causal_aligned_max_risk.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_hodge_curl_attenuation") {
            if (0.0..=1.0).contains(&v) { self.council_hodge_curl_attenuation.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_hodge_laminar_floor") {
            if (0.0..=1.0).contains(&v) { self.council_hodge_laminar_floor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_ym_ref_current") {
            if v.is_finite() && v > 0.0 { self.council_ym_ref_current.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_ym_max_weight") {
            if (0.0..=1.0).contains(&v) { self.council_ym_max_weight.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_ml_dead_zone") {
            if (0.0..=1.0).contains(&v) { self.council_ml_dead_zone.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_ml_full_edge") {
            if (0.0..=1.0).contains(&v) { self.council_ml_full_edge.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_meta_dd_penalty_mult") {
            if v.is_finite() && v >= 0.0 { self.council_meta_dd_penalty_mult.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_meta_max_dd_penalty") {
            if (0.0..=1.0).contains(&v) { self.council_meta_max_dd_penalty.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_meta_majority_factor") {
            if (0.0..=1.0).contains(&v) { self.council_meta_majority_factor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_meta_solitary_factor") {
            if (0.0..=1.0).contains(&v) { self.council_meta_solitary_factor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_whale_z_threshold") {
            if v.is_finite() && v > 0.0 { self.council_whale_z_threshold.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_whale_min_floor") {
            if (0.0..=1.0).contains(&v) { self.council_whale_min_floor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_liq_decay_mult") {
            if v.is_finite() && v > 0.0 { self.council_liq_decay_mult.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_liq_min_floor") {
            if (0.0..=1.0).contains(&v) { self.council_liq_min_floor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_crowd_imbalance_decay") {
            if (0.0..=1.0).contains(&v) { self.council_crowd_imbalance_decay.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_crowd_min_floor") {
            if (0.0..=1.0).contains(&v) { self.council_crowd_min_floor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_devil_advocate_weight") {
            if v.is_finite() && v >= 0.0 { self.council_devil_advocate_weight.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_discrepancy_veto_threshold") {
            if (0.0..=1.0).contains(&v) { self.council_discrepancy_veto_threshold.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_tracker_weight_floor") {
            if v.is_finite() && v > 0.0 { self.council_tracker_weight_floor.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("council_tracker_weight_cap") {
            if v.is_finite() && v >= 1.0 { self.council_tracker_weight_cap.store(v, Relaxed); }
        }

        // --- 2. Física de Navier-Stokes, Feynman, Tsallis, Renyi, Yang-Mills ---
        if let Some(v) = reg.get_value_fast("navier_critical_laminar") {
            if v.is_finite() && v > 0.0 { self.navier_critical_laminar.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("navier_critical_turbulent") {
            if v.is_finite() && v > 0.0 { self.navier_critical_turbulent.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("navier_relaxation_time_s") {
            if v.is_finite() && v > 0.0 { self.navier_relaxation_time_s.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("ofi_ewma_alpha") {
            if (0.0..=1.0).contains(&v) { self.ofi_ewma_alpha.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("feynman_hbar") {
            if v.is_finite() && v > 0.0 { self.feynman_hbar.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("feynman_mass") {
            if v.is_finite() && v > 0.0 { self.feynman_mass.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("feynman_k_spring") {
            if v.is_finite() && v > 0.0 { self.feynman_k_spring.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("feynman_coherence_threshold") {
            if (0.0..=1.0).contains(&v) { self.feynman_coherence_threshold.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("yang_mills_adaptation_rate") {
            if v.is_finite() && v > 0.0 { self.yang_mills_adaptation_rate.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("hodge_return_sensitivity") {
            if v.is_finite() && v > 0.0 { self.hodge_return_sensitivity.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("tsallis_q") {
            if v.is_finite() && v > 0.0 { self.tsallis_q.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("renyi_alpha") {
            if v.is_finite() && v > 0.0 { self.renyi_alpha.store(v, Relaxed); }
        }

        // --- 3. Hawkes, PPO, Plasticidad y Perceptrón ---
        if let Some(v) = reg.get_value_fast("hawkes_alpha") {
            if v.is_finite() && v >= 0.0 { self.hawkes_alpha.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("hawkes_beta") {
            if v.is_finite() && v > 0.0 { self.hawkes_beta.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("perceptron_ganancia") {
            if v.is_finite() && v > 0.0 { self.perceptron_ganancia.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("ppo_alpha_ema") {
            if (0.0..=1.0).contains(&v) { self.ppo_alpha_ema.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("plasticity_min_mult") {
            if (0.0..=1.0).contains(&v) { self.plasticity_min_mult.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("plasticity_max_mult") {
            if v.is_finite() && v >= 1.0 { self.plasticity_max_mult.store(v, Relaxed); }
        }

        // --- 4. Espectral y Régimen ---
        if let Some(v) = reg.get_value_fast("espectral_recencia_x") {
            if (0.0..=1.0).contains(&v) { self.espectral_recencia_x.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("espectral_olvido") {
            if (0.0..=1.0).contains(&v) { self.espectral_olvido.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("espectral_muestras_maduras") {
            if v.is_finite() && v >= 5.0 { self.espectral_muestras_maduras.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("regime_w_accel_fast") {
            if (0.0..=1.0).contains(&v) { self.regime_w_accel_fast.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("regime_w_adverse_tide") {
            if (0.0..=1.0).contains(&v) { self.regime_w_adverse_tide.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("multifractal_tanh_scale") {
            if v.is_finite() && v > 0.0 { self.multifractal_tanh_scale.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("burst_threshold_zscore") {
            if v.is_finite() && v > 0.0 { self.burst_threshold_zscore.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("hidden_liquidity_weight") {
            if (0.0..=1.0).contains(&v) { self.hidden_liquidity_weight.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("conformal_gate_strictness") {
            if v.is_finite() && v > 0.0 { self.conformal_gate_strictness.store(v, Relaxed); }
        }
        if let Some(v) = reg.get_value_fast("lg22_arl") {
            if v.is_finite() && v >= 10.0 { self.lg22_arl.store(v, Relaxed); }
        }
    }
}
