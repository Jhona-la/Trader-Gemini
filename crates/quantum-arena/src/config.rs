use crate::atomic_float::AtomicF64;


/// Axioma V: El Config Omnisciente (5 Capas Adaptativas)
/// Todo expuesto como atómicos para mutación lock-free (O(1)) desde el Evolver.
/// Alineado a 64-bytes para evitar False Sharing entre núcleos.
#[repr(C, align(64))]
pub struct QuantumConfig {
    // 1. Capa GLOBAL (Límites de supervivencia)
    pub base_capital: AtomicF64,
    pub global_max_drawdown: AtomicF64,
    pub global_leverage: AtomicF64,
    pub min_notional: AtomicF64,

    // 2. Capa ASSET CLASS
    pub btc_volatility_multiplier: AtomicF64,
    pub eth_volatility_multiplier: AtomicF64,

    // 3. Capa ASSET INDIVIDUAL (Ej: BTCUSDT)
    pub funding_rate_sensitivity: AtomicF64,
    
    // 4. Capa RÉGIMEN DE MERCADO (Trend, Range, Volatile)
    pub trend_threshold: AtomicF64,
    pub range_threshold: AtomicF64,
    
    // 5. Capa ESTRATEGIA (Scalp / Swing)
    pub scalp_kelly_fraction: AtomicF64,
    pub swing_kelly_fraction: AtomicF64,
    pub scalp_obi_threshold: AtomicF64,
    pub scalp_tp_base: AtomicF64,
    pub scalp_sl_base: AtomicF64,
    pub swing_tp_base: AtomicF64,
    pub swing_sl_base: AtomicF64,

    // --- Umbrales Dinámicos (Extraídos del Genoma) ---
    pub sl_atr_mult_btc: AtomicF64,
    pub tp_rr_ratio_btc: AtomicF64,
    pub min_confidence_btc: AtomicF64,
    pub veto_threshold_btc: AtomicF64,
    pub tech_threshold: AtomicF64,
    pub ml_threshold_long: AtomicF64,
    pub ml_threshold_short: AtomicF64,

    // --- Maker Engine (Phase 69) ---
    pub maker_spread_pct: AtomicF64,
    pub maker_obi_threshold: AtomicF64,

    // --- Umbrales Dinámicos de Micro-Estructura (Evolution Engine) ---
    pub dynamic_atr_min: AtomicF64,
    pub dynamic_obi_threshold: AtomicF64,
    pub dynamic_ema_trend: AtomicF64,
    pub dynamic_ofi_threshold: AtomicF64,

    // --- Hardcoding Eradication (Phase 67) ---
    pub capital_split_scalp: AtomicF64, // 0.0 to 1.0 (remainder is Swing)
    pub kelly_clamp_min: AtomicF64,
    pub kelly_clamp_max: AtomicF64,
    pub leverage_cap: AtomicF64,
    pub explosive_confidence_threshold: AtomicF64,
    pub explosive_leverage_multiplier: AtomicF64,
    pub sim_fee_rate: AtomicF64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            base_capital: AtomicF64::new(13.0),
            global_max_drawdown: AtomicF64::new(0.95), // GA needs survival room
            global_leverage: AtomicF64::new(30.0), // Reduced from 100 to survive 1-min synthetic candle gaps compounding with tight TP
            min_notional: AtomicF64::new(5.05),

            btc_volatility_multiplier: AtomicF64::new(1.0),
            eth_volatility_multiplier: AtomicF64::new(1.2),

            funding_rate_sensitivity: AtomicF64::new(0.5),

            trend_threshold: AtomicF64::new(0.3), // GA Supreme
            range_threshold: AtomicF64::new(0.45),

            scalp_kelly_fraction: AtomicF64::new(1.0),  // G-01: Full Kelly for Exponential Growth
            swing_kelly_fraction: AtomicF64::new(0.10), 
            scalp_obi_threshold: AtomicF64::new(0.18), // G-01: Increased from 0.08 to filter low conviction trades
            scalp_tp_base: AtomicF64::new(0.0060),    // 0.60% (30% ROE at 50x)
            scalp_sl_base: AtomicF64::new(0.0030),   // 0.30% (15% ROE at 50x)
            swing_tp_base: AtomicF64::new(0.045),    // 4.5%
            swing_sl_base: AtomicF64::new(0.015),    // 1.5%

            sl_atr_mult_btc: AtomicF64::new(1.0),
            tp_rr_ratio_btc: AtomicF64::new(4.0), // Swing RR ratio ~4.0
            min_confidence_btc: AtomicF64::new(0.58), // GA Supreme
            veto_threshold_btc: AtomicF64::new(0.65),
            tech_threshold: AtomicF64::new(0.005),
            ml_threshold_long: AtomicF64::new(0.72),
            ml_threshold_short: AtomicF64::new(0.71),

            maker_spread_pct: AtomicF64::new(0.001671), // 0.16% GA Supreme
            maker_obi_threshold: AtomicF64::new(0.59),  // GA Supreme

            dynamic_atr_min: AtomicF64::new(0.0005),
            dynamic_obi_threshold: AtomicF64::new(0.35),
            dynamic_ema_trend: AtomicF64::new(0.0002),
            dynamic_ofi_threshold: AtomicF64::new(0.20),

            capital_split_scalp: AtomicF64::new(0.90), // 90% to Scalp
            kelly_clamp_min: AtomicF64::new(0.01),
            kelly_clamp_max: AtomicF64::new(0.30), // Max 30% of account per trade
            leverage_cap: AtomicF64::new(100.0),
            explosive_confidence_threshold: AtomicF64::new(0.95),
            explosive_leverage_multiplier: AtomicF64::new(1.0), // No need to multiply if cap is 100
            sim_fee_rate: AtomicF64::new(0.0001), // Maker fee simulation (1 bp) for post-only
        }
    }
}
