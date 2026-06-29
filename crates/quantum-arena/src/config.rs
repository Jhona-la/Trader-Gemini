use crate::atomic_float::AtomicF64;
use std::sync::atomic::{AtomicU64, Ordering};

/// Axioma V: El Config Omnisciente (5 Capas Adaptativas)
/// Todo expuesto como atómicos para mutación lock-free (O(1)) desde el Evolver.
/// Alineado a 64-bytes para evitar False Sharing entre núcleos.
#[repr(C, align(64))]
pub struct QuantumConfig {
    // 1. Capa GLOBAL (Límites de supervivencia)
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
    pub scalp_tp_base: AtomicF64,
    pub scalp_sl_base: AtomicF64,
    pub swing_tp_base: AtomicF64,
    pub swing_sl_base: AtomicF64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            global_max_drawdown: AtomicF64::new(0.05), // 5% límite duro
            global_leverage: AtomicF64::new(50.0),     // Hasta 50x (margen micro)
            min_notional: AtomicF64::new(5.05),        // Binance minimum notional

            btc_volatility_multiplier: AtomicF64::new(1.0),
            eth_volatility_multiplier: AtomicF64::new(1.2),

            funding_rate_sensitivity: AtomicF64::new(0.5),

            trend_threshold: AtomicF64::new(0.65), // Hurst > 0.65
            range_threshold: AtomicF64::new(0.45), // Hurst < 0.45

            scalp_tp_base: AtomicF64::new(0.005), // 0.5% TP base
            scalp_sl_base: AtomicF64::new(0.002), // 0.2% SL base
            swing_tp_base: AtomicF64::new(0.030), // 3.0% TP base
            swing_sl_base: AtomicF64::new(0.010), // 1.0% SL base
        }
    }
}
