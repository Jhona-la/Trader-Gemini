/// ⚡ ALGORITMO #66: DISPARADOR ADAPTATIVO DE MICRO-SCALPING HAWKES/OBI (MICRO SCALP TRIGGER ENGINE)
/// Dispara entradas de micro-scalp de alta probabilidad calibrando dinámicamente el umbral a partir del
/// Ratio Hawkes de Auto-Excitación y Orderbook Imbalance (OBI) Z-Score en O(1).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct MicroScalpTriggerEngine;

impl MicroScalpTriggerEngine {
    /// Evalúa si las condiciones de micro-scalping de alta precisión están activas
    #[inline(always)]
    pub fn should_trigger_micro_scalp(
        arena: &quantum_arena::GlobalArena,
        hawkes_ratio: f64,
        obi_zscore: f64,
        ml_prob: f64,
        is_long: bool,
    ) -> bool {
        use std::sync::atomic::Ordering;
        let hawkes_thresh = arena.config.hawkes_scalp_threshold.load(Ordering::Relaxed);
        let obi_thresh = arena.config.obi_zscore_threshold.load(Ordering::Relaxed);
        let ml_long_thresh = arena.config.ml_threshold_long.load(Ordering::Relaxed);
        let ml_short_thresh = arena.config.ml_threshold_short.load(Ordering::Relaxed);
        
        let hawkes_ok = hawkes_ratio >= hawkes_thresh;
        let obi_ok = if is_long { obi_zscore >= obi_thresh } else { obi_zscore <= -obi_thresh };
        let ml_ok = if is_long { ml_prob >= ml_long_thresh } else { ml_prob <= ml_short_thresh };
        hawkes_ok && obi_ok && ml_ok
    }
}
