use std::sync::atomic::{AtomicU64, Ordering};

/// Sniffing de Mempool (RBF) y WebSockets de DEX
/// Alimenta al QuantumStateArena con señales de presión de liquidez y pánico de red.
/// Totalmente lock-free usando AtomicU64 bitcasting (f64 <-> u64) para Axioma V y VIII.
#[repr(C, align(64))]
pub struct DarkAlphaRouter {
    pub mempool_panic_score: AtomicU64,      // f64
    pub net_liq_pressure: AtomicU64,         // f64
    pub liquidation_cascade_risk: AtomicU64, // f64
    pub last_update_ts: AtomicU64,           // ms

    // Concurrency stats
    pub processed_packets: AtomicU64,
}

impl Default for DarkAlphaRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl DarkAlphaRouter {
    pub fn new() -> Self {
        Self {
            mempool_panic_score: AtomicU64::new(0f64.to_bits()),
            net_liq_pressure: AtomicU64::new(0f64.to_bits()),
            liquidation_cascade_risk: AtomicU64::new(0f64.to_bits()),
            last_update_ts: AtomicU64::new(0),
            processed_packets: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn set_mempool_panic_score(&self, score: f64) {
        self.mempool_panic_score
            .store(score.to_bits(), Ordering::Release);
    }

    #[inline(always)]
    pub fn get_mempool_panic_score(&self) -> f64 {
        f64::from_bits(self.mempool_panic_score.load(Ordering::Acquire))
    }

    #[inline(always)]
    pub fn set_net_liq_pressure(&self, pressure: f64) {
        self.net_liq_pressure
            .store(pressure.to_bits(), Ordering::Release);
    }

    #[inline(always)]
    pub fn get_net_liq_pressure(&self) -> f64 {
        f64::from_bits(self.net_liq_pressure.load(Ordering::Acquire))
    }

    #[inline(always)]
    pub fn set_liquidation_cascade_risk(&self, risk: f64) {
        self.liquidation_cascade_risk
            .store(risk.to_bits(), Ordering::Release);
    }

    #[inline(always)]
    pub fn get_liquidation_cascade_risk(&self) -> f64 {
        f64::from_bits(self.liquidation_cascade_risk.load(Ordering::Acquire))
    }

    #[inline(always)]
    pub fn inc_processed_packets(&self) {
        self.processed_packets.fetch_add(1, Ordering::Relaxed);
    }

    /// Ingresa un pulso de liquidez oscuro (MEV, Liquidaciones DEX)
    /// Aplica un decaimiento exponencial estricto O(1) basado en dt (Axioma XVIII)
    #[inline(always)]
    pub fn ingest_dex_liquidation(&self, qty: f64, impact: f64, ts_ms: u64) {
        // FIX #1452: Sanitización estricta de inputs antes del decaimiento y CAS loop
        let safe_qty = if qty.is_finite() && qty >= 0.0 {
            qty
        } else {
            0.0
        };
        let safe_impact = if impact.is_finite() {
            impact.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let last_ts = self.last_update_ts.load(Ordering::Acquire);
        let dt = if ts_ms > last_ts {
            (ts_ms - last_ts) as f64
        } else {
            0.0
        };

        // Decaimiento Exponencial: lambda = 0.001 (media vida de ~693ms)
        let lambda = 0.001;
        let decay_factor = (-lambda * dt).exp();

        // FIX #1412: CAS loop atómico para liquidation_cascade_risk (Cero Lost Updates)
        let mut curr_bits = self.liquidation_cascade_risk.load(Ordering::Acquire);
        loop {
            let curr_val = f64::from_bits(curr_bits);
            let new_val = (curr_val * decay_factor) + (safe_impact * 10.0);
            let safe_new_val = if new_val.is_finite() { new_val } else { 0.0 };
            match self.liquidation_cascade_risk.compare_exchange_weak(
                curr_bits,
                safe_new_val.to_bits(),
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => curr_bits = actual,
            }
        }

        // FIX #1412: CAS loop atómico para net_liq_pressure
        let mut curr_press_bits = self.net_liq_pressure.load(Ordering::Acquire);
        loop {
            let curr_press = f64::from_bits(curr_press_bits);
            let new_press = (curr_press * decay_factor) + (safe_qty * safe_impact);
            let safe_new_press = if new_press.is_finite() {
                new_press
            } else {
                0.0
            };
            match self.net_liq_pressure.compare_exchange_weak(
                curr_press_bits,
                safe_new_press.to_bits(),
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => curr_press_bits = actual,
            }
        }

        // Update TS only if newer
        let mut curr_ts = last_ts;
        while ts_ms > curr_ts {
            match self.last_update_ts.compare_exchange_weak(
                curr_ts,
                ts_ms,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => curr_ts = actual,
            }
        }

        self.inc_processed_packets();
    }

    // Proxy for backward compatibility with god_engine currently passing 4 params
    #[inline(always)]
    pub fn ingest_l2_snapshot(&self, qty: f64, _obi: f64, impact: f64, ts_ms: u64) {
        self.ingest_dex_liquidation(qty, impact, ts_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dark_alpha_router_atomic_get_set() {
        let router = DarkAlphaRouter::default();
        router.set_mempool_panic_score(0.85);
        router.set_net_liq_pressure(120.5);
        router.set_liquidation_cascade_risk(0.65);

        assert_eq!(router.get_mempool_panic_score(), 0.85);
        assert_eq!(router.get_net_liq_pressure(), 120.5);
        assert_eq!(router.get_liquidation_cascade_risk(), 0.65);
    }

    #[test]
    fn test_dark_alpha_router_ingest_dex_liquidation_and_nan_immunity() {
        let router = DarkAlphaRouter::new();
        router.ingest_dex_liquidation(10.0, 0.5, 1000);
        assert!(router.get_liquidation_cascade_risk() > 0.0);
        assert!(router.get_net_liq_pressure() > 0.0);

        // NaN inputs must not corrupt internal state
        router.ingest_dex_liquidation(f64::NAN, f64::NAN, 2000);
        assert!(router.get_liquidation_cascade_risk().is_finite());
        assert!(router.get_net_liq_pressure().is_finite());
    }
}
