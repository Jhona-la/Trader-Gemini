use std::sync::atomic::{AtomicU64, Ordering};

/// Sniffing de Mempool (RBF) y WebSockets de DEX
/// Alimenta al QuantumStateArena con señales de presión de liquidez y pánico de red.
/// Totalmente lock-free usando AtomicU64 bitcasting (f64 <-> u64) para Axioma V y VIII.
#[repr(C, align(64))]
pub struct DarkAlphaRouter {
    pub mempool_panic_score: AtomicU64, // f64
    pub net_liq_pressure: AtomicU64,    // f64
    pub liquidation_cascade_risk: AtomicU64, // f64
    pub last_update_ts: AtomicU64, // ms
    
    // Concurrency stats
    pub processed_packets: AtomicU64,
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
        self.mempool_panic_score.store(score.to_bits(), Ordering::Release);
    }

    #[inline(always)]
    pub fn get_mempool_panic_score(&self) -> f64 {
        f64::from_bits(self.mempool_panic_score.load(Ordering::Acquire))
    }

    #[inline(always)]
    pub fn set_net_liq_pressure(&self, pressure: f64) {
        self.net_liq_pressure.store(pressure.to_bits(), Ordering::Release);
    }

    #[inline(always)]
    pub fn get_net_liq_pressure(&self) -> f64 {
        f64::from_bits(self.net_liq_pressure.load(Ordering::Acquire))
    }

    #[inline(always)]
    pub fn set_liquidation_cascade_risk(&self, risk: f64) {
        self.liquidation_cascade_risk.store(risk.to_bits(), Ordering::Release);
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
        let current_liq = self.get_liquidation_cascade_risk();
        let current_pressure = self.get_net_liq_pressure();
        
        let last_ts = self.last_update_ts.load(Ordering::Acquire);
        let dt = if ts_ms > last_ts { (ts_ms - last_ts) as f64 } else { 0.0 };
        
        // Decaimiento Exponencial: lambda = 0.001 (media vida de ~693ms)
        let lambda = 0.001;
        let decay_factor = (-lambda * dt).exp(); 
        
        let new_liq = (current_liq * decay_factor) + (impact * 10.0);
        let new_pressure = (current_pressure * decay_factor) + (qty * impact);
        
        self.set_liquidation_cascade_risk(new_liq);
        self.set_net_liq_pressure(new_pressure);
        
        // Update TS only if newer
        let mut curr_ts = last_ts;
        while ts_ms > curr_ts {
            match self.last_update_ts.compare_exchange_weak(curr_ts, ts_ms, Ordering::Release, Ordering::Relaxed) {
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

