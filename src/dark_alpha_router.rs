use std::sync::atomic::{AtomicU64, Ordering};

/// Sniffing de Mempool (RBF) y WebSockets de DEX
/// Alimenta al QuantumStateArena con señales de presión de liquidez y pánico de red.
/// Totalmente lock-free usando AtomicU64 bitcasting (f64 <-> u64) para Axioma V y VIII.
#[repr(C, align(64))]
pub struct DarkAlphaRouter {
    pub mempool_panic_score: AtomicU64, // f64
    pub net_liq_pressure: AtomicU64,    // f64
    pub liquidation_cascade_risk: AtomicU64, // f64
    
    // Concurrency stats
    pub processed_packets: AtomicU64,
}

impl DarkAlphaRouter {
    pub fn new() -> Self {
        Self {
            mempool_panic_score: AtomicU64::new(0f64.to_bits()),
            net_liq_pressure: AtomicU64::new(0f64.to_bits()),
            liquidation_cascade_risk: AtomicU64::new(0f64.to_bits()),
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
    /// Aplica un decaimiento básico y lo funde con el estado atómico O(1).
    #[inline(always)]
    pub fn ingest_l2_snapshot(&self, qty: f64, _delay_ms: f64, impact: f64, _tick_count: u64) {
        let current_liq = self.get_liquidation_cascade_risk();
        let current_pressure = self.get_net_liq_pressure();
        
        // Decaimiento Exponencial simplificado (lambda = 0.05) para efecto del tick
        let decay = 0.95; 
        let new_liq = (current_liq * decay) + (impact * 10.0);
        let new_pressure = (current_pressure * decay) + (qty * impact);
        
        self.set_liquidation_cascade_risk(new_liq);
        self.set_net_liq_pressure(new_pressure);
        self.inc_processed_packets();
    }
}

