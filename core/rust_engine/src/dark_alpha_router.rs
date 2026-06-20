use std::sync::atomic::{AtomicU64, Ordering};

/// Sniffing de Mempool (RBF) y WebSockets de DEX
/// Alimenta al QuantumStateArena con señales de presión de liquidez y pánico de red.
pub struct DarkAlphaRouter {
    pub mempool_panic_score: f32,
    pub net_liq_pressure: f32,
    pub liquidation_cascade_risk: f32,
    
    // Concurrency stats
    processed_packets: AtomicU64,
}

impl DarkAlphaRouter {
    pub fn new() -> Self {
        Self {
            mempool_panic_score: 0.0,
            net_liq_pressure: 0.0,
            liquidation_cascade_risk: 0.0,
            processed_packets: AtomicU64::new(0),
        }
    }

    /// Update with L2/L3 Orderbook data and DEX events
    pub fn ingest_l2_snapshot(&mut self, bids_vol: f32, asks_vol: f32, spread: f32) {
        self.processed_packets.fetch_add(1, Ordering::Relaxed);
        
        let total_vol = bids_vol + asks_vol;
        if total_vol > 0.0 {
            // Pressure ranges from -1.0 to +1.0
            self.net_liq_pressure = (bids_vol - asks_vol) / total_vol;
        }

        // Extremely basic cascading liquidation risk based on spread widening and lack of liquidity
        if spread > 0.005 && total_vol < 10.0 {
            self.liquidation_cascade_risk = (self.liquidation_cascade_risk + 0.1).clamp(0.0, 1.0);
        } else {
            self.liquidation_cascade_risk *= 0.9; // Decay
        }
    }

    /// Ingests Mempool RBF (Replace-By-Fee) transactions as a proxy for network panic
    pub fn ingest_mempool_rbf(&mut self, rbf_tx_count: usize, avg_fee_surge: f32) {
        self.processed_packets.fetch_add(1, Ordering::Relaxed);
        
        // If mempool RBF surges, panic increases.
        let surge = avg_fee_surge * (rbf_tx_count as f32) * 0.01;
        self.mempool_panic_score = (self.mempool_panic_score + surge).clamp(0.0, 1.0);
        self.mempool_panic_score *= 0.95; // Decay
    }
}
