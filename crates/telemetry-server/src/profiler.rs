use arc_swap::ArcSwap;
use crossbeam::queue::ArrayQueue;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

lazy_static! {
    // Lock-free queue for telemetry events (capacity 65536)
    // Completely wait-free. O(1).
    pub static ref TELEMETRY_QUEUE: ArrayQueue<(&'static str, u64)> = ArrayQueue::new(65536);
    // Slow-path aggregator for Web UI, reads are wait-free using ArcSwap
    pub static ref GLOBAL_AGGREGATOR: ArcSwap<Profiler> = ArcSwap::from_pointee(Profiler::new());

    // Atomic financials to avoid cloning HashMap in hot loop
    pub static ref TOTAL_GROSS_PNL: AtomicU64 = AtomicU64::new(0);
    pub static ref TOTAL_FEES: AtomicU64 = AtomicU64::new(0);
    pub static ref WIN_RATE: AtomicU64 = AtomicU64::new(0);
    pub static ref ROI_NET: AtomicU64 = AtomicU64::new(0);
}

#[derive(Debug, Default, Clone)]
pub struct Profiler {
    pub metrics: HashMap<&'static str, u64>,
    pub counts: HashMap<&'static str, u64>,
    pub roi_net: f64,
    pub total_fees: f64,
    pub win_rate: f64,
    pub total_gross_pnl: f64,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            counts: HashMap::new(),
            roi_net: 0.0,
            total_fees: 0.0,
            win_rate: 0.0,
            total_gross_pnl: 0.0,
        }
    }

    pub fn drain_queue(&mut self) {
        while let Some((name, latency)) = TELEMETRY_QUEUE.pop() {
            let total = self.metrics.entry(name).or_insert(0);
            *total += latency;
            let count = self.counts.entry(name).or_insert(0);
            *count += 1;
        }
    }

    pub fn get_averages(&self) -> HashMap<&'static str, u64> {
        let mut averages = HashMap::with_capacity(self.metrics.len());
        for (&name, total) in &self.metrics {
            #[allow(clippy::collapsible_if)]
            if let Some(count) = self.counts.get(name) {
                if *count > 0 {
                    averages.insert(name, total / count);
                }
            }
        }
        averages
    }

    pub fn update_financials(&mut self, gross: f64, net: f64, win_rate: f64, capital: f64) {
        self.total_gross_pnl = gross;
        self.total_fees = gross - net;
        self.win_rate = win_rate;
        if capital > 0.0 {
            self.roi_net = (net / capital) * 100.0;
        }
    }
}

pub fn update_financials_atomic(gross: f64, net: f64, win_rate: f64, capital: f64) {
    TOTAL_GROSS_PNL.store(gross.to_bits(), Ordering::Relaxed);
    TOTAL_FEES.store((gross - net).to_bits(), Ordering::Relaxed);
    WIN_RATE.store(win_rate.to_bits(), Ordering::Relaxed);
    if capital > 0.0 {
        ROI_NET.store(((net / capital) * 100.0).to_bits(), Ordering::Relaxed);
    }
}

/// Helper macro to measure execution time using _rdtsc
#[macro_export]
macro_rules! profile_node {
    ($name:expr, $block:block) => {{
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        let result = $block;
        let end = unsafe { core::arch::x86_64::_rdtsc() };
        // Approximate conversion: 1 CPU cycle ~ 0.3ns (on 3GHz CPU)
        let latency_ns = (end - start) / 3;
        // Wait-free, allocation-free push in the hot path. Drops if full (sampler).
        let _ = $crate::profiler::TELEMETRY_QUEUE.push(($name, latency_ns));
        result
    }};
}

/// Inicia el auditor de latencias en un hilo secundario.
pub fn start_profiler_auditor() {
    thread::spawn(move || {
        crate::telemetry_log!("[TELEMETRY] ⏱️ Auditor de Latencia y Rendimiento Iniciado.");
        loop {
            thread::sleep(Duration::from_secs(10));

            // Read-Copy-Update (RCU) wait-free update
            let mut aggregator = (**GLOBAL_AGGREGATOR.load()).clone();
            aggregator.drain_queue();
            let averages = aggregator.get_averages();
            let roi_net = f64::from_bits(ROI_NET.load(Ordering::Relaxed));
            let total_fees = f64::from_bits(TOTAL_FEES.load(Ordering::Relaxed));
            let win_rate = f64::from_bits(WIN_RATE.load(Ordering::Relaxed));
            let gross_pnl = f64::from_bits(TOTAL_GROSS_PNL.load(Ordering::Relaxed));

            aggregator.roi_net = roi_net;
            aggregator.total_fees = total_fees;
            aggregator.win_rate = win_rate;
            aggregator.total_gross_pnl = gross_pnl;

            // Mostrar promedios si existen
            if !averages.is_empty() {
                crate::telemetry_log!("--- [TELEMETRY LATENCY & FINANCIAL REPORT] ---");
                crate::telemetry_log!(
                    "💰 Gross PnL: {:.4} | Net PnL: {:.4} | Fees: {:.4} | ROI Net: {:.2}% | Win-Rate: {:.2}%",
                    gross_pnl,
                    gross_pnl - total_fees,
                    total_fees,
                    roi_net,
                    win_rate
                );

                let mut all_latencies: Vec<u64> = averages.values().cloned().collect();
                all_latencies.sort_unstable();
                let median_latency = *all_latencies.get(all_latencies.len() / 2).unwrap_or(&0);
                let threshold = median_latency.saturating_mul(5); // Umbral de anomalía relativo

                for (name, avg_ns) in &averages {
                    if *avg_ns > threshold && *avg_ns > 1000 {
                        crate::telemetry_log!(
                            "⚠️ ANOMALÍA LENTITUD RELATIVA | {}: {} ns (Mediana del sistema: {} ns)",
                            name,
                            avg_ns,
                            median_latency
                        );
                    } else {
                        crate::telemetry_log!("✅ {}: {} ns", name, avg_ns);
                    }
                }
            }

            GLOBAL_AGGREGATOR.store(Arc::new(aggregator));
        }
    });
}
