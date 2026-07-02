use std::collections::HashMap;
use crossbeam::queue::ArrayQueue;
use arc_swap::ArcSwap;
use lazy_static::lazy_static;
use std::thread;
use std::time::Duration;
use std::sync::Arc;

lazy_static! {
    // Lock-free queue for telemetry events (capacity 65536)
    // Completely wait-free. O(1).
    pub static ref TELEMETRY_QUEUE: ArrayQueue<(&'static str, u64)> = ArrayQueue::new(65536);
    // Slow-path aggregator for Web UI, reads are wait-free using ArcSwap
    pub static ref GLOBAL_AGGREGATOR: ArcSwap<Profiler> = ArcSwap::from_pointee(Profiler::new());
}

#[derive(Debug, Default, Clone)]
pub struct Profiler {
    pub metrics: HashMap<&'static str, u64>,
    pub counts: HashMap<&'static str, u64>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            counts: HashMap::new(),
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
            if let Some(count) = self.counts.get(name)
                && *count > 0 {
                    averages.insert(name, total / count);
                }
        }
        averages
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
        println!("[TELEMETRY] ⏱️ Auditor de Latencia y Rendimiento Iniciado.");
        loop {
            thread::sleep(Duration::from_secs(10));
            
            // Read-Copy-Update (RCU) wait-free update
            let mut aggregator = (**GLOBAL_AGGREGATOR.load()).clone();
            aggregator.drain_queue();
            let averages = aggregator.get_averages();
            
            // Mostrar promedios si existen
            if !averages.is_empty() {
                println!("--- [TELEMETRY LATENCY REPORT] ---");
                for (name, avg_ns) in &averages {
                    if *avg_ns > 50_000 {
                        println!("⚠️ PELIGRO LENTITUD | {}: {} ns", name, avg_ns);
                    } else {
                        println!("✅ {}: {} ns", name, avg_ns);
                    }
                }
            }
            
            GLOBAL_AGGREGATOR.store(Arc::new(aggregator));
        }
    });
}
