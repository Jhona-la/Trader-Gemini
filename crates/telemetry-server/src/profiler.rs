use arc_swap::ArcSwap;
use crossbeam::queue::ArrayQueue;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

// Process-lifetime lower bound on samples rejected by the bounded profiler
// queue. Saturation means at least u64::MAX drops, never a fabricated reset.
static DROPPED_PROFILE_SAMPLES: AtomicU64 = AtomicU64::new(0);

lazy_static! {
    // Bounded lock-free queue for profiler samples (capacity 65536).
    // ArrayQueue may retry under contention: do not claim wait-free latency.
    pub static ref TELEMETRY_QUEUE: ArrayQueue<(&'static str, u64)> = ArrayQueue::new(65536);
    // Slow-path aggregator for Web UI, reads are wait-free using ArcSwap
    pub static ref GLOBAL_AGGREGATOR: ArcSwap<Profiler> = ArcSwap::from_pointee(Profiler::new());

    // Atomic financials to avoid cloning HashMap in hot loop
    pub static ref TOTAL_GROSS_PNL: AtomicU64 = AtomicU64::new(0);
    pub static ref TOTAL_FEES: AtomicU64 = AtomicU64::new(0);
    pub static ref WIN_RATE: AtomicU64 = AtomicU64::new(0);
    pub static ref ROI_NET: AtomicU64 = AtomicU64::new(0);
}

fn count_rejected_sample(counter: &AtomicU64) {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        if current == u64::MAX {
            return;
        }
        match counter.compare_exchange_weak(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

fn enqueue_profile_sample(
    queue: &ArrayQueue<(&'static str, u64)>,
    dropped: &AtomicU64,
    sample: (&'static str, u64),
) {
    match queue.push(sample) {
        Ok(()) => {}
        Err(_) => count_rejected_sample(dropped),
    }
}

/// Record a sample without blocking for queue capacity, allocating a fallback
/// buffer or formatting on the producer path. A full queue rejects the NEW
/// sample and increments the process-lifetime loss counter. Relaxed atomics
/// count losses only; ArrayQueue supplies its own publication synchronization.
/// This is not random sampling, and no unbiased-average claim is made.
pub fn record_profile_sample(name: &'static str, latency: u64) {
    enqueue_profile_sample(&TELEMETRY_QUEUE, &DROPPED_PROFILE_SAMPLES, (name, latency));
}

#[derive(Debug, Default, Clone)]
pub struct Profiler {
    pub metrics: HashMap<&'static str, u64>,
    pub counts: HashMap<&'static str, u64>,
    /// Process-lifetime rejected samples observed after the last drain.
    /// u64::MAX is a saturated lower bound, not an exact count. While
    /// producers are active this snapshot is not a transactional queue census.
    pub dropped_samples: u64,
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
            dropped_samples: 0,
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
        self.dropped_samples = DROPPED_PROFILE_SAMPLES.load(Ordering::Relaxed);
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
        // FIX #1442: Sanitización de métricas financieras de telemetría
        let safe_gross = if gross.is_finite() { gross } else { 0.0 };
        let safe_net = if net.is_finite() { net } else { 0.0 };
        let safe_win_rate = if win_rate.is_finite() { win_rate } else { 0.0 };
        self.total_gross_pnl = safe_gross;
        self.total_fees = safe_gross - safe_net;
        self.win_rate = safe_win_rate;
        // FIX #1530 / FASE 2: Erradicación del 13.0 inventado (CL-15); ROI 0.0 ante capital no positivo o no finito
        self.roi_net = if capital.is_finite() && capital > 0.0 {
            let raw_roi = (safe_net / capital) * 100.0;
            if raw_roi.is_finite() {
                raw_roi.clamp(-1000.0, 100000.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
    }
}

pub fn update_financials_atomic(gross: f64, net: f64, win_rate: f64, capital: f64) {
    // FIX #1442: Sanitización de métricas financieras atómicas
    let safe_gross = if gross.is_finite() { gross } else { 0.0 };
    let safe_net = if net.is_finite() { net } else { 0.0 };
    let safe_win_rate = if win_rate.is_finite() { win_rate } else { 0.0 };
    TOTAL_GROSS_PNL.store(safe_gross.to_bits(), Ordering::Relaxed);
    TOTAL_FEES.store((safe_gross - safe_net).to_bits(), Ordering::Relaxed);
    WIN_RATE.store(safe_win_rate.to_bits(), Ordering::Relaxed);
    // FIX #1530 / FASE 2: Erradicación del 13.0 inventado (CL-15); ROI 0.0 ante capital no positivo o no finito
    let safe_roi = if capital.is_finite() && capital > 0.0 {
        let raw_roi = (safe_net / capital) * 100.0;
        if raw_roi.is_finite() {
            raw_roi.clamp(-1000.0, 100000.0)
        } else {
            0.0
        }
    } else {
        0.0
    };
    ROI_NET.store(safe_roi.to_bits(), Ordering::Relaxed);
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
        // Bounded recording; rejected samples are counted, never silently
        // treated as unbiased sampling. Formatting stays on the auditor thread.
        $crate::profiler::record_profile_sample($name, latency_ns);
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

            // Publish loss even if there are no surviving averages. Queue
            // overflow can correlate with load: the averages are conditional
            // on admitted samples, not evidence of complete hot-path coverage.
            if aggregator.dropped_samples > 0 {
                crate::telemetry_log!(
                    "[PROFILER COVERAGE] rejected_samples={} saturated_lower_bound={} averages_conditioned_on_admission=true",
                    aggregator.dropped_samples,
                    aggregator.dropped_samples == u64::MAX
                );
            }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_queue_and_averages() {
        let mut profiler = Profiler::new();
        record_profile_sample("OrderbookParse", 500);
        record_profile_sample("OrderbookParse", 700);

        profiler.drain_queue();
        let averages = profiler.get_averages();
        assert_eq!(averages.get("OrderbookParse"), Some(&600));
    }

    #[test]
    fn profiler_loss_rejects_new_sample_without_overwriting_retained_data() {
        let queue = ArrayQueue::new(2);
        let dropped = AtomicU64::new(0);
        enqueue_profile_sample(&queue, &dropped, ("first", 500));
        enqueue_profile_sample(&queue, &dropped, ("second", 700));
        enqueue_profile_sample(&queue, &dropped, ("rejected", 900));
        assert_eq!(queue.len(), 2);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(queue.pop(), Some(("first", 500)));
        assert_eq!(queue.pop(), Some(("second", 700)));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn profiler_loss_counter_survives_drain_and_recovery() {
        let queue = ArrayQueue::new(1);
        let dropped = AtomicU64::new(0);
        enqueue_profile_sample(&queue, &dropped, ("first", 1));
        enqueue_profile_sample(&queue, &dropped, ("lost", 2));
        assert_eq!(queue.pop(), Some(("first", 1)));
        enqueue_profile_sample(&queue, &dropped, ("recovered", 3));
        assert_eq!(queue.pop(), Some(("recovered", 3)));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn profiler_loss_counter_saturates_instead_of_wrapping_to_zero() {
        let queue = ArrayQueue::new(1);
        let dropped = AtomicU64::new(u64::MAX - 1);
        enqueue_profile_sample(&queue, &dropped, ("retained", 1));
        for _ in 0..3 {
            enqueue_profile_sample(&queue, &dropped, ("lost", 1));
        }
        assert_eq!(dropped.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(queue.pop(), Some(("retained", 1)));
    }

    #[test]
    fn profiler_loss_conservation_with_concurrent_real_producers() {
        let queue = ArrayQueue::new(64);
        let dropped = AtomicU64::new(0);
        thread::scope(|scope| {
            for _ in 0..4 {
                let queue = &queue;
                let dropped = &dropped;
                scope.spawn(move || {
                    for _ in 0..1024 {
                        enqueue_profile_sample(queue, dropped, ("parallel", 1));
                    }
                });
            }
        });
        // Producers have joined, so this is a quiescent conservation check,
        // not an unjustified transactional claim about concurrent snapshots.
        assert_eq!(queue.len(), 64);
        assert_eq!(dropped.load(Ordering::Relaxed), 4096 - 64);
        let mut received = 0;
        while let Some(sample) = queue.pop() {
            assert_eq!(sample, ("parallel", 1));
            received += 1;
        }
        assert_eq!(received, 64);
    }

    #[test]
    fn test_profiler_financials_nan_sanitization() {
        let mut profiler = Profiler::new();
        profiler.update_financials(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        assert_eq!(profiler.total_gross_pnl, 0.0);
        assert_eq!(profiler.roi_net, 0.0);

        update_financials_atomic(10.0, 9.5, 0.75, 13.0);
        assert_eq!(f64::from_bits(WIN_RATE.load(Ordering::Relaxed)), 0.75);
    }

    #[test]
    fn test_profiler_roi_clamping_and_fee_calculation() {
        let mut profiler = Profiler::new();
        // Gross 20.0, Net 18.0 -> Fees 2.0, ROI (18.0 / 13.0) * 100 = 138.46%
        profiler.update_financials(20.0, 18.0, 0.80, 13.0);
        assert!((profiler.total_fees - 2.0).abs() < 1e-6);
        assert!((profiler.roi_net - 138.461538).abs() < 1e-3);

        // Clamping extreme ROI (infinite profit simulation)
        profiler.update_financials(2_000_000.0, 2_000_000.0, 1.0, 1.0);
        assert_eq!(profiler.roi_net, 100_000.0);
    }
}
