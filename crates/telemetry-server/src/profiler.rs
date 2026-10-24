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

    // F1-099: financial telemetry atomics. They start as NaN bits (unknown),
    // not 0.0: before the first real update there is no financial data yet, and
    // a zero would assert "nothing was earned". A non-finite value read back is
    // reported as unavailable instead of being silently replaced by 0.0.
    pub static ref TOTAL_GROSS_PNL: AtomicU64 = AtomicU64::new(f64::NAN.to_bits());
    pub static ref TOTAL_NET_PNL: AtomicU64 = AtomicU64::new(f64::NAN.to_bits());
    pub static ref TOTAL_FEES: AtomicU64 = AtomicU64::new(f64::NAN.to_bits());
    pub static ref WIN_RATE: AtomicU64 = AtomicU64::new(f64::NAN.to_bits());
    pub static ref ROI_NET: AtomicU64 = AtomicU64::new(f64::NAN.to_bits());
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

/// Snapshot of profiler state published to the Web UI and the auditor log.
///
/// F1-099: the financial fields use `NaN` to mean **unavailable**, never a
/// fabricated zero. A genuine zero PnL stays `0.0` (finite) and is reported as
/// a real zero. This mirrors the existing convention in
/// `backtest-engine/src/metrics.rs`, where `win_rate` is `NaN` without a sample.
#[derive(Debug, Clone)]
pub struct Profiler {
    pub metrics: HashMap<&'static str, u64>,
    pub counts: HashMap<&'static str, u64>,
    /// Process-lifetime rejected samples observed after the last drain.
    /// u64::MAX is a saturated lower bound, not an exact count. While
    /// producers are active this snapshot is not a transactional queue census.
    pub dropped_samples: u64,
    /// Net ROI over wallet capital. `NaN` = unavailable (capital not finite
    /// or not positive, or the net PnL was unreadable).
    pub roi_net: f64,
    /// Fees actually charged, `gross - net`. `NaN` = unavailable.
    pub total_fees: f64,
    /// Net realized PnL. `NaN` = unavailable.
    pub net_pnl: f64,
    /// Win rate in percent. `NaN` = unavailable.
    pub win_rate: f64,
    /// Gross realized PnL. `NaN` = unavailable (non-finite input).
    pub total_gross_pnl: f64,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            counts: HashMap::new(),
            dropped_samples: 0,
            // F1-099: no financial data has arrived yet, so every financial
            // field starts unavailable rather than as an asserted zero.
            roi_net: f64::NAN,
            total_fees: f64::NAN,
            net_pnl: f64::NAN,
            win_rate: f64::NAN,
            total_gross_pnl: f64::NAN,
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

    /// Publish financial telemetry.
    ///
    /// F1-099: an unreadable input stays unreadable. The previous behaviour
    /// replaced a non-finite `net` with `0.0` and then computed
    /// `fees = gross - 0.0`, i.e. it **invented a fee equal to the entire gross
    /// PnL** while reporting `Net PnL: 0.0000`. Now `gross - NaN` is `NaN`, so
    /// fees are unknown instead of fabricated. A genuine zero is preserved as a
    /// real zero because `0.0` is finite.
    ///
    /// ROI is only defined with finite, positive capital; otherwise it is `NaN`
    /// (unavailable), not `0.0` — a zero would assert "nothing was earned" when
    /// the truth is "it cannot be computed". No clamp is applied: a finite ROI
    /// is a real measurement and an arbitrary range would misreport it.
    pub fn update_financials(&mut self, gross: f64, net: f64, win_rate: f64, capital: f64) {
        self.total_gross_pnl = gross;
        self.net_pnl = net;
        self.total_fees = gross - net;
        self.win_rate = win_rate;
        self.roi_net = if capital.is_finite() && capital > 0.0 {
            (net / capital) * 100.0
        } else {
            f64::NAN
        };
    }
}

/// Atomic variant of [`Profiler::update_financials`], read by the auditor
/// thread. Stores the raw bits, so a forensic reader can recover exactly what
/// arrived; the report decides availability from `is_finite()`.
///
/// F1-099: same contract as the struct version — no silent zero substitution,
/// no fee derived from a missing net, no clamped ROI.
pub fn update_financials_atomic(gross: f64, net: f64, win_rate: f64, capital: f64) {
    TOTAL_GROSS_PNL.store(gross.to_bits(), Ordering::Relaxed);
    TOTAL_NET_PNL.store(net.to_bits(), Ordering::Relaxed);
    TOTAL_FEES.store((gross - net).to_bits(), Ordering::Relaxed);
    WIN_RATE.store(win_rate.to_bits(), Ordering::Relaxed);
    let roi = if capital.is_finite() && capital > 0.0 {
        (net / capital) * 100.0
    } else {
        f64::NAN
    };
    ROI_NET.store(roi.to_bits(), Ordering::Relaxed);
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

/// Render one financial field for the auditor report.
///
/// F1-099: a finite value is shown as measured; a non-finite one is declared
/// `UNAVAILABLE`. Nothing is coerced into `0.0`, so a reader can always tell
/// "nothing was earned" (a real zero) from "this could not be computed".
///
/// Runs on the auditor thread at a 10 s cadence, so the allocation is harmless;
/// the hot path never formats.
fn campo_financiero(valor: f64, decimales: usize) -> String {
    if valor.is_finite() {
        format!("{:.*}", decimales, valor)
    } else {
        "UNAVAILABLE".to_string()
    }
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
            let net_pnl = f64::from_bits(TOTAL_NET_PNL.load(Ordering::Relaxed));

            aggregator.roi_net = roi_net;
            aggregator.total_fees = total_fees;
            aggregator.net_pnl = net_pnl;
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

            // F1-099: the financial line is emitted every tick, independently
            // of whether latency samples exist. Previously it sat inside
            // `if !averages.is_empty()`, so an idle profiler hid the financial
            // state entirely instead of reporting it as unavailable.
            //
            // Availability is decided PER FIELD: a readable gross is shown as a
            // number even when the net is unreadable, instead of demoting the
            // whole line to hex. Allocation here is fine — this is the auditor
            // thread at a 10 s cadence, never the hot path.
            crate::telemetry_log!(
                "💰 Gross PnL: {} | Net PnL: {} | Fees: {} | Win-Rate: {}%",
                campo_financiero(gross_pnl, 4),
                campo_financiero(net_pnl, 4),
                campo_financiero(total_fees, 4),
                campo_financiero(win_rate, 2)
            );
            if roi_net.is_finite() {
                crate::telemetry_log!("📈 ROI Net (on wallet capital): {:.4}%", roi_net);
            } else {
                crate::telemetry_log!(
                    "📈 ROI Net: UNAVAILABLE — capital was not finite or not positive; 0.0% would falsely assert that nothing was earned"
                );
            }

            // Mostrar promedios si existen
            if !averages.is_empty() {
                crate::telemetry_log!("--- [TELEMETRY LATENCY REPORT] ---");

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
    fn test_profiler_financials_atomics_end_to_end() {
        // F1-099: ONE test owns the process-global financial atomics. The
        // scenarios are deliberately in a single function: under the default
        // parallel runner, two separate tests storing into and loading from the
        // same globals would race and read each other's values.
        unreadable_stays_unreadable();
        finite_update_lands_exactly();
        atomic_roundtrip_preserves_bits_for_forensics();
        extreme_finite_roi_is_not_clamped();
    }

    fn unreadable_stays_unreadable() {
        // F1-099: this scenario previously asserted the DEFECT (that a NaN
        // input became 0.0). An unreadable input must stay unreadable: no
        // fabricated zero, and no fee derived from a missing net.
        let mut profiler = Profiler::new();
        profiler.update_financials(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        assert!(
            profiler.total_gross_pnl.is_nan(),
            "gross ilegible no puede ser 0.0"
        );
        assert!(profiler.net_pnl.is_nan(), "net ilegible no puede ser 0.0");
        assert!(
            profiler.total_fees.is_nan(),
            "fees no pueden derivarse de un neto inexistente (antes valían todo el bruto)"
        );
        assert!(profiler.win_rate.is_nan());
        assert!(
            profiler.roi_net.is_nan(),
            "ROI sin capital válido no es 0.0%"
        );
    }

    fn finite_update_lands_exactly() {
        // A finite update still lands exactly.
        update_financials_atomic(10.0, 9.5, 0.75, 13.0);
        assert_eq!(f64::from_bits(WIN_RATE.load(Ordering::Relaxed)), 0.75);
        assert_eq!(
            f64::from_bits(TOTAL_GROSS_PNL.load(Ordering::Relaxed)),
            10.0
        );
        assert_eq!(f64::from_bits(TOTAL_NET_PNL.load(Ordering::Relaxed)), 9.5);
        assert_eq!(f64::from_bits(TOTAL_FEES.load(Ordering::Relaxed)), 0.5);
    }

    fn atomic_roundtrip_preserves_bits_for_forensics() {
        // Raw bits are stored, so what arrived can be recovered exactly.
        update_financials_atomic(-7.25, -6.0, 0.4, 50.0);
        assert_eq!(
            f64::from_bits(TOTAL_GROSS_PNL.load(Ordering::Relaxed)),
            -7.25
        );
        assert_eq!(f64::from_bits(TOTAL_NET_PNL.load(Ordering::Relaxed)), -6.0);
        assert_eq!(f64::from_bits(TOTAL_FEES.load(Ordering::Relaxed)), -1.25);
        assert_eq!(f64::from_bits(WIN_RATE.load(Ordering::Relaxed)), 0.4);
        assert!((f64::from_bits(ROI_NET.load(Ordering::Relaxed)) - (-12.0)).abs() < 1e-9);
    }

    fn extreme_finite_roi_is_not_clamped() {
        // The old clamp(-1000.0, 100000.0) silently rewrote a real measurement.
        update_financials_atomic(2_000_000.0, 2_000_000.0, 1.0, 1.0);
        assert_eq!(f64::from_bits(TOTAL_FEES.load(Ordering::Relaxed)), 0.0);
        assert_eq!(
            f64::from_bits(ROI_NET.load(Ordering::Relaxed)),
            200_000_000.0
        );
    }

    #[test]
    fn test_profiler_financials_genuine_zero_is_not_hidden() {
        // A real zero PnL (no trades, or an exactly break-even book) must remain
        // a reported zero. Unavailable-ness is NaN, never "everything is zero".
        let mut profiler = Profiler::new();
        profiler.update_financials(0.0, 0.0, 0.0, 13.0);
        assert_eq!(profiler.total_gross_pnl, 0.0);
        assert_eq!(profiler.net_pnl, 0.0);
        assert_eq!(profiler.total_fees, 0.0);
        assert_eq!(profiler.win_rate, 0.0);
        assert!(profiler.roi_net.is_finite());
        assert_eq!(profiler.roi_net, 0.0);
    }

    #[test]
    fn test_profiler_financials_partial_corruption_does_not_fabricate_fees() {
        // The dangerous live case: gross readable, net not. The old code set
        // net=0.0 and reported fees == gross, i.e. it invented the entire gross
        // as a fee.
        let mut profiler = Profiler::new();
        profiler.update_financials(20.0, f64::NAN, 0.8, 13.0);
        assert_eq!(profiler.total_gross_pnl, 20.0, "gross legible se conserva");
        assert!(profiler.net_pnl.is_nan(), "net ilegible sigue ilegible");
        assert!(
            profiler.total_fees.is_nan(),
            "fees desconocidas, no inventadas como el bruto completo"
        );
        assert!(profiler.roi_net.is_nan(), "ROI depende del neto");
    }

    #[test]
    fn test_profiler_financials_capital_zero_makes_roi_unavailable_not_zero() {
        let mut profiler = Profiler::new();
        profiler.update_financials(5.0, 5.0, 1.0, 0.0);
        assert_eq!(profiler.total_gross_pnl, 5.0);
        assert_eq!(profiler.net_pnl, 5.0);
        assert_eq!(profiler.total_fees, 0.0);
        assert!(
            profiler.roi_net.is_nan(),
            "sin capital positivo el ROI es desconocido; 0.0% afirmaría que no se ganó nada"
        );
    }

    #[test]
    fn test_profiler_financials_fresh_snapshot_is_unavailable_not_zero() {
        // Before any financial update, a snapshot must not claim zeros.
        let profiler = Profiler::new();
        assert!(profiler.total_gross_pnl.is_nan());
        assert!(profiler.net_pnl.is_nan());
        assert!(profiler.total_fees.is_nan());
        assert!(profiler.win_rate.is_nan());
        assert!(profiler.roi_net.is_nan());
        // Default must agree with new(): a Default-built snapshot is equally
        // "no data yet", not an asserted zero.
        let via_default = Profiler::default();
        assert!(via_default.total_gross_pnl.is_nan());
        assert!(via_default.roi_net.is_nan());
    }

    #[test]
    fn test_profiler_financials_roi_reports_the_real_value_without_arbitrary_clamp() {
        // F1-099: the old clamp(-1000.0, 100000.0) silently rewrote a real
        // measurement. A finite ROI is now reported exactly.
        let mut profiler = Profiler::new();
        // Gross 20.0, Net 18.0 -> Fees 2.0, ROI (18.0 / 13.0) * 100 = 138.46%
        profiler.update_financials(20.0, 18.0, 0.80, 13.0);
        assert!((profiler.total_fees - 2.0).abs() < 1e-6);
        assert!((profiler.roi_net - 138.461538).abs() < 1e-3);

        // Extreme but finite ROI: reported as measured, not rewritten to 1e5.
        profiler.update_financials(2_000_000.0, 2_000_000.0, 1.0, 1.0);
        assert_eq!(profiler.total_fees, 0.0);
        assert_eq!(profiler.roi_net, 200_000_000.0);
    }
}
