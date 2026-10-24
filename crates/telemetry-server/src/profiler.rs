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

// F1-100: TSC calibration. `profile_node!` divided the cycle delta by a
// hardcoded `3` — an assumption of exactly 3 GHz — so every reported latency
// was wrong by the ratio between the real TSC rate and 3 GHz, and the integer
// division quantised the measurement to 3-cycle steps.
//
// The rate is now measured once against the monotonic clock and stored as a
// Q32 fixed-point nanoseconds-per-cycle, so the hot path converts with a
// multiply and a shift instead of a division. Zero means "not calibrated": the
// span is then counted, never assigned a fabricated duration.
static TSC_HZ: AtomicU64 = AtomicU64::new(0);
static TSC_NS_PER_CYCLE_Q32: AtomicU64 = AtomicU64::new(0);
/// Cost of the instrument itself: the `lfence`+`rdtsc` pair the macro runs
/// inside every measured span. Reported so a reader can tell a fast node from
/// the profiler's own resolution floor.
static TSC_INSTRUMENT_FLOOR_NS: AtomicU64 = AtomicU64::new(0);
/// 0 = never calibrated, 1 = calibration claimed or done. Guarantees a single
/// background calibration instead of one per unmeasured span.
static CALIBRATION_STATE: AtomicU64 = AtomicU64::new(0);

/// No x86_64 part this system can run on ticks below 100 MHz; a reading under
/// that is a broken measurement, not a slow clock.
const MIN_TSC_HZ: u64 = 100_000_000;

// Spans that could not be turned into a duration, split by reason so the report
// distinguishes "no samples yet" from "samples we could not time".
static UNMEASURED_PROFILE_SAMPLES: AtomicU64 = AtomicU64::new(0);
static REVERSED_TSC_SAMPLES: AtomicU64 = AtomicU64::new(0);

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

/// Convert a TSC delta to nanoseconds with a Q32 fixed-point factor.
///
/// Pure function, so the arithmetic can be contracted without touching the
/// process-global calibration (parallel tests would race otherwise). `q32` is
/// nanoseconds-per-cycle scaled by 2^32; zero means uncalibrated and yields
/// `None` — an untimed span is unknown, not zero.
///
/// Rounding is to nearest, not truncation: truncating would systematically
/// understate latency, which is the wrong direction for a slowness detector.
/// The resolution is one nanosecond, so a span shorter than that reports `0` —
/// a real quantisation floor, not a fabricated zero. The macro's own
/// instrument floor is measured and published separately.
#[inline]
fn nanos_from_q32(cycles: u64, q32: u64) -> Option<u64> {
    if q32 == 0 {
        return None;
    }
    // u128 keeps the product exact for any u64 cycle count: even a full u64
    // span cannot overflow before the shift.
    let exact = cycles as u128 * q32 as u128 + (1_u128 << 31);
    let ns = exact >> 32;
    // Saturate rather than truncate: a wrapped duration would look plausible.
    Some(ns.min(u64::MAX as u128) as u64)
}

/// Convert a measured TSC delta to nanoseconds with the calibrated rate.
///
/// `None` when calibration has not run. Callers must not substitute a default.
#[inline]
pub fn cycles_to_nanos(cycles: u64) -> Option<u64> {
    nanos_from_q32(cycles, TSC_NS_PER_CYCLE_Q32.load(Ordering::Acquire))
}

/// Start a background calibration at most once, for binaries that never call
/// [`start_profiler_auditor`] (backtest, simulators, latency audits).
///
/// Calibration must never run on the caller's thread: a ~50 ms spin inside a
/// measured block would stall the engine it is measuring. Spans emitted while
/// it runs are counted as unmeasured, so a process that never calibrates
/// reports "no latency data" instead of a wrong number.
fn ensure_tsc_calibrated() {
    if CALIBRATION_STATE
        .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        // Detached on purpose: the caller must never wait for the ~50 ms
        // measurement. `Builder::spawn` returns a Result instead of panicking
        // when the OS refuses a thread, so telemetry cannot abort the measured
        // block. A refusal releases the claim so a later span can retry.
        match thread::Builder::new()
            .name("tsc-calibration".to_string())
            .spawn(calibrate_tsc)
        {
            Ok(handle) => drop(handle),
            Err(_) => CALIBRATION_STATE.store(0, Ordering::Relaxed),
        }
    }
}

/// Record one measured span without ever inventing a duration.
///
/// Telemetry must not change the control flow of the measured block, so every
/// failure mode is counted and reported instead of panicking or defaulting.
#[inline]
pub fn record_tsc_span(name: &'static str, start: u64, end: u64) {
    match end.checked_sub(start) {
        // A span that runs backwards is not a duration: the reads were
        // reordered around the block. Counting it beats underflowing into a
        // ~1.8e19 ns "measurement" that poisons every average and every
        // relative-anomaly threshold derived from them.
        None => count_rejected_sample(&REVERSED_TSC_SAMPLES),
        Some(cycles) => match cycles_to_nanos(cycles) {
            Some(ns) => record_profile_sample(name, ns),
            None => {
                ensure_tsc_calibrated();
                count_rejected_sample(&UNMEASURED_PROFILE_SAMPLES);
            }
        },
    }
}

/// Whether the CPU advertises an invariant TSC (CPUID 0x80000007 EDX bit 8).
///
/// With an invariant TSC the rate is fixed for the life of the process, so one
/// calibration stays valid across turbo, C-states and core migration. Without
/// it the rate follows the core clock and a single calibration is only an
/// approximation — the report says so instead of pretending otherwise.
fn tsc_is_invariant() -> Option<bool> {
    let max_extended = unsafe { core::arch::x86_64::__cpuid(0x8000_0000) }.eax;
    if max_extended < 0x8000_0007 {
        return None;
    }
    let features = unsafe { core::arch::x86_64::__cpuid(0x8000_0007) };
    Some(features.edx & (1 << 8) != 0)
}

/// Measure the TSC rate against the monotonic system clock.
///
/// Five 10 ms rounds, median. On a CPU with an invariant TSC the rate is fixed
/// and preemption cannot bias the ratio (both clocks advance in real time
/// together), so a descheduled round stays on the same ratio and the median
/// only discards outliers. Without an invariant TSC the rate follows the core
/// clock and this single calibration is an approximation — the log says so.
///
/// Also measures the instrument floor: the macro's own `lfence`+`rdtsc` pair
/// runs inside every span, so it is a real, reportable resolution floor.
///
/// Returns the measured rate in Hz. On failure the previous calibration, if
/// any, is left untouched.
pub fn calibrate_tsc() -> Option<u64> {
    // Claim the state so a concurrent `ensure_tsc_calibrated` does not spawn a
    // second calibration thread.
    CALIBRATION_STATE.store(1, Ordering::Relaxed);

    let mut samples: Vec<u64> = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let c0 = unsafe { core::arch::x86_64::_rdtsc() };
        while t0.elapsed() < Duration::from_millis(10) {
            std::hint::spin_loop();
        }
        let c1 = unsafe { core::arch::x86_64::_rdtsc() };
        let seconds = t0.elapsed().as_secs_f64();
        if seconds <= 0.0 {
            continue;
        }
        let Some(cycles) = c1.checked_sub(c0) else {
            continue;
        };
        let hz = (cycles as f64 / seconds).round();
        if !hz.is_finite() {
            continue;
        }
        if hz < MIN_TSC_HZ as f64 {
            continue;
        }
        if hz >= u64::MAX as f64 {
            continue;
        }
        samples.push(hz as u64);
    }
    if samples.is_empty() {
        return None;
    }
    samples.sort_unstable();
    let hz = samples[samples.len() / 2];
    let q32 = (1_000_000_000.0 / hz as f64 * 4_294_967_296.0).round();
    if q32 < 1.0 {
        return None;
    }
    let q32 = q32 as u64;

    let mut floors: Vec<u64> = Vec::with_capacity(64);
    for _ in 0..64 {
        unsafe { core::arch::x86_64::_mm_lfence() };
        let a = unsafe { core::arch::x86_64::_rdtsc() };
        unsafe { core::arch::x86_64::_mm_lfence() };
        let b = unsafe { core::arch::x86_64::_rdtsc() };
        if let Some(cycles) = b.checked_sub(a) {
            floors.push(cycles);
        }
    }
    floors.sort_unstable();
    let floor_ns = floors
        .get(floors.len() / 2)
        .and_then(|cycles| nanos_from_q32(*cycles, q32))
        .unwrap_or(0);

    TSC_HZ.store(hz, Ordering::Relaxed);
    TSC_INSTRUMENT_FLOOR_NS.store(floor_ns, Ordering::Relaxed);
    // Release publishes the rate together with the fixed-point factor, so a
    // reader that observes a non-zero factor also observes the matching TSC_HZ.
    TSC_NS_PER_CYCLE_Q32.store(q32, Ordering::Release);
    crate::telemetry_log!(
        "[TSC] frecuencia medida {:.4} GHz (invariante={:?}); conversion en punto fijo Q32, no una constante de 3 GHz; suelo del instrumento={} ns",
        hz as f64 / 1_000_000_000.0,
        tsc_is_invariant(),
        floor_ns
    );
    Some(hz)
}

/// Helper macro to measure execution time with the calibrated TSC rate.
///
/// F1-100: the previous body computed `(end - start) / 3`, i.e. it assumed the
/// CPU ticks at exactly 3 GHz and truncated to 3-cycle steps. On a 4.5 GHz part
/// every latency was reported ~33 % too low; on a 2.0 GHz part ~50 % too high.
///
/// The two `lfence` reads keep the block from being hoisted above the opening
/// read or sunk below the closing one, which otherwise yields a backwards span.
/// The closing `lfence` executes inside the measured window, so every span
/// carries the instrument's own serialisation cost; that cost is measured at
/// calibration and published as [`TSC_INSTRUMENT_FLOOR_NS`] instead of being
/// left as an undisclosed offset.
#[macro_export]
macro_rules! profile_node {
    ($name:expr, $block:block) => {{
        unsafe { core::arch::x86_64::_mm_lfence() };
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        let result = $block;
        unsafe { core::arch::x86_64::_mm_lfence() };
        let end = unsafe { core::arch::x86_64::_rdtsc() };
        // Bounded recording; rejected samples are counted, never silently
        // treated as unbiased sampling. Formatting stays on the auditor thread.
        $crate::profiler::record_tsc_span($name, start, end);
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
        // F1-100: calibrate before the first report. This runs on the auditor
        // thread, not on the engine runtime, so the ~50 ms of measurement never
        // stalls a tick. Spans emitted before this completes are counted as
        // unmeasured rather than given a fabricated duration.
        if calibrate_tsc().is_none() {
            crate::telemetry_log!(
                "⚠️ [TSC] no se pudo calibrar el reloj; las latencias se reportaran como NO DISPONIBLES en vez de inventarse"
            );
        }
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

            // F1-100: the header states which rate produced the numbers and how
            // many spans could not be timed. Without it a reader cannot tell a
            // 3 GHz assumption from a measured rate, nor see silent gaps.
            //
            // The Q32 factor is read FIRST: it is the Acquire that publishes
            // the rate and the floor, so reading it first guarantees the header
            // can never print `calibrated=true` beside a zero rate.
            let tsc_q32 = TSC_NS_PER_CYCLE_Q32.load(Ordering::Acquire);
            let tsc_hz = TSC_HZ.load(Ordering::Relaxed);
            crate::telemetry_log!(
                "--- [TELEMETRY LATENCY REPORT] tsc={:.4}GHz calibrated={} instrument_floor={}ns unmeasured_spans={} reversed_spans={} ---",
                tsc_hz as f64 / 1_000_000_000.0,
                tsc_q32 != 0,
                TSC_INSTRUMENT_FLOOR_NS.load(Ordering::Relaxed),
                UNMEASURED_PROFILE_SAMPLES.load(Ordering::Relaxed),
                REVERSED_TSC_SAMPLES.load(Ordering::Relaxed)
            );

            // Mostrar promedios si existen
            if !averages.is_empty() {
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

    #[test]
    fn tsc_conversion_is_exact_and_uncalibrated_stays_unknown() {
        // 2.5 GHz -> 0.4 ns per cycle.
        let q32 = (0.4_f64 * 4_294_967_296.0) as u64;
        assert_eq!(nanos_from_q32(0, q32), Some(0));
        assert_eq!(nanos_from_q32(1_000_000_000, q32), Some(400_000_000));
        // Uncalibrated is unknown, not zero: no fabricated duration.
        assert_eq!(nanos_from_q32(1_000_000_000, 0), None);
    }

    #[test]
    fn tsc_conversion_erradicates_the_three_cycle_floor() {
        // The old body computed `cycles / 3`, i.e. it assumed exactly 3 GHz and
        // quantised to 3-cycle steps. A 25-cycle span is 10 ns at 2.5 GHz, but
        // the old formula reported 8 ns — a 20 % understatement of latency.
        assert_eq!(25_u64 / 3, 8);
        let q32 = (0.4_f64 * 4_294_967_296.0) as u64;
        assert_eq!(nanos_from_q32(25, q32), Some(10));
        assert_eq!(nanos_from_q32(3, q32), Some(1));
    }

    #[test]
    fn tsc_backwards_span_is_counted_and_never_becomes_a_duration() {
        // Without serialisation a short block can be reordered so that the
        // closing read precedes the opening one. `end - start` would underflow
        // to ~1.8e19 ns and poison every average derived from it; it must be
        // counted instead of recorded.
        let before = REVERSED_TSC_SAMPLES.load(Ordering::Relaxed);
        record_tsc_span("F1100_reversed", 1_000, 999);
        let after = REVERSED_TSC_SAMPLES.load(Ordering::Relaxed);
        assert_eq!(after, before + 1, "un span invertido debe contarse");
    }

    #[test]
    fn tsc_calibration_measures_a_plausible_rate_and_one_second_round_trips() {
        let hz = calibrate_tsc().expect("el reloj TSC debe poder calibrarse en x86_64");
        assert!(
            hz > 500_000_000 && hz < 10_000_000_000,
            "frecuencia TSC fuera del rango fisico plausible: {hz} Hz"
        );
        assert!(
            TSC_NS_PER_CYCLE_Q32.load(Ordering::Acquire) > 0,
            "la calibracion debe publicar el factor Q32"
        );
        // One second's worth of cycles must convert back to one second.
        let ns = cycles_to_nanos(hz).expect("calibrado");
        let error = (ns as f64 - 1_000_000_000.0).abs() / 1_000_000_000.0;
        assert!(error < 0.01, "1 s de ciclos debe dar 1 s, error {error}");
        // The instrument floor is measured, not assumed: positive, and far
        // below a microsecond on any part this system runs on.
        let floor = TSC_INSTRUMENT_FLOOR_NS.load(Ordering::Relaxed);
        assert!(floor > 0, "el suelo del instrumento debe medirse");
        assert!(
            floor < 10_000,
            "suelo del instrumento fuera de rango plausible: {floor} ns"
        );
    }
}
