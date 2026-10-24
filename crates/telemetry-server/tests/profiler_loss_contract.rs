//! F1-098: actual macro -> bounded producer -> actual drain/report snapshot.
//! Deterministic telemetry samples, no sockets, orders, account or model access.
//! One test owns the process-local profiler globals, so the scenarios cannot
//! steal each other's samples even under the default parallel test runner.
//! No background auditor is started in this executable.

use telemetry_server::profiler::{Profiler, TELEMETRY_QUEUE, calibrate_tsc, record_profile_sample};

#[test]
fn profiler_loss_end_to_end_real_queue_macro_and_snapshot() {
    canary_reaches_real_report_without_loss();
    full_real_ring_reports_rejections_and_recovers();
    exported_macro_records_rejection_without_changing_result();
}

fn canary_reaches_real_report_without_loss() {
    let mut report = Profiler::new();
    report.drain_queue();
    let losses_before = report.dropped_samples;
    record_profile_sample("F1098_canary", 500);
    record_profile_sample("F1098_canary", 700);
    report.drain_queue();
    assert_eq!(report.counts.get("F1098_canary"), Some(&2));
    assert_eq!(report.get_averages().get("F1098_canary"), Some(&600));
    assert_eq!(report.dropped_samples, losses_before);
}

fn full_real_ring_reports_rejections_and_recovers() {
    let mut report = Profiler::new();
    report.drain_queue();
    let losses_before = report.dropped_samples;
    let capacity = TELEMETRY_QUEUE.capacity();
    for _ in 0..capacity {
        record_profile_sample("F1098_retained", 1);
    }
    for _ in 0..3 {
        record_profile_sample("F1098_rejected", 2);
    }
    assert_eq!(TELEMETRY_QUEUE.len(), capacity);
    report.drain_queue();
    assert_eq!(
        report.counts.get("F1098_retained"),
        Some(&(capacity as u64))
    );
    assert!(!report.counts.contains_key("F1098_rejected"));
    assert_eq!(report.dropped_samples, losses_before + 3);
    // Real record/drain is usable again; rejected-sample evidence persists.
    record_profile_sample("F1098_recovery", 17);
    report.drain_queue();
    assert_eq!(report.metrics.get("F1098_recovery"), Some(&17));
    assert_eq!(report.dropped_samples, losses_before + 3);
}

fn exported_macro_records_rejection_without_changing_result() {
    // F1-100: the macro converts its cycle span with the measured TSC rate.
    // Without calibration it counts the span instead of inventing a duration,
    // so this contract must calibrate first. A failure here is a real
    // measurement failure, not a skipped assertion.
    assert!(
        calibrate_tsc().is_some(),
        "el reloj TSC debe poder calibrarse para que el macro registre spans"
    );
    let mut report = Profiler::new();
    report.drain_queue();
    let losses_before = report.dropped_samples;
    for _ in 0..TELEMETRY_QUEUE.capacity() {
        record_profile_sample("F1098_fill", 1);
    }
    let result = telemetry_server::profile_node!("F1098_macro", { 42_u64 });
    assert_eq!(result, 42);
    report.drain_queue();
    assert_eq!(report.dropped_samples, losses_before + 1);
    assert!(!report.counts.contains_key("F1098_macro"));
    // After capacity recovers, the SAME exported macro reaches the report.
    let result = telemetry_server::profile_node!("F1098_macro", { 43_u64 });
    assert_eq!(result, 43);
    report.drain_queue();
    assert_eq!(report.counts.get("F1098_macro"), Some(&1));
    assert_eq!(report.dropped_samples, losses_before + 1);
    // This contract checks lineage, not the converted duration: the nanosecond
    // value depends on the host's measured TSC rate and is contracted by the
    // unit tests for `nanos_from_q32` and `calibrate_tsc`.
}
