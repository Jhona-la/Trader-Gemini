//! # Contrato de Verificación Formal F27-001: Flight Recorder LZCNT Histograms
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Batería de contratos formales para el registrador de vuelo y el histograma logarítmico LZCNT.
//! - **POR QUÉ:** La telemetría de latencias en microsegundos y nanosegundos debe ser evaluada en O(1)
//!   sin bloqueos ni alocaciones dinámicas, garantizando monotonía de percentiles y alineación de silicio a 64 bytes.
//! - **PARA QUÉ:** Demostrar que el sistema puede rastrear y diagnosticar latencias con fidelidad de nanosegundos
//!   en hardware de pocos recursos sin perturbar el hot-path.
//! - **CÓMO:** Ejecución de pruebas unitarias sobre bins logarítmicos, percentiles y estructura de memoria.

#![allow(clippy::all)]

use flight_recorder::mmap_flight_recorder::{
    FlightEventRecord, LOG_HISTOGRAM_BINS, LzcntLogHistogram, MmapFlightRecordRing,
};

#[test]
fn c1_lzcnt_histogram_zero_and_boundary_bins() {
    let hist = LzcntLogHistogram::new();

    // 1. Latencias pequeñas
    hist.record(0);
    hist.record(1); // 2^0 -> bin 0
    hist.record(2); // 2^1 -> bin 1
    hist.record(1024); // 2^10 -> bin 10
    hist.record(1_048_576); // 2^20 -> bin 20

    assert_eq!(hist.sample_count(), 5);
    assert_eq!(hist.min_latency_ns(), 0);
    assert_eq!(hist.max_latency_ns(), 1_048_576);
}

#[test]
fn c2_lzcnt_histogram_sample_count_and_moments() {
    let hist = LzcntLogHistogram::new();

    // Inyectar 100 muestras de 50 ns
    for _ in 0..100 {
        hist.record(50);
    }

    assert_eq!(hist.sample_count(), 100);
    assert_eq!(hist.min_latency_ns(), 50);
    assert_eq!(hist.max_latency_ns(), 50);
    assert!((hist.mean_latency_ns() - 50.0).abs() < 1e-6);
}

#[test]
fn c3_lzcnt_histogram_percentile_monotonicity() {
    let hist = LzcntLogHistogram::new();

    // Inyectar una distribución sintética con cola pesada (de 10 ns a 10,000 ns)
    for i in 1..=1000 {
        let lat = i * 10;
        hist.record(lat);
    }

    let p50 = hist.percentile(0.50);
    let p90 = hist.percentile(0.90);
    let p99 = hist.percentile(0.99);
    let p999 = hist.percentile(0.999);
    let max = hist.max_latency_ns();

    assert!(
        p50 <= p90,
        "p50 ({}) debe ser menor o igual que p90 ({})",
        p50,
        p90
    );
    assert!(
        p90 <= p99,
        "p90 ({}) debe ser menor o igual que p99 ({})",
        p90,
        p99
    );
    assert!(
        p99 <= p999,
        "p99 ({}) debe ser menor o igual que p99.9 ({})",
        p99,
        p999
    );
    assert!(
        p999 <= max,
        "p99.9 ({}) debe ser menor o igual que max ({})",
        p999,
        max
    );
}

#[test]
fn c4_mmap_flight_record_ring_sequential_wrap() {
    let mut ring = MmapFlightRecordRing::<8>::new();

    // Registrar 12 eventos (más que la capacidad de 8, probando el wrap circular)
    for i in 1..=12 {
        ring.record_event(
            1_000_000 + i as u64,
            1, // Evento tick
            42,
            50 + i as u32,
            100.0 + i as f64,
            1.5,
        );
    }

    let latest = ring.latest_event();
    assert!(latest.is_some());
    let ev = latest.unwrap();
    assert_eq!(
        ev.sequence, 11,
        "Último sequence index debe ser 11 (evento 12)"
    );
    assert_eq!(ev.latency_ns, 62);
    assert_eq!(ring.histogram().sample_count(), 12);
}

#[test]
fn c5_flight_event_record_memory_alignment() {
    use std::mem::{align_of, size_of};

    // Verificar que la estructura está alineada a 64 bytes (1 cache-line de CPU x86_64)
    let align = align_of::<FlightEventRecord>();
    assert_eq!(
        align, 64,
        "FlightEventRecord debe tener alineación estricta a 64 bytes para evitar false-sharing"
    );

    let size = size_of::<FlightEventRecord>();
    assert_eq!(
        size % 64,
        0,
        "FlightEventRecord tamaño ({}) debe ser múltiplo exacto de 64 bytes",
        size
    );
}

#[test]
fn c6_lzcnt_histogram_constant_total_bins() {
    assert_eq!(
        LOG_HISTOGRAM_BINS, 64,
        "El número total de bins debe ser exactamente 64 para mapeo O(1) con 64-bit LZCNT"
    );
}
