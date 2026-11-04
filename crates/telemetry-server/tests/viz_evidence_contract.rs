use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use telemetry_server::hyperdimensional_graph::HyperdimensionalGraphSnapshot;
use telemetry_server::mmap_viz_stream::{
    MmapVizStreamReader, MmapVizStreamWriter, TelemetrySnapshot,
};

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(0);

fn temp_file(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "trader_gemini_task7_{label}_{}_{}.bin",
        std::process::id(),
        NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn valid_snapshot() -> TelemetrySnapshot {
    TelemetrySnapshot {
        tick_counter: 91,
        unified_capital: 7.25,
        pnl_realized: -1.5,
        global_leverage: 2.5,
        ai_ml_prob: 0.65,
        hurst_exponent: 0.7,
        memory_used_mb: 128.0,
        global_roi: -20.689_655_172_413_794,
        win_rate: 0.4,
        tensor_drift: 0.125,
    }
}

fn packet_value(
    reader: &mut MmapVizStreamReader,
    symbols: &[String],
    vn: &[f64],
    tsallis: &[f64],
    gauge: &[f64],
    void_risks: &[f64],
    betti_1: &[f64],
) -> Value {
    serde_json::to_value(
        reader.build_unified_packet(symbols, vn, tsallis, gauge, void_risks, betti_1),
    )
    .expect("el paquete debe ser serializable")
}

#[test]
fn absent_file_produces_no_snapshot_or_scene() {
    let path = temp_file("absent");
    let mut reader = MmapVizStreamReader::new(&path);

    assert!(reader.read_snapshot().is_none());
    let packet = packet_value(&mut reader, &[], &[], &[], &[], &[], &[]);
    assert!(
        packet.is_null(),
        "se fabricó una escena sin snapshot: {packet}"
    );
}

#[test]
fn invalid_required_fields_are_rejected_without_defaults() {
    let path = temp_file("invalid");
    let valid = valid_snapshot();
    let invalid = [
        TelemetrySnapshot {
            unified_capital: f64::NAN,
            ..valid
        },
        TelemetrySnapshot {
            pnl_realized: f64::NAN,
            ..valid
        },
        TelemetrySnapshot {
            global_leverage: f64::INFINITY,
            ..valid
        },
        TelemetrySnapshot {
            ai_ml_prob: f64::NAN,
            ..valid
        },
        TelemetrySnapshot {
            hurst_exponent: f64::NAN,
            ..valid
        },
        TelemetrySnapshot {
            memory_used_mb: f64::NAN,
            ..valid
        },
        TelemetrySnapshot {
            global_roi: f64::NAN,
            ..valid
        },
        TelemetrySnapshot {
            win_rate: f64::NAN,
            ..valid
        },
        TelemetrySnapshot {
            unified_capital: 0.0,
            ..valid
        },
        TelemetrySnapshot {
            global_leverage: 0.0,
            ..valid
        },
    ];
    let mut writer = MmapVizStreamWriter::new(&path).expect("writer");
    let mut reader = MmapVizStreamReader::new(&path);

    for snapshot in invalid {
        writer
            .write_snapshot(&snapshot)
            .expect("write invalid fixture");
        assert!(
            reader.read_snapshot().is_none(),
            "campo requerido inválido aceptado o sustituido: {snapshot:?}"
        );
    }

    drop(reader);
    drop(writer);
    std::fs::remove_file(path).expect("cleanup");
}

#[test]
fn nan_tensor_drift_remains_an_explicit_absence() {
    let path = temp_file("tensor");
    let snapshot = TelemetrySnapshot {
        tensor_drift: f64::NAN,
        ..valid_snapshot()
    };
    let mut writer = MmapVizStreamWriter::new(&path).expect("writer");
    writer.write_snapshot(&snapshot).expect("write");
    let mut reader = MmapVizStreamReader::new(&path);

    let actual = reader.read_snapshot().expect("snapshot requerido válido");
    assert!(actual.tensor_drift.is_nan());
    assert_eq!(actual.tick_counter, 91);
    assert_eq!(actual.unified_capital, 7.25);

    drop(reader);
    drop(writer);
    std::fs::remove_file(path).expect("cleanup");
}

#[test]
fn absent_tensor_has_no_lattice_or_fabricated_causality() {
    let path = temp_file("packet");
    let snapshot = TelemetrySnapshot {
        tensor_drift: f64::NAN,
        ..valid_snapshot()
    };
    let mut writer = MmapVizStreamWriter::new(&path).expect("writer");
    writer.write_snapshot(&snapshot).expect("write");
    let mut reader = MmapVizStreamReader::new(&path);

    let packet = packet_value(&mut reader, &[], &[], &[], &[], &[], &[]);
    assert!(packet.is_object(), "snapshot válido sin paquete: {packet}");
    assert_eq!(packet["unified_capital"], 7.25);
    assert_eq!(packet["pnl_realized"], -1.5);
    assert!(packet["timestamp_ns"].as_u64().expect("timestamp") > 0);
    assert!(packet["lattice"].is_null(), "lattice sin tensor: {packet}");
    assert!(packet["tail_ridge"].is_object());
    assert!(
        packet["hypergraph"]["nodes"]
            .as_array()
            .expect("nodes")
            .is_empty()
    );
    assert!(
        packet["causal_report"].is_null(),
        "causalidad fabricada: {packet}"
    );

    drop(reader);
    drop(writer);
    std::fs::remove_file(path).expect("cleanup");
}

#[test]
fn empty_auxiliary_state_produces_an_empty_graph() {
    let graph = serde_json::to_value(HyperdimensionalGraphSnapshot::build_from_state(
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ))
    .expect("graph serialization");

    assert!(graph["nodes"].as_array().expect("nodes").is_empty());
    assert!(graph["edges"].as_array().expect("edges").is_empty());
    assert!(graph["pods"].as_array().expect("pods").is_empty());
    assert_eq!(graph["total_energy"], 0.0);
    assert_eq!(graph["global_betti_1_cycles"], 0);
    assert!(!graph.to_string().contains("COIN_0"));
}

#[test]
fn writer_flushes_real_fields_for_an_independent_reader() {
    let path = temp_file("roundtrip");
    let expected = valid_snapshot();
    let mut writer = MmapVizStreamWriter::new(&path).expect("writer");
    writer.write_snapshot(&expected).expect("write and flush");
    let mut reader = MmapVizStreamReader::new(&path);

    let actual = reader.read_snapshot().expect("readback");
    assert_eq!(actual, expected);

    drop(reader);
    drop(writer);
    std::fs::remove_file(path).expect("cleanup");
}

#[test]
fn arena_snapshot_preserves_real_fields() {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(5.0);
    arena.tick_counter.store(314, Ordering::Relaxed);
    arena.config.global_leverage.store(2.75, Ordering::Relaxed);

    let mut ml_sum = 0.0;
    let mut hurst_sum = 0.0;
    for (index, coin) in arena.coins.iter().enumerate() {
        let ml = if index % 2 == 0 { 0.2 } else { 0.65 };
        let hurst = if index % 2 == 0 { 0.4 } else { 0.8 };
        coin.ml_prob.store(ml, Ordering::Relaxed);
        coin.hurst_exponent.store(hurst, Ordering::Relaxed);
        ml_sum += ml;
        hurst_sum += hurst;
    }
    telemetry_server::profiler::update_financials_atomic(23.0, 17.0, 0.625, 4.0);

    let snapshot = telemetry_server::snapshot_from_arena(&arena);

    assert_eq!(snapshot.tick_counter, 314);
    assert_eq!(snapshot.unified_capital, 5.0);
    assert_eq!(snapshot.global_leverage, 2.75);
    assert_eq!(snapshot.ai_ml_prob, ml_sum / arena.coins.len() as f64);
    assert_eq!(
        snapshot.hurst_exponent,
        hurst_sum / arena.coins.len() as f64
    );
    assert!(snapshot.memory_used_mb.is_finite() && snapshot.memory_used_mb > 0.0);
    assert_eq!(snapshot.pnl_realized, 17.0);
    assert_eq!(snapshot.global_roi, 425.0);
    assert_eq!(snapshot.win_rate, 0.625);
    assert!(snapshot.tensor_drift.is_nan());
}

#[test]
fn dashboard_source_guards_null_and_non_array_viz_payloads() {
    let source = include_str!("../src/lib.rs");

    assert!(source.contains("Array.isArray(causalData.attributions)"));
    assert!(source.contains("Array.isArray(hyperData.nodes)"));
    assert!(source.contains("Array.isArray(hyperData.pods)"));
    assert!(source.contains("N/D"));
    assert!(!source.contains("causalData.attributions.forEach"));
    assert!(!source.contains("hyperData.pods.forEach"));
}

#[test]
fn producer_and_trader_sources_share_the_environment_path() {
    let telemetry_source = include_str!("../src/lib.rs");
    let trader_source = include_str!("../../../src/bin/trader_viz.rs");

    assert!(telemetry_source.contains("pub fn telemetry_viz_path() -> String"));
    assert!(
        telemetry_source.contains("quantum_arena::paths::env_data_path(\"telemetry_mmap.bin\")")
    );
    assert!(trader_source.contains("telemetry_server::telemetry_viz_path()"));
    assert!(!trader_source.contains("possible_paths"));
    assert!(!trader_source.contains("data/prod/telemetry_mmap.bin"));
    assert!(!trader_source.contains("data/demo/telemetry_mmap.bin"));
}

#[test]
fn infinite_tensor_drift_invalidates_the_snapshot() {
    let path = temp_file("infinite_tensor");
    let mut writer = MmapVizStreamWriter::new(&path).expect("writer");
    let mut reader = MmapVizStreamReader::new(&path);

    for tensor_drift in [f64::INFINITY, f64::NEG_INFINITY] {
        let snapshot = TelemetrySnapshot {
            tensor_drift,
            ..valid_snapshot()
        };
        writer.write_snapshot(&snapshot).expect("write");
        assert!(
            reader.read_snapshot().is_none(),
            "tensor_drift infinito aceptado: {tensor_drift}"
        );
    }

    drop(reader);
    drop(writer);
    std::fs::remove_file(path).expect("cleanup");
}

#[test]
fn telemetry_viz_path_matches_demo_and_production_policy() {
    let _restore = EnvironmentRestore::capture();

    for (demo, expected) in [
        (true, "data/demo/telemetry_mmap.bin"),
        (false, "data/prod/telemetry_mmap.bin"),
    ] {
        // SAFETY: este binario de pruebas se ejecuta con --test-threads=1 y el
        // guard restaura las cuatro variables al terminar el escenario.
        unsafe {
            std::env::set_var("TG_GENOME_ENV", if demo { "demo" } else { "prod" });
            std::env::set_var("USE_TESTNET", if demo { "true" } else { "false" });
            std::env::remove_var("SHADOW_MODE");
            std::env::remove_var("BINANCE_USE_DEMO");
        }
        let policy = quantum_arena::paths::env_data_path("telemetry_mmap.bin");
        assert_eq!(telemetry_server::telemetry_viz_path(), policy);
        assert_eq!(policy, expected);
    }
}

struct EnvironmentRestore {
    entries: Vec<(&'static str, Option<String>)>,
}

impl EnvironmentRestore {
    fn capture() -> Self {
        const KEYS: [&str; 4] = [
            "TG_GENOME_ENV",
            "USE_TESTNET",
            "SHADOW_MODE",
            "BINANCE_USE_DEMO",
        ];
        Self {
            entries: KEYS
                .into_iter()
                .map(|key| (key, std::env::var(key).ok()))
                .collect(),
        }
    }
}

impl Drop for EnvironmentRestore {
    fn drop(&mut self) {
        for (key, value) in &self.entries {
            // SAFETY: restauración del entorno en el mismo binario monohilo.
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}
