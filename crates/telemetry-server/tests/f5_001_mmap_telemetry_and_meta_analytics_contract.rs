//! Contratos Formales de Verificación de Fase 5 (F5-001)
//!
//! Suite de pruebas y contratos matemáticos para:
//! - Streaming Visual Desacoplado vía `mmap` lock-free.
//! - Simulación Probabilística Mirofish & 3D Tail Ridge en 32 escalas.
//! - Grafo Hiperdimensional 3D & Pods de Aislamiento Cuántico.
//! - Atribución Causal de PnL Aumann-Shapley, Valores de Owen & Eficiencia Silicio eta_CPU.
//! - Detector Secuencial de Puntos de Cambio de Shiryaev-Roberts-Pollak (SRP).
//! - Cumplimiento estricto del invariante U-ERR-6 (cero sufijos de banda scalp/swing).

use telemetry_server::causal_attribution::{
    CausalAttributionEngine, ShiryaevRobertsPollakDetector, SUBSYSTEM_COUNT,
};
use telemetry_server::hyperdimensional_graph::{
    HyperdimensionalGraphSnapshot, PodIsolationState,
};
use telemetry_server::mirofish_lattice::{MirofishLattice, TailRidge3D};
use telemetry_server::mmap_viz_stream::{
    MmapVizStreamReader, MmapVizStreamWriter, TelemetrySnapshot,
};

/// Contrato C1: Desacoplamiento Lock-Free & Integridad Mmap Zero-Copy
#[test]
fn test_c1_mmap_lock_free_decoupling_and_readback() {
    let temp_dir = std::env::temp_dir();
    let unique_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = temp_dir.join(format!("contract_c1_mmap_{}.bin", unique_id));
    let path_str = path.to_string_lossy().to_string();

    let snap_init = TelemetrySnapshot {
        tick_counter: 4242,
        unified_capital: 150.0,
        pnl_realized: 25.0,
        global_leverage: 1.0,
        ai_ml_prob: 0.75,
        hurst_exponent: 0.60,
        memory_used_mb: 120.0,
        global_roi: 50.0,
        win_rate: 0.80,
        tensor_drift: 0.0,
    };

    let mut writer = MmapVizStreamWriter::new(&path).expect("writer init");
    writer.write_snapshot(&snap_init).expect("write initial snapshot");

    // Lector desacoplado
    let mut reader = MmapVizStreamReader::new(&path_str);
    let snap_opt = reader.read_snapshot();
    assert!(snap_opt.is_some(), "Snapshot debe ser leído exitosamente");

    let snap = snap_opt.unwrap();
    assert_eq!(snap.tick_counter, 4242);
    assert_eq!(snap.unified_capital, 150.0);
    assert_eq!(snap.pnl_realized, 25.0);
    assert!((snap.global_roi - 50.0).abs() < 1e-4); // (150 - 100)/100 = 50%

    // Segunda escritura concurrente vía el escritor mmap compartido
    let snap_update = TelemetrySnapshot {
        tick_counter: 5000,
        unified_capital: 200.0,
        ..snap_init
    };
    writer.write_snapshot(&snap_update).expect("second write");

    let snap2 = reader.read_snapshot().expect("segunda lectura");
    assert_eq!(snap2.tick_counter, 5000);
    assert_eq!(snap2.unified_capital, 200.0);

    drop(reader);
    drop(writer);
    let _ = std::fs::remove_file(path);
}

/// Contrato C2: Conservación de Probabilidad y Relieve de 32 Escalas en Mirofish
#[test]
fn test_c2_mirofish_probability_conservation_and_32_scales() {
    let spot = 250.0;
    let vol = 0.55;
    let ofi = 0.35;
    let ising = 0.15;
    let alpha = 0.05;

    let lattice = MirofishLattice::compute(spot, vol, ofi, ising, alpha, 8, 1.0);

    // Suma de probabilidades exacta en cada nivel
    for level in &lattice.levels {
        let sum_p: f64 = level.nodes.iter().map(|n| n.probability).sum();
        assert!(
            (sum_p - 1.0).abs() < 1e-9,
            "Nivel {} violó la conservación de probabilidad: sum={}",
            level.step,
            sum_p
        );
        assert!(level.expected_price > 0.0);
        assert!(level.variance >= 0.0);
    }

    // Comprobación del relieve de colas 3D
    let ridge = TailRidge3D::generate(spot, vol, 0.65, alpha);
    assert_eq!(ridge.points.len(), 32, "Debe abarcar exactamente las 32 escalas diádicas");

    // Monotonicidad del exceso de curtosis (la microestructura es más leptocúrtica que el horizonte secular)
    assert!(
        ridge.points[0].excess_kurtosis > ridge.points[31].excess_kurtosis,
        "Curtosis de microestructura ({}) debe exceder la secular ({})",
        ridge.points[0].excess_kurtosis,
        ridge.points[31].excess_kurtosis
    );

    for p in &ridge.points {
        assert!(p.conformal_band_low <= spot);
        assert!(p.conformal_band_high >= spot);
        assert!(p.left_tail_cvar >= 0.0);
    }
}

/// Contrato C3: Bisección Espectral y Cuarentena de Pods en Grafo Hiperdimensional
#[test]
fn test_c3_hyperdimensional_graph_isolation_pods() {
    let symbols = vec![
        "BTCUSDT".to_string(),
        "ETHUSDT".to_string(),
        "SOLUSDT".to_string(),
        "AVAXUSDT".to_string(),
    ];
    let vn = vec![0.30, 0.40, 0.95, 0.35]; // SOLUSDT con alta entropía
    let tsallis = vec![0.35, 0.45, 0.98, 0.40];
    let gauge = vec![0.5, 1.0, 15.0, 0.8]; // SOLUSDT con acción gauge divergente
    let void_risks = vec![0.05, 0.10, 0.88, 0.08]; // SOLUSDT con riesgo de vacío extremo
    let betti_1 = vec![0.0, 0.0, 3.0, 0.0];

    let snap = HyperdimensionalGraphSnapshot::build_from_state(
        &symbols,
        &vn,
        &tsallis,
        &gauge,
        &void_risks,
        &betti_1,
    );

    assert_eq!(snap.nodes.len(), 4);
    assert_eq!(snap.global_betti_1_cycles, 3);

    // SOLUSDT (índice 2) debe estar en cuarentena estricta
    assert_eq!(snap.nodes[2].isolation_state, PodIsolationState::Quarantined);
    assert_eq!(snap.nodes[0].isolation_state, PodIsolationState::Optimal);
    assert_eq!(snap.nodes[1].isolation_state, PodIsolationState::Optimal);
    assert_eq!(snap.nodes[3].isolation_state, PodIsolationState::Optimal);

    // Debe existir al menos un pod en cuarentena que contenga el nodo 2
    let q_pod = snap.pods.iter().find(|p| p.status == PodIsolationState::Quarantined);
    assert!(q_pod.is_some(), "Debe existir un pod de aislamiento en cuarentena");
    assert!(q_pod.unwrap().member_nodes.contains(&2));
}

/// Contrato C4: Axioma de Eficiencia Exacta de Aumann-Shapley & Owen
#[test]
fn test_c4_aumann_shapley_and_owen_exact_efficiency() {
    let mut engine = CausalAttributionEngine::new();
    let activity = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
    let cycles = [3000, 4500, 2500, 1800, 1200, 5200];
    let misses = [25, 40, 15, 10, 8, 45];
    let gross_pnl = 320.0;
    let fees = 32.0;
    let expected_net = gross_pnl - fees;

    let report = engine.evaluate(&activity, &cycles, &misses, gross_pnl, fees);

    assert_eq!(report.attributions.len(), SUBSYSTEM_COUNT);

    // Suma de Aumann-Shapley == Net PnL exacto
    let as_sum: f64 = report.attributions.iter().map(|a| a.aumann_shapley_pnl).sum();
    assert!(
        (as_sum - expected_net).abs() < 1e-9,
        "Aumann-Shapley {} != {}",
        as_sum,
        expected_net
    );

    // Suma de Owen == Net PnL exacto
    let owen_sum: f64 = report.attributions.iter().map(|a| a.owen_coalition_pnl).sum();
    assert!(
        (owen_sum - expected_net).abs() < 1e-9,
        "Owen {} != {}",
        owen_sum,
        expected_net
    );

    // Comisiones atribuidas == Total fees
    let fee_sum: f64 = report.attributions.iter().map(|a| a.attributed_fees).sum();
    assert!(
        (fee_sum - fees).abs() < 1e-9,
        "Fees atribuidas {} != {}",
        fee_sum,
        fees
    );
}

/// Contrato C5: Detector Secuencial Shiryaev-Roberts-Pollak & Gobernador Silicio-Alpha
#[test]
fn test_c5_srp_change_detection_and_silicon_efficiency() {
    let mut detector = ShiryaevRobertsPollakDetector::new(60.0, 0.05, -0.05, 0.02);

    // 10 muestras en régimen positivo -> no alarma
    for _ in 0..10 {
        let (r, alarm) = detector.update(0.06);
        assert!(!alarm);
        assert!(r < 60.0);
    }

    // Inyección de degradación severa sostenida
    let mut alarm_triggered = false;
    for _ in 0..30 {
        let (_, alarm) = detector.update(-0.10);
        if alarm {
            alarm_triggered = true;
            break;
        }
    }
    assert!(alarm_triggered, "SRP debió detectar el punto de cambio y disparar alarma");

    // Gobernador de Silicio
    let mut engine = CausalAttributionEngine::new();
    let activity = [1.0; 6];
    let cycles = [1000, 10_000, 1000, 1000, 1000, 1000];
    let misses = [10, 10, 10, 10, 10, 10];
    let report = engine.evaluate(&activity, &cycles, &misses, 100.0, 10.0);

    let eta_0 = report.attributions[0].silicon_efficiency_eta;
    let eta_1 = report.attributions[1].silicon_efficiency_eta;
    assert!(
        eta_0 > eta_1,
        "Mayor consumo de ciclos debe reducir eta_CPU: {} vs {}",
        eta_0,
        eta_1
    );
}

/// Contrato C6: Cumplimiento U-ERR-6 (Cero Falsas Divisiones Scalp/Swing en Wire)
#[test]
fn test_c6_u_err_6_zero_scalp_swing_in_wire_protocol() {
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("contract_c6_u_err_6.bin");
    let mut reader = MmapVizStreamReader::new(&path);

    let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
    let vn = vec![0.5, 0.5];
    let tsallis = vec![0.5, 0.5];
    let gauge = vec![0.0, 0.0];
    let void = vec![0.0, 0.0];
    let betti_1 = vec![0.0, 0.0];

    let packet = reader.build_unified_packet(&symbols, &vn, &tsallis, &gauge, &void, &betti_1);
    let json_packet = serde_json::to_string(&packet).expect("serialization packet");
    let lower_json = json_packet.to_ascii_lowercase();

    // Verificación estricta: ninguna clave en el wire lleva scalp ni swing
    assert!(
        !lower_json.contains("\"scalp"),
        "Wire contiene claves con scalp: {}",
        json_packet
    );
    assert!(
        !lower_json.contains("\"swing"),
        "Wire contiene claves con swing: {}",
        json_packet
    );
}
