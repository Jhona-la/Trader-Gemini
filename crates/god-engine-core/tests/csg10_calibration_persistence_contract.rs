//! CSG-10/LH-12: the persisted record must describe the evidence actually learned.
//! These contracts use Core ingestion, the real snapshot, disk save/load and
//! restoration. They do not copy Platt's timestamp normalization or optimizer.
use god_engine_core::calibration::{
    PlattCalibrator, MAX_CLOCK_JUMP_MS, MAX_PLAUSIBLE_EPOCH_MS, SEMIVIDA_EVIDENCIA_MS, WINDOW,
};
use god_engine_core::estado_aprendido::{self, Carga, EstadoAprendido, MonedaGuardada};
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, GlobalArena};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

static SYMBOLS: Mutex<()> = Mutex::new(());
static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
const T: u64 = 1_800_000_000_000;

fn symbol_guard() -> MutexGuard<'static, ()> {
    // A designated baseline assertion may panic. It must not turn later
    // contracts into mutex-poison failures instead of their own assertions.
    SYMBOLS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn core() -> GodEngineCore {
    quantum_arena::symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "ETHUSDT".into()]);
    let symbols = ["BTCUSDT", "ETHUSDT"]
        .into_iter()
        .map(|symbol| symbol_registry::SymbolSpec {
            symbol: symbol.into(),
            step_size: 0.001,
            tick_size: 0.01,
            min_qty: 0.001,
            min_notional: 5.0,
            max_leverage: 20,
            maker_fee: 0.0002,
            taker_fee: 0.0004,
            is_shadow: false,
        })
        .collect();
    symbol_registry::update_registry(symbols);
    GodEngineCore::new(GlobalArena::build_in_own_stack(1000.0))
}

struct StateDir(PathBuf);

impl StateDir {
    fn new() -> Self {
        loop {
            let nonce = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "tg-csg10-calibration-persistence-{}-{nonce}",
                std::process::id()
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create isolated state directory: {error}"),
            }
        }
    }

    fn file(&self) -> PathBuf {
        self.0.join("state.json")
    }
}

impl Drop for StateDir {
    fn drop(&mut self) {
        // Only the two known files in this exclusively created directory.
        let _ = std::fs::remove_file(self.file());
        let _ = std::fs::remove_file(self.file().with_extension("tmp"));
        let _ = std::fs::remove_dir(&self.0);
    }
}

fn coin<'a>(state: &'a EstadoAprendido, symbol: &str) -> &'a MonedaGuardada {
    state.monedas.iter().find(|row| row.simbolo == symbol).unwrap()
}

fn timestamps(state: &EstadoAprendido, symbol: &str) -> Vec<u64> {
    coin(state, symbol).calibrador.iter().map(|row| row.2).collect()
}

fn global_timestamps(state: &EstadoAprendido) -> Vec<u64> {
    state.calibrador_global.iter().map(|row| row.2).collect()
}

fn assert_same_calibrator(before: &PlattCalibrator, after: &PlattCalibrator, now: u64) {
    assert_eq!(after.observations(), before.observations());
    assert_eq!(after.evidence_clock_ms(), before.evidence_clock_ms());
    let (a0, b0) = before.coefficients();
    let (a1, b1) = after.coefficients();
    // The constrained Newton solver may take different warm starts during
    // replay; compare its coefficients numerically, not bitwise. Its ridge is
    // 1e-4 and projected-gradient stopping tolerance is 1e-9.
    for (left, right) in [(a0, a1), (b0, b1)] {
        assert!(left.is_finite() && right.is_finite());
        assert!(
            (left - right).abs() < 5e-5,
            "coefficient drift: {left} vs {right}"
        );
    }
    // At converged iterates, ridge 1e-4 and KKT tolerance 1e-9 allow
    // coefficient differences up to about 2*sqrt(2)*1e-5 between fits.
    // Sigmoid is 1/4-Lipschitz. A 1e-8 probability tolerance at unobserved
    // scores would be stricter than this numerical contract justifies.
    // This does not assert universal convergence of the bounded solver.
    for time in [now, now + SEMIVIDA_EVIDENCIA_MS as u64] {
        for score in [0.2, 0.5, 0.8] {
            let p0 = before.calibrate_at(score, time);
            let p1 = after.calibrate_at(score, time);
            assert!(p0.is_finite() && p1.is_finite());
            assert!(
                (p0 - p1).abs() < 4e-5,
                "probability drift: {p0} vs {p1}"
            );
        }
    }
}

fn assert_disk_roundtrip(live: &GodEngineCore, now: u64) {
    let state = live.estado_aprendido();
    estado_aprendido::validar(&state).expect("learned evidence must remain persistable");
    let directory = StateDir::new();
    estado_aprendido::guardar(&directory.file(), &state).expect("save real learned state");
    let loaded = match estado_aprendido::cargar(&directory.file()) {
        Carga::Leido(state) => state,
        other => panic!("load saved learned state: {other:?}"),
    };
    assert_eq!(loaded.calibrador_global, state.calibrador_global);
    assert_eq!(loaded.calibrador_global_reloj_ms, state.calibrador_global_reloj_ms);
    for symbol in ["BTCUSDT", "ETHUSDT"] {
        assert_eq!(
            coin(&loaded, symbol).calibrador,
            coin(&state, symbol).calibrador
        );
        assert_eq!(
            coin(&loaded, symbol).calibrador_reloj_ms,
            coin(&state, symbol).calibrador_reloj_ms
        );
    }
    let mut restored = core();
    assert_eq!(restored.restaurar_estado_aprendido(&loaded), 2);
    for index in 0..2 {
        assert_same_calibrator(
            &live.calibrator_by_coin[index],
            &restored.calibrator_by_coin[index],
            now,
        );
    }
    assert_same_calibrator(
        &live.confidence_calibrator,
        &restored.confidence_calibrator,
        now,
    );
    let after = restored.estado_aprendido();
    estado_aprendido::validar(&after).unwrap();
    assert_eq!(after.calibrador_global, loaded.calibrador_global);
    for symbol in ["BTCUSDT", "ETHUSDT"] {
        assert_eq!(
            coin(&after, symbol).calibrador,
            coin(&loaded, symbol).calibrador
        );
    }
}

#[test]
fn csg10_anchored_corrupt_timestamp_stays_persistable() {
    let _guard = symbol_guard();
    let mut live = core();
    let corrupt = T * 1000;
    assert!(corrupt > MAX_PLAUSIBLE_EPOCH_MS);
    live.alimentar_calibradores(0, 0.8, true, T);
    live.alimentar_calibradores(0, 0.8, false, corrupt);
    assert_eq!(live.calibrator_by_coin[0].observations(), 2);
    assert_eq!(live.confidence_calibrator.observations(), 2);
    assert_eq!(live.calibrator_by_coin[0].evidence_clock_ms(), Some(T));
    assert_eq!(live.confidence_calibrator.evidence_clock_ms(), Some(T));
    let state = live.estado_aprendido();
    estado_aprendido::validar(&state).expect("corrupt anchored sample must remain persistable");
    assert_eq!(timestamps(&state, "BTCUSDT"), vec![T, T]);
    assert_eq!(global_timestamps(&state), vec![T, T]);
    assert_disk_roundtrip(&live, T);
}

#[test]
fn csg10_unanchored_producer_rejects_corrupt_sample_then_accepts_valid_date() {
    let _guard = symbol_guard();
    let mut live = core();
    live.alimentar_calibradores(0, 0.8, false, T * 1000);
    assert_eq!(
        live.calibrator_by_coin[0].observations(),
        0,
        "unanchored persistent producer must not learn an unrecordable date"
    );
    assert_eq!(live.confidence_calibrator.observations(), 0);
    for calibrator in [&live.calibrator_by_coin[0], &live.confidence_calibrator] {
        assert_eq!(calibrator.evidence_clock_ms(), None);
        assert_eq!(calibrator.coefficients(), (1.0, 0.0));
    }
    let empty = live.estado_aprendido();
    assert!(coin(&empty, "BTCUSDT").calibrador.is_empty());
    assert!(empty.calibrador_global.is_empty());
    assert_disk_roundtrip(&live, T);

    // The direct, non-persistent Platt API keeps its documented behavior.
    let mut direct = PlattCalibrator::new();
    direct.update_at(0.8, false, T * 1000);
    assert_eq!(direct.observations(), 1);
    assert_eq!(direct.evidence_clock_ms(), None);

    live.alimentar_calibradores(0, 0.8, true, T);
    assert_eq!(live.calibrator_by_coin[0].observations(), 1);
    assert_eq!(live.confidence_calibrator.observations(), 1);
    let state = live.estado_aprendido();
    assert_eq!(timestamps(&state, "BTCUSDT"), vec![T]);
    assert_eq!(global_timestamps(&state), vec![T]);
    assert_eq!(coin(&state, "BTCUSDT").calibrador_reloj_ms, Some(T));
    assert_disk_roundtrip(&live, T);
}

#[test]
fn csg10_global_and_coin_use_their_own_evidence_clocks() {
    let _guard = symbol_guard();
    let mut live = core();
    let global_clock = T + MAX_CLOCK_JUMP_MS / 2;
    live.alimentar_calibradores(1, 0.7, true, global_clock);
    live.alimentar_calibradores(0, 0.8, false, T * 1000);
    assert_eq!(
        live.calibrator_by_coin[0].observations(),
        0,
        "an anchored global model must not supply a fabricated clock to an unanchored coin"
    );
    assert_eq!(live.confidence_calibrator.observations(), 2);
    live.alimentar_calibradores(0, 0.8, true, T);
    live.alimentar_calibradores(0, 0.8, false, T * 1000);
    assert_eq!(live.calibrator_by_coin[0].observations(), 2);
    assert_eq!(live.confidence_calibrator.observations(), 4);
    assert_eq!(live.calibrator_by_coin[0].evidence_clock_ms(), Some(T));
    assert_eq!(
        live.confidence_calibrator.evidence_clock_ms(),
        Some(global_clock)
    );
    let state = live.estado_aprendido();
    assert_eq!(timestamps(&state, "BTCUSDT"), vec![T, T]);
    assert_eq!(timestamps(&state, "ETHUSDT"), vec![global_clock]);
    assert_eq!(
        global_timestamps(&state),
        vec![global_clock, global_clock, T, global_clock]
    );
    assert_disk_roundtrip(&live, global_clock);
}

#[test]
fn csg10_legitimate_late_and_zero_timestamps_keep_their_identity() {
    let _guard = symbol_guard();
    let mut live = core();
    for time in [0, 1000, 500] {
        live.alimentar_calibradores(0, 0.8, time == 0, time);
    }
    assert_eq!(live.calibrator_by_coin[0].observations(), 3);
    assert_eq!(live.confidence_calibrator.observations(), 3);
    assert_eq!(live.calibrator_by_coin[0].evidence_clock_ms(), Some(1000));
    let state = live.estado_aprendido();
    assert_eq!(timestamps(&state, "BTCUSDT"), vec![0, 1000, 500]);
    assert_eq!(global_timestamps(&state), vec![0, 1000, 500]);
    assert_disk_roundtrip(&live, 1000);
}

#[test]
fn csg10_excessive_clock_jump_reuses_the_accepted_anchor() {
    let _guard = symbol_guard();
    let mut live = core();
    let jump = T + MAX_CLOCK_JUMP_MS + 1;
    assert!(jump < MAX_PLAUSIBLE_EPOCH_MS);
    live.alimentar_calibradores(0, 0.8, true, T);
    live.alimentar_calibradores(0, 0.8, false, jump);
    assert_eq!(live.calibrator_by_coin[0].observations(), 2);
    assert_eq!(live.confidence_calibrator.observations(), 2);
    let state = live.estado_aprendido();
    estado_aprendido::validar(&state).expect("rejected annual jump must remain persistable");
    assert_eq!(timestamps(&state, "BTCUSDT"), vec![T, T]);
    assert_eq!(global_timestamps(&state), vec![T, T]);
    assert_disk_roundtrip(&live, T);
}

#[test]
fn csg10_evicted_maximum_clock_survives_disk_roundtrip() {
    let _guard = symbol_guard();
    let mut live = core();
    let maximum = 1_700_000_000_000_u64;
    let late = maximum - MAX_CLOCK_JUMP_MS - 1;
    live.alimentar_calibradores(0, 0.8, true, maximum);
    for _ in 0..WINDOW {
        live.alimentar_calibradores(0, 0.8, false, late);
    }
    assert_eq!(live.calibrator_by_coin[0].observations(), WINDOW);
    assert_eq!(live.confidence_calibrator.observations(), WINDOW);
    assert_eq!(live.calibrator_by_coin[0].evidence_clock_ms(), Some(maximum));
    assert_eq!(live.confidence_calibrator.evidence_clock_ms(), Some(maximum));
    let state = live.estado_aprendido();
    estado_aprendido::validar(&state).unwrap();
    assert_eq!(coin(&state, "BTCUSDT").calibrador_reloj_ms, Some(maximum));
    assert_eq!(state.calibrador_global_reloj_ms, Some(maximum));
    assert_eq!(timestamps(&state, "BTCUSDT"), vec![late; WINDOW]);
    assert_eq!(global_timestamps(&state), vec![late; WINDOW]);
    assert_disk_roundtrip(&live, maximum);
}

// Legacy snapshots are constructed explicitly. Feeding a future runtime event
// across a >365-day gap would legitimately trigger CSG-10's ingestion policy
// and would not represent the historical state being migrated here.
fn legacy_snapshot(rows: &[(f64, bool, u64)]) -> EstadoAprendido {
    let empty = core();
    let mut state = empty.estado_aprendido();
    let btc = state.monedas.iter_mut().find(|row| row.simbolo == "BTCUSDT").unwrap();
    btc.calibrador = rows.iter().copied().collect();
    btc.calibrador_reloj_ms = None;
    state.calibrador_global = rows.iter().copied().collect();
    state.calibrador_global_reloj_ms = None;
    state
}

fn restore_legacy_from_disk(state: &EstadoAprendido) -> GodEngineCore {
    estado_aprendido::validar(state).expect("legacy fixture must enter through the real loader");
    let directory = StateDir::new();
    let path = directory.file();
    estado_aprendido::guardar(&path, state).expect("write explicit legacy fixture");
    let original_bytes = std::fs::read(&path).unwrap();
    let loaded = match estado_aprendido::cargar(&path) {
        Carga::Leido(state) => state,
        other => panic!("load explicit legacy fixture: {other:?}"),
    };
    assert_eq!(loaded.calibrador_global, state.calibrador_global);
    assert_eq!(loaded.calibrador_global_reloj_ms, None);
    assert_eq!(coin(&loaded, "BTCUSDT").calibrador, coin(state, "BTCUSDT").calibrador);
    assert_eq!(coin(&loaded, "BTCUSDT").calibrador_reloj_ms, None);
    let mut restored = core();
    assert_eq!(restored.restaurar_estado_aprendido(&loaded), 2);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        original_bytes,
        "migration must not silently overwrite its legacy source file"
    );
    // The caller saves the migrated result through assert_disk_roundtrip,
    // which allocates a distinct directory and file from the legacy fixture.
    restored
}

#[test]
fn csg10_legacy_without_clock_infers_maximum_before_replay() {
    let _guard = symbol_guard();
    let maximum = 1_700_000_000_000_u64;
    let late = maximum - MAX_CLOCK_JUMP_MS - 1;
    let legacy = legacy_snapshot(&[(0.8, false, late), (0.8, true, maximum)]);
    let restored = restore_legacy_from_disk(&legacy);
    assert_eq!(
        restored.calibrator_by_coin[0].evidence_clock_ms(),
        Some(maximum),
        "legacy maximum must be inferred before replay across the annual jump"
    );
    assert_eq!(restored.confidence_calibrator.evidence_clock_ms(), Some(maximum));
    assert_eq!(restored.calibrator_by_coin[0].observations(), 2);
    assert_eq!(restored.confidence_calibrator.observations(), 2);
    let migrated = restored.estado_aprendido();
    assert_eq!(timestamps(&migrated, "BTCUSDT"), vec![late, maximum]);
    assert_eq!(global_timestamps(&migrated), vec![late, maximum]);
    assert_eq!(coin(&migrated, "BTCUSDT").calibrador, coin(&legacy, "BTCUSDT").calibrador);

    // The reverse arrival order is legitimate runtime ingestion. It carries
    // the same dated likelihood without submitting a forbidden future jump.
    let mut reference = core();
    reference.alimentar_calibradores(0, 0.8, true, maximum);
    reference.alimentar_calibradores(0, 0.8, false, late);
    assert_same_calibrator(
        &reference.calibrator_by_coin[0],
        &restored.calibrator_by_coin[0],
        maximum,
    );
    assert_same_calibrator(
        &reference.confidence_calibrator,
        &restored.confidence_calibrator,
        maximum,
    );
    assert_disk_roundtrip(&restored, maximum);
}

#[test]
fn csg10_legacy_without_clock_discards_only_corrupt_dates() {
    let _guard = symbol_guard();
    let corrupt = 1_700_000_000_000_000_u64;
    assert!(corrupt > MAX_PLAUSIBLE_EPOCH_MS);
    let legacy = legacy_snapshot(&[(0.8, false, corrupt)]);
    let restored = restore_legacy_from_disk(&legacy);
    assert_eq!(
        restored.calibrator_by_coin[0].observations(),
        0,
        "legacy impossible date must not become undated learned evidence"
    );
    assert_eq!(restored.confidence_calibrator.observations(), 0);
    for calibrator in [&restored.calibrator_by_coin[0], &restored.confidence_calibrator] {
        assert_eq!(calibrator.evidence_clock_ms(), None);
        assert_eq!(calibrator.coefficients(), (1.0, 0.0));
    }
    let migrated = restored.estado_aprendido();
    assert!(coin(&migrated, "BTCUSDT").calibrador.is_empty());
    assert!(migrated.calibrador_global.is_empty());
    assert_eq!(coin(&migrated, "BTCUSDT").calibrador_reloj_ms, None);
    assert_eq!(migrated.calibrador_global_reloj_ms, None);
    let reference = core();
    assert_same_calibrator(
        &reference.calibrator_by_coin[0],
        &restored.calibrator_by_coin[0],
        T,
    );
    assert_same_calibrator(&reference.confidence_calibrator, &restored.confidence_calibrator, T);
    assert_disk_roundtrip(&restored, T);
}

#[test]
fn csg10_legacy_without_clock_keeps_valid_row_and_discards_corrupt_row() {
    let _guard = symbol_guard();
    let maximum = 1_700_000_000_000_u64;
    let corrupt = maximum * 1000;
    let legacy = legacy_snapshot(&[(0.8, true, maximum), (0.8, false, corrupt)]);
    let restored = restore_legacy_from_disk(&legacy);
    assert_eq!(
        restored.calibrator_by_coin[0].observations(),
        1,
        "an existing anchor must not rehabilitate an impossible legacy date"
    );
    assert_eq!(restored.confidence_calibrator.observations(), 1);
    assert_eq!(restored.calibrator_by_coin[0].evidence_clock_ms(), Some(maximum));
    assert_eq!(restored.confidence_calibrator.evidence_clock_ms(), Some(maximum));
    let migrated = restored.estado_aprendido();
    assert_eq!(timestamps(&migrated, "BTCUSDT"), vec![maximum]);
    assert_eq!(global_timestamps(&migrated), vec![maximum]);
    assert_eq!(
        coin(&migrated, "BTCUSDT").calibrador.iter().copied().collect::<Vec<_>>(),
        vec![(0.8, true, maximum)]
    );
    let mut reference = core();
    reference.alimentar_calibradores(0, 0.8, true, maximum);
    assert_same_calibrator(
        &reference.calibrator_by_coin[0],
        &restored.calibrator_by_coin[0],
        maximum,
    );
    assert_same_calibrator(
        &reference.confidence_calibrator,
        &restored.confidence_calibrator,
        maximum,
    );
    assert_disk_roundtrip(&restored, maximum);
}
