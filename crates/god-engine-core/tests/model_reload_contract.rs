//! Exercise the exact std-only policy module without starting the trading host.
//! GLM 134 (fix CI E0433 de Ω144): el módulo se importa del CRATE en vez de
//! la inclusión aislada `#[path]` — Ω144 añadió `preload_models_from_disk()`
//! que referencia `crate::ml_inference::NanoForest` (resolvible sólo dentro
//! de god-engine-core). El módulo es `pub` (lib.rs) y ningún test de este
//! archivo necesita el aislamiento std-only: el fixture y el tracker son
//! del propio módulo público.
use god_engine_core::model_reload::ModelReloadTracker;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "tg_mw_{}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str, contents: &[u8], seconds: u64) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, contents).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new()
                    .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds)),
            )
            .unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn mw_json_and_bin_are_one_candidate_without_stable_poll_oscillation() {
    let f = Fixture::new();
    let json = f.file("A.json", b"source", 10_000);
    f.file("A.bin", b"cache", 20_000);
    let mut tracker = ModelReloadTracker::default();
    let mut calls = Vec::new();
    let events = tracker
        .scan(&f.0, |key, path| {
            calls.push((key.to_owned(), path.to_owned()));
            Ok(())
        })
        .unwrap();
    assert_eq!(calls, vec![("A".to_owned(), json.clone())]);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].key, "A");
    assert_eq!(events[0].path, json);
    assert!(events[0].result.is_ok());
    assert!(
        tracker
            .scan(&f.0, |_, _| panic!("unchanged source reloaded"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mw_failed_startup_retries_same_timestamp_and_reports_failure() {
    let f = Fixture::new();
    f.file("A.json", b"valid after retry", 10_000);
    let mut tracker = ModelReloadTracker::default();
    let first = tracker
        .scan(&f.0, |_, _| Err("transient failure".into()))
        .unwrap();
    assert_eq!(first[0].result, Err("transient failure".into()));
    let retry = tracker.scan(&f.0, |_, _| Ok(())).unwrap();
    assert_eq!(retry.len(), 1, "failure must not advance applied stamp");
    assert!(retry[0].result.is_ok());
}

#[test]
fn mw_failed_update_keeps_retrying_until_success() {
    let f = Fixture::new();
    f.file("A.json", b"old", 10_000);
    let mut tracker = ModelReloadTracker::default();
    tracker.scan(&f.0, |_, _| Ok(())).unwrap();
    f.file("A.json", b"new", 20_000);
    assert!(
        tracker.scan(&f.0, |_, _| Err("reject".into())).unwrap()[0]
            .result
            .is_err()
    );
    assert_eq!(tracker.scan(&f.0, |_, _| Ok(())).unwrap().len(), 1);
    assert!(
        tracker
            .scan(&f.0, |_, _| panic!("already applied"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mw_bin_legacy_to_json_transition_same_timestamp_is_observed() {
    let f = Fixture::new();
    let bin = f.file("A.bin", b"legacy", 10_000);
    let mut tracker = ModelReloadTracker::default();
    assert_eq!(tracker.scan(&f.0, |_, _| Ok(())).unwrap()[0].path, bin);
    let json = f.file("A.json", b"source", 10_000);
    let next = tracker.scan(&f.0, |_, _| Ok(())).unwrap();
    assert_eq!(
        next.len(),
        1,
        "source identity changed despite identical mtime"
    );
    assert_eq!(next[0].path, json);
}

#[test]
fn mw_file_size_change_is_observed_even_with_same_mtime() {
    let f = Fixture::new();
    f.file("A.json", b"old", 10_000);
    let mut tracker = ModelReloadTracker::default();
    tracker.scan(&f.0, |_, _| Ok(())).unwrap();
    f.file("A.json", b"different length", 10_000);
    assert_eq!(tracker.scan(&f.0, |_, _| Ok(())).unwrap().len(), 1);
}

#[test]
fn mw_one_rejection_does_not_block_another_asset() {
    let f = Fixture::new();
    f.file("A.json", b"bad", 10_000);
    f.file("B.json", b"good", 10_000);
    let mut tracker = ModelReloadTracker::default();
    let first = tracker
        .scan(&f.0, |key, _| {
            if key == "A" {
                Err("bad".into())
            } else {
                Ok(())
            }
        })
        .unwrap();
    assert_eq!(first.iter().filter(|e| e.result.is_ok()).count(), 1);
    let retry = tracker
        .scan(&f.0, |key, _| {
            assert_eq!(key, "A");
            Ok(())
        })
        .unwrap();
    assert_eq!(retry.len(), 1);
}

#[test]
fn mw_ignores_non_model_entries_and_model_named_directories() {
    let f = Fixture::new();
    f.file("notes.txt", b"text", 10_000);
    f.file("A.json.tmp", b"partial", 10_000);
    std::fs::create_dir(f.0.join("not_a_model.json")).unwrap();
    assert!(
        ModelReloadTracker::default()
            .scan(&f.0, |_, _| panic!("not a model file"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mw_valid_standalone_bin_still_loads_once() {
    let f = Fixture::new();
    f.file("A.bin", b"legacy", 10_000);
    let mut tracker = ModelReloadTracker::default();
    assert_eq!(tracker.scan(&f.0, |_, _| Ok(())).unwrap().len(), 1);
    assert!(
        tracker
            .scan(&f.0, |_, _| panic!("unchanged legacy"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mw_missing_directory_is_an_error_not_a_successful_empty_scan() {
    let f = Fixture::new();
    assert!(
        ModelReloadTracker::default()
            .scan(&f.0.join("absent"), |_, _| Ok(()))
            .is_err()
    );
}

#[test]
fn mw_cache_created_by_load_does_not_trigger_duplicate_work_next_scan() {
    let f = Fixture::new();
    f.file("A.json", b"source", 10_000);
    let mut tracker = ModelReloadTracker::default();
    tracker
        .scan(&f.0, |_, _| {
            f.file("A.bin", b"new cache", 20_000);
            Ok(())
        })
        .unwrap();
    assert!(
        tracker
            .scan(&f.0, |_, _| panic!("cache is not a second source"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mw_host_wires_one_policy_into_both_startup_and_polling() {
    // Static wiring check; does not start the daemon or certify runtime timing.
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert_eq!(
        host.matches("refresh_ml_models(&mut forest_reload);")
            .count(),
        2
    );
    assert!(!host.contains("forest_timestamps.insert"));
    assert!(host.contains("NanoForest::load_global(key, source)"));
    // A requested JSON path does not certify whether the loader used its BIN.
    assert!(host.contains("Loaded {} (requested path: {})"));
}

#[test]
fn mw_open_diagnostic_same_mtime_and_size_are_not_content_identity() {
    let f = Fixture::new();
    f.file("A.json", b"one", 10_000);
    let mut tracker = ModelReloadTracker::default();
    tracker.scan(&f.0, |_, _| Ok(())).unwrap();
    f.file("A.json", b"two", 10_000);
    // Explicit remaining limitation; a content/generation contract is separate.
    assert!(
        tracker
            .scan(&f.0, |_, _| panic!("metadata cannot detect this"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mw_open_diagnostic_demoting_json_does_not_revoke_legacy_bin() {
    let f = Fixture::new();
    let json = f.file("A_MOTOR.json", b"source", 10_000);
    let bin = f.file("A_MOTOR.bin", b"cached", 20_000);
    let mut tracker = ModelReloadTracker::default();
    tracker.scan(&f.0, |_, _| Ok(())).unwrap();
    std::fs::rename(json, f.0.join("A_MOTOR_CANDIDATE.json")).unwrap();
    let events = tracker.scan(&f.0, |_, _| Ok(())).unwrap();
    assert!(events.iter().any(|e| e.key == "A_MOTOR" && e.path == bin));
}

/// CL-141 — Esperado: un modelo inválido se reintenta en cada escaneo, pero
/// sólo el primer rechazo pide aviso; otro error o un archivo cambiado
/// vuelven a pedirlo, y la carga correcta lo olvida. Antes, cada escaneo
/// (cada 10 s) escribía la misma línea: 8 640 al día por archivo.
#[test]
fn cl141_un_rechazo_repetido_se_reintenta_sin_repetir_el_aviso() {
    let f = Fixture::new();
    f.file("A.json", b"bad", 10_000);
    let mut tracker = ModelReloadTracker::default();
    let mut intentos = 0;
    let mut avisos = 0;
    for _ in 0..100 {
        let ev = tracker
            .scan(&f.0, |_, _| {
                intentos += 1;
                Err("split has missing child".into())
            })
            .unwrap();
        assert_eq!(ev.len(), 1);
        avisos += !ev[0].repetido as u32;
    }
    assert_eq!(intentos, 100, "el reintento por si el fallo es transitorio no cambia");
    assert_eq!(avisos, 1, "el mismo rechazo se avisa una vez");
    let otro = tracker.scan(&f.0, |_, _| Err("otro error".into())).unwrap();
    assert!(!otro[0].repetido, "otro error vuelve a avisar");
    f.file("A.json", b"bad, other version", 20_000);
    let cambiado = tracker.scan(&f.0, |_, _| Err("otro error".into())).unwrap();
    assert!(!cambiado[0].repetido, "un archivo cambiado vuelve a avisar");
    let ok = tracker.scan(&f.0, |_, _| Ok(())).unwrap();
    assert!(ok[0].result.is_ok() && !ok[0].repetido);
    f.file("A.json", b"bad", 30_000);
    let de_nuevo = tracker.scan(&f.0, |_, _| Err("otro error".into())).unwrap();
    assert!(!de_nuevo[0].repetido, "tras cargar bien, un rechazo es nuevo otra vez");
}

/// CL-141 — Esperado: un directorio ilegible avisa una vez por error.
#[test]
fn cl141_el_error_de_escaneo_se_avisa_una_vez() {
    let mut tracker = ModelReloadTracker::default();
    let falta = std::env::temp_dir().join("tg_cl141_no_existe_nunca");
    let mut avisos = 0;
    for _ in 0..10 {
        let e = tracker.scan(&falta, |_, _| Ok(())).unwrap_err().to_string();
        avisos += tracker.error_de_escaneo_nuevo(&e) as u32;
    }
    assert_eq!(avisos, 1);
    let f = Fixture::new();
    assert!(tracker.scan(&f.0, |_, _| Ok(())).unwrap().is_empty());
    let e = tracker.scan(&falta, |_, _| Ok(())).unwrap_err().to_string();
    assert!(tracker.error_de_escaneo_nuevo(&e), "un escaneo correcto lo olvida");
}

/// CL-141b — Esperado: los pesos de DarkAlpha no son candidatos a bosque.
/// Antes cada reescritura del entrenador automático era un rechazo «nuevo»
/// y volvía a avisar.
#[test]
fn cl141b_los_pesos_de_dark_alpha_no_son_un_bosque() {
    let f = Fixture::new();
    f.file("DarkAlpha_BTCUSDT.json", b"{\"weights\": []}", 10_000);
    f.file("BTCUSDT_MOTOR.json", b"bosque", 10_000);
    let mut tracker = ModelReloadTracker::default();
    for (i, mtime) in [10_000u64, 20_000, 30_000].into_iter().enumerate() {
        f.file("DarkAlpha_BTCUSDT.json", format!("{{\"weights\": [{i}]}}").as_bytes(), mtime);
        let mut claves = Vec::new();
        let ev = tracker
            .scan(&f.0, |k, _| {
                claves.push(k.to_owned());
                Ok(())
            })
            .unwrap();
        assert!(claves.iter().all(|k| !k.starts_with("DarkAlpha_")), "{claves:?}");
        assert!(ev.iter().all(|e| !e.key.starts_with("DarkAlpha_")));
    }
}
