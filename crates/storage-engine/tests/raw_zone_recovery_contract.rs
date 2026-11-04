use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use storage_engine::raw_zone::{verify_segment, RawTick, RawTickArchive};

fn test_dir(name: &str) -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("reloj de prueba posterior a UNIX_EPOCH")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "raw_zone_recovery_{name}_{}_{unique}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("crear directorio temporal");
    dir
}

fn tick(i: u64) -> RawTick {
    RawTick {
        ts_ms: 1_700_000_000_000 + i,
        bid: 60_000.0 + i as f64,
        ask: 60_000.5 + i as f64,
        bid_qty: 1.5 + i as f64,
        ask_qty: 2.5 + i as f64,
    }
}

fn active_path(dir: &Path) -> PathBuf {
    dir.join("BTCUSDT_active.tgrw.tmp")
}

fn segments(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = fs::read_dir(dir)
        .expect("leer directorio")
        .map(|entry| entry.expect("entrada de directorio").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "tgrw"))
        .collect();
    paths.sort();
    paths
}

#[test]
fn open_recovers_complete_ticks_without_footer_and_appends() {
    let dir = test_dir("append_without_footer");
    let mut initial = RawTickArchive::open(&dir, "BTCUSDT").expect("crear activo");
    initial.append(tick(0)).expect("primer tick");
    initial.append(tick(1)).expect("segundo tick");
    drop(initial);

    let mut recovered = RawTickArchive::open(&dir, "BTCUSDT").expect("recuperar activo");
    recovered.append(tick(2)).expect("tercer tick");
    recovered.rotate().expect("publicar segmento");

    let published = segments(&dir);
    assert_eq!(published.len(), 1, "un sólo segmento final");
    let ticks = verify_segment(&published[0]).expect("segmento íntegro");
    assert_eq!(ticks, vec![tick(0), tick(1), tick(2)]);

    fs::remove_dir_all(dir).expect("limpiar directorio temporal");
}

#[test]
fn open_publishes_active_with_valid_footer_before_new_appends() {
    let dir = test_dir("active_with_footer");
    let mut initial = RawTickArchive::open(&dir, "BTCUSDT").expect("crear activo");
    initial.append(tick(0)).expect("primer tick");
    initial.append(tick(1)).expect("segundo tick");
    initial.rotate().expect("escribir footer");
    drop(initial);

    let published = segments(&dir);
    assert_eq!(published.len(), 1);
    fs::remove_file(active_path(&dir)).expect("retirar activo vacío");
    fs::rename(&published[0], active_path(&dir)).expect("activo con footer");

    let mut recovered = RawTickArchive::open(&dir, "BTCUSDT").expect("publicar activo previo");
    recovered.append(tick(2)).expect("tick del nuevo activo");
    recovered.rotate().expect("publicar nuevo segmento");

    let published = segments(&dir);
    assert_eq!(published.len(), 2, "segmento recuperado y segmento nuevo");
    let mut ticks = Vec::new();
    for path in published {
        ticks.extend(verify_segment(path).expect("cada segmento debe verificar"));
    }
    ticks.sort_by_key(|tick| tick.ts_ms);
    assert_eq!(ticks, vec![tick(0), tick(1), tick(2)]);

    fs::remove_dir_all(dir).expect("limpiar directorio temporal");
}

#[test]
fn open_rejects_partial_suffix_without_touching_active_file() {
    let dir = test_dir("partial_suffix");
    let mut initial = RawTickArchive::open(&dir, "BTCUSDT").expect("crear activo");
    initial.append(tick(0)).expect("primer tick");
    initial.append(tick(1)).expect("segundo tick");
    drop(initial);
    fs::OpenOptions::new()
        .append(true)
        .open(active_path(&dir))
        .expect("abrir sufijo de prueba")
        .write_all(&[0xa5, 0x5a, 0xff])
        .expect("escribir sufijo parcial");
    let before = fs::read(active_path(&dir)).expect("bytes antes de open");

    let opened = RawTickArchive::open(&dir, "BTCUSDT");

    assert!(opened.is_err(), "un sufijo parcial debe rechazarse");
    let after = fs::read(active_path(&dir)).expect("bytes después de open");
    assert_eq!(after, before, "el activo debe quedar byte-idéntico");
    assert!(segments(&dir).is_empty(), "no se publica ningún segmento");

    fs::remove_dir_all(dir).expect("limpiar directorio temporal");
}

#[test]
fn repeated_writers_in_one_second_never_replace_published_segments() {
    let dir = test_dir("unique_segment_names");
    for i in 0..12 {
        let mut archive = RawTickArchive::open(&dir, "BTCUSDT").expect("nuevo writer");
        archive.append(tick(i)).expect("tick distinto");
        archive
            .rotate()
            .expect("cada writer debe poder publicar su segmento");
    }

    let published = segments(&dir);
    assert_eq!(published.len(), 12, "ningún nombre puede colisionar");
    let mut ticks = Vec::new();
    for path in published {
        let segment = verify_segment(path).expect("cada segmento conserva su tick");
        assert_eq!(segment.len(), 1);
        ticks.extend(segment);
    }
    ticks.sort_by_key(|tick| tick.ts_ms);
    assert_eq!(ticks, (0..12).map(tick).collect::<Vec<_>>());

    fs::remove_dir_all(dir).expect("limpiar directorio temporal");
}
