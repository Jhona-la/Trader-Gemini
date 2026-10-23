//! Contrato F1-094 (Sol) — UN SEGMENTO CORRUPTO SE RECHAZA, NO ABORTA EL PROCESO.
//!
//! # QUÉ
//! La zona raw del lakehouse es la fuente de verdad de la que se reconstruye
//! todo lo derivado. `verify_segment` la valida contra su footer (count + FNV-1a)
//! y devuelve `Result`. Este contrato comprueba que las cuatro formas de
//! corrupción real —truncado, cuerpo desalineado, byte mutado y count mentido—
//! se rechazan con un `Err` descriptivo y NUNCA con un pánico de proceso.
//!
//! # POR QUÉ
//! El perfil release usa `panic = "abort"` (LG-28). La decodificación hacía
//! `b[0..8].try_into().unwrap()` por campo: correcto sólo mientras el llamador
//! garantizara los 40 bytes. Un archivo truncado por un corte de energía o una
//! escritura parcial —el escenario que la propia cabecera del módulo dice
//! evitar— convertía una lectura de archivo en un aborto del motor con
//! posiciones abiertas. Ahora los índices son CONSTANTES (el compilador los
//! verifica), un buffer corto devuelve `None`, y el segmento se rechaza entero.
//!
//! # DÓNDE
//! `crates/storage-engine/src/raw_zone.rs`; el crate se conecta a
//! `replay-contracts.yml` en este mismo átomo: hasta hoy `storage-engine` no
//! aparecía en NINGÚN workflow, así que ni estos contratos ni los de `mmap`
//! corrían en CI.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use storage_engine::raw_zone::{verify_segment, RawTick, RawTickArchive};

fn dir_de_prueba(nombre: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "tgrw_f1094_{}_{}",
        nombre,
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("crear dir de prueba");
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

/// Crea un segmento finalizado y devuelve su ruta.
fn segmento_finalizado(dir: &Path, n: u64) -> PathBuf {
    let mut arch = RawTickArchive::open(dir, "BTCUSDT").expect("abrir archivo");
    for i in 0..n {
        arch.append(tick(i)).expect("append");
    }
    arch.rotate().expect("rotar");
    let mut segmentos: Vec<PathBuf> = fs::read_dir(dir)
        .expect("leer dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "tgrw"))
        .collect();
    assert_eq!(segmentos.len(), 1, "debe haber exactamente un segmento");
    segmentos.remove(0)
}

#[test]
fn f1_094_segmento_sano_se_verifica_con_valores_exactos() {
    let dir = dir_de_prueba("sano");
    let ruta = segmento_finalizado(&dir, 3);

    let ticks = verify_segment(&ruta).expect("un segmento íntegro debe verificar");
    assert_eq!(ticks.len(), 3);
    assert_eq!(ticks[0], tick(0));
    assert_eq!(ticks[2], tick(2));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn f1_094_segmento_truncado_se_rechaza_sin_panico() {
    let dir = dir_de_prueba("truncado");
    let ruta = segmento_finalizado(&dir, 4);

    // Corte de energía / escritura parcial: se pierde el último byte.
    let bytes = fs::read(&ruta).expect("leer");
    fs::write(&ruta, &bytes[..bytes.len() - 1]).expect("truncar");

    let err = verify_segment(&ruta).expect_err("un segmento truncado debe rechazarse");
    assert!(
        err.contains("múltiplo") || err.contains("truncado"),
        "el error debe decir POR QUÉ se rechaza: {err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn f1_094_segmento_mas_corto_que_el_footer_se_rechaza() {
    let dir = dir_de_prueba("corto");
    let ruta = dir.join("vacio.tgrw");
    fs::write(&ruta, [0u8; 5]).expect("escribir 5 bytes");

    let err = verify_segment(&ruta).expect_err("5 bytes no son un segmento");
    assert!(err.contains("truncado"), "error esperado de truncado: {err}");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn f1_094_byte_mutado_se_detecta_por_checksum() {
    let dir = dir_de_prueba("mutado");
    let ruta = segmento_finalizado(&dir, 3);

    // Corrupción silenciosa: un bit cambiado en el cuerpo (sector dañado).
    let mut bytes = fs::read(&ruta).expect("leer");
    bytes[7] ^= 0x01;
    fs::write(&ruta, &bytes).expect("reescribir");

    let err = verify_segment(&ruta).expect_err("el checksum debe detectar la mutación");
    assert!(
        err.contains("checksum"),
        "el error debe nombrar el checksum: {err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn f1_094_count_mentido_se_rechaza() {
    let dir = dir_de_prueba("count");
    let ruta = segmento_finalizado(&dir, 3);

    let mut bytes = fs::read(&ruta).expect("leer");
    let body_len = bytes.len() - 12;
    // El footer declara 99 ticks donde el cuerpo tiene 3.
    bytes[body_len..body_len + 4].copy_from_slice(&99u32.to_le_bytes());
    fs::write(&ruta, &bytes).expect("reescribir");

    let err = verify_segment(&ruta).expect_err("un count incoherente debe rechazarse");
    assert!(
        err.contains("declara") || err.contains("truncado"),
        "el error debe señalar la incoherencia: {err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn f1_094_archivo_inexistente_es_error_no_panico() {
    let err = verify_segment("/ruta/que/no/existe/f1094.tgrw")
        .expect_err("un archivo ausente es un Err, no un pánico");
    assert!(!err.is_empty());
}

#[test]
fn f1_094_decodificar_un_tick_corto_no_inventa_valores() {
    // El decodificador público sólo se alcanza vía `verify_segment`, que ya
    // filtra por tamaño; aquí se fija el invariante de que un cuerpo de 39
    // bytes NO produce un tick con ceros fabricados, sino un segmento
    // desalineado (rechazado antes de decodificar).
    let dir = dir_de_prueba("corto39");
    let ruta = dir.join("desalineado.tgrw");
    let mut f = fs::File::create(&ruta).expect("crear");
    f.write_all(&[7u8; 39]).expect("escribir 39");
    f.write_all(&[0u8; 12]).expect("footer");
    drop(f);

    let err = verify_segment(&ruta).expect_err("39B de cuerpo no es múltiplo de 40");
    assert!(err.contains("múltiplo"), "error de alineación esperado: {err}");

    let _ = fs::remove_dir_all(&dir);
}
