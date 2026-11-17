//! TGB-C: API de la cinta logica (representacion, transporte e identidad). La conducta del lector de ticks la
//! fija `crates/backtest-engine/tests/tgb_reader_contract.rs`.

mod comun;

use comun::sha;
use std::io::ErrorKind;
use tick_tape::logica::{hay_cinta, largo_logico, leer_logica, open_logical_tape, open_logical_tape_con_identidad, representacion, ruta_tgb, Representacion, Transporte};
use tick_tape::tgb::{fallo_de, FalloTgb, IdentidadEsperada, Presupuesto};

const F1: &[u8] = include_bytes!("fixtures/f1_real.bin");
const F1T: &[u8] = include_bytes!("fixtures/f1_real.bin.tgb");
const F3: &[u8] = include_bytes!("fixtures/f3_sintetico.bin");

#[test]
fn logica_representacion_transporte_e_identidad() {
    let d = std::env::temp_dir().join(format!("tick_tape_logica_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let x = d.join("X_REAL.bin");
    assert_eq!(ruta_tgb(&x), d.join("X_REAL.bin.tgb"));
    assert!(!hay_cinta(&x).unwrap());
    let e = representacion(&x).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::NotFound);
    assert!(e.to_string().contains("X_REAL.bin.tgb"), "NotFound nombra las dos rutas: {e}");

    std::fs::write(ruta_tgb(&x), F1T).unwrap();
    assert_eq!(representacion(&x).unwrap(), Representacion::Tgb);
    let t = open_logical_tape(&x, Presupuesto::sin_tope()).unwrap();
    assert!(t.as_bytes() == F1);
    assert_eq!(t.ruta_logica(), x.as_path());
    assert_eq!(t.ruta_fisica(), ruta_tgb(&x).as_path());
    assert_eq!(t.sha256_original(), sha(F1));
    match t.transporte() {
        Transporte::Tgb { bytes_contenedor, sha256_contenedor, modo } => {
            assert_eq!(bytes_contenedor, F1T.len() as u64);
            assert_eq!(sha256_contenedor, sha(F1T), "el SHA del transporte es el del archivo entero (sha256_tgb)");
            assert_eq!(modo, 1);
        }
        otro => panic!("transporte {otro:?}"),
    }
    assert_eq!(largo_logico(&x).unwrap(), F1.len() as u64);
    assert_eq!(leer_logica(&x).unwrap(), F1);
    let buena = IdentidadEsperada { bytes: F1.len() as u64, sha256: sha(F1) };
    assert!(open_logical_tape_con_identidad(&x, Presupuesto::sin_tope(), &buena).is_ok());
    let mala = IdentidadEsperada { sha256: sha(F3), ..buena };
    let e = open_logical_tape_con_identidad(&x, Presupuesto::sin_tope(), &mala).err().unwrap();
    assert_eq!(fallo_de(&e), Some(FalloTgb::Identidad));

    // Con los dos, manda el .bin (aqui con otro contenido), y la identidad esperada se mide sobre el .bin.
    std::fs::write(&x, F3).unwrap();
    let t = open_logical_tape(&x, Presupuesto::sin_tope()).unwrap();
    assert_eq!(t.representacion(), Representacion::Bin);
    assert!(t.as_bytes() == F3);
    let e = open_logical_tape_con_identidad(&x, Presupuesto::sin_tope(), &buena).err().unwrap();
    assert_eq!(fallo_de(&e), Some(FalloTgb::Identidad), "un .bin distinto del original esperado se rechaza");
    let e = open_logical_tape(&x, Presupuesto::hasta(F3.len() as u64 - 1)).err().unwrap();
    assert_eq!(fallo_de(&e), Some(FalloTgb::Presupuesto));
    std::fs::remove_dir_all(&d).unwrap();
}
