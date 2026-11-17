//! TGB-E — la seleccion REAL frente a sintetico de `continuous_evolution_backtest` (antes en
//! continuous_evolution_backtest.rs:126-127, 209 y 220) mira la cinta LOGICA: un REAL guardado solo como
//! `.bin.tgb` gana al sintetico; el sintetico solo entra si faltan las DOS representaciones del REAL; y un REAL
//! corrupto o inaccesible aborta en vez de ceder al sintetico.

use backtest_engine::tick_replayer::{cintas_presentes, elegir_cinta, load_binary_ticks};
use std::io::ErrorKind;
use std::path::PathBuf;

const F1: &[u8] = include_bytes!("../../tick-tape/tests/fixtures/f1_real.bin");
const F1T: &[u8] = include_bytes!("../../tick-tape/tests/fixtures/f1_real.bin.tgb");
const F3: &[u8] = include_bytes!("../../tick-tape/tests/fixtures/f3_sintetico.bin");

fn dir(nombre: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tgb_selector_{}_{nombre}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn e_el_real_comprimido_gana_al_sintetico() {
    let d = dir("real_tgb");
    let real = d.join("BTCUSDT_ticks_REAL.bin");
    let sint = d.join("BTCUSDT_ticks.bin");
    std::fs::write(&sint, F3).unwrap();
    std::fs::write(d.join("BTCUSDT_ticks_REAL.bin.tgb"), F1T).unwrap();
    let elegida = elegir_cinta(&real, &sint).unwrap();
    assert_eq!(elegida, real.as_path(), "con el REAL solo en .bin.tgb no se puede elegir el sintetico");
    let ticks = load_binary_ticks(elegida, 0).unwrap();
    assert_eq!(ticks.len(), (F1.len() - 8) / 40);

    // El sintetico solo si faltan las dos representaciones del REAL.
    std::fs::remove_file(d.join("BTCUSDT_ticks_REAL.bin.tgb")).unwrap();
    assert_eq!(elegir_cinta(&real, &sint).unwrap(), sint.as_path());

    // Multimoneda: el REAL comprimido cuenta como presente, el ausente no, y nada se duplica.
    std::fs::write(d.join("BTCUSDT_ticks_REAL.bin.tgb"), F1T).unwrap();
    let eth = d.join("ETHUSDT_ticks.bin");
    let presentes = cintas_presentes(vec![(real.as_path(), 0), (eth.as_path(), 1), (sint.as_path(), 2)]).unwrap();
    assert_eq!(presentes, vec![(real.as_path(), 0), (sint.as_path(), 2)]);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn e_un_real_corrupto_o_inaccesible_no_cede_al_sintetico() {
    let d = dir("real_malo");
    let real = d.join("X_REAL.bin");
    let sint = d.join("X.bin");
    std::fs::write(&sint, F3).unwrap();
    let mut danado = F1T.to_vec();
    danado[70] ^= 0x04;
    std::fs::write(d.join("X_REAL.bin.tgb"), &danado).unwrap();
    let elegida = elegir_cinta(&real, &sint).unwrap();
    assert_eq!(elegida, real.as_path(), "un REAL corrupto sigue siendo el REAL: la carga tiene que fallar");
    assert_eq!(load_binary_ticks(elegida, 0).err().map(|e| e.kind()), Some(ErrorKind::InvalidData));
    // Un directorio en la ruta del REAL no es ausencia.
    let otro = d.join("Y_REAL.bin");
    std::fs::create_dir_all(&otro).unwrap();
    assert!(elegir_cinta(&otro, &sint).is_err(), "un REAL inaccesible aborta la seleccion");
    assert!(cintas_presentes(vec![(otro.as_path(), 0)]).is_err());
    let _ = std::fs::remove_dir_all(&d);
}
