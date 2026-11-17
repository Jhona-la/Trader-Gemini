//! TGB-C/D — el lector comun de ticks abre la cinta logica `X.bin` aunque solo exista `X.bin.tgb`, con los mismos
//! eventos y errores que el original, y el `.bin` manda si existen los dos (adenda (d) del coordinador: la
//! convivencia se compara registro a registro). Prefijo y multimoneda verifican el contenedor ENTERO.
//!
//! Fixtures: los contenedores que escribio `tgb comprimir` en `crates/tick-tape/tests/fixtures` (procedencia en
//! `crates/tick-tape/src/lib.rs`). `f1_real_sha_original_mal.bin.tgb` es f1 con el SHA del original alterado y
//! el sello recalculado: solo falla la comprobacion final.

use backtest_engine::tick_replayer::{load_binary_ticks, load_binary_ticks_prefix, load_multi_coin_binary_ticks};
use quantum_arena::TickEvent;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

const F: &[(&str, &[u8], &[u8])] = &[
    ("f1_real", include_bytes!("../../tick-tape/tests/fixtures/f1_real.bin"), include_bytes!("../../tick-tape/tests/fixtures/f1_real.bin.tgb")),
    ("f2_legado", include_bytes!("../../tick-tape/tests/fixtures/f2_legado.bin"), include_bytes!("../../tick-tape/tests/fixtures/f2_legado.bin.tgb")),
    ("f3_sintetico", include_bytes!("../../tick-tape/tests/fixtures/f3_sintetico.bin"), include_bytes!("../../tick-tape/tests/fixtures/f3_sintetico.bin.tgb")),
    ("f4_otro", include_bytes!("../../tick-tape/tests/fixtures/f4_otro.bin"), include_bytes!("../../tick-tape/tests/fixtures/f4_otro.bin.tgb")),
    ("f5_vacio", include_bytes!("../../tick-tape/tests/fixtures/f5_vacio.bin"), include_bytes!("../../tick-tape/tests/fixtures/f5_vacio.bin.tgb")),
    ("f6_solo_cabecera", include_bytes!("../../tick-tape/tests/fixtures/f6_solo_cabecera.bin"), include_bytes!("../../tick-tape/tests/fixtures/f6_solo_cabecera.bin.tgb")),
    ("f7_tgm_ajena", include_bytes!("../../tick-tape/tests/fixtures/f7_tgm_ajena.bin"), include_bytes!("../../tick-tape/tests/fixtures/f7_tgm_ajena.bin.tgb")),
];
const F1_SHA_MAL: &[u8] = include_bytes!("../../tick-tape/tests/fixtures/f1_real_sha_original_mal.bin.tgb");

struct Dir(PathBuf);
impl Dir {
    fn nuevo(nombre: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("tgb_reader_contract_{}_{nombre}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
    fn poner(&self, nombre: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(nombre);
        if let Some(padre) = p.parent() {
            std::fs::create_dir_all(padre).unwrap();
        }
        std::fs::write(&p, bytes).unwrap();
        p
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

type Clave = (usize, u64, u64, u64, u64, u64);
fn claves(v: &[TickEvent]) -> Vec<Clave> {
    v.iter()
        .map(|t| (t.coin_id, t.timestamp, t.bid_price.to_bits(), t.ask_price.to_bits(), t.bid_qty.to_bits(), t.ask_qty.to_bits()))
        .collect()
}

/// Resultado comparable: eventos por bits, o el tipo de error.
fn resultado(r: std::io::Result<Vec<TickEvent>>) -> Result<Vec<Clave>, ErrorKind> {
    r.map(|v| claves(&v)).map_err(|e| e.kind())
}

fn igual_registro_a_registro(nombre: &str, a: &Result<Vec<Clave>, ErrorKind>, b: &Result<Vec<Clave>, ErrorKind>) {
    match (a, b) {
        (Ok(x), Ok(y)) => {
            assert_eq!(x.len(), y.len(), "{nombre}: cantidad de eventos");
            for (k, (p, q)) in x.iter().zip(y).enumerate() {
                assert_eq!(p, q, "{nombre}: evento {k} distinto");
            }
        }
        _ => assert_eq!(a, b, "{nombre}: resultado o error distinto"),
    }
}

#[test]
fn c_solo_tgb_da_los_mismos_eventos_y_errores_que_el_bin() {
    let con_bin = Dir::nuevo("bin");
    let con_tgb = Dir::nuevo("tgb");
    for (nombre, bin, tgb) in F {
        let pb = con_bin.poner(&format!("{nombre}.bin"), bin);
        con_tgb.poner(&format!("{nombre}.bin.tgb"), tgb);
        let pt = con_tgb.0.join(format!("{nombre}.bin"));
        let a = resultado(load_binary_ticks(&pb, 3));
        let b = resultado(load_binary_ticks(&pt, 3));
        if *nombre == "f1_real" {
            assert!(b.is_ok(), "con solo {nombre}.bin.tgb el lector tiene que cargar la cinta logica: {b:?}");
        }
        igual_registro_a_registro(nombre, &a, &b);
    }
}

#[test]
fn c_con_ambos_manda_el_bin() {
    let (_, f1, f1t) = F[0];
    let (_, f3, f3t) = F[2];
    let (_, f7, _) = F[6];
    let d = Dir::nuevo("ambos");
    // Contenidos distintos: gana el .bin.
    let p = d.poner("x.bin", f1);
    d.poner("x.bin.tgb", f3t);
    let solo_f1 = Dir::nuevo("ref_f1");
    let r1 = solo_f1.poner("r.bin", f1);
    igual_registro_a_registro("ambos distintos", &resultado(load_binary_ticks(&p, 0)), &resultado(load_binary_ticks(&r1, 0)));
    // .bin valido + .tgb danado: gana el .bin y carga.
    let mut danado = f1t.to_vec();
    danado[60] ^= 0x01;
    let p2 = d.poner("y.bin", f3);
    d.poner("y.bin.tgb", &danado);
    let r3 = solo_f1.poner("r3.bin", f3);
    igual_registro_a_registro("tgb danado", &resultado(load_binary_ticks(&p2, 0)), &resultado(load_binary_ticks(&r3, 0)));
    // .bin invalido + .tgb valido: el error del .bin se conserva; no se cambia de fuente.
    let p3 = d.poner("z.bin", f7);
    d.poner("z.bin.tgb", f1t);
    let e = load_binary_ticks(&p3, 0).err().expect("el .bin invalido manda aunque el .tgb sea valido");
    assert_eq!(e.kind(), ErrorKind::InvalidData);
}

#[test]
fn c_ausencias_directorios_y_rutas_no_ascii() {
    let d = Dir::nuevo("ausencia");
    let e = load_binary_ticks(&d.0.join("nada.bin"), 0).err().unwrap();
    assert_eq!(e.kind(), ErrorKind::NotFound);
    // Un directorio en la ruta del .bin no es ausencia: no se usa el .tgb.
    std::fs::create_dir_all(d.0.join("dir.bin")).unwrap();
    d.poner("dir.bin.tgb", F[0].2);
    let r = load_binary_ticks(&d.0.join("dir.bin"), 0);
    assert!(r.is_err(), "un directorio en la ruta del .bin no puede caer al .tgb");
    assert_ne!(r.err().unwrap().kind(), ErrorKind::NotFound);
    // Espacios y no ASCII: se anade .tgb al nombre completo.
    let p = d.0.join("cinta con espacios ñ 東").join("BTC é_REAL.bin");
    d.poner("cinta con espacios ñ 東/BTC é_REAL.bin.tgb", F[0].2);
    let bin = Dir::nuevo("ref_noascii");
    let r = bin.poner("r.bin", F[0].1);
    igual_registro_a_registro("no ascii", &resultado(load_binary_ticks(&p, 1)), &resultado(load_binary_ticks(&r, 1)));
    // Un .tgb danado (solo el) es un error de datos, nunca un vacio aparentemente valido.
    let mut danado = F[0].2.to_vec();
    danado[100] ^= 0x01;
    let p = d.poner("mal.bin.tgb", &danado);
    let _ = p;
    let e = load_binary_ticks(&d.0.join("mal.bin"), 0).err().expect("un .tgb danado no se carga");
    assert_eq!(e.kind(), ErrorKind::InvalidData);
}

#[test]
fn d_prefijo_y_multimoneda_verifican_el_contenedor_entero() {
    let d = Dir::nuevo("prefijo");
    let ref_dir = Dir::nuevo("prefijo_ref");
    let (_, f1, f1t) = F[0];
    let pt = d.0.join("f1.bin");
    d.poner("f1.bin.tgb", f1t);
    let pb = ref_dir.poner("f1.bin", f1);
    let completo = claves(&load_binary_ticks(&pt, 2).expect("completo desde .tgb"));
    for n in [0usize, 1, 2, 100, 256, 257, 10_000] {
        let pre = claves(&load_binary_ticks_prefix(&pt, 2, n).expect("prefijo desde .tgb"));
        assert_eq!(pre[..], completo[..n.min(completo.len())], "prefijo {n}");
        let pre_bin = claves(&load_binary_ticks_prefix(&pb, 2, n).unwrap());
        assert_eq!(pre, pre_bin, "prefijo {n} contra el .bin");
    }
    // Error solo en el SHA final del original: el prefijo (incluso de 0) tambien lo rechaza.
    d.poner("sha_mal.bin.tgb", F1_SHA_MAL);
    for n in [0usize, 1, 257] {
        let r = load_binary_ticks_prefix(&d.0.join("sha_mal.bin"), 2, n);
        assert_eq!(r.err().map(|e| e.kind()), Some(ErrorKind::InvalidData), "prefijo {n} con el SHA final malo");
    }
    // Multimoneda: misma union ordenada (estable con tiempos iguales) desde .tgb que desde .bin.
    let (_, f2, f2t) = F[1];
    d.poner("f2.bin.tgb", f2t);
    let pb2 = ref_dir.poner("f2.bin", f2);
    let mt = load_multi_coin_binary_ticks(&[(pt.as_path(), 0), (d.0.join("f2.bin").as_path(), 1)]).expect("multi desde .tgb");
    let mb = load_multi_coin_binary_ticks(&[(pb.as_path(), 0), (pb2.as_path(), 1)]).unwrap();
    assert_eq!(claves(&mt), claves(&mb));
    // Una cinta mala aborta la carga multimoneda entera.
    let r = load_multi_coin_binary_ticks(&[(pt.as_path(), 0), (Path::new(&d.0.join("sha_mal.bin")), 1)]);
    assert!(r.is_err(), "una cinta mala no se omite en multimoneda");
}
