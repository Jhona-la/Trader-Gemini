//! TGB-B: integridad y publicacion transaccional. Las 20 corrupciones del pedido CODEX-CLI (Parte 3), cada una
//! con la comprobacion que debe detectarla. Las marcadas "resellar" recalculan el sello para llegar a las
//! validaciones internas. Un `Err` no publica nada: `restituir_tgb` no entrega bytes antes de terminar.

mod comun;

use comun::*;
use tick_tape::tgb::{fallo_de, restituir_tgb, restituir_tgb_con, FalloTgb, IdentidadEsperada, Presupuesto};

const F1_BIN: &[u8] = include_bytes!("fixtures/f1_real.bin");
const F1_TGB: &[u8] = include_bytes!("fixtures/f1_real.bin.tgb");

fn rechazo(c: &[u8]) -> FalloTgb {
    match restituir_tgb(c, Presupuesto::sin_tope()) {
        Ok(r) => panic!("aceptado: {} B publicados", r.len()),
        Err(e) => fallo_de(&e).unwrap_or_else(|| panic!("error sin causa TGB: {e}")),
    }
}

fn sellado(mut c: Vec<u8>) -> Vec<u8> {
    resellar(&mut c);
    c
}

/// Sustituye la columna `f` del bloque `b` por `nuevo` (bytes comprimidos), con su largo.
fn con_columna(c: &[u8], m: &Mapa, b: usize, f: usize, nuevo: &[u8]) -> Vec<u8> {
    let col = &m.bloques[b].columnas[f];
    let mut pieza = (nuevo.len() as u64).to_le_bytes().to_vec();
    pieza.extend_from_slice(nuevo);
    reemplazar(c, col.pos_largo, col.pos_datos + col.largo, &pieza)
}

/// Columna de tiempos del bloque 0 de f1, sin comprimir (zigzag de diferencias, como la escribe el formato).
fn tiempos_f1() -> Vec<u8> {
    let regs = &F1_BIN[8..];
    let mut v = Vec::new();
    let mut previo = 0u64;
    for r in regs.chunks_exact(REC) {
        let t = u64::from_le_bytes(r[..8].try_into().unwrap());
        let d = t.wrapping_sub(previo) as i64;
        previo = t;
        v.extend_from_slice(&(((d << 1) ^ (d >> 63)) as u64).to_le_bytes());
    }
    v
}

#[test]
fn tgb_rechaza_20_corrupciones() {
    let base = F1_TGB.to_vec();
    let m = mapa(&base);
    assert_eq!(m.modo, 1);
    assert_eq!(m.bloques.len(), 1);
    assert!(restituir_tgb(&base, Presupuesto::sin_tope()).is_ok(), "la base tiene que ser valida");
    let n = m.bloques[0].n;
    let pos_n = m.bloques[0].pos_n;
    let pos_total = m.pos_total.unwrap();
    let pos_cab = m.pos_cabecera.unwrap();
    let t0 = &m.bloques[0].columnas[0];
    let mut casos: Vec<(&str, Vec<u8>, FalloTgb)> = Vec::new();

    // C01-C05: sin resellar.
    let mut c = base.clone();
    c[0] = b'X';
    casos.push(("C01 magia", c, FalloTgb::Magia));
    casos.push(("C02 cabecera cortada", base[..40].to_vec(), FalloTgb::Cortado));
    casos.push(("C03 falta un byte del sello", base[..base.len() - 1].to_vec(), FalloTgb::Sello));
    let mut c = base.clone();
    let k = c.len() - 5;
    c[k] ^= 0x01;
    casos.push(("C04 sello alterado", c, FalloTgb::Sello));
    let mut c = base.clone();
    c[t0.pos_datos + 3] ^= 0x10;
    casos.push(("C05 payload sin resellar", c, FalloTgb::Sello));

    // C06-C20: reselladas.
    let mut c = base.clone();
    escribir_u64(&mut c, 8, F1_BIN.len() as u64 + 40);
    casos.push(("C06 largo original", sellado(c), FalloTgb::LargoOriginal));
    let mut c = base.clone();
    c[20] ^= 0xFF;
    casos.push(("C07 SHA original", sellado(c), FalloTgb::ShaOriginal));
    let mut c = base.clone();
    c[48] = 7;
    casos.push(("C08 modo desconocido", sellado(c), FalloTgb::Modo));
    let mut c = base.clone();
    escribir_u64(&mut c, pos_total, n + 1);
    casos.push(("C09 total mayor que los bloques", sellado(c), FalloTgb::FaltanBloques));
    let mut c = base.clone();
    escribir_u64(&mut c, pos_total, n - 1);
    casos.push(("C10 total menor que el bloque", sellado(c), FalloTgb::BloqueExcedeTotal));
    let mut c = base.clone();
    escribir_u64(&mut c, pos_n, 0);
    casos.push(("C11 bloque con cero registros", sellado(c), FalloTgb::BloqueCero));
    let mut c = base.clone();
    escribir_u64(&mut c, pos_n, (1u64 << 22) + 1);
    casos.push(("C12 bloque mayor que el maximo", sellado(c), FalloTgb::BloqueMayorQueMax));
    let mut c = base.clone();
    escribir_u64(&mut c, t0.pos_largo, base.len() as u64);
    casos.push(("C13 largo comprimido fuera del archivo", sellado(c), FalloTgb::FueraDelCuerpo));
    let c = reemplazar(&base, t0.pos_largo, t0.pos_datos + t0.largo, &0u64.to_le_bytes());
    casos.push(("C14 columna comprimida vacia", sellado(c), FalloTgb::ColumnaVacia));
    let basura = vec![0xA5u8; t0.largo];
    casos.push(("C15 frame Zstandard invalido", sellado(con_columna(&base, &m, 0, 0, &basura)), FalloTgb::Zstd));
    let tiempos = tiempos_f1();
    let corta = z(&tiempos[..tiempos.len() - 8]);
    casos.push(("C16 columna que expande a menos", sellado(con_columna(&base, &m, 0, 0, &corta)), FalloTgb::LargoColumna));
    let mut c = base.clone();
    let fin = c.len() - 32;
    c.splice(fin..fin, [0u8; 5]);
    casos.push(("C17 bytes sobrantes antes del sello", sellado(c), FalloTgb::Sobrante));
    let mut c = base.clone();
    c[pos_cab + 7] = b'2';
    casos.push(("C18 cabecera original modificada", sellado(c), FalloTgb::ShaOriginal));
    let mut otros = tiempos.clone();
    otros[8 * 30] ^= 0x02;
    casos.push(("C19 tiempos validos y distintos", sellado(con_columna(&base, &m, 0, 0, &z(&otros))), FalloTgb::ShaOriginal));
    let mut c = base.clone();
    escribir_u64(&mut c, 8, u64::MAX);
    casos.push(("C20 largo original u64 maximo", sellado(c), FalloTgb::LargoOriginal));

    assert_eq!(casos.len(), 20);
    let mut mal = Vec::new();
    for (nombre, c, esperado) in &casos {
        let obtenido = match restituir_tgb(c, Presupuesto::sin_tope()) {
            Ok(r) => format!("ACEPTADO ({} B publicados)", r.len()),
            Err(e) => match fallo_de(&e) {
                Some(f) if f == *esperado => continue,
                Some(f) => format!("{f:?}"),
                None => format!("error sin causa TGB: {e}"),
            },
        };
        mal.push(format!("{nombre}: esperado {esperado:?}, obtenido {obtenido}"));
    }
    assert!(mal.is_empty(), "corrupciones mal detectadas:\n{}", mal.join("\n"));
}

#[test]
fn tgb_descompresion_acotada_y_frames_concatenados() {
    let tiempos = tiempos_f1();
    let base = F1_TGB.to_vec();
    let m = mapa(&base);
    // Una columna que expande a MAS de lo anotado: la cota de lectura lo detecta.
    let mut larga = tiempos.clone();
    larga.extend_from_slice(&[0u8; 8]);
    assert_eq!(rechazo(&sellado(con_columna(&base, &m, 0, 0, &z(&larga)))), FalloTgb::LargoColumna);
    // Un frame valido seguido de basura dentro de su columna: no se olvidan los bytes restantes.
    let mut con_cola = z(&tiempos);
    con_cola.extend_from_slice(&[0x11u8; 9]);
    assert_eq!(rechazo(&sellado(con_columna(&base, &m, 0, 0, &con_cola))), FalloTgb::Zstd);
    // Dos frames concatenados que juntos dan la columna: se leen los dos.
    let mitad = tiempos.len() / 2;
    let mut dos = z(&tiempos[..mitad]);
    dos.extend_from_slice(&z(&tiempos[mitad..]));
    let r = restituir_tgb(&sellado(con_columna(&base, &m, 0, 0, &dos)), Presupuesto::sin_tope()).unwrap();
    assert!(r.as_bytes() == F1_BIN);
    // Modo 0: un frame que expande a otro largo.
    let otro: Vec<u8> = (0..999u32).map(|k| (k % 251) as u8).collect();
    let mut c0 = codificar(&otro, 0, 1);
    escribir_u64(&mut c0, 8, 998);
    assert_eq!(rechazo(&sellado(c0)), FalloTgb::LargoColumna);
}

#[test]
fn tgb_presupuesto_identidad_y_alineacion() {
    let largo = F1_BIN.len() as u64;
    let e = restituir_tgb(F1_TGB, Presupuesto::hasta(largo - 1)).err().unwrap();
    assert_eq!(fallo_de(&e), Some(FalloTgb::Presupuesto));
    let r = restituir_tgb(F1_TGB, Presupuesto::hasta(largo)).unwrap();
    assert_eq!(r.as_bytes().as_ptr() as usize % 8, 0, "el original va alineado para los registros de 8 B");
    let buena = IdentidadEsperada { bytes: largo, sha256: sha(F1_BIN) };
    assert!(restituir_tgb_con(F1_TGB, Presupuesto::sin_tope(), Some(&buena)).is_ok());
    for mala in [
        IdentidadEsperada { bytes: largo + 40, ..buena },
        IdentidadEsperada { sha256: [0u8; 32], ..buena },
    ] {
        let e = restituir_tgb_con(F1_TGB, Presupuesto::sin_tope(), Some(&mala)).err().unwrap();
        assert_eq!(fallo_de(&e), Some(FalloTgb::Identidad));
    }
    // Un original vacio (modo 0) se restituye vacio y verificado.
    let vacio = restituir_tgb(include_bytes!("fixtures/f5_vacio.bin.tgb"), Presupuesto::hasta(0)).unwrap();
    assert!(vacio.is_empty());
    assert_eq!(vacio.as_bytes().as_ptr() as usize % 8, 0, "tambien el original vacio va alineado");
}
