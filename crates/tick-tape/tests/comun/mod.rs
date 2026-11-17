//! Utilidades de prueba: un codificador TGB00001 de prueba (bloques de cualquier tamano, para ejercitar el reinicio
//! del delta temporal sin fixtures de 2^22 registros), el resellado y el mapa de desplazamientos de un contenedor.
#![allow(dead_code)]

use sha2::{Digest, Sha256};

pub const REC: usize = 40;

pub fn sha(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn zigzag(d: i64) -> u64 {
    ((d << 1) ^ (d >> 63)) as u64
}

/// Columnas de un bloque: el tiempo como diferencia en zigzag (previo 0 al empezar el bloque); el resto tal cual.
fn columnas(registros: &[u8]) -> Vec<Vec<u8>> {
    let mut cols: Vec<Vec<u8>> = vec![Vec::new(); 5];
    let mut previo = 0u64;
    for r in registros.chunks_exact(REC) {
        for (f, col) in cols.iter_mut().enumerate() {
            let v = u64::from_le_bytes(r[f * 8..f * 8 + 8].try_into().unwrap());
            let w = if f == 0 {
                let d = v.wrapping_sub(previo) as i64;
                previo = v;
                zigzag(d)
            } else {
                v
            };
            col.extend_from_slice(&w.to_le_bytes());
        }
    }
    cols
}

pub fn z(datos: &[u8]) -> Vec<u8> {
    zstd::bulk::compress(datos, 3).unwrap()
}

/// Codifica `orig` en el modo pedido, con bloques de `bloque` registros. Modo 1: los primeros 8 bytes son la cabecera.
pub fn codificar(orig: &[u8], modo: u8, bloque: usize) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(b"TGB00001");
    c.extend_from_slice(&(orig.len() as u64).to_le_bytes());
    c.extend_from_slice(&sha(orig));
    c.push(modo);
    if modo == 0 {
        let zz = z(orig);
        c.extend_from_slice(&(zz.len() as u64).to_le_bytes());
        c.extend_from_slice(&zz);
    } else {
        let cab = if modo == 1 { 8 } else { 0 };
        c.extend_from_slice(&orig[..cab]);
        let regs = &orig[cab..];
        assert_eq!(regs.len() % REC, 0);
        let total = regs.len() / REC;
        c.extend_from_slice(&(total as u64).to_le_bytes());
        for trozo in regs.chunks(bloque * REC) {
            c.extend_from_slice(&((trozo.len() / REC) as u64).to_le_bytes());
            for col in columnas(trozo) {
                let zz = z(&col);
                c.extend_from_slice(&(zz.len() as u64).to_le_bytes());
                c.extend_from_slice(&zz);
            }
        }
    }
    let s = sha(&c);
    c.extend_from_slice(&s);
    c
}

/// Recalcula el sello final (los ultimos 32 bytes) sobre el resto.
pub fn resellar(c: &mut Vec<u8>) {
    let n = c.len() - 32;
    let s = sha(&c[..n]);
    c[n..].copy_from_slice(&s);
}

pub fn leer_u64(c: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(c[i..i + 8].try_into().unwrap())
}

pub fn escribir_u64(c: &mut [u8], i: usize, v: u64) {
    c[i..i + 8].copy_from_slice(&v.to_le_bytes());
}

#[derive(Debug, Clone)]
pub struct Columna {
    pub pos_largo: usize,
    pub pos_datos: usize,
    pub largo: usize,
}

#[derive(Debug, Clone)]
pub struct Bloque {
    pub pos_n: usize,
    pub n: u64,
    pub columnas: Vec<Columna>,
}

#[derive(Debug, Clone)]
pub struct Mapa {
    pub modo: u8,
    pub pos_cabecera: Option<usize>,
    pub pos_total: Option<usize>,
    pub bloques: Vec<Bloque>,
    pub modo0: Option<Columna>,
    pub fin_cuerpo: usize,
}

/// Mapa de desplazamientos de un contenedor VALIDO.
pub fn mapa(c: &[u8]) -> Mapa {
    let fin = c.len() - 32;
    let modo = c[48];
    let mut i = 49;
    let mut m = Mapa { modo, pos_cabecera: None, pos_total: None, bloques: vec![], modo0: None, fin_cuerpo: fin };
    if modo == 0 {
        let l = leer_u64(c, i) as usize;
        m.modo0 = Some(Columna { pos_largo: i, pos_datos: i + 8, largo: l });
        return m;
    }
    if modo == 1 {
        m.pos_cabecera = Some(i);
        i += 8;
    }
    m.pos_total = Some(i);
    i += 8;
    while i < fin {
        let n = leer_u64(c, i);
        let mut b = Bloque { pos_n: i, n, columnas: vec![] };
        i += 8;
        for _ in 0..5 {
            let l = leer_u64(c, i) as usize;
            b.columnas.push(Columna { pos_largo: i, pos_datos: i + 8, largo: l });
            i += 8 + l;
        }
        m.bloques.push(b);
    }
    m
}

/// Reemplaza los bytes [a, b) del contenedor por `nuevo` (sin resellar).
pub fn reemplazar(c: &[u8], a: usize, b: usize, nuevo: &[u8]) -> Vec<u8> {
    let mut v = c[..a].to_vec();
    v.extend_from_slice(nuevo);
    v.extend_from_slice(&c[b..]);
    v
}

/// Registros sinteticos deterministas con casos de bits (duplicados, salto modular, -0.0, NaN con payload, inf).
pub fn registros(cuantos: usize, semilla: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(cuantos * REC);
    let mut ts = 1_700_000_000_000u64.wrapping_add(semilla);
    for k in 0..cuantos as u64 {
        ts = ts.wrapping_add(k % 5);
        let t = match k % 97 {
            50 => u64::MAX,
            51 => 3,
            _ => ts,
        };
        let bid = 100.0 + (k as f64) * 0.25;
        let campos: [u64; 5] = [
            t,
            if k % 89 == 7 { (-0.0f64).to_bits() } else { bid.to_bits() },
            if k % 83 == 9 { 0x7ff8_0000_0000_beefu64 } else { (bid + 0.5).to_bits() },
            if k % 79 == 3 { f64::INFINITY.to_bits() } else { ((k % 11) as f64 * 0.01).to_bits() },
            if k % 73 == 2 { f64::NEG_INFINITY.to_bits() } else { ((k % 7) as f64 * 0.02).to_bits() },
        ];
        for x in campos {
            v.extend_from_slice(&x.to_le_bytes());
        }
    }
    v
}

/// Cinta con cabecera de 8 bytes.
pub fn cinta(cabecera: &[u8; 8], cuantos: usize, semilla: u64) -> Vec<u8> {
    let mut v = cabecera.to_vec();
    v.extend_from_slice(&registros(cuantos, semilla));
    v
}
