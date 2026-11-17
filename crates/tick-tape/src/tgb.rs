//! Decodificador del contenedor TGB00001 (paso A: restitucion de contenedores validos).

use sha2::{Digest, Sha256};
use std::io::{self, Read};

/// Magia del contenedor.
pub const MAGIA_TGB: &[u8; 8] = b"TGB00001";
/// Registro de la cinta: u64 tiempo + 4 f64 (formato, `tgb/src/main.rs:14`).
pub const REGISTRO: usize = 40;
/// Campos de 8 bytes por registro.
pub const CAMPOS: usize = 5;
/// Ventana maxima de Zstandard del formato (`tgb/src/main.rs:49`).
pub const VENTANA_LOG_MAX: u32 = 27;
const SELLO: usize = 32;
const PREFIJO: usize = 8 + 8 + 32 + 1;

/// Tope de lectura que fija quien consume la cinta (bytes del original restituido).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Presupuesto {
    max_bytes_original: Option<u64>,
}

impl Presupuesto {
    /// Sin tope propio: el consumidor va a retener la cinta entera de todos modos.
    pub fn sin_tope() -> Self {
        Presupuesto { max_bytes_original: None }
    }
    /// Tope derivado de la tarea del consumidor.
    pub fn hasta(bytes: u64) -> Self {
        Presupuesto { max_bytes_original: Some(bytes) }
    }
    pub fn tope(&self) -> Option<u64> {
        self.max_bytes_original
    }
}

/// Original restituido y verificado.
pub struct Restituido {
    datos: Vec<u8>,
    sha256_original: [u8; 32],
    modo: u8,
}

impl Restituido {
    pub fn as_bytes(&self) -> &[u8] {
        &self.datos
    }
    pub fn sha256_original(&self) -> [u8; 32] {
        self.sha256_original
    }
    pub fn modo(&self) -> u8 {
        self.modo
    }
}

fn invalido(m: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m.to_string())
}

fn u64_en(b: &[u8], i: usize) -> io::Result<u64> {
    let t = b.get(i..i.saturating_add(8)).ok_or_else(|| invalido("contenedor cortado"))?;
    let a: [u8; 8] = t.try_into().map_err(|_| invalido("contenedor cortado"))?;
    Ok(u64::from_le_bytes(a))
}

fn trozo(b: &[u8], i: usize, n: u64) -> io::Result<&[u8]> {
    let n = usize::try_from(n).map_err(|_| invalido("largo"))?;
    b.get(i..i.saturating_add(n)).ok_or_else(|| invalido("contenedor cortado"))
}

fn descomprimir(z: &[u8]) -> io::Result<Vec<u8>> {
    let mut d = zstd::stream::read::Decoder::with_buffer(z)?;
    d.window_log_max(VENTANA_LOG_MAX)?;
    let mut v = Vec::new();
    d.read_to_end(&mut v)?;
    Ok(v)
}

fn zigzag_inverso(t: u64) -> u64 {
    (t >> 1) ^ 0u64.wrapping_sub(t & 1)
}

pub fn restituir_tgb(contenedor: &[u8], _presupuesto: Presupuesto) -> io::Result<Restituido> {
    if contenedor.len() < PREFIJO + SELLO || &contenedor[..8] != MAGIA_TGB {
        return Err(invalido("no es un contenedor TGB00001"));
    }
    let (cuerpo, sello) = contenedor.split_at(contenedor.len() - SELLO);
    let h: [u8; 32] = Sha256::digest(cuerpo).into();
    if h[..] != sello[..] {
        return Err(invalido("sello del contenedor distinto"));
    }
    let mut sha256_original = [0u8; 32];
    sha256_original.copy_from_slice(&cuerpo[16..48]);
    let modo = cuerpo[48];
    let mut i = PREFIJO;
    let mut datos = Vec::new();
    if modo == 0 {
        let zl = u64_en(cuerpo, i)?;
        i += 8;
        datos = descomprimir(trozo(cuerpo, i, zl)?)?;
    } else {
        if modo == 1 {
            datos.extend_from_slice(trozo(cuerpo, i, 8)?);
            i += 8;
        }
        let total = u64_en(cuerpo, i)?;
        i += 8;
        let mut hechos = 0u64;
        while hechos < total {
            let n = u64_en(cuerpo, i)?;
            i += 8;
            let mut cols = Vec::new();
            for _ in 0..CAMPOS {
                let zl = u64_en(cuerpo, i)?;
                i += 8;
                cols.push(descomprimir(trozo(cuerpo, i, zl)?)?);
                i += zl as usize;
            }
            let mut previo = 0u64;
            for k in 0..n as usize {
                for (f, col) in cols.iter().enumerate() {
                    let t = u64_en(col, k * 8)?;
                    let v = if f == 0 {
                        previo = previo.wrapping_add(zigzag_inverso(t));
                        previo
                    } else {
                        t
                    };
                    datos.extend_from_slice(&v.to_le_bytes());
                }
            }
            hechos += n;
        }
    }
    Ok(Restituido { datos, sha256_original, modo })
}
