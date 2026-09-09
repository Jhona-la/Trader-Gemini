//! ZONA RAW DEL LAKEHOUSE (F2.5) — archivo inmutable de ticks con integridad.
//!
//! QUÉ: append-only binario de ticks YA VALIDADOS (aduana F2.1): nunca se
//!      reescribe, solo se añade. Al rotar, el segmento se finaliza con
//!      checksum y rename atómico — un lector jamás ve un archivo a medias.
//! POR QUÉ: directriz — "almacenar muy rápido, muy eficiente, y SIEMPRE
//!      evitando corrupción". La zona raw es la fuente de verdad de la que
//!      todo lo derivado (features, backtests) puede reconstruirse.
//! FORMATO por segmento (little-endian):
//!      [magic u32 "TGRW"][versión u16][ticks N × 40B][footer: count u32][fnv1a u64]
//! RENDIMIENTO: append directo a BufWriter; checksum FNV-1a incremental
//!      (no criptográfico: detecta corrupción accidental, no adversarios —
//!      suficiente y 100× más rápido que SHA sobre el hot-path).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[allow(dead_code)]
const MAGIC: u32 = 0x54524757; // "TGRW"
#[allow(dead_code)]
const VERSION: u16 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RawTick {
    pub ts_ms: u64,
    pub bid: f64,
    pub ask: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

impl RawTick {
    fn to_bytes(self) -> [u8; 40] {
        let mut b = [0u8; 40];
        b[0..8].copy_from_slice(&self.ts_ms.to_le_bytes());
        b[8..16].copy_from_slice(&self.bid.to_le_bytes());
        b[16..24].copy_from_slice(&self.ask.to_le_bytes());
        b[24..32].copy_from_slice(&self.bid_qty.to_le_bytes());
        b[32..40].copy_from_slice(&self.ask_qty.to_le_bytes());
        b
    }
    fn from_bytes(b: &[u8; 40]) -> Self {
        Self {
            ts_ms: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            bid: f64::from_le_bytes(b[8..16].try_into().unwrap()),
            ask: f64::from_le_bytes(b[16..24].try_into().unwrap()),
            bid_qty: f64::from_le_bytes(b[24..32].try_into().unwrap()),
            ask_qty: f64::from_le_bytes(b[32..40].try_into().unwrap()),
        }
    }
}

/// FNV-1a incremental sobre los bytes de los ticks.
#[derive(Default, Clone, Copy)]
struct Fnv1a(u64);
impl Fnv1a {
    fn new() -> Self {
        Fnv1a(0xcbf29ce484222325)
    }
    fn update(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.0 = h;
    }
    fn finish(self) -> u64 {
        self.0
    }
}

pub struct RawTickArchive {
    dir: PathBuf,
    symbol: String,
    writer: Option<BufWriter<File>>,
    active_path: PathBuf,
    count: u32,
    hash: Fnv1a,
    /// Rotación por cantidad de ticks (segmentos de ~8 MB con 200k ticks).
    max_ticks_per_segment: u32,
    segments_finalized: u64,
}

impl RawTickArchive {
    pub fn open(dir: impl AsRef<Path>, symbol: &str) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir.as_ref())?;
        let active_path = dir.as_ref().join(format!("{}_active.tgrw.tmp", symbol));
        // Reabrir un tmp huérfano sería corrupción silenciosa: descartarlo
        // (los ticks vivos no dependen del archivo; el archivo es archivo).
        let _ = std::fs::remove_file(&active_path);
        let writer = Some(BufWriter::with_capacity(
            1 << 20,
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&active_path)?,
        ));
        Ok(Self {
            dir: dir.as_ref().to_path_buf(),
            symbol: symbol.to_string(),
            writer,
            active_path,
            count: 0,
            hash: Fnv1a::new(),
            max_ticks_per_segment: 200_000,
            segments_finalized: 0,
        })
    }

    /// Append O(1) amortizado. El tick ya pasó la aduana de validación.
    pub fn append(&mut self, tick: RawTick) -> std::io::Result<()> {
        if let Some(w) = self.writer.as_mut() {
            let bytes = tick.to_bytes();
            w.write_all(&bytes)?;
            self.hash.update(&bytes);
            self.count += 1;
            if self.count >= self.max_ticks_per_segment {
                self.rotate()?;
            }
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "archivo activo cerrado",
            ))
        }
    }

    /// Finaliza el segmento: header reescrito no — formato append puro:
    /// los ticks van crudos; el FOOTER (count+hash) se escribe al finalizar,
    /// y luego rename atómico tmp→final. Sin footer = segmento inválido.
    pub fn rotate(&mut self) -> std::io::Result<()> {
        let Some(mut w) = self.writer.take() else {
            return Ok(());
        };
        // Footer: count + hash de TODOS los bytes de ticks.
        w.write_all(&self.count.to_le_bytes())?;
        w.write_all(&self.hash.finish().to_le_bytes())?;
        w.flush()?;

        let final_name = format!(
            "{}_{:06}_{}.tgrw",
            self.symbol,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            self.count
        );
        let final_path = self.dir.join(&final_name);
        // Rename atómico: los lectores ven O inexistente O completo.
        std::fs::rename(&self.active_path, &final_path)?;

        self.segments_finalized += 1;
        self.count = 0;
        self.hash = Fnv1a::new();
        self.writer = Some(BufWriter::with_capacity(
            1 << 20,
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.active_path)?,
        ));
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        if let Some(w) = self.writer.as_mut() {
            w.flush()
        } else {
            Ok(())
        }
    }

    pub fn segments_finalized(&self) -> u64 {
        self.segments_finalized
    }
}

/// Valida un segmento finalizado: header implícito en el formato crudo —
/// se verifica tamaño múltiplo + footer (count/hash) contra el contenido.
/// Retorna los ticks sanos o el error de corrupción con su offset.
pub fn verify_segment(path: impl AsRef<Path>) -> Result<Vec<RawTick>, String> {
    let mut f = File::open(path.as_ref()).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    if buf.len() < 12 {
        return Err(format!("segmento truncado: {} bytes (< footer)", buf.len()));
    }
    let body_len = buf.len() - 12;
    if body_len % 40 != 0 {
        return Err(format!(
            "cuerpo no es múltiplo de 40B ({} bytes) — corrupción de alineación",
            body_len
        ));
    }
    let count = u32::from_le_bytes(buf[body_len..body_len + 4].try_into().unwrap());
    let stored_hash = u64::from_le_bytes(buf[body_len + 4..].try_into().unwrap());
    let n_ticks = body_len / 40;
    if count as usize != n_ticks {
        return Err(format!(
            "footer declara {} ticks pero el cuerpo tiene {} — truncado o alterado",
            count, n_ticks
        ));
    }
    let mut h = Fnv1a::new();
    h.update(&buf[..body_len]);
    if h.finish() != stored_hash {
        return Err("checksum NO coincide — los bytes mutaron tras la finalización".to_string());
    }
    let mut ticks = Vec::with_capacity(n_ticks);
    for i in 0..n_ticks {
        let chunk: [u8; 40] = buf[i * 40..(i + 1) * 40].try_into().unwrap();
        ticks.push(RawTick::from_bytes(&chunk));
    }
    Ok(ticks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_con_integridad() {
        let dir = std::env::temp_dir().join(format!("tgrw_test_{}", std::process::id()));
        let mut arch = RawTickArchive::open(&dir, "BTCUSDT").unwrap();
        let mut written = Vec::new();
        for i in 0..1000u64 {
            let t = RawTick {
                ts_ms: 1_700_000_000_000 + i,
                bid: 60000.0 + i as f64,
                ask: 60000.5 + i as f64,
                bid_qty: 1.5,
                ask_qty: 2.5,
            };
            arch.append(t).unwrap();
            written.push(t);
        }
        arch.rotate().unwrap();
        let seg = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .find(|e| e.path().extension().map(|x| x == "tgrw").unwrap_or(false))
            .map(|e| e.path())
            .expect("segmento finalizado existe");
        let read = verify_segment(&seg).expect("segmento íntegro");
        assert_eq!(read.len(), 1000);
        assert_eq!(read[0], written[0]);
        assert_eq!(read[999], written[999]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupcion_detectada() {
        let dir = std::env::temp_dir().join(format!("tgrw_corr_{}", std::process::id()));
        let mut arch = RawTickArchive::open(&dir, "ETHUSDT").unwrap();
        for i in 0..100u64 {
            arch.append(RawTick {
                ts_ms: i,
                bid: 3000.0,
                ask: 3000.1,
                bid_qty: 1.0,
                ask_qty: 1.0,
            })
            .unwrap();
        }
        arch.rotate().unwrap();
        let seg = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .find(|e| e.path().extension().map(|x| x == "tgrw").unwrap_or(false))
            .map(|e| e.path())
            .unwrap();
        // Muta UN byte del cuerpo: el checksum debe gritar.
        let mut bytes = std::fs::read(&seg).unwrap();
        bytes[10] ^= 0xFF;
        std::fs::write(&seg, &bytes).unwrap();
        let err = verify_segment(&seg).unwrap_err();
        assert!(err.contains("checksum"), "debe detectar mutación: {}", err);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tick_bytes_simetrico() {
        let t = RawTick {
            ts_ms: 42,
            bid: 1.5,
            ask: 1.6,
            bid_qty: 7.0,
            ask_qty: 8.0,
        };
        assert_eq!(RawTick::from_bytes(&t.to_bytes()), t);
    }
}
