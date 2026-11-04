//! ZONA RAW DEL LAKEHOUSE (F2.5) — archivo inmutable de ticks con integridad.
//!
//! QUÉ: append-only binario de ticks YA VALIDADOS (aduana F2.1): nunca se
//!      reescribe, solo se añade. Al rotar, el segmento se finaliza con
//!      checksum y rename atómico — un lector jamás ve un archivo a medias.
//! POR QUÉ: directriz — "almacenar muy rápido, muy eficiente, y SIEMPRE
//!      evitando corrupción". La zona raw es la fuente de verdad de la que
//!      todo lo derivado (features, backtests) puede reconstruirse.
//! FORMATO por segmento (little-endian), tal como lo escribe `rotate`:
//!      [ticks N × 40B][footer: count u32][fnv1a u64]
//! SIN cabecera: F1-094 corrigió este comentario, que anunciaba un
//! `[magic u32][versión u16]` que el escritor NUNCA emite (las constantes
//! `MAGIC`/`VERSION` existían sin uso, ocultas tras `#[allow(dead_code)]`).
//! El lector (`verify_segment`) calcula el cuerpo como `len - 12`, que es
//! correcto precisamente porque no hay cabecera. Añadirla ahora invalidaría
//! todos los segmentos ya escritos: si algún día se versiona el formato, debe
//! hacerse con un discriminador al final (footer) y compatibilidad de lectura.
//! RENDIMIENTO: append directo a BufWriter; checksum FNV-1a incremental
//!      (no criptográfico: detecta corrupción accidental, no adversarios —
//!      suficiente y 100× más rápido que SHA sobre el hot-path).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

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
    /// Decodifica un tick de 40 bytes little-endian.
    ///
    /// F1-094 — `Option` en vez de `unwrap`: la aritmética de índices es
    /// CONSTANTE (`b[0]`…`b[39]`), así que el compilador la verifica y no
    /// puede paniquear; y un buffer corto se RECHAZA en vez de rebanarlo a
    /// ciegas. Antes cada campo hacía `b[a..c].try_into().unwrap()`: correcto
    /// sólo mientras el llamador garantizara los 40 bytes, y un pánico de
    /// proceso (`panic = "abort"`) si esa garantía se relajaba.
    fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < 40 {
            return None;
        }
        Some(Self {
            ts_ms: u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            bid: f64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]),
            ask: f64::from_le_bytes([b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23]]),
            bid_qty: f64::from_le_bytes([b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]]),
            ask_qty: f64::from_le_bytes([b[32], b[33], b[34], b[35], b[36], b[37], b[38], b[39]]),
        })
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

static NEXT_SEGMENT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NEXT_ACTIVE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn open_active_writer(path: &Path) -> std::io::Result<BufWriter<File>> {
    Ok(BufWriter::with_capacity(
        1 << 20,
        OpenOptions::new().append(true).open(path)?,
    ))
}

fn next_final_path(dir: &Path, symbol: &str, count: u32) -> std::io::Result<PathBuf> {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        .as_secs();
    let id = NEXT_SEGMENT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(dir.join(format!(
        "{}_{:06}_{}_{}_{}.tgrw",
        symbol,
        seconds,
        std::process::id(),
        count,
        id
    )))
}

fn next_active_path(dir: &Path, symbol: &str) -> std::io::Result<PathBuf> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        .as_nanos();
    let id = NEXT_ACTIVE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(dir.join(format!(
        "{}_active_{}_{}_{}.tgrw.tmp",
        symbol,
        nanos,
        std::process::id(),
        id
    )))
}

fn lock_active(file: &File) -> std::io::Result<()> {
    match file.try_lock() {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "el activo pertenece a otro writer",
        )),
        Err(std::fs::TryLockError::Error(e)) => Err(e),
    }
}

fn active_lock_path(active_path: &Path) -> PathBuf {
    let mut path = active_path.as_os_str().to_os_string();
    path.push(".lock");
    path.into()
}

fn create_active(dir: &Path, symbol: &str) -> std::io::Result<(PathBuf, File)> {
    let path = next_active_path(dir, symbol)?;
    let tick_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(active_lock_path(&path))?;
    lock_active(&lock_file)?;
    drop(tick_file);
    Ok((path, lock_file))
}

fn recovery_candidates(dir: &Path, symbol: &str) -> std::io::Result<Vec<PathBuf>> {
    let legacy_name = format!("{symbol}_active.tgrw.tmp");
    let unique_prefix = format!("{symbol}_active_");
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let matches = path.file_name().is_some_and(|name| {
            let name = name.to_string_lossy();
            name.as_ref() == legacy_name
                || (name.starts_with(&unique_prefix) && name.ends_with(".tgrw.tmp"))
        });
        if matches {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn claim_or_create_active(dir: &Path, symbol: &str) -> std::io::Result<(PathBuf, File, Vec<u8>)> {
    for candidate in recovery_candidates(dir, symbol)? {
        let lock_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(active_lock_path(&candidate))?;
        match lock_active(&lock_file) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(e),
        }
        match std::fs::read(&candidate) {
            Ok(bytes) => return Ok((candidate, lock_file, bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        }
    }
    let (path, lock_file) = create_active(dir, symbol)?;
    Ok((path, lock_file, Vec::new()))
}

#[cfg(unix)]
fn sync_directory(dir: &Path) -> std::io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}

pub struct RawTickArchive {
    dir: PathBuf,
    symbol: String,
    writer: Option<BufWriter<File>>,
    active_lock: Option<File>,
    active_path: PathBuf,
    rotation_path: PathBuf,
    count: u32,
    hash: Fnv1a,
    /// Rotación por cantidad de ticks (segmentos de ~8 MB con 200k ticks).
    max_ticks_per_segment: u32,
    segments_finalized: u64,
}

impl RawTickArchive {
    pub fn open(dir: impl AsRef<Path>, symbol: &str) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir.as_ref())?;
        let (mut active_path, mut active_lock, bytes) =
            claim_or_create_active(dir.as_ref(), symbol)?;
        let mut count = 0;
        let mut hash = Fnv1a::new();
        let mut segments_finalized = 0;

        if !bytes.is_empty() {
            match bytes.len() % 40 {
                0 => {
                    for (index, chunk) in bytes.as_chunks::<40>().0.iter().enumerate() {
                        RawTick::from_bytes(chunk).ok_or_else(|| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("tick corrupto en el offset {}", index * 40),
                            )
                        })?;
                    }
                    count = u32::try_from(bytes.len() / 40).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "el activo excede el contador de ticks del formato",
                        )
                    })?;
                    hash.update(&bytes);
                }
                12 => {
                    let ticks = verify_segment(&active_path)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    let count = u32::try_from(ticks.len()).map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "el segmento excede el contador de ticks del formato",
                        )
                    })?;
                    let final_path = next_final_path(dir.as_ref(), symbol, count)?;
                    std::fs::rename(&active_path, final_path)?;
                    drop(active_lock);
                    sync_directory(dir.as_ref())?;
                    let (new_path, new_lock) = create_active(dir.as_ref(), symbol)?;
                    active_path = new_path;
                    active_lock = new_lock;
                    segments_finalized = 1;
                }
                suffix => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("sufijo parcial de {suffix} bytes en el activo"),
                    ))
                }
            }
        }

        let rotation_path = next_final_path(dir.as_ref(), symbol, count)?;
        Ok(Self {
            dir: dir.as_ref().to_path_buf(),
            symbol: symbol.to_string(),
            writer: Some(open_active_writer(&active_path)?),
            active_lock: Some(active_lock),
            active_path,
            rotation_path,
            count,
            hash,
            max_ticks_per_segment: 200_000,
            segments_finalized,
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

    /// Finaliza el segmento: los ticks van crudos; el FOOTER (count+hash) se
    /// escribe al finalizar, se sincroniza y se publica con rename atómico.
    pub fn rotate(&mut self) -> std::io::Result<()> {
        let Some(mut w) = self.writer.take() else {
            self.active_lock = None;
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "archivo activo cerrado; no se puede rotar",
            ));
        };
        let active_lock = self.active_lock.take();
        let final_path = self.rotation_path.clone();
        let outcome = (|| -> std::io::Result<()> {
            w.write_all(&self.count.to_le_bytes())?;
            w.write_all(&self.hash.finish().to_le_bytes())?;
            w.flush()?;
            w.get_ref().sync_all()?;
            std::fs::rename(&self.active_path, &final_path)?;
            drop(w);
            drop(active_lock);

            self.segments_finalized += 1;
            self.count = 0;
            self.hash = Fnv1a::new();
            sync_directory(&self.dir)?;
            let (new_path, new_lock) = create_active(&self.dir, &self.symbol)?;
            let new_writer = open_active_writer(&new_path)?;
            let next_rotation_path = next_final_path(&self.dir, &self.symbol, self.count)?;
            self.active_path = new_path;
            self.rotation_path = next_rotation_path;
            self.active_lock = Some(new_lock);
            self.writer = Some(new_writer);
            Ok(())
        })();
        if let Err(e) = outcome {
            self.writer = None;
            self.active_lock = None;
            return Err(e);
        }
        Ok(())
    }

    pub fn rotation_path(&self) -> &Path {
        &self.rotation_path
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer
            .as_mut()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "archivo activo cerrado; no se puede flush",
                )
            })?
            .flush()
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
    // F1-094 — índices constantes sobre la ventana del footer: con
    // `buf.len() >= 12` (guardado arriba) y `body_len = buf.len() - 12`, las
    // doce posiciones existen por aritmética, sin `try_into().unwrap()`.
    let count = u32::from_le_bytes([
        buf[body_len],
        buf[body_len + 1],
        buf[body_len + 2],
        buf[body_len + 3],
    ]);
    let stored_hash = u64::from_le_bytes([
        buf[body_len + 4],
        buf[body_len + 5],
        buf[body_len + 6],
        buf[body_len + 7],
        buf[body_len + 8],
        buf[body_len + 9],
        buf[body_len + 10],
        buf[body_len + 11],
    ]);
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
    for chunk in buf[..body_len].chunks_exact(40) {
        match RawTick::from_bytes(chunk) {
            Some(t) => ticks.push(t),
            // Inalcanzable: `chunks_exact(40)` sólo entrega trozos de 40.
            // Si algún día dejara de serlo, el segmento se rechaza entero en
            // vez de abortar el proceso con posiciones abiertas.
            None => {
                return Err(format!(
                    "tick ilegible en el offset {} — segmento rechazado",
                    ticks.len() * 40
                ))
            }
        }
    }
    if ticks.len() != n_ticks {
        return Err(format!(
            "se decodificaron {} ticks de {} declarados — segmento incoherente",
            ticks.len(),
            n_ticks
        ));
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
        // F1-094: el decodificador devuelve `Option` — un buffer corto se
        // rechaza en vez de rebanarse a ciegas.
        assert_eq!(RawTick::from_bytes(&t.to_bytes()), Some(t));
        assert_eq!(
            RawTick::from_bytes(&t.to_bytes()[..39]),
            None,
            "39 bytes no son un tick: se rechaza, no se rellena con ceros"
        );
    }
}
