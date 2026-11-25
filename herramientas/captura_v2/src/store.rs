//! Archivo de horas selladas: re-codificacion sin perdida (TGLL), verificacion de ida y
//! vuelta, borrado del JSON.gz solo tras verificar, retencion por tiempo UTC y guarda
//! de disco libre.

use crate::salidafallo::{CompuertaStore, OpStore};
use crate::{CaptureResult, codec, digest, write_json};
use flate2::read::GzDecoder;
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub struct StoreConfig {
    pub root: PathBuf,
    pub retention_ms: u64,
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub min_free_bytes: u64,
    pub codec: codec::Params,
    pub scan_every: Duration,
    /// SALIDAFALLO P4c: compuerta nativa (comparte `cancel`). Ordena cada trabajo y cada borrado.
    pub gate: std::sync::Arc<CompuertaStore>,
    /// FALLAS-1: reloj de store (monotono para STATUS y esperas, UTC para retencion), inyectable.
    pub reloj: std::sync::Arc<dyn RelojStore>,
}

/// FALLAS-1: reloj que usa `run`. El monotono gobierna la cadencia de STATUS y la espera de avisos;
/// el UTC solo la retencion y las marcas de los registros. Son independientes.
pub trait RelojStore: Send + Sync {
    /// Tiempo monotono transcurrido desde el origen fijo de este reloj (cero o positivo).
    fn mono(&self) -> Duration;
    /// Marca UTC en ms (retencion por edad).
    fn utc_ms(&self) -> u64;
    /// Espera un aviso de hora sellada como mucho `max`.
    fn esperar(
        &self,
        sealed: &mpsc::Receiver<(PathBuf, String)>,
        max: Duration,
    ) -> Result<(), mpsc::RecvTimeoutError>;
}

/// Reloj de produccion: el origen es el `Instant` de su creacion; nunca se resta nada a un `Instant`.
pub struct RelojStoreReal {
    origen: Instant,
}

impl RelojStoreReal {
    pub fn nuevo() -> Self {
        Self {
            origen: Instant::now(),
        }
    }
}

impl RelojStore for RelojStoreReal {
    fn mono(&self) -> Duration {
        self.origen.elapsed()
    }
    fn utc_ms(&self) -> u64 {
        utc_ms()
    }
    fn esperar(
        &self,
        sealed: &mpsc::Receiver<(PathBuf, String)>,
        max: Duration,
    ) -> Result<(), mpsc::RecvTimeoutError> {
        sealed.recv_timeout(max).map(|_| ())
    }
}

/// FALLAS-1: cadencia de STATUS sin inventar un instante anterior al actual. La primera publicacion
/// queda pendiente (`None`) y se hace en la primera vuelta que supera las guardas; despues toca cuando
/// el tiempo monotono transcurrido desde la ultima publicacion, comprobado con `checked_sub`, alcanza
/// `cada`. Un reloj que retrocediera no adelanta la publicacion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CadenciaStatus {
    ultima: Option<Duration>,
}

impl CadenciaStatus {
    pub(crate) fn pendiente() -> Self {
        Self { ultima: None }
    }
    pub(crate) fn toca(&self, ahora: Duration, cada: Duration) -> bool {
        match self.ultima {
            None => true,
            Some(t) => ahora
                .checked_sub(t)
                .is_some_and(|transcurrido| transcurrido >= cada),
        }
    }
    pub(crate) fn publicada(&mut self, ahora: Duration) {
        self.ultima = Some(ahora);
    }
}

/// Borrado de recode: con configuracion de captura pasa por la compuerta; sin ella (CLI y
/// pruebas de recode) es el borrado directo de siempre.
fn remove(cfg: Option<&StoreConfig>, op: OpStore, path: &Path) -> io::Result<()> {
    match cfg {
        Some(cfg) => cfg.gate.borrar(op, path),
        None => fs::remove_file(path),
    }
}

// Politica solicitada en encargo_retencion_v2.md: nunca vencer datos antes de un ano.
// Se usa el FINAL del cubo UTC, para proteger incluso la ultima fila de la hora.
fn expiry_anchor(manifest: &Value, name: &str) -> Option<u64> {
    let hour = manifest["utc_hour_ms"].as_u64()?;
    let first = manifest["first_recv_utc_ms"].as_u64()?;
    let last = manifest["last_recv_utc_ms"].as_u64()?;
    let end = hour.checked_add(crate::HOUR_MS)?;
    let tail = name.strip_prefix(&format!("hour-{hour}-"))?;
    if hour == 0
        || hour % crate::HOUR_MS != 0
        || tail.is_empty()
        || !tail.bytes().all(|b| b.is_ascii_digit())
        || tail.parse::<u64>().ok()? == 0
        || first < hour
        || first > last
        || last >= end
    {
        return None;
    }
    Some(end)
}

fn expired(anchor: Option<u64>, now: u64, retention_ms: u64) -> bool {
    retention_ms > 0
        && anchor
            .and_then(|a| a.checked_add(retention_ms))
            .is_some_and(|deadline| now >= deadline)
}

#[derive(Debug)]
pub enum StoreStop {
    LowDisk { free: u64, minimum: u64 },
    StorageError(String),
}

impl StoreStop {
    pub fn code(&self) -> u8 {
        match self {
            Self::LowDisk { .. } => 3,
            Self::StorageError(_) => 1,
        }
    }
    pub fn reason(&self) -> &'static str {
        match self {
            Self::LowDisk { .. } => "low_disk",
            Self::StorageError(_) => "store_error",
        }
    }
    pub fn message(&self) -> String {
        match self {
            Self::LowDisk { free, minimum } => {
                format!("DISCO-BAJO: {free} bytes libres < {minimum}")
            }
            Self::StorageError(error) => format!("ARCHIVO-PARADO: {error}"),
        }
    }
}

fn space(root: &Path, minimum: u64) -> Result<u64, StoreStop> {
    // La inyeccion solo existe en tests y es local al hilo; produccion mide el disco.
    #[cfg(test)]
    let measured = retention_tests::measured_free_bytes(root);
    #[cfg(not(test))]
    let measured = free_bytes(root);
    let free =
        measured.map_err(|e| StoreStop::StorageError(format!("no se puede medir disco: {e}")))?;
    if free < minimum {
        Err(StoreStop::LowDisk { free, minimum })
    } else {
        Ok(free)
    }
}

// La guarda de captura no depende de que el hilo de recodificacion acabe su trabajo.
pub async fn watch_space(root: &Path, minimum: u64, every: Duration) -> StoreStop {
    loop {
        if let Err(stop) = space(root, minimum) {
            return stop;
        }
        tokio::time::sleep(every).await;
    }
}

struct SpaceWriter<'a> {
    inner: File,
    cfg: Option<&'a StoreConfig>,
}

impl Write for SpaceWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(cfg) = self.cfg {
            if cfg.cancel.load(std::sync::atomic::Ordering::Acquire) {
                return Err(io::Error::other("archivo cancelado; se conserva el gzip"));
            }
            space(&cfg.root, cfg.min_free_bytes).map_err(|s| io::Error::other(s.message()))?;
        }
        self.inner.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn utc_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(windows)]
pub fn free_bytes(path: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut avail = 0u64;
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut avail,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(avail)
}

#[cfg(not(windows))]
pub fn free_bytes(_path: &Path) -> io::Result<u64> {
    Err(io::Error::other("free_bytes only on Windows"))
}

fn log_line(root: &Path, file: &str, row: &Value) {
    if let Ok(mut f) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(file))
    {
        let _ = writeln!(f, "{row}");
    }
}

/// Re-codifica una hora sellada. Orden: 1) el .jsonl.gz coincide con el SHA-256 de su
/// manifiesto; 2) codificar a .tgll.partial y publicarlo; 3) decodificar el .tgll desde disco
/// y comparar lineas, bytes y SHA-256 del crudo con los del original; 4) escribir el recibo
/// .recode.json; 5) solo entonces borrar el .jsonl.gz.
pub fn recode(dir: &Path, name: &str, p: codec::Params) -> CaptureResult<Value> {
    recode_checked(dir, name, p, None)
}

fn recode_checked(
    dir: &Path,
    name: &str,
    p: codec::Params,
    cfg: Option<&StoreConfig>,
) -> CaptureResult<Value> {
    let gz = dir.join(format!("{name}.jsonl.gz"));
    let manifest_path = dir.join(format!("{name}.manifest.json"));
    // P4c: el trabajo entero queda admitido (y delimitado si llega el corte) o no empieza.
    let _job = match cfg {
        Some(cfg) => Some(cfg.gate.admitir(OpStore::Recode).ok_or_else(|| {
            io::Error::other(format!(
                "{name}: compuerta de store cerrada; no se recodifica"
            ))
        })?),
        None => None,
    };
    let manifest: Value = serde_json::from_reader(File::open(&manifest_path)?)?;
    let started = Instant::now();
    let (gz_bytes, gz_sha) = digest(&gz)?;
    if manifest["sha256"].as_str() != Some(gz_sha.as_str())
        || manifest["bytes"].as_u64() != Some(gz_bytes)
    {
        return Err(io::Error::other(format!(
            "{name}: el .jsonl.gz no coincide con su manifiesto"
        ))
        .into());
    }
    let out = dir.join(format!("{name}.tgll"));
    let partial = dir.join(format!("{name}.tgll.partial"));
    let _ = remove(cfg, OpStore::RecodeBorraParcialViejo, &partial);
    let enc = {
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)?;
        let mut w = BufWriter::with_capacity(1 << 20, SpaceWriter { inner: file, cfg });
        let r = codec::encode(
            GzDecoder::new(io::BufReader::new(File::open(&gz)?)),
            &mut w,
            p,
        )?;
        w.flush()?;
        w.get_ref().inner.sync_all()?;
        r
    };
    let encode_s = started.elapsed().as_secs_f64();
    if manifest["rows"].as_u64() != Some(enc.lines) {
        let _ = remove(cfg, OpStore::RecodeBorraParcial, &partial);
        return Err(io::Error::other(format!(
            "{name}: filas del manifiesto {} != lineas leidas {}",
            manifest["rows"], enc.lines
        ))
        .into());
    }
    fs::rename(&partial, &out)?;
    let t = Instant::now();
    let dec = codec::decode(io::BufReader::new(File::open(&out)?), io::sink())?;
    let decode_s = t.elapsed().as_secs_f64();
    if dec.raw_sha256 != enc.raw_sha256 || dec.raw_bytes != enc.raw_bytes || dec.lines != enc.lines
    {
        return Err(io::Error::other(format!(
            "{name}: la decodificacion no es identica; se conserva el .jsonl.gz"
        ))
        .into());
    }
    let (tg_bytes, tg_sha) = digest(&out)?;
    let receipt = json!({
        "schema":1, "format":"TGLL0001", "file":format!("{name}.tgll"), "bytes":tg_bytes, "sha256":tg_sha,
        "source_file":manifest["file"], "source_bytes":gz_bytes, "source_sha256":gz_sha,
        "raw_jsonl_bytes":enc.raw_bytes, "raw_jsonl_sha256":enc.raw_sha256, "rows":enc.lines,
        "decoded_rows":dec.lines, "decoded_bytes":dec.raw_bytes, "decoded_sha256":dec.raw_sha256,
        "identical":true, "skeletons":enc.skeletons, "contexts":enc.contexts,
        "raw_fallback_lines":enc.raw_fallback_lines,
        "zstd_level":p.zstd_level, "zstd_window_log":p.window_log, "block_raw_bytes":p.block_raw_bytes,
        "encode_s":encode_s, "decode_verify_s":decode_s,
        "ratio_vs_gz": gz_bytes as f64 / tg_bytes.max(1) as f64,
        "utc_hour_ms":manifest["utc_hour_ms"], "first_recv_utc_ms":manifest["first_recv_utc_ms"],
        "last_recv_utc_ms":manifest["last_recv_utc_ms"], "recoded_utc_ms":utc_ms(),
    });
    let receipt_path = dir.join(format!("{name}.recode.json"));
    let _ = remove(
        cfg,
        OpStore::RecodeBorraReciboParcial,
        &receipt_path.with_extension("json.partial"),
    );
    write_json(&receipt_path, &receipt)?;
    if let Some(cfg) = cfg {
        if cfg.cancel.load(std::sync::atomic::Ordering::Acquire) {
            return Err(
                io::Error::other("archivo cancelado tras verificar; se conserva el gzip").into(),
            );
        }
        space(&cfg.root, cfg.min_free_bytes).map_err(|s| io::Error::other(s.message()))?;
    }
    if digest(&gz)? != (gz_bytes, gz_sha.clone()) {
        return Err(io::Error::other("gzip cambio durante la recodificacion; se conserva").into());
    }
    remove(cfg, OpStore::RecodeBorraGzip, &gz)?;
    Ok(receipt)
}

struct Group {
    dir: PathBuf,
    name: String,
    hour: u64,
    span_ms: u64,
    bytes: u64,
    tgll_bytes: u64,
    recoded: bool,
    expiry_anchor_ms: Option<u64>,
}

fn scan(root: &Path) -> (Vec<Group>, u64, Vec<(PathBuf, String)>) {
    let mut groups = Vec::new();
    let mut pending = Vec::new();
    let mut total = 0u64;
    let Ok(runs) = fs::read_dir(root) else {
        return (groups, total, pending);
    };
    for run in runs.flatten() {
        let path = run.path();
        if let Ok(m) = run.metadata() {
            if m.is_file() {
                total += m.len();
                continue;
            }
        }
        let Ok(files) = fs::read_dir(&path) else {
            continue;
        };
        let mut names = std::collections::BTreeSet::new();
        for f in files.flatten() {
            let len = f.metadata().map(|m| m.len()).unwrap_or(0);
            total += len;
            let n = f.file_name().to_string_lossy().to_string();
            if let Some(base) = n.strip_suffix(".manifest.json") {
                names.insert(base.to_string());
            }
        }
        for name in names {
            let size = |ext: &str| {
                fs::metadata(path.join(format!("{name}{ext}")))
                    .map(|m| m.len())
                    .ok()
            };
            let manifest: Value = fs::read(path.join(format!("{name}.manifest.json")))
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or(Value::Null);
            let recoded =
                path.join(format!("{name}.recode.json")).exists() && size(".tgll").is_some();
            let gz = size(".jsonl.gz");
            if !recoded && gz.is_some() {
                pending.push((path.clone(), name.clone()));
            }
            let first = manifest["first_recv_utc_ms"].as_u64().unwrap_or(0);
            let last = manifest["last_recv_utc_ms"].as_u64().unwrap_or(first);
            let bytes = [".tgll", ".jsonl.gz", ".manifest.json", ".recode.json"]
                .iter()
                .filter_map(|e| size(e))
                .sum();
            groups.push(Group {
                dir: path.clone(),
                name: name.clone(),
                hour: manifest["utc_hour_ms"].as_u64().unwrap_or(0),
                span_ms: last.saturating_sub(first),
                bytes,
                tgll_bytes: if recoded {
                    size(".tgll").unwrap_or(0)
                } else {
                    0
                },
                recoded,
                expiry_anchor_ms: expiry_anchor(&manifest, &name),
            });
        }
    }
    groups.sort_by_key(|g| (g.hour, g.name.clone()));
    (groups, total, pending)
}

fn signal_stop(
    root: &Path,
    tx: &mut Option<tokio::sync::oneshot::Sender<StoreStop>>,
    stop: StoreStop,
) {
    eprintln!("{}", stop.message());
    log_line(
        root,
        "RETENTION.log",
        &json!({"utc_ms":utc_ms(),"event":stop.reason(),"message":stop.message()}),
    );
    if let Some(tx) = tx.take() {
        let _ = tx.send(stop);
    }
}

// Un recibo presente no basta: cotejarlo con el manifiesto y con el contenedor en disco.
// Solo se invoca sobre candidatos vencidos; no relee un ano de datos cada minuto.
fn verify_archive_group(g: &Group) -> CaptureResult<()> {
    let manifest: Value =
        serde_json::from_reader(File::open(g.dir.join(format!("{}.manifest.json", g.name)))?)?;
    let receipt: Value =
        serde_json::from_reader(File::open(g.dir.join(format!("{}.recode.json", g.name)))?)?;
    let (bytes, sha) = digest(&g.dir.join(format!("{}.tgll", g.name)))?;
    let valid = receipt["identical"] == true
        && receipt["format"] == "TGLL0001"
        && receipt["file"].as_str() == Some(format!("{}.tgll", g.name).as_str())
        && receipt["bytes"].as_u64() == Some(bytes)
        && receipt["sha256"].as_str() == Some(sha.as_str())
        && receipt["source_file"] == manifest["file"]
        && receipt["source_bytes"] == manifest["bytes"]
        && receipt["source_sha256"] == manifest["sha256"]
        && receipt["rows"].as_u64().is_some()
        && receipt["rows"] == manifest["rows"]
        && receipt["rows"] == receipt["decoded_rows"]
        && receipt["raw_jsonl_bytes"].as_u64().is_some()
        && receipt["raw_jsonl_bytes"] == receipt["decoded_bytes"]
        && receipt["raw_jsonl_sha256"].as_str().is_some()
        && receipt["raw_jsonl_sha256"] == receipt["decoded_sha256"]
        && expiry_anchor(&manifest, &g.name) == g.expiry_anchor_ms;
    if !valid {
        return Err(io::Error::other(format!(
            "{}: recibo o manifiesto no valido; no se borra",
            g.name
        ))
        .into());
    }
    Ok(())
}

/// Retencion por edad. Con disco bajo/incierto se para sin empezar recode ni purga.
/// SALIDAFALLO P2: el aviso es `&mut Option` para que el envoltorio del hilo fije su estado de
/// salida antes de soltarlo.
pub fn run(
    cfg: StoreConfig,
    sealed: mpsc::Receiver<(PathBuf, String)>,
    stop_tx: &mut Option<tokio::sync::oneshot::Sender<StoreStop>>,
) {
    crate::inyeccion::store_al_empezar(&cfg.gate);
    let mut failed = std::collections::HashSet::new();
    let mut status = CadenciaStatus::pendiente();
    loop {
        if cfg.cancel.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let free = match space(&cfg.root, cfg.min_free_bytes) {
            Ok(free) => free,
            Err(stop) => {
                signal_stop(&cfg.root, stop_tx, stop);
                return;
            }
        };
        let now = cfg.reloj.utc_ms();
        let (groups, total, _) = scan(&cfg.root);
        let mut deleted = Vec::new();
        for g in groups
            .iter()
            .filter(|g| g.recoded && expired(g.expiry_anchor_ms, now, cfg.retention_ms))
        {
            // Un gzip coexistente indica una transaccion sin cerrar. No se toca su grupo.
            if g.dir.join(format!("{}.jsonl.gz", g.name)).exists() {
                continue;
            }
            if cfg.cancel.load(std::sync::atomic::Ordering::Acquire) {
                return;
            }
            let Some(_job) = cfg.gate.admitir(OpStore::Retencion) else {
                return;
            };
            if let Err(e) = verify_archive_group(g) {
                signal_stop(&cfg.root, stop_tx, StoreStop::StorageError(e.to_string()));
                return;
            }
            let intent = json!({"utc_ms":now,"event":"expiry_intent","hour":g.name,
                "run":g.dir,"expiry_anchor_ms":g.expiry_anchor_ms,"retention_ms":cfg.retention_ms});
            log_line(&cfg.root, "RETENTION.log", &intent);
            for ext in [".tgll", ".recode.json", ".manifest.json"] {
                if cfg.cancel.load(std::sync::atomic::Ordering::Acquire) {
                    return;
                }
                if let Err(stop) = space(&cfg.root, cfg.min_free_bytes) {
                    signal_stop(&cfg.root, stop_tx, stop);
                    return;
                }
                if let Err(e) = cfg.gate.borrar(
                    OpStore::RetencionBorra,
                    &g.dir.join(format!("{}{ext}", g.name)),
                ) {
                    signal_stop(
                        &cfg.root,
                        stop_tx,
                        StoreStop::StorageError(format!("purga {}{ext}: {e}", g.name)),
                    );
                    return;
                }
            }
            deleted.push(json!({"run":g.dir,"hour":g.name,"bytes":g.bytes}));
        }
        if !deleted.is_empty() {
            log_line(
                &cfg.root,
                "RETENTION.log",
                &json!({"utc_ms":now,"event":"expired_365d",
                "retention_ms":cfg.retention_ms,"total_before":total,"deleted":deleted}),
            );
        }
        // Releer el disco: nunca publicar como existentes grupos ya borrados ni restas supuestas.
        let (groups, total_now, pending) = scan(&cfg.root);
        let ahora = cfg.reloj.mono();
        if status.toca(ahora, cfg.scan_every) || !deleted.is_empty() {
            status.publicada(ahora);
            let recoded: Vec<&Group> = groups.iter().filter(|g| g.recoded).collect();
            let bytes: u64 = recoded.iter().map(|g| g.tgll_bytes).sum();
            let span: u64 = recoded.iter().map(|g| g.span_ms).sum();
            let rate = (span > 0).then(|| bytes as f64 * 86_400_000.0 / span as f64);
            let status = json!({
                "utc_ms":now,"free_bytes":free,"min_free_bytes":cfg.min_free_bytes,
                "v2_total_bytes":total_now,"retention_ms":cfg.retention_ms,"retention_policy":"time_only",
                "hours_recoded":recoded.len(),"hours_pending":groups.iter().filter(|g| !g.recoded).count(),
                "hours_invalid_age":groups.iter().filter(|g| g.expiry_anchor_ms.is_none()).count(),
                "tgll_bytes":bytes,"tgll_span_ms":span,"measured_tgll_bytes_per_day":rate,
                "measured_tgll_gib_per_day":rate.map(|r| r / (1u64 << 30) as f64),
                "oldest_hour_ms":groups.first().map(|g| g.hour),"newest_hour_ms":groups.last().map(|g| g.hour),
            });
            if let Err(e) = write_json_replace(&cfg.root.join("STATUS.json"), &status) {
                signal_stop(
                    &cfg.root,
                    stop_tx,
                    StoreStop::StorageError(format!("STATUS: {e}")),
                );
                return;
            }
        }
        // Recuperar el reinicio entre recibo durable y borrado del gzip, sin re-encode ni sobreescritura.
        for g in groups
            .iter()
            .filter(|g| g.recoded && g.dir.join(format!("{}.jsonl.gz", g.name)).exists())
        {
            let Some(_job) = cfg.gate.admitir(OpStore::Recuperacion) else {
                return;
            };
            let recovered = (|| -> CaptureResult<()> {
                verify_archive_group(g)?;
                let gz = g.dir.join(format!("{}.jsonl.gz", g.name));
                let m: Value = serde_json::from_reader(File::open(
                    g.dir.join(format!("{}.manifest.json", g.name)),
                )?)?;
                let r: Value = serde_json::from_reader(File::open(
                    g.dir.join(format!("{}.recode.json", g.name)),
                )?)?;
                let (bytes, sha) = digest(&gz)?;
                let decoded = codec::decode(
                    io::BufReader::new(File::open(g.dir.join(format!("{}.tgll", g.name)))?),
                    io::sink(),
                )?;
                if m["bytes"].as_u64() != Some(bytes)
                    || m["sha256"].as_str() != Some(sha.as_str())
                    || r["raw_jsonl_bytes"].as_u64() != Some(decoded.raw_bytes)
                    || r["rows"].as_u64() != Some(decoded.lines)
                    || r["raw_jsonl_sha256"].as_str() != Some(decoded.raw_sha256.as_str())
                {
                    return Err(
                        io::Error::other("recuperacion no identica: gzip conservado").into(),
                    );
                }
                if cfg.cancel.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(io::Error::other("recuperacion cancelada: gzip conservado").into());
                }
                space(&cfg.root, cfg.min_free_bytes).map_err(|s| io::Error::other(s.message()))?;
                cfg.gate.borrar(OpStore::RecuperacionBorraGzip, &gz)?;
                Ok(())
            })();
            if let Err(e) = recovered {
                signal_stop(&cfg.root, stop_tx, StoreStop::StorageError(e.to_string()));
                return;
            }
        }
        if let Some((dir, name)) = pending
            .iter()
            .find(|(d, n)| !failed.contains(&(d.clone(), n.clone())))
        {
            match recode_checked(dir, name, cfg.codec, Some(&cfg)) {
                Ok(r) => log_line(&cfg.root, "RECODE.log", &r),
                Err(e) => {
                    log_line(
                        &cfg.root,
                        "RECODE.log",
                        &json!({"utc_ms":utc_ms(),"file":name,"error":e.to_string()}),
                    );
                    failed.insert((dir.clone(), name.clone()));
                }
            }
            continue;
        }
        match cfg.reloj.esperar(&sealed, cfg.scan_every) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn write_json_replace(path: &Path, value: &Value) -> io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(&tmp, path)
}

#[cfg(test)]
#[path = "store_reloj_tests.rs"]
mod reloj_tests;

#[cfg(test)]
mod retention_tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use futures_util::FutureExt;
    use std::{
        cell::RefCell,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicU64, Ordering},
        },
    };

    const RETAIN: u64 = crate::RETENTION_MS;
    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    pub(super) struct DiskReading {
        free: u64,
        calls: usize,
        falla: bool,
    }
    thread_local! {
        static DISK_READING: RefCell<Option<DiskReading>> = const { RefCell::new(None) };
    }

    // No se llena el SSD ni se cambia su umbral real. Cada test controla solo su hilo.
    pub(super) struct DiskProbe;
    impl DiskProbe {
        pub(super) fn new(free: u64) -> Self {
            DISK_READING.with(|slot| {
                let mut slot = slot.borrow_mut();
                assert!(slot.is_none(), "inyeccion de disco anidada");
                *slot = Some(DiskReading {
                    free,
                    calls: 0,
                    falla: false,
                });
            });
            Self
        }
        pub(super) fn set(&self, free: u64) {
            DISK_READING.with(|slot| slot.borrow_mut().as_mut().unwrap().free = free);
        }
        pub(super) fn calls(&self) -> usize {
            DISK_READING.with(|slot| slot.borrow().as_ref().unwrap().calls)
        }
    }
    impl Drop for DiskProbe {
        fn drop(&mut self) {
            DISK_READING.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
    pub(super) fn measured_free_bytes(root: &Path) -> io::Result<u64> {
        let injected = DISK_READING.with(|slot| {
            slot.borrow_mut().as_mut().map(|reading| {
                reading.calls += 1;
                (reading.free, reading.falla)
            })
        });
        match injected {
            // FALLAS-1 (A32): error real devuelto por la sonda inyectada, no una cifra supuesta.
            Some((_, true)) => Err(io::Error::other(
                "sonda de disco inyectada: fallo de lectura",
            )),
            Some((free, false)) => Ok(free),
            None => free_bytes(root),
        }
    }
    /// FALLAS-1: lectura de disco de la sonda del hilo, desde el guion de un reloj de prueba.
    pub(super) fn fijar_disco(free: u64, falla: bool) {
        DISK_READING.with(|slot| {
            if let Some(r) = slot.borrow_mut().as_mut() {
                r.free = free;
                r.falla = falla;
            }
        });
    }
    pub(super) fn llamadas_disco() -> usize {
        DISK_READING.with(|slot| slot.borrow().as_ref().map_or(0, |r| r.calls))
    }

    pub(super) struct Fixture {
        pub(super) root: PathBuf,
        pub(super) dir: PathBuf,
        pub(super) name: String,
        pub(super) gzip: Vec<u8>,
    }
    impl Fixture {
        pub(super) fn sealed() -> CaptureResult<Self> {
            let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(".retention-tests")
                .join(format!(
                    "{}-{stamp}-{}",
                    std::process::id(),
                    NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
                ));
            let dir = root.join("run");
            fs::create_dir_all(&dir)?;
            let hour = crate::HOUR_MS;
            let name = format!("hour-{hour}-0001");
            let raw = format!(
                "{{\"recv_utc_ms\":{hour},\"seq\":1}}\n{{\"recv_utc_ms\":{},\"seq\":2}}\n",
                hour + crate::HOUR_MS - 1
            );
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(raw.as_bytes())?;
            let gzip = encoder.finish()?;
            let fixture = Self {
                root,
                dir,
                name,
                gzip,
            };
            fs::write(fixture.path(".jsonl.gz"), &fixture.gzip)?;
            let (bytes, sha) = digest(&fixture.path(".jsonl.gz"))?;
            let mut m = manifest(hour);
            m["file"] = json!(format!("{}.jsonl.gz", fixture.name));
            m["bytes"] = json!(bytes);
            m["sha256"] = json!(sha);
            m["rows"] = json!(raw.lines().count() as u64);
            fixture.put_json(".manifest.json", &m)?;
            Ok(fixture)
        }
        pub(super) fn archived_with_gzip() -> CaptureResult<Self> {
            let fixture = Self::sealed()?;
            recode(&fixture.dir, &fixture.name, Self::params())?;
            assert!(!fixture.path(".jsonl.gz").exists());
            // Recrea exactamente el estado de caida: recibo durable y gzip coexistente.
            fs::write(fixture.path(".jsonl.gz"), &fixture.gzip)?;
            Ok(fixture)
        }
        pub(super) fn params() -> codec::Params {
            // Parametros pequenos del fixture ya usados por los tests de main.rs.
            codec::Params {
                zstd_level: 3,
                block_raw_bytes: 1 << 20,
                window_log: 20,
            }
        }
        pub(super) fn path(&self, ext: &str) -> PathBuf {
            self.dir.join(format!("{}{ext}", self.name))
        }
        fn get_json(&self, ext: &str) -> CaptureResult<Value> {
            Ok(serde_json::from_reader(File::open(self.path(ext))?)?)
        }
        fn put_json(&self, ext: &str, value: &Value) -> CaptureResult<()> {
            Ok(fs::write(
                self.path(ext),
                serde_json::to_vec_pretty(value)?,
            )?)
        }
        fn group(&self) -> Group {
            let (mut groups, _, pending) = scan(&self.root);
            assert!(pending.is_empty());
            assert_eq!(groups.len(), 1);
            let group = groups.pop().unwrap();
            assert!(group.recoded);
            assert!(group.expiry_anchor_ms.is_some());
            group
        }
        pub(super) fn cfg(&self) -> StoreConfig {
            let cancel = Arc::new(AtomicBool::new(false));
            StoreConfig {
                root: self.root.clone(),
                retention_ms: RETAIN,
                cancel: Arc::clone(&cancel),
                min_free_bytes: crate::MIN_FREE_BYTES,
                codec: Self::params(),
                scan_every: crate::STORE_SCAN_EVERY,
                gate: Arc::new(CompuertaStore::real(cancel)),
                reloj: Arc::new(RelojStoreReal::nuevo()),
            }
        }
        fn run_once(&self) -> Option<StoreStop> {
            let (sealed_tx, sealed_rx) = mpsc::channel();
            drop(sealed_tx); // Una pasada real de run, sin hilo ni espera de scan_every.
            let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
            let mut stop_tx = Some(stop_tx);
            run(self.cfg(), sealed_rx, &mut stop_tx);
            drop(stop_tx);
            match stop_rx.try_recv() {
                Ok(stop) => Some(stop),
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => None,
                Err(error) => panic!("run no cerro el canal: {error}"),
            }
        }
        pub(super) fn assert_gzip_kept(&self) -> CaptureResult<()> {
            assert_eq!(fs::read(self.path(".jsonl.gz"))?, self.gzip);
            Ok(())
        }
    }

    #[test]
    fn verify_archive_group_accepts_valid_manifest_receipt_and_tgll() -> CaptureResult<()> {
        let fixture = Fixture::archived_with_gzip()?;
        verify_archive_group(&fixture.group())?;
        fixture.assert_gzip_kept()
    }

    #[test]
    fn verify_archive_group_rejects_invalid_manifest_receipt_and_tgll() -> CaptureResult<()> {
        let fixture = Fixture::archived_with_gzip()?;
        let group = fixture.group();
        let m = fixture.get_json(".manifest.json")?;
        let r = fixture.get_json(".recode.json")?;
        for (field, value) in [
            ("file", json!("otro.gz")),
            ("bytes", json!(0)),
            ("sha256", json!("alterado")),
            ("rows", Value::Null),
            ("utc_hour_ms", json!(0)),
            ("first_recv_utc_ms", json!(0)),
            ("last_recv_utc_ms", json!(0)),
        ] {
            let mut invalid = m.clone();
            invalid[field] = value;
            fixture.put_json(".manifest.json", &invalid)?;
            assert!(verify_archive_group(&group).is_err(), "manifiesto: {field}");
        }
        fixture.put_json(".manifest.json", &m)?;
        for (field, value) in [
            ("identical", json!(false)),
            ("format", json!("otro")),
            ("file", json!("otro.tgll")),
            ("bytes", json!(0)),
            ("sha256", json!("alterado")),
            ("source_file", json!("otro.gz")),
            ("source_bytes", json!(0)),
            ("source_sha256", json!("alterado")),
            ("rows", Value::Null),
            ("decoded_rows", json!(0)),
            ("raw_jsonl_bytes", Value::Null),
            ("decoded_bytes", json!(0)),
            ("raw_jsonl_sha256", Value::Null),
            ("decoded_sha256", json!("alterado")),
        ] {
            let mut invalid = r.clone();
            invalid[field] = value;
            fixture.put_json(".recode.json", &invalid)?;
            assert!(verify_archive_group(&group).is_err(), "recibo: {field}");
        }
        fixture.put_json(".recode.json", &r)?;
        for (ext, valid) in [(".manifest.json", &m), (".recode.json", &r)] {
            fs::write(fixture.path(ext), b"{")?;
            assert!(
                verify_archive_group(&group).is_err(),
                "JSON corrupto: {ext}"
            );
            fixture.put_json(ext, valid)?;
        }
        let tgll = fs::read(fixture.path(".tgll"))?;
        let mut altered = tgll.clone();
        altered[0] ^= 1; // Igual longitud: se exige el SHA, no basta el tamano.
        fs::write(fixture.path(".tgll"), altered)?;
        assert!(verify_archive_group(&group).is_err());
        fs::write(fixture.path(".tgll"), &tgll[..tgll.len() - 1])?;
        assert!(verify_archive_group(&group).is_err());
        fs::write(fixture.path(".tgll"), tgll)?;
        verify_archive_group(&group)?;
        fixture.assert_gzip_kept()
    }

    fn assert_writer_stops(cancelled: bool) -> CaptureResult<()> {
        let fixture = Fixture::sealed()?;
        let cfg = fixture.cfg();
        cfg.cancel.store(cancelled, Ordering::Release);
        let probe = DiskProbe::new(cfg.min_free_bytes - 1);
        assert!(matches!(
            space(&cfg.root, cfg.min_free_bytes),
            Err(StoreStop::LowDisk { .. })
        ));
        let before = probe.calls();
        let path = fixture.path(".write-probe");
        let file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        let mut writer = SpaceWriter {
            inner: file,
            cfg: Some(&cfg),
        };
        let error = writer.write(codec::MAGIC).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(if cancelled { "cancelado" } else { "DISCO-BAJO" })
        );
        assert_eq!(writer.inner.metadata()?.len(), 0);
        assert_eq!(probe.calls() - before, if cancelled { 0 } else { 1 });
        // Tambien pasa por el BufWriter y el encoder usados en produccion.
        let error = recode_checked(&fixture.dir, &fixture.name, cfg.codec, Some(&cfg)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(if cancelled { "cancelado" } else { "DISCO-BAJO" })
        );
        assert_eq!(fs::metadata(fixture.path(".tgll.partial"))?.len(), 0);
        assert!(!fixture.path(".tgll").exists());
        assert!(!fixture.path(".recode.json").exists());
        fixture.assert_gzip_kept()
    }

    #[test]
    fn space_writer_low_disk_stops_write_and_preserves_gzip() -> CaptureResult<()> {
        assert_writer_stops(false)
    }
    #[test]
    fn space_writer_cancelled_stops_write_and_preserves_gzip() -> CaptureResult<()> {
        assert_writer_stops(true)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn watch_space_detects_low_disk_after_paused_interval() -> CaptureResult<()> {
        tokio::time::pause();
        let started = tokio::time::Instant::now();
        let fixture = Fixture::sealed()?;
        let minimum = fixture.cfg().min_free_bytes;
        let probe = DiskProbe::new(minimum); // La igualdad no debe parar el monitor.
        let every = crate::STORE_SCAN_EVERY;
        let monitor = watch_space(&fixture.root, minimum, every);
        tokio::pin!(monitor);
        assert!(monitor.as_mut().now_or_never().is_none());
        assert_eq!(probe.calls(), 1);
        probe.set(minimum - 1);
        let tick = Duration::from_millis(1);
        tokio::time::advance(every - tick).await;
        assert!(monitor.as_mut().now_or_never().is_none());
        assert_eq!(
            probe.calls(),
            1,
            "no se recrea el future ni se adelanta la medida"
        );
        tokio::time::advance(tick).await;
        let stop = monitor.await;
        assert!(
            matches!(stop, StoreStop::LowDisk { free, minimum: bound } if free == minimum - 1 && bound == minimum)
        );
        assert_eq!(stop.code(), 3);
        assert_eq!(stop.reason(), "low_disk");
        assert_eq!(probe.calls(), 2);
        assert!(started.elapsed() >= every);
        fixture.assert_gzip_kept()
    }

    #[test]
    fn recovery_closes_receipt_gzip_crash_without_reencoding() -> CaptureResult<()> {
        let fixture = Fixture::archived_with_gzip()?;
        let probe = DiskProbe::new(fixture.cfg().min_free_bytes);
        let preserved: Vec<_> = [".tgll", ".recode.json", ".manifest.json"]
            .into_iter()
            .map(|ext| Ok((ext, fs::read(fixture.path(ext))?)))
            .collect::<io::Result<_>>()?;
        assert!(fixture.run_once().is_none());
        assert!(!fixture.path(".jsonl.gz").exists());
        assert!(!fixture.path(".tgll.partial").exists());
        for (ext, bytes) in preserved {
            assert_eq!(fs::read(fixture.path(ext))?, bytes, "{ext}");
        }
        verify_archive_group(&fixture.group())?;
        assert_eq!(
            probe.calls(),
            2,
            "comprobacion inicial y justo antes de borrar el gzip"
        );
        Ok(())
    }

    #[test]
    fn recovery_keeps_gzip_when_receipt_is_invalid() -> CaptureResult<()> {
        let fixture = Fixture::archived_with_gzip()?;
        let _probe = DiskProbe::new(fixture.cfg().min_free_bytes);
        let mut receipt = fixture.get_json(".recode.json")?;
        receipt["identical"] = json!(false);
        fixture.put_json(".recode.json", &receipt)?;
        assert!(matches!(
            fixture.run_once(),
            Some(StoreStop::StorageError(_))
        ));
        assert!(fixture.path(".tgll").exists());
        assert!(fixture.path(".manifest.json").exists());
        fixture.assert_gzip_kept()
    }

    #[test]
    fn recovery_keeps_changed_gzip() -> CaptureResult<()> {
        let mut fixture = Fixture::archived_with_gzip()?;
        let _probe = DiskProbe::new(fixture.cfg().min_free_bytes);
        fixture.gzip.push(0);
        fs::write(fixture.path(".jsonl.gz"), &fixture.gzip)?;
        assert!(matches!(
            fixture.run_once(),
            Some(StoreStop::StorageError(_))
        ));
        assert!(fixture.path(".tgll").exists());
        assert!(fixture.path(".recode.json").exists());
        fixture.assert_gzip_kept()
    }

    #[test]
    fn recovery_keeps_gzip_when_tgll_cannot_decode() -> CaptureResult<()> {
        let fixture = Fixture::archived_with_gzip()?;
        let _probe = DiskProbe::new(fixture.cfg().min_free_bytes);
        fs::write(fixture.path(".tgll"), codec::MAGIC)?; // Cabecera sin cuerpo zstd.
        let (bytes, sha) = digest(&fixture.path(".tgll"))?;
        let mut receipt = fixture.get_json(".recode.json")?;
        receipt["bytes"] = json!(bytes);
        receipt["sha256"] = json!(sha);
        fixture.put_json(".recode.json", &receipt)?;
        verify_archive_group(&fixture.group())?; // Obliga a probar el decode de recuperacion.
        assert!(matches!(
            fixture.run_once(),
            Some(StoreStop::StorageError(_))
        ));
        fixture.assert_gzip_kept()
    }
    fn manifest(hour: u64) -> Value {
        json!({"utc_hour_ms":hour,"first_recv_utc_ms":hour,"last_recv_utc_ms":hour+crate::HOUR_MS-1})
    }
    #[test]
    fn expiration_uses_end_of_hour() {
        let hour = crate::HOUR_MS;
        let a = expiry_anchor(&manifest(hour), &format!("hour-{hour}-0001"));
        assert_eq!(a, Some(hour + crate::HOUR_MS));
        assert!(!expired(a, hour + RETAIN, RETAIN));
        assert!(!expired(a, hour + crate::HOUR_MS + RETAIN - 1, RETAIN));
        assert!(expired(a, hour + crate::HOUR_MS + RETAIN, RETAIN));
    }
    #[test]
    fn missing_or_invalid_age_cannot_expire() {
        for m in [Value::Null, json!({}), manifest(0), manifest(1)] {
            assert_eq!(expiry_anchor(&m, "hour-0-0001"), None);
        }
        assert!(!expired(None, u64::MAX, RETAIN));
        assert!(!expired(Some(u64::MAX), u64::MAX, RETAIN));
        assert!(!expired(Some(1), u64::MAX, 0));
        assert!(!expired(Some(100), 99, RETAIN));
    }
    #[test]
    fn future_or_mismatched_manifest_is_kept() {
        let hour = crate::HOUR_MS;
        assert_eq!(expiry_anchor(&manifest(hour), "hour-7200000-0001"), None);
        let mut m = manifest(hour);
        m["last_recv_utc_ms"] = json!(hour + crate::HOUR_MS);
        assert_eq!(expiry_anchor(&m, &format!("hour-{hour}-0001")), None);
    }
}
