//! Captura en vivo v2 (40 simbolos) de Trader Gemini. Base: kit v1 capture-kit-v2
//! (src/main.rs sha256 f8d3647c...). Especificacion: qoder_cli_especificacion_captura_v2_40_simbolos.md.
//! Solo flujos PUBLICOS de Binance USD-M: sin llaves, sin cuenta y sin ordenes.
//! Retencion del archivo: retention_ms es fijo en 365 dias UTC desde el final del
//! cubo horario (RETENTION_MS). No se configura por argumentos: retention_ms=N y
//! retention_days=N no son opciones de configuracion de la CLI.

mod codec;
mod gate;
mod l2;
mod public_l2;
mod salidafallo;
mod store;

use salidafallo::{Admision, Clase, Control};

#[cfg(test)]
mod banco_local;

use flate2::{Compression, write::GzEncoder};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc as blocking_channel,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch},
    task::JoinSet,
    time::{self, MissedTickBehavior},
};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message};

pub type CaptureResult<T> = Result<T, Box<dyn Error + Send + Sync>>;
type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

// ---------------------------------------------------------------------------------------------
// Parametros. Cada uno con su derivacion (tambien se copian a RUN.json).
// ---------------------------------------------------------------------------------------------
const HOUR_MS: u64 = 3_600_000;
/// Spec 2.1: 8 conexiones = 4 shards /public + 4 shards /market (10 simbolos por shard).
const SHARDS_PER_ROUTE: usize = 4;
/// Spec 2.7: 8.192 filas (v1) x 10,74 (carga v2 / pico horario v1) = 87.844 -> potencia de 2 superior.
const QUEUE_CAPACITY: usize = 131_072;
/// Spec 2.7: 32 MiB (v1) x 10,74 = 342 MiB -> potencia de 2 superior.
const QUEUE_BYTE_BUDGET: usize = 512 * 1024 * 1024;
/// v1 main.rs:34.
const ROW_CHARGE_OVERHEAD: usize = 2_048;
/// v1 main.rs:35.
const SEAL_QUEUE_CAPACITY: usize = 2;
/// v1 main.rs:36.
const ROTATION_GRACE_MS: u64 = 2_000;
/// Spec 2.3: validez documentada de una conexion (24 h) menos 1 h de margen.
const ROTATE_AFTER: Duration = Duration::from_secs(23 * 3600);
/// Spec 2.3: plazo de gracia del solapamiento = tope probado del backoff v1 (30 s, main.rs:776).
const OVERLAP_GRACE: Duration = Duration::from_secs(30);
/// Spec 2.3: retencion maxima de la conexion vieja = margen de rotacion (24 h - 23 h).
const HOLD_MAX: Duration = Duration::from_secs(3600);
/// v1 main.rs:744 y 776: espera inicial 1 s, se duplica, tope 30 s.
const BACKOFF_START: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(30);
/// v1 main.rs:756-759: la espera vuelve a 1 s si la conexion vivio 10 s o mas.
const BACKOFF_RESET_LIFE: Duration = Duration::from_secs(10);
/// Spec 2.4: parte al azar uniforme (0, 1 s).
const JITTER_MAX_MS: u64 = 1_000;
/// Spec 2.4: desfase por shard = hueco de reconexion medido (1,924 s) redondeado a 2 s.
const SHARD_OFFSET: Duration = Duration::from_secs(2);
/// v1 main.rs:750, 712 y 726.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const SILENCE_LIMIT: Duration = Duration::from_secs(90);
const PONG_TIMEOUT: Duration = Duration::from_secs(5);
/// Spec 2.5: markPrice@1s llega una vez por segundo; 3 periodos sin mensaje = stall.
const MARK_PERIOD: Duration = Duration::from_secs(1);
const STALL_PERIODS: u32 = 3;
/// v1 main.rs:693: espera por defecto tras un fallo REST.
const REST_DEFAULT_COOLDOWN_S: u64 = 5;
/// Spec 2.6.1: exchangeInfo al arrancar y cada 24 h.
const INFO_REFRESH: Duration = Duration::from_secs(24 * 3600);
/// Doc de Binance /fapi/v1/aggTrades: limit maximo 1000.
const AGG_LIMIT_MAX: u64 = 1_000;
// R3: 365 dias, mandato de Jhonatan en encargo_retencion_v2.md, 18 nov 2026.
const RETENTION_DAYS: u64 = 365;
const RETENTION_MS: u64 = RETENTION_DAYS * 24 * HOUR_MS;
/// REGLAS_DE_TRABAJO.md regla 4: 40 GB libres; en GiB como el supervisor v1 (`/1GB`), mas estricto.
const MIN_FREE_BYTES: u64 = 40 * (1 << 30);
/// Archivo: bloque de 64 MiB de JSONL crudo por bloque del codec (cota de memoria del recodificador).
const CODEC_BLOCK_RAW: usize = 64 << 20;
/// zstd: ventana de 2^27 = 128 MiB (cubre dos bloques). Nivel 9: medido en la prueba de 2 h sobre la hora
/// completa 0002 (6,69 GB crudos) con el PC al 100 %: nivel 19 = 232,4 MB en 963 s; nivel 9 = 257,8 MB
/// en 152 s. El 19 no garantiza re-codificar una hora en menos de una hora con carga; el 9 si, con margen.
const CODEC_WINDOW_LOG: u32 = 27;
const CODEC_LEVEL_DEFAULT: i32 = 9;
/// Vuelta del hilo de archivo (guarda de disco, ventana y STATUS.json).
const STORE_SCAN_EVERY: Duration = Duration::from_secs(60);
/// Vigilancia del archivo de parada.
const STOP_FILE: &str = "PARAR_CAPTURA2.txt";

#[derive(Clone, Copy)]
pub struct Clock(Instant);

#[derive(Clone, Copy)]
pub struct Stamp {
    pub utc_ms: u64,
    pub mono_ns: u64,
}

impl Clock {
    pub fn stamp(self) -> CaptureResult<Stamp> {
        Ok(Stamp {
            utc_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_millis()
                .try_into()?,
            mono_ns: self.0.elapsed().as_nanos().try_into()?,
        })
    }
}

#[derive(Clone, Copy)]
struct Venue {
    name: &'static str,
    ws: &'static str,
    rest: &'static str,
}

impl Venue {
    fn parse(value: &str) -> CaptureResult<Self> {
        match value {
            "mainnet" => Ok(Self {
                name: "mainnet",
                ws: "fstream.binance.com",
                rest: "https://fapi.binance.com",
            }),
            "testnet" => Ok(Self {
                name: "testnet",
                ws: "stream.binancefuture.com",
                rest: "https://testnet.binancefuture.com",
            }),
            _ => Err(io::Error::other("venue must be mainnet or testnet").into()),
        }
    }

    fn url(self, route: &str, symbols: &[String], with_force: bool) -> String {
        let mut names = Vec::new();
        for symbol in symbols {
            let s = symbol.to_ascii_lowercase();
            if route == "public" {
                names.extend([format!("{s}@bookTicker"), format!("{s}@depth@100ms")]);
            } else {
                names.extend([format!("{s}@aggTrade"), format!("{s}@markPrice@1s")]);
            }
        }
        if with_force {
            names.push("!forceOrder@arr".into());
        }
        format!("wss://{}/{route}/stream?streams={}", self.ws, names.join("/"))
    }
}

fn new_file(path: &Path) -> io::Result<File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

const SHARING_VIOLATION: i32 = 32;
const SHARING_RETRY_DELAYS_MS: [u64; 8] = [25, 50, 100, 200, 400, 800, 1_600, 3_200];

fn retry_archive_sharing_violation<T>(
    operation: &str,
    path: &Path,
    mut action: impl FnMut() -> io::Result<T>,
) -> io::Result<T> {
    let mut delays = SHARING_RETRY_DELAYS_MS.into_iter();
    loop {
        match action() {
            Ok(value) => return Ok(value),
            Err(error) => {
                if cfg!(windows) && error.raw_os_error() == Some(SHARING_VIOLATION) {
                    if let Some(delay_ms) = delays.next() {
                        thread::sleep(Duration::from_millis(delay_ms));
                        continue;
                    }
                }
                return Err(io::Error::new(
                    error.kind(),
                    format!("{operation} {}: {error}", path.display()),
                ));
            }
        }
    }
}

pub fn write_json(path: &Path, value: &Value) -> CaptureResult<()> {
    let partial = path.with_extension("json.partial");
    let mut file = new_file(&partial)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    retry_archive_sharing_violation("publish JSON", &partial, || fs::rename(&partial, path))?;
    Ok(())
}

pub fn digest(path: &Path) -> CaptureResult<(u64, String)> {
    let file = retry_archive_sharing_violation("open for SHA-256", path, || File::open(path))?;
    let mut file = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut bytes = 0;
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok((bytes, format!("{:x}", hasher.finalize())));
        }
        bytes += count as u64;
        hasher.update(&buffer[..count]);
    }
}

// ---------------------------------------------------------------------------------------------
// Continuidad que el manifiesto de cada hora resume por simbolo (spec 2.5, schema 2).
// ---------------------------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub enum Cont {
    Depth { symbol: String, first: u64, last: u64, prev: u64 },
    Book { symbol: String, u: u64 },
    Agg { symbol: String, a: u64 },
    Stall { symbol: String },
}

#[derive(Default, Clone)]
struct SymbolCont {
    depth: Option<(u64, u64, u64)>,
    book_u: Option<u64>,
    agg_a: Option<u64>,
    stalls: u64,
}

#[derive(Default)]
struct Counts {
    rows: u64,
    kinds: BTreeMap<String, u64>,
    streams: BTreeMap<String, u64>,
    routes: BTreeMap<String, u64>,
    cont: BTreeMap<String, SymbolCont>,
}

impl Counts {
    fn record(&mut self, row: &Value, cont: Option<&Cont>) {
        self.rows += 1;
        if let Some(kind) = row["kind"].as_str() {
            increment(&mut self.kinds, kind);
        }
        if let Some(stream) = row["stream"].as_str() {
            increment(&mut self.streams, stream);
        }
        if row["kind"] == "frame" {
            if let Some(route) = row["route"].as_str() {
                increment(&mut self.routes, route);
            }
        }
        match cont {
            Some(Cont::Depth { symbol, first, last, prev }) => {
                self.cont.entry(symbol.clone()).or_default().depth = Some((*first, *last, *prev));
            }
            Some(Cont::Book { symbol, u }) => {
                let e = self.cont.entry(symbol.clone()).or_default();
                e.book_u = Some(e.book_u.map_or(*u, |x| x.max(*u)));
            }
            Some(Cont::Agg { symbol, a }) => {
                let e = self.cont.entry(symbol.clone()).or_default();
                e.agg_a = Some(e.agg_a.map_or(*a, |x| x.max(*a)));
            }
            Some(Cont::Stall { symbol }) => self.cont.entry(symbol.clone()).or_default().stalls += 1,
            None => {}
        }
    }

    fn merge(&mut self, other: Self) {
        self.rows += other.rows;
        for (total, segment) in [
            (&mut self.kinds, other.kinds),
            (&mut self.streams, other.streams),
            (&mut self.routes, other.routes),
        ] {
            for (key, count) in segment {
                *total.entry(key).or_default() += count;
            }
        }
    }

    fn cont_json(&self) -> Value {
        let m: BTreeMap<&String, Value> = self
            .cont
            .iter()
            .map(|(k, c)| {
                (
                    k,
                    // Nombres sin choque de mayusculas: ConvertFrom-Json de PowerShell no admite U y u.
                    json!({"last_depth_first_U":c.depth.map(|d| d.0),"last_depth_final_u":c.depth.map(|d| d.1),
                        "last_depth_pu":c.depth.map(|d| d.2),"last_book_u":c.book_u,
                        "last_agg_a":c.agg_a,"stalls":c.stalls}),
                )
            })
            .collect();
        json!(m)
    }
}

fn increment(counts: &mut BTreeMap<String, u64>, key: &str) {
    if let Some(value) = counts.get_mut(key) {
        *value += 1;
    } else {
        counts.insert(key.to_owned(), 1);
    }
}

// ---------------------------------------------------------------------------------------------
// Cola de archivo fail-closed (v1 main.rs:216-314, con la continuidad al lado de cada fila).
// ---------------------------------------------------------------------------------------------
struct QueuedRow {
    row: Value,
    cont: Option<Cont>,
    queued_at: Instant,
    _charge: OwnedSemaphorePermit,
}

#[derive(Default)]
struct QueueAudit {
    accepted: AtomicUsize,
    dequeued: AtomicUsize,
    peak_rows: AtomicUsize,
    peak_charged_bytes: AtomicUsize,
    max_dequeue_age_ns: AtomicU64,
    first_failure: Mutex<Option<&'static str>>,
    rejected: Mutex<Vec<Value>>,
}

impl QueueAudit {
    fn record_dequeue(&self, row: &QueuedRow) {
        self.max_dequeue_age_ns
            .fetch_max(elapsed_ns(row.queued_at), Ordering::Relaxed);
        self.dequeued.fetch_add(1, Ordering::Relaxed);
    }

    fn failure_reason(&self) -> Option<&'static str> {
        match self.first_failure.lock() {
            Ok(reason) => *reason,
            Err(poisoned) => *poisoned.into_inner(),
        }
    }

    fn stop_reason(&self, fallback: &'static str, exit: u8) -> &'static str {
        match self.failure_reason() {
            Some("row_capacity" | "byte_budget") if exit == 0 => "archive_queue_full",
            _ => fallback,
        }
    }
}

struct ArchiveReceiver {
    rx: mpsc::Receiver<QueuedRow>,
    audit: Arc<QueueAudit>,
}

impl ArchiveReceiver {
    fn try_recv(&mut self) -> Result<QueuedRow, mpsc::error::TryRecvError> {
        let row = self.rx.try_recv()?;
        self.audit.record_dequeue(&row);
        Ok(row)
    }

    /// SALIDAFALLO P4a: siguiente fila del escritor sin depender del EOF del canal. El escritor
    /// duerme con `park` y lo despiertan: cada oferta al salir (`Vuelo::drop`), el corte de
    /// admision y el vigilante. Con el corte resuelto (sin ofertas en vuelo) toda confirmada ya
    /// esta en el canal: se vacia y se termina aunque queden emisores clonados vivos.
    fn next_row(&mut self, admision: &Admision) -> Result<QueuedRow, QueueEnd> {
        loop {
            match self.try_recv() {
                Ok(row) => return Ok(row),
                Err(mpsc::error::TryRecvError::Disconnected) => return Err(QueueEnd::Eof),
                Err(mpsc::error::TryRecvError::Empty) => {}
            }
            if admision.resuelta() {
                return match self.try_recv() {
                    Ok(row) => Ok(row),
                    Err(_) => Err(QueueEnd::CutResolved),
                };
            }
            thread::park();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueueEnd {
    Eof,
    CutResolved,
}

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Puntos de pausa de prueba dentro de `offer` (T6): el codigo de produccion es el mismo.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OfferPoint {
    AfterReserve,
    AfterConfirm,
}

#[cfg(test)]
type OfferPause = Arc<dyn Fn(OfferPoint) + Send + Sync>;

#[derive(Clone)]
struct ArchiveQueue {
    tx: mpsc::Sender<QueuedRow>,
    budget: Arc<Semaphore>,
    byte_limit: usize,
    audit: Arc<QueueAudit>,
    admission: Arc<Admision>,
    #[cfg(test)]
    pause: Option<OfferPause>,
}

impl ArchiveQueue {
    #[cfg(test)]
    fn new(capacity: usize, byte_limit: usize) -> (Self, ArchiveReceiver) {
        Self::with_admission(capacity, byte_limit, Arc::new(Admision::nueva()))
    }

    fn with_admission(capacity: usize, byte_limit: usize, admission: Arc<Admision>) -> (Self, ArchiveReceiver) {
        let (tx, rx) = mpsc::channel(capacity);
        let audit = Arc::new(QueueAudit::default());
        let queue = Self {
            tx,
            budget: Arc::new(Semaphore::new(byte_limit)),
            byte_limit,
            audit: Arc::clone(&audit),
            admission,
            #[cfg(test)]
            pause: None,
        };
        (queue, ArchiveReceiver { rx, audit })
    }

    #[cfg(test)]
    fn pause_at(&self, point: OfferPoint) {
        if let Some(p) = &self.pause {
            p(point);
        }
    }

    fn offer(&self, row: Value, cont: Option<Cont>) -> CaptureResult<()> {
        // SALIDAFALLO P4a: entrar en vuelo ANTES de reservar; tras el corte no se reserva nada.
        let Some(mut flight) = self.admission.entrar() else {
            return Err(io::Error::other(format!(
                "archive admission cut; {} connection {} seq {} not admitted",
                row["route"], row["connection"], row["seq"]
            ))
            .into());
        };
        let text_bytes = row
            .as_object()
            .into_iter()
            .flat_map(|fields| fields.values())
            .filter_map(Value::as_str)
            .fold(0usize, |sum, value| sum.saturating_add(value.len()));
        let charged = ROW_CHARGE_OVERHEAD.saturating_add(text_bytes);
        // Reserve and send under the same admission guard as the first rejection.
        let mut failure = self.audit.first_failure.lock()
            .map_err(|_| io::Error::other("archive queue admission lock poisoned"))?;
        if let Some(reason) = *failure {
            return Err(io::Error::other(format!(
                "archive admission already closed ({reason}); capture stopped"
            )).into());
        }
        if self.tx.is_closed() {
            return self.reject(&mut failure, &row, "writer_unavailable", charged);
        }
        let charge = match u32::try_from(charged)
            .ok()
            .and_then(|size| self.budget.clone().try_acquire_many_owned(size).ok())
        {
            Some(permit) => permit,
            None => return self.reject(&mut failure, &row, "byte_budget", charged),
        };
        let slot = match self.tx.try_reserve() {
            Ok(slot) => slot,
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.audit.peak_rows.fetch_max(self.tx.max_capacity(), Ordering::Relaxed);
                return self.reject(&mut failure, &row, "row_capacity", charged);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return self.reject(&mut failure, &row, "writer_unavailable", charged);
            }
        };
        self.audit.peak_rows.fetch_max(
            self.tx.max_capacity() - self.tx.capacity(),
            Ordering::Relaxed,
        );
        self.audit.peak_charged_bytes.fetch_max(
            self.byte_limit - self.budget.available_permits(),
            Ordering::Relaxed,
        );
        #[cfg(test)]
        self.pause_at(OfferPoint::AfterReserve);
        // Confirmacion ordenada con el corte (CAS sobre la palabra de admision). Si el corte llego
        // durante la reserva, se sueltan plaza y presupuesto y la fila no cuenta como aceptada.
        if !flight.confirmar() {
            drop(slot);
            drop(charge);
            return Err(io::Error::other(format!(
                "archive admission cut during reservation; {} connection {} seq {} released",
                row["route"], row["connection"], row["seq"]
            ))
            .into());
        }
        self.audit.accepted.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        self.pause_at(OfferPoint::AfterConfirm);
        slot.send(QueuedRow {
            row,
            cont,
            queued_at: Instant::now(),
            _charge: charge,
        });
        drop(flight);
        Ok(())
    }

    fn reject(
        &self,
        first_failure: &mut Option<&'static str>,
        row: &Value,
        reason: &'static str,
        charged: usize,
    ) -> CaptureResult<()> {
        *first_failure = Some(reason);
        let failure = json!({
            "reason":reason, "kind":row["kind"], "route":row["route"], "shard":row["shard"],
            "stream":row["stream"], "symbol":row["symbol"],
            "connection":row["connection"], "seq":row["seq"],
            "recv_utc_ms":row["recv_utc_ms"], "recv_mono_ns":row["recv_mono_ns"],
            "row_charge_bytes":charged,
            "queue_occupied":self.tx.max_capacity() - self.tx.capacity(),
            "charged_bytes":self.byte_limit - self.budget.available_permits(),
        });
        match self.audit.rejected.lock() {
            Ok(mut rows) => rows.push(failure),
            Err(poisoned) => poisoned.into_inner().push(failure),
        }
        Err(io::Error::other(format!(
            "archive offer failed ({reason}) for {} connection {} seq {}; capture stopped",
            row["route"], row["connection"], row["seq"]
        ))
        .into())
    }
}

/// Emisor de filas con la envolvente v1 (kind, route, connection, seq, recv_utc_ms,
/// recv_mono_ns) mas `shard`. `seq` es por shard; `ingest_seq` lo pone el escritor (global).
#[derive(Clone)]
pub struct Emitter {
    queue: ArchiveQueue,
    clock: Clock,
    route: &'static str,
    shard: Option<usize>,
    seq: Arc<AtomicU64>,
}

impl Emitter {
    fn emit(&self, kind: &str, connection: u64, at: Stamp, mut row: Value, cont: Option<Cont>) -> CaptureResult<()> {
        row["kind"] = json!(kind);
        row["route"] = json!(self.route);
        row["shard"] = json!(self.shard);
        row["connection"] = json!(connection);
        row["seq"] = json!(self.seq.fetch_add(1, Ordering::Relaxed) + 1);
        row["recv_utc_ms"] = json!(at.utc_ms);
        row["recv_mono_ns"] = json!(at.mono_ns);
        self.queue.offer(row, cont)
    }

    pub fn emit_now(&self, kind: &str, connection: u64, row: Value, cont: Option<Cont>) -> CaptureResult<()> {
        self.emit(kind, connection, self.clock.stamp()?, row, cont)
    }
}

// ---------------------------------------------------------------------------------------------
// Segmentos por hora UTC (v1 main.rs:316-452), manifiesto schema 2.
// SALIDAFALLO P4b: el disco pasa por un adaptador (`Disk`) para probar en memoria cada fase del
// sellado; en produccion (`RealDisk`) son las mismas llamadas de antes.
// ---------------------------------------------------------------------------------------------
pub(crate) trait Disk: Send + Sync + 'static {
    type File: Write + Send;
    fn create(&self, path: &Path) -> io::Result<Self::File>;
    fn sync(&self, file: &Self::File) -> io::Result<()>;
    fn digest(&self, path: &Path) -> CaptureResult<(u64, String)>;
    fn publish(&self, from: &Path, to: &Path) -> io::Result<()>;
    fn write_json(&self, path: &Path, value: &Value) -> CaptureResult<()>;
}

pub(crate) struct RealDisk;

impl Disk for RealDisk {
    type File = File;
    fn create(&self, path: &Path) -> io::Result<File> {
        new_file(path)
    }
    fn sync(&self, file: &File) -> io::Result<()> {
        file.sync_all()
    }
    fn digest(&self, path: &Path) -> CaptureResult<(u64, String)> {
        digest(path)
    }
    fn publish(&self, from: &Path, to: &Path) -> io::Result<()> {
        retry_archive_sharing_violation("publish gzip", from, || fs::rename(from, to))
    }
    fn write_json(&self, path: &Path, value: &Value) -> CaptureResult<()> {
        write_json(path, value)
    }
}

struct Segment<F: Write> {
    name: String,
    hour: u64,
    first: u64,
    last: u64,
    counts: Counts,
    gzip: GzEncoder<BufWriter<F>>,
    line: Vec<u8>,
    handoff_at: Option<Instant>,
}

/// Fallo de sellado de un segmento: fase, error y estado alcanzado en disco (no se fuerza
/// rollback, renombrado compensatorio ni borrado).
#[derive(Debug)]
struct SealFailure {
    phase: &'static str,
    error: String,
    state: &'static str,
}

impl<F: Write + Send> Segment<F> {
    fn open<D: Disk<File = F>>(disk: &D, dir: &Path, hour: u64, timestamp: u64, number: u64) -> CaptureResult<Self> {
        let name = format!("hour-{hour}-{number:04}");
        let path = dir.join(format!("{name}.jsonl.gz.partial"));
        let gzip = GzEncoder::new(
            BufWriter::with_capacity(65536, disk.create(&path)?),
            Compression::fast(),
        );
        Ok(Self {
            name,
            hour,
            first: timestamp,
            last: timestamp,
            counts: Counts::default(),
            gzip,
            line: Vec::new(),
            handoff_at: None,
        })
    }

    fn append(&mut self, row: &Value, cont: Option<&Cont>, timestamp: u64) -> CaptureResult<()> {
        self.line.clear();
        serde_json::to_writer(&mut self.line, row)?;
        self.line.push(b'\n');
        self.gzip.write_all(&self.line)?;
        self.counts.record(row, cont);
        self.first = self.first.min(timestamp);
        self.last = self.last.max(timestamp);
        Ok(())
    }

    fn seal<D: Disk<File = F>>(self, disk: &D, dir: &Path, times: &mut PhaseTimes) -> Result<(String, Counts), SealFailure> {
        const PARTIAL: &str = "jsonl.gz.partial";
        let fail = |phase: &'static str, state: &'static str| move |e: &dyn std::fmt::Display| SealFailure { phase, error: e.to_string(), state };
        if let Some(handoff) = self.handoff_at {
            times.insert("handoff_to_seal_ns", elapsed_ns(handoff));
        }
        let file = format!("{}.jsonl.gz", self.name);
        let partial = dir.join(format!("{file}.partial"));
        let mut writer = time_phase(times, "gzip_finish_ns", || self.gzip.finish())
            .map_err(|e| fail("gzip_finish", PARTIAL)(&e))?;
        time_phase(times, "flush_ns", || writer.flush()).map_err(|e| fail("flush", PARTIAL)(&e))?;
        time_phase(times, "sync_all_ns", || disk.sync(writer.get_ref())).map_err(|e| fail("sync_all", PARTIAL)(&e))?;
        time_phase(times, "file_close_ns", || drop(writer));
        let (bytes, sha256) = time_phase(times, "sha256_ns", || disk.digest(&partial))
            .map_err(|e| fail("sha256", PARTIAL)(&e))?;
        let final_path = dir.join(&file);
        time_phase(times, "publish_gzip_ns", || disk.publish(&partial, &final_path))
            .map_err(|e| fail("publish_gzip", PARTIAL)(&e))?;
        // Desde aqui el gzip ya es final: un fallo del manifiesto lo deja sin manifiesto valido.
        time_phase(times, "publish_manifest_ns", || {
            disk.write_json(
                &dir.join(format!("{}.manifest.json", self.name)),
                &json!({
                    "schema": 2, "file": file, "bytes": bytes, "sha256": sha256, "rows": self.counts.rows,
                    "utc_hour_ms": self.hour, "first_recv_utc_ms": self.first, "last_recv_utc_ms": self.last,
                    "kinds": &self.counts.kinds, "streams": &self.counts.streams,
                    "frames_by_route": &self.counts.routes, "continuity": self.counts.cont_json(),
                }),
            )
        })
        .map_err(|e| fail("publish_manifest", "jsonl.gz_final_sin_manifiesto_valido")(&e))?;
        println!("SEALED {file} {} rows {bytes} bytes", self.counts.rows);
        Ok((self.name, self.counts))
    }
}

type PhaseTimes = BTreeMap<&'static str, u64>;

fn time_phase<T>(times: &mut PhaseTimes, phase: &'static str, job: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = job();
    times.insert(phase, elapsed_ns(start));
    result
}

/// Resultado del escritor (P4b): filas escritas, no escritas, fallos por segmento y como acabo
/// la cola. Ningun fallo de un segmento abandona a los demas.
#[derive(Default)]
struct ArchiveOutcome {
    rows_written: u64,
    rows_unwritten: u64,
    failures: Vec<Value>,
    end: Option<QueueEnd>,
}

impl ArchiveOutcome {
    fn json(&self) -> Value {
        json!({
            "rows_written": self.rows_written, "rows_unwritten": self.rows_unwritten,
            "failures": self.failures, "queue_end": self.end.map(|e| format!("{e:?}")),
        })
    }
}

fn archive<D: Disk>(
    disk: &D,
    dir: &Path,
    mut rx: ArchiveReceiver,
    seal: blocking_channel::SyncSender<Segment<D::File>>,
    ctl: &Control,
) -> ArchiveOutcome {
    ctl.admision.registrar_escritor(thread::current());
    let mut out = ArchiveOutcome::default();
    let mut open = BTreeMap::<u64, Segment<D::File>>::new();
    let mut failed_hours = std::collections::BTreeSet::<u64>::new();
    let mut segment_number = 0;
    let mut sequence = 0_u64;
    let mut newest = 0;
    let fail = |out: &mut ArchiveOutcome, what: &str, segment: Option<&str>, error: String| {
        let message = format!("escritor {what}: {error}");
        ctl.fatal(Clase::Escritor, &message);
        out.failures.push(json!({"phase": what, "segment": segment, "error": error}));
    };
    loop {
        let QueuedRow { mut row, cont, _charge, .. } = match rx.next_row(&ctl.admision) {
            Ok(row) => row,
            Err(end) => {
                out.end = Some(end);
                break;
            }
        };
        let Some(timestamp) = row["recv_utc_ms"].as_u64() else {
            out.rows_unwritten += 1;
            fail(&mut out, "timestamp", None, "archive record lacks timestamp".into());
            continue;
        };
        let hour = timestamp / HOUR_MS * HOUR_MS;
        newest = newest.max(timestamp);
        sequence += 1;
        row["ingest_seq"] = json!(sequence);
        if failed_hours.contains(&hour) {
            out.rows_unwritten += 1;
            continue;
        }
        let segment = match open.entry(hour) {
            std::collections::btree_map::Entry::Occupied(slot) => slot.into_mut(),
            std::collections::btree_map::Entry::Vacant(slot) => {
                segment_number += 1;
                match Segment::open(disk, dir, hour, timestamp, segment_number) {
                    Ok(segment) => slot.insert(segment),
                    Err(e) => {
                        out.rows_unwritten += 1;
                        failed_hours.insert(hour);
                        fail(&mut out, "open", Some(&format!("hour-{hour}-{segment_number:04}")), e.to_string());
                        continue;
                    }
                }
            }
        };
        if let Err(e) = segment.append(&row, cont.as_ref(), timestamp) {
            // El segmento queda como .jsonl.gz.partial en el estado alcanzado; no se sella.
            out.rows_written -= segment.counts.rows;
            out.rows_unwritten += 1 + segment.counts.rows;
            let name = segment.name.clone();
            open.remove(&hour);
            failed_hours.insert(hour);
            fail(&mut out, "append", Some(&name), e.to_string());
            continue;
        }
        out.rows_written += 1;
        while open.first_key_value().is_some_and(|(&oldest, _)| {
            open.len() > 1
                && newest.saturating_sub(oldest.saturating_add(HOUR_MS)) >= ROTATION_GRACE_MS
        }) {
            if let Some((_, completed)) = open.pop_first() {
                let name = completed.name.clone();
                if let Err(rows) = handoff_segment(&seal, completed) {
                    out.rows_written -= rows;
                    out.rows_unwritten += rows;
                    fail(&mut out, "handoff", Some(&name), "archive sealer unavailable".into());
                }
            }
        }
    }
    for (_, segment) in open {
        let name = segment.name.clone();
        if let Err(rows) = handoff_segment(&seal, segment) {
            out.rows_written -= rows;
            out.rows_unwritten += rows;
            fail(&mut out, "handoff", Some(&name), "archive sealer unavailable".into());
        }
    }
    out
}

/// Entrega al sellador. Si el sellador no esta, devuelve las filas del segmento perdido.
fn handoff_segment<F: Write>(
    seal: &blocking_channel::SyncSender<Segment<F>>,
    mut segment: Segment<F>,
) -> Result<(), u64> {
    segment.handoff_at = Some(Instant::now());
    seal.send(segment).map_err(|e| e.0.counts.rows)
}

#[derive(Default)]
struct SealReport {
    segments: u64,
    counts: Counts,
    error: Option<String>,
    attempted_segments: u64,
    phase_max_ns: PhaseTimes,
    failures: Vec<Value>,
}

impl SealReport {
    fn record_timing(&mut self, name: &str, times: &PhaseTimes, success: bool) {
        self.attempted_segments += 1;
        for (&phase, &ns) in times {
            let peak = self.phase_max_ns.entry(phase).or_default();
            *peak = (*peak).max(ns);
        }
        println!("SEAL_TIMING {}", json!({
            "segment": name,
            "ok": success,
            "phase_ns": times,
        }));
    }

    fn telemetry_json(&self) -> Value {
        json!({
            "attempted_segments": self.attempted_segments,
            "phase_max_ns": self.phase_max_ns,
            "failures": self.failures,
        })
    }
}

fn seal_segments<D: Disk>(
    disk: &D,
    dir: &Path,
    segments: blocking_channel::Receiver<Segment<D::File>>,
    sealed: Option<blocking_channel::Sender<(PathBuf, String)>>,
    ctl: Option<&Control>,
) -> SealReport {
    let mut report = SealReport::default();
    for segment in segments {
        let name = segment.name.clone();
        let rows = segment.counts.rows;
        let mut times = PhaseTimes::new();
        let sealed_segment = segment.seal(disk, dir, &mut times);
        report.record_timing(&name, &times, sealed_segment.is_ok());
        match sealed_segment {
            Ok((name, counts)) => {
                report.segments += 1;
                report.counts.merge(counts);
                if let Some(tx) = &sealed {
                    let _ = tx.send((dir.to_path_buf(), name));
                }
            }
            Err(failure) => {
                // P4b: se registra y se sigue con los demas segmentos.
                let message = format!("could not seal {name} ({}): {}", failure.phase, failure.error);
                if let Some(ctl) = ctl {
                    ctl.fatal(Clase::Sellador, &message);
                }
                report.error.get_or_insert(message);
                report.failures.push(json!({
                    "segment": name, "phase": failure.phase, "error": failure.error,
                    "state": failure.state, "rows": rows, "admitted": false,
                    "path": dir.join(format!("{name}.{}", if failure.state == "jsonl.gz.partial" { "jsonl.gz.partial" } else { "jsonl.gz" })),
                }));
            }
        }
    }
    report
}

/// Hilo nativo de archivo con aviso de terminacion independiente del runtime: el oneshot para
/// el `select!` y una guarda que, tambien durante un panic, marca el fin y despierta al
/// finalizador y al vigilante.
struct ArchiveWorker<T> {
    done: oneshot::Receiver<()>,
    thread: thread::JoinHandle<T>,
}

struct WorkerEnd {
    finished: Arc<std::sync::atomic::AtomicBool>,
    ctl: Arc<Control>,
}

impl Drop for WorkerEnd {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::SeqCst);
        self.ctl.despertar_finalizador();
        self.ctl.despertar_vigilante();
    }
}

impl<T: Send + 'static> ArchiveWorker<T> {
    fn spawn(
        name: &str,
        ctl: Arc<Control>,
        finished: Arc<std::sync::atomic::AtomicBool>,
        job: impl FnOnce() -> T + Send + 'static,
    ) -> CaptureResult<Self> {
        let (done_tx, done) = oneshot::channel();
        let thread = thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let _end = WorkerEnd { finished, ctl };
                let result = job();
                let _ = done_tx.send(());
                result
            })?;
        Ok(Self { done, thread })
    }
}

/// Banco local (solo pruebas): las firmas de antes sobre las de SALIDAFALLO.
#[cfg(test)]
impl<X: Send + 'static> ArchiveWorker<CaptureResult<X>> {
    fn spawn_local(name: &str, job: impl FnOnce() -> CaptureResult<X> + Send + 'static) -> CaptureResult<Self> {
        let (ctl, _, _) = salidafallo::tests::control_de_prueba(salidafallo::PLAZO_FATAL);
        Self::spawn(name, ctl, Arc::new(std::sync::atomic::AtomicBool::new(false)), job)
    }

    async fn finish(self, completion_observed: bool) -> CaptureResult<X> {
        if !completion_observed {
            let _ = self.done.await;
        }
        self.thread
            .join()
            .map_err(|_| io::Error::other("archive worker panicked"))?
    }
}

#[cfg(test)]
fn archive_local(dir: &Path, rx: ArchiveReceiver, seal: blocking_channel::SyncSender<Segment<File>>) -> CaptureResult<()> {
    let (ctl, _, _) = salidafallo::tests::control_de_prueba(salidafallo::PLAZO_FATAL);
    let outcome = archive(&RealDisk, dir, rx, seal, &ctl);
    match outcome.failures.first() {
        None => Ok(()),
        Some(f) => Err(io::Error::other(f.to_string()).into()),
    }
}

// ---------------------------------------------------------------------------------------------
// Estado de continuidad por shard (compartido por la conexion activa y la de relevo).
// ---------------------------------------------------------------------------------------------
#[derive(Default)]
struct ShardBook {
    depth: HashMap<String, public_l2::DepthState>,
    tickers: HashMap<String, l2::Ticker<public_l2::PublicFrame>>,
    public_connections: HashSet<u64>,
    agg: HashMap<String, u64>,
    mark: HashMap<String, Instant>,
    stalled: HashSet<String>,
    seen: HashMap<u64, HashMap<String, Instant>>,
}

#[derive(Debug)]
enum Life {
    Open(u64),
    Stream(u64),
    Ended(u64, String, bool),
}

struct ShardCtx {
    venue: Venue,
    route: &'static str,
    index: usize,
    total: usize,
    symbols: Vec<String>,
    url: String,
    emitter: Emitter,
    clock: Clock,
    book: Arc<Mutex<ShardBook>>,
    gate: mpsc::UnboundedSender<gate::Request>,
    stop: watch::Receiver<bool>,
    fatal: mpsc::UnboundedSender<String>,
    rotate_after: Duration,
    connections: Arc<AtomicU64>,
    // R2 (solo ruta public): plan de snapshots periodicos por simbolo.
    gate_info: watch::Receiver<Value>,
    r2_max_gap: Duration,
    r2_symbols_total: usize,
    r2_rank: HashMap<String, usize>,
}

fn lock(book: &Mutex<ShardBook>) -> std::sync::MutexGuard<'_, ShardBook> {
    match book.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

struct Conn {
    id: u64,
    emitter: Emitter,
    route: &'static str,
    book: Arc<Mutex<ShardBook>>,
    gate: mpsc::UnboundedSender<gate::Request>,
    shard: usize,
    life: mpsc::UnboundedSender<Life>,
    l2_streams: HashMap<String, String>,
}

impl Conn {
    /// Una trama del venue: continuidad + fila. Devuelve false si la trama no es un mensaje
    /// combinado valido (v1 main.rs:654).
    fn frame(&mut self, text: String, at: Stamp) -> CaptureResult<bool> {
        let parsed: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                self.emitter.emit("frame", self.id, at, json!({"payload":text,"parse_error":true}), None)?;
                return Ok(false);
            }
        };
        let stream = parsed["stream"].as_str().map(str::to_owned);
        let data = &parsed["data"];
        let symbol = data["s"].as_str().or_else(|| data["o"]["s"].as_str()).map(str::to_owned);
        if self.route == "public" {
            return self.public_frame(text, at, stream, symbol, data);
        }
        let mut cont = None;
        let now = Instant::now();
        let mut first_of_stream = false;
        if let Some(stream) = &stream {
            let mut b = lock(&self.book);
            let seen = b.seen.entry(self.id).or_default();
            first_of_stream = seen.insert(stream.clone(), now).is_none();
            if let Some(symbol) = &symbol {
                if stream.ends_with("@aggTrade") {
                    if let Some(a) = data["a"].as_u64() {
                        match b.agg.get(symbol).copied() {
                            Some(last) if a > last + 1 => {
                                b.agg.insert(symbol.clone(), a);
                                self.emitter.emit("agg_gap", self.id, at, json!({"symbol":symbol,"stream":stream,"previous_a":last,"a":a,"missing":a - last - 1}), None)?;
                                // Relleno por fromId a traves de la puerta (spec 2.6, encargo punto 3).
                                let mut from = last + 1;
                                while from < a {
                                    let n = (a - from).min(AGG_LIMIT_MAX);
                                    let _ = self.gate.send(gate::Request { kind: gate::Kind::AggFill, priority: 2, shard: self.shard,
                                        symbol: symbol.clone(), generation: 0, trigger_u: a, from_id: from, limit: n, periodic: false, reply: None });
                                    from += n;
                                }
                            }
                            Some(last) if a <= last => {}
                            _ => {
                                b.agg.insert(symbol.clone(), a);
                            }
                        }
                        cont = Some(Cont::Agg { symbol: symbol.clone(), a });
                    }
                } else if stream.ends_with("@markPrice@1s") {
                    b.mark.insert(symbol.clone(), now);
                    if b.stalled.remove(symbol) {
                        self.emitter.emit("stall_end", self.id, at, json!({"symbol":symbol,"stream":stream}), None)?;
                    }
                }
            }
        }
        if first_of_stream {
            let _ = self.life.send(Life::Stream(self.id));
        }
        self.emitter.emit(
            "frame",
            self.id,
            at,
            json!({"stream":stream,"symbol":symbol,"payload":text,"depth_generation":None::<u64>}),
            cont,
        )?;
        Ok(stream.is_some() && data.is_object())
    }

    async fn connected(&mut self, mut ws: Socket, mut close: oneshot::Receiver<String>, mut stop: watch::Receiver<bool>) -> CaptureResult<String> {
        let mut tick = time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut last_frame = Instant::now();
        loop {
            if *stop.borrow() {
                let _ = time::timeout(Duration::from_secs(2), ws.close(None)).await;
                return Ok("stop".into());
            }
            tokio::select! {
                _ = stop.changed() => {}
                why = &mut close => {
                    let why = why.unwrap_or_else(|_| "supervisor gone".into());
                    let _ = time::timeout(Duration::from_secs(2), ws.close(None)).await;
                    return Ok(why);
                }
                _ = tick.tick() => {
                    if last_frame.elapsed() >= SILENCE_LIMIT {
                        return Ok(format!("{}s without market frames", SILENCE_LIMIT.as_secs()));
                    }
                }
                message = ws.next() => {
                    let at = self.emitter.clock.stamp()?;
                    match message {
                        Some(Ok(Message::Text(text))) => {
                            last_frame = Instant::now();
                            if !self.frame(text, at)? {
                                return Ok("invalid combined market frame".into());
                            }
                        }
                        Some(Ok(Message::Ping(payload))) => {
                            match time::timeout(PONG_TIMEOUT, ws.send(Message::Pong(payload))).await {
                                Ok(Ok(())) => {}
                                Ok(Err(e)) => return Ok(format!("WebSocket pong: {e}")),
                                Err(_) => return Ok("WebSocket pong deadline".into()),
                            }
                        }
                        Some(Ok(Message::Pong(_))) => {}
                        Some(Ok(Message::Close(frame))) => return Ok(format!("venue closed WebSocket: {frame:?}")),
                        Some(Ok(_)) => return Ok("unexpected binary WebSocket frame".into()),
                        Some(Err(e)) => return Ok(format!("WebSocket read: {e}")),
                        None => return Ok("WebSocket ended".into()),
                    }
                }
            }
        }
    }
}

/// Una conexion de un shard: abre, lee hasta cerrar o fallar, y avisa al supervisor del shard.
async fn connection(mut conn: Conn, url: String, mut close: oneshot::Receiver<String>, mut stop: watch::Receiver<bool>, fatal: mpsc::UnboundedSender<String>) {
    let id = conn.id;
    let life = conn.life.clone();
    let result: CaptureResult<String> = async {
        if *stop.borrow() {
            return Ok("stop".into());
        }
        let opened = tokio::select! {
            biased;
            _ = stop.changed() => return Ok("stop".into()),
            why = &mut close => return Ok(why.unwrap_or_else(|_| "supervisor gone".into())),
            opened = time::timeout(CONNECT_TIMEOUT, connect_async(url.as_str())) => opened,
        };
        match opened {
            Ok(Ok((ws, _))) => {
                conn.emitter.emit_now("socket_open", id, json!({}), None)?;
                if conn.route == "public" {
                    lock(&conn.book).public_connections.insert(id);
                }
                let _ = life.send(Life::Open(id));
                let why = conn.connected(ws, close, stop.clone()).await?;
                conn.emitter.emit_now("socket_close", id, json!({"reason":why}), None)?;
                Ok(why)
            }
            Ok(Err(e)) => Ok(format!("WebSocket connect: {e}")),
            Err(_) => Ok("WebSocket connect deadline".into()),
        }
    }
    .await;
    let (result, coverage_lost) = {
        let mut book = lock(&conn.book);
        book.seen.remove(&id);
        let coverage_lost = book.public_connections.remove(&id)
            && book.public_connections.is_empty()
            && !*stop.borrow()
            && result.is_ok();
        let result = result.and_then(|why| {
            if coverage_lost {
                book.l2_lost(&conn.emitter, Instant::now())?;
            }
            Ok(why)
        });
        (result, coverage_lost)
    };
    match result {
        Ok(why) => {
            let _ = life.send(Life::Ended(id, why, coverage_lost));
        }
        Err(e) => {
            let _ = fatal.send(format!("shard {} connection {id}: {e}", conn.shard));
            let _ = life.send(Life::Ended(id, format!("fatal: {e}"), false));
        }
    }
}

fn jitter() -> Duration {
    Duration::from_millis(fastrand::u64(0..JITTER_MAX_MS))
}

struct Live {
    id: u64,
    close: Option<oneshot::Sender<String>>,
    opened: Option<Instant>,
    started: Instant,
}

impl ShardCtx {
    fn spawn_conn(&self, life: &mpsc::UnboundedSender<Life>, tasks: &mut JoinSet<()>) -> Live {
        let id = self.connections.fetch_add(1, Ordering::Relaxed) + 1;
        let (tx, rx) = oneshot::channel();
        let l2_streams = if self.route == "public" {
            self.symbols.iter().flat_map(|symbol| {
                let lower = symbol.to_ascii_lowercase();
                let upper = symbol.to_ascii_uppercase();
                [(format!("{lower}@depth@100ms"), upper.clone()), (format!("{lower}@bookTicker"), upper)]
            }).collect()
        } else {
            HashMap::new()
        };
        let conn = Conn {
            id,
            emitter: self.emitter.clone(),
            route: self.route,
            book: Arc::clone(&self.book),
            gate: self.gate.clone(),
            shard: self.index,
            life: life.clone(),
            l2_streams,
        };
        tasks.spawn(connection(conn, self.url.clone(), rx, self.stop.clone(), self.fatal.clone()));
        Live { id, close: Some(tx), opened: None, started: Instant::now() }
    }

    fn row(&self, kind: &str, conn: u64, row: Value) -> CaptureResult<()> {
        self.emitter.emit_now(kind, conn, row, None)
    }

    /// Flujos que la conexion vieja entrego en el ultimo plazo de gracia y que la nueva aun no.
    fn missing_on(&self, old: u64, new: u64) -> Vec<String> {
        let b = lock(&self.book);
        let now = Instant::now();
        let empty = HashMap::new();
        let o = b.seen.get(&old).unwrap_or(&empty);
        let n = b.seen.get(&new).unwrap_or(&empty);
        let mut m: Vec<String> = o
            .iter()
            .filter(|(s, t)| now.duration_since(**t) <= OVERLAP_GRACE && !n.contains_key(*s))
            .map(|(s, _)| s.clone())
            .collect();
        m.sort();
        m
    }

    async fn run(self) -> CaptureResult<()> {
        if self.route == "public" {
            self.run_l2().await
        } else {
            self.run_market().await
        }
    }

    async fn run_market(self) -> CaptureResult<()> {
        let (life_tx, mut life) = mpsc::unbounded_channel();
        let mut tasks = JoinSet::new();
        let mut stop = self.stop.clone();
        let mut tick = time::interval(MARK_PERIOD);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        // Apertura inicial escalonada por shard (cuidado de IP, spec 2.4).
        let first_delay = SHARD_OFFSET * self.index as u32;
        tokio::select! {
            _ = stop.changed() => return Ok(()),
            _ = time::sleep(first_delay) => {}
        }
        let mut active: Option<Live> = Some(self.spawn_conn(&life_tx, &mut tasks));
        let mut candidate: Option<Live> = None;
        let mut holding_logged = false;
        let mut rotation_due: Option<Instant> = None;
        let mut first_rotation = true;
        let mut delay = BACKOFF_START;
        let mut reconnect_at: Option<Instant> = None;
        let mut rotation_retry_at: Option<Instant> = None;
        let shard_start = Instant::now();
        loop {
            if *stop.borrow() {
                break;
            }
            tokio::select! {
                _ = stop.changed() => {}
                // SALIDAFALLO P2: un join exitoso de conexion efimera es normal (Life::Ended
                // gobierna la reconexion, en cualquier orden); un JoinError es fatal aunque
                // Life::Ended nunca llegue.
                Some(joined) = tasks.join_next(), if !tasks.is_empty() => {
                    salidafallo::conexion_terminada(joined)?;
                }
                Some(ev) = life.recv() => match ev {
                    Life::Open(id) => {
                        if active.as_ref().is_some_and(|a| a.id == id) {
                            let a = active.as_mut().expect("active");
                            a.opened = Some(Instant::now());
                            // Primera rotacion escalonada: el shard i rota a (N - i)/N del intervalo.
                            let span = if first_rotation {
                                first_rotation = false;
                                self.rotate_after.mul_f64((self.total - self.index) as f64 / self.total as f64)
                            } else {
                                self.rotate_after
                            };
                            rotation_due = Some(Instant::now() + span);
                        } else if let Some(c) = candidate.as_mut().filter(|c| c.id == id) {
                            c.opened = Some(Instant::now());
                            self.row("rotation_open", id, json!({"old_connection":active.as_ref().map(|a| a.id),"new_connection":id}))?;
                        }
                    }
                    Life::Stream(_) => {}
                    Life::Ended(_, _, _) if *stop.borrow() => {}
                    Life::Ended(id, why, _) => {
                        if active.as_ref().is_some_and(|a| a.id == id) {
                            let a = active.take().expect("active");
                            let lived = a.opened.map(|o| o.elapsed()).unwrap_or_default();
                            if let Some(c) = candidate.take().filter(|c| c.opened.is_some()) {
                                // La de relevo ya esta viva: sin hueco, pasa a ser la activa.
                                self.row("rotation_promote", c.id, json!({"old_connection":id,"reason":why}))?;
                                active = Some(c);
                                rotation_due = Some(Instant::now() + self.rotate_after);
                                holding_logged = false;
                            } else {
                                if lived >= BACKOFF_RESET_LIFE {
                                    delay = BACKOFF_START;
                                }
                                let sleep = delay + jitter() + SHARD_OFFSET * self.index as u32;
                                self.row("socket_gap", id, json!({"reason":why,"retry_ms":sleep.as_millis() as u64,"backoff_ms":delay.as_millis() as u64}))?;
                                reconnect_at = Some(Instant::now() + sleep);
                                delay = (delay * 2).min(BACKOFF_MAX);
                                rotation_due = None;
                            }
                        } else if candidate.as_ref().is_some_and(|c| c.id == id) {
                            candidate = None;
                            holding_logged = false;
                            let sleep = delay + jitter() + SHARD_OFFSET * self.index as u32;
                            self.row("rotation_failed", id, json!({"reason":why,"retry_ms":sleep.as_millis() as u64}))?;
                            rotation_retry_at = Some(Instant::now() + sleep);
                            delay = (delay * 2).min(BACKOFF_MAX);
                        }
                    }
                },
                _ = tick.tick() => {
                    let now = Instant::now();
                    if active.is_none() && reconnect_at.is_some_and(|t| now >= t) {
                        reconnect_at = None;
                        active = Some(self.spawn_conn(&life_tx, &mut tasks));
                    }
                    // Rotacion programada con solapamiento (spec 2.3).
                    if candidate.is_none()
                        && active.as_ref().is_some_and(|a| a.opened.is_some())
                        && rotation_due.is_some_and(|t| now >= t)
                        && rotation_retry_at.is_none_or(|t| now >= t)
                    {
                        rotation_retry_at = None;
                        candidate = Some(self.spawn_conn(&life_tx, &mut tasks));
                    }
                    if let (Some(a), Some(c)) = (active.as_ref(), candidate.as_ref()) {
                        if let Some(opened) = c.opened {
                            let missing = self.missing_on(a.id, c.id);
                            let waited = opened.elapsed();
                            let ready = missing.is_empty() && lock(&self.book).seen.get(&c.id).is_some_and(|s| !s.is_empty());
                            if ready || waited >= HOLD_MAX {
                                let kind = if ready { "rotation_ready" } else { "rotation_forced" };
                                let c_id = c.id;
                                let streams: Vec<String> = lock(&self.book).seen.get(&c_id).map(|s| {
                                    let mut v: Vec<String> = s.keys().cloned().collect(); v.sort(); v }).unwrap_or_default();
                                self.row(kind, c_id, json!({"old_connection":a.id,"new_connection":c_id,
                                    "overlap_ms":waited.as_millis() as u64,"streams_on_new":streams,"missing":missing}))?;
                                let mut old = active.take().expect("active");
                                if let Some(tx) = old.close.take() {
                                    let _ = tx.send(format!("rotation to connection {c_id}"));
                                }
                                self.row("rotation_close", old.id, json!({"old_connection":old.id,"new_connection":c_id,
                                    "old_lived_ms":old.opened.map(|o| o.elapsed().as_millis() as u64)}))?;
                                active = candidate.take();
                                rotation_due = Some(now + self.rotate_after);
                                holding_logged = false;
                                delay = BACKOFF_START;
                            } else if waited >= OVERLAP_GRACE && !holding_logged {
                                holding_logged = true;
                                self.row("rotation_hold", c.id, json!({"old_connection":a.id,"new_connection":c.id,"missing":missing,
                                    "hold_max_ms":HOLD_MAX.as_millis() as u64}))?;
                            }
                        } else if now.duration_since(c.started) > CONNECT_TIMEOUT * 2 {
                            // El intento de relevo ni abrio: lo reporta su propia tarea (Ended).
                        }
                    }
                    // Latido markPrice: 3 periodos sin mensaje = stall (spec 2.5).
                    let limit = MARK_PERIOD * STALL_PERIODS;
                    let mut rows = Vec::new();
                    {
                        let mut b = lock(&self.book);
                        for symbol in &self.symbols {
                            let last = b.mark.get(symbol).copied().unwrap_or(shard_start);
                            if now.duration_since(last) >= limit && !b.stalled.contains(symbol) {
                                b.stalled.insert(symbol.clone());
                                rows.push((symbol.clone(), now.duration_since(last)));
                            }
                        }
                    }
                    let conn = active.as_ref().map(|a| a.id).unwrap_or(0);
                    for (symbol, quiet) in rows {
                        self.emitter.emit_now("stall", conn, json!({"symbol":symbol,"stream":format!("{}@markPrice@1s", symbol.to_ascii_lowercase()),
                            "quiet_ms":quiet.as_millis() as u64}), Some(Cont::Stall { symbol: symbol.clone() }))?;
                    }
                }
            }
        }
        for live in [active, candidate].into_iter().flatten() {
            if let Some(tx) = live.close {
                let _ = tx.send("stop".into());
            }
        }
        // Recoger los joins de cierre sin descartarlos: el primer JoinError se devuelve.
        let mut first = None;
        while let Some(joined) = tasks.join_next().await {
            if let Err(e) = salidafallo::conexion_terminada(joined) {
                first.get_or_insert(e);
            }
        }
        let _ = self.venue;
        first.map_or(Ok(()), Err)
    }
}

async fn deadline(clock: Clock, until: Option<u64>) -> CaptureResult<()> {
    let Some(until) = until else {
        return std::future::pending().await;
    };
    loop {
        let remaining = until.saturating_sub(clock.stamp()?.utc_ms);
        if remaining == 0 {
            return Ok(());
        }
        time::sleep(Duration::from_millis(remaining.min(1000))).await;
    }
}

async fn stop_file(path: PathBuf) {
    loop {
        if path.exists() {
            return;
        }
        time::sleep(Duration::from_secs(1)).await;
    }
}

/// Roster: simbolos unicos del CSV del manifiesto de Binance Vision cuyo tipo no es BVOLIndex
/// (indices de opciones sin libro ni trades de futuros). Spec 2.0.
fn roster(csv: &Path) -> CaptureResult<Vec<String>> {
    let text = fs::read_to_string(csv)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let col = |name: &str| header.iter().position(|h| h.trim_matches('"') == name);
    let (Some(sym), Some(kind)) = (col("simbolo"), col("tipo")) else {
        return Err(io::Error::other("roster CSV needs columns simbolo,tipo").into());
    };
    let mut all = std::collections::BTreeMap::<String, bool>::new();
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        if let (Some(s), Some(k)) = (f.get(sym), f.get(kind)) {
            let s = s.trim_matches('"').to_string();
            let tradable = k.trim_matches('"') != "BVOLIndex";
            let e = all.entry(s).or_insert(false);
            *e = *e || tradable;
        }
    }
    Ok(all.into_iter().filter(|(_, t)| *t).map(|(s, _)| s).collect())
}

fn opt<T: std::str::FromStr>(args: &[String], key: &str) -> CaptureResult<Option<T>> {
    for a in args {
        if let Some(v) = a.strip_prefix(&format!("{key}=")) {
            return Ok(Some(v.parse().map_err(|_| io::Error::other(format!("bad value for {key}")))?));
        }
    }
    Ok(None)
}

fn codec_params(args: &[String]) -> CaptureResult<codec::Params> {
    Ok(codec::Params {
        zstd_level: opt(args, "zstd_level")?.unwrap_or(CODEC_LEVEL_DEFAULT),
        block_raw_bytes: opt(args, "block_raw_bytes")?.unwrap_or(CODEC_BLOCK_RAW),
        window_log: opt(args, "window_log")?.unwrap_or(CODEC_WINDOW_LOG),
    })
}

async fn capture(args: &[String], ctl: Arc<Control>) -> CaptureResult<CaptureEnd> {
    let [venue, out, roster_csv, rest @ ..] = args else {
        return Err(io::Error::other(
            "usage: captura_v2 capture <mainnet|testnet> <out-root> <roster.csv> [until_ms=N] [rotate_after_s=N] [r2_max_gap_s=N] [min_free_bytes=N] [zstd_level=N]",
        )
        .into());
    };
    let venue = Venue::parse(venue)?;
    let until: Option<u64> = opt(rest, "until_ms")?;
    let rotate_after = opt::<u64>(rest, "rotate_after_s")?.map(Duration::from_secs).unwrap_or(ROTATE_AFTER);
    let r2_max_gap = opt::<u64>(rest, "r2_max_gap_s")?.map(Duration::from_secs).unwrap_or(public_l2::R2_MAX_GAP_DEFAULT);
    if rest.iter().any(|a| a.starts_with("alloc_bytes=")) {
        return Err(io::Error::other("R3 no admite alloc_bytes; retencion fija de 365 dias").into());
    }
    let min_free_bytes = opt(rest, "min_free_bytes")?.unwrap_or(MIN_FREE_BYTES);
    let codec_p = codec_params(rest)?;
    let clock = Clock(Instant::now());
    let start = clock.stamp()?;
    if until.is_some_and(|u| u <= start.utc_ms) {
        return Err(io::Error::other("until_ms must be later than current UTC time").into());
    }
    let root = PathBuf::from(out);
    fs::create_dir_all(&root)?;
    let free = store::free_bytes(&root)?;
    if free < min_free_bytes {
        eprintln!("DISCO-BAJO: {free} bytes libres < {min_free_bytes}; no arranca");
        return Ok(CaptureEnd::Early(3));
    }
    let symbols = roster(Path::new(roster_csv))?;
    let (_, roster_sha) = digest(Path::new(roster_csv))?;
    let r2_symbols_total = symbols.len();
    let per = symbols.len().div_ceil(SHARDS_PER_ROUTE).max(1);
    let dir = root.join(format!("run-{}-{}-{}", venue.name, start.utc_ms, std::process::id()));
    fs::create_dir(&dir)?;
    let mut shard_defs = Vec::new();
    for (route_i, route) in ["public", "market"].into_iter().enumerate() {
        for (k, chunk) in symbols.chunks(per).enumerate() {
            let index = route_i * SHARDS_PER_ROUTE + k;
            let with_force = route == "market" && k == 0;
            // Posicion del primer simbolo del shard dentro del roster completo: base de la
            // fase R2 para escalonar los temporizadores entre shards (public_l2::r2_phase).
            let roster_base = if route == "public" { k * per } else { 0 };
            shard_defs.push((index, route, chunk.to_vec(), venue.url(route, chunk, with_force), roster_base));
        }
    }
    let total_shards = shard_defs.len();
    let r2_snapshot = json!({
        "goal":"ancla fresca por simbolo aunque el stream calle (objetivo de intervalo entre anclas)",
        "max_gap_ms":r2_max_gap.as_millis() as u64,
        "budget_factor":0.8,
        "formula":"T = max(max_gap, ceil(N*w*window / max(1, 0.8*L - N*w - baseline)))",
        "weight_depth_default":20,"limit_default":2400,
        "baseline_snapshots_per_symbol_h":1.2,
        "priority":3,"stagger":"fase = T * (posicion_en_roster + 1) / N",
        "measured":{"pilot_300s_weight_per_min":160,"anti_ban_budget_per_min":1920,
            "storm_reserve_per_min":800,"baseline_weight_per_min":16,
            "pilot_interval_floor_s":44,"pilot_interval_s":300},
    });
    write_json(
        &dir.join("RUN.json"),
        &json!({
            "schema":2, "tool":"captura_v2", "venue":venue.name, "symbols":symbols, "roster_csv":roster_csv,
            "roster_csv_sha256":roster_sha,
            "shards": shard_defs.iter().map(|(i,r,s,u,b)| json!({"shard":i,"route":r,"symbols":s,"url":u,"roster_base":b,"streams":u.split("streams=").nth(1).map(|x| x.split('/').count())})).collect::<Vec<_>>(),
            "snapshot_rest":venue.rest,"started_utc_ms":start.utc_ms,"until_utc_ms":until,
            "parameters":{
                "queue_capacity":QUEUE_CAPACITY, "queue_byte_budget":QUEUE_BYTE_BUDGET,
                "queue_row_overhead_charge":ROW_CHARGE_OVERHEAD, "seal_queue_capacity":SEAL_QUEUE_CAPACITY,
                "rotation_grace_ms":ROTATION_GRACE_MS, "rotate_after_ms":rotate_after.as_millis() as u64,
                "rotation_stagger_ms":(rotate_after / total_shards as u32).as_millis() as u64,
                "overlap_grace_ms":OVERLAP_GRACE.as_millis() as u64, "hold_max_ms":HOLD_MAX.as_millis() as u64,
                "backoff_ms":[BACKOFF_START.as_millis() as u64, BACKOFF_MAX.as_millis() as u64],
                "backoff_reset_life_ms":BACKOFF_RESET_LIFE.as_millis() as u64, "jitter_max_ms":JITTER_MAX_MS,
                "shard_offset_ms":SHARD_OFFSET.as_millis() as u64, "connect_timeout_ms":CONNECT_TIMEOUT.as_millis() as u64,
                "silence_limit_ms":SILENCE_LIMIT.as_millis() as u64, "pong_timeout_ms":PONG_TIMEOUT.as_millis() as u64,
                "stall_ms":(MARK_PERIOD * STALL_PERIODS).as_millis() as u64,
                "rest":"una puerta; limite de exchangeInfo; peso por X-MBX-USED-WEIGHT; balde limite/ventana; 429 pausa y mitad; 418 cierre",
                "r2_snapshot":r2_snapshot,
                "retention_days":RETENTION_DAYS,"retention_ms":RETENTION_MS,"retention_policy":"time_only", "min_free_bytes":min_free_bytes,
                "codec":{"format":"TGLL0001","zstd_level":codec_p.zstd_level,"window_log":codec_p.window_log,"block_raw_bytes":codec_p.block_raw_bytes},
                "derivations":"ver PARAMETROS.md y la cabecera de src/main.rs",
            },
            "archive_order":"ingest_seq is global across UTC-hour segments; seq is per shard",
            "payload_encoding":"original UTF-8 text as a JSON string",
            "overlap_note":"public: arbitrado por flujo y simbolo; market: deduplicar por (stream, a/E) al leer",
            "public_redundancy":{
                "sockets_per_shard":2,"arbiter_tick_ms":public_l2::ARBITER_TICK.as_millis() as u64,
                "depth_wait_ms":l2::DEPTH_WAIT.as_millis() as u64,"depth_rows":l2::DEPTH_ROWS,"depth_bytes":l2::DEPTH_BYTES,
                "book_wait_ms":l2::TICKER_WAIT.as_millis() as u64,"book_rows":l2::TICKER_ROWS,
                "book_bytes":l2::TICKER_BYTES,"book_recent_ids":l2::TICKER_HISTORY
            },
        }),
    )?;
    println!("CAPTURE {}", dir.display());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(12))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let (queue, rx) = ArchiveQueue::with_admission(QUEUE_CAPACITY, QUEUE_BYTE_BUDGET, Arc::clone(&ctl.admision));
    let audit = Arc::clone(&queue.audit);
    let (sealed_tx, sealed_rx) = blocking_channel::channel();
    let (low_tx, low_rx) = oneshot::channel::<store::StoreStop>();
    // SALIDAFALLO P4c: la compuerta del control despierta a store por el mismo canal de horas
    // selladas (envio no bloqueante) y comparte la bandera de cancelacion ordinaria.
    {
        let wake = sealed_tx.clone();
        ctl.store.fijar_despertador(Box::new(move || {
            let _ = wake.send((PathBuf::new(), String::new()));
        }));
    }
    let store_cfg = store::StoreConfig { root: root.clone(), retention_ms: RETENTION_MS,
        cancel: Arc::clone(&ctl.store.cancel), min_free_bytes, codec: codec_p, scan_every: STORE_SCAN_EVERY,
        gate: Arc::clone(&ctl.store) };
    let store_exit = Arc::new(salidafallo::SalidaStore::nueva());
    let store_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let store_thread = {
        let store_exit = Arc::clone(&store_exit);
        let end = WorkerEnd { finished: Arc::clone(&store_finished), ctl: Arc::clone(&ctl) };
        thread::Builder::new().name("capture-store".into()).spawn(move || {
            let _end = end;
            // Re-codificar con la prioridad de hilo mas baja sobre la clase del proceso (CPU): la captura va
            // primero y, con el proceso en AboveNormal, el hilo queda al nivel de un hilo normal ajeno (C3).
            // No se usa THREAD_MODE_BACKGROUND_BEGIN: recorta E/S y memoria y en la prueba de 2 h dejo la hora
            // 0002 sin terminar en 28 min.
            #[cfg(windows)]
            unsafe {
                use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
            }
            // SALIDAFALLO P2: el estado de salida se fija ANTES de soltar el aviso oneshot, para que
            // el select distinga fin, panic y cancelacion del canal.
            let mut low_tx = Some(low_tx);
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| store::run(store_cfg, sealed_rx, &mut low_tx)));
            store_exit.fijar(if ran.is_ok() { salidafallo::EstadoStore::Volvio } else { salidafallo::EstadoStore::Panic });
            drop(low_tx);
        })?
    };
    let (seal_tx, seal_rx) = blocking_channel::sync_channel(SEAL_QUEUE_CAPACITY);
    let sealer_dir = dir.clone();
    let sealer_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sealer_ctl = Arc::clone(&ctl);
    let sealer = ArchiveWorker::spawn("capture-seal", Arc::clone(&ctl), Arc::clone(&sealer_finished), move || {
        seal_segments(&RealDisk, &sealer_dir, seal_rx, Some(sealed_tx), Some(&sealer_ctl))
    })?;
    let writer_dir = dir.clone();
    let writer_finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writer_ctl = Arc::clone(&ctl);
    let writer = ArchiveWorker::spawn("capture-gzip", Arc::clone(&ctl), Arc::clone(&writer_finished), move || {
        archive(&RealDisk, &writer_dir, rx, seal_tx, &writer_ctl)
    })?;
    let ArchiveWorker { done: mut writer_done, thread: writer_thread } = writer;
    let ArchiveWorker { done: mut sealer_done, thread: sealer_thread } = sealer;
    let connections = Arc::new(AtomicU64::new(0));
    // SALIDAFALLO P4b: finalizador unico de END.json, nativo. Espera la entrega del recorrido async
    // o, con fatal, el fin del escritor y del sellador (no depende de que Tokio progrese).
    let finisher = {
        let parts = FinishParts {
            ctl: Arc::clone(&ctl), dir: dir.clone(), clock, audit: Arc::clone(&audit),
            connections: Arc::clone(&connections), writer_thread, sealer_thread, store_thread,
            writer_finished: Arc::clone(&writer_finished), sealer_finished: Arc::clone(&sealer_finished),
            store_finished: Arc::clone(&store_finished),
        };
        thread::Builder::new().name("capture-end".into()).spawn(move || finish_capture(parts))?
    };
    ctl.registrar_finalizador(finisher.thread().clone());
    drop(finisher);
    let (stop_tx, stop) = watch::channel(false);
    let (fatal_tx, mut fatal_rx) = mpsc::unbounded_channel::<String>();
    let (gate_tx, gate_rx) = mpsc::unbounded_channel();
    let gate_emitter = Emitter { queue: queue.clone(), clock, route: "rest", shard: None, seq: Arc::new(AtomicU64::new(0)) };
    let (gate, info_rx) = gate::Gate::new(
        gate::GateConfig { rest: venue.rest, root: root.clone(), agg_limit_max: AGG_LIMIT_MAX, default_cooldown_s: REST_DEFAULT_COOLDOWN_S, info_refresh: INFO_REFRESH },
        client.clone(), gate_emitter, clock);
    // SALIDAFALLO P2: el JoinSet superior se observa durante la captura con la identidad de cada
    // tarea; un JoinError o un final antes de la orden de parada es fatal.
    let mut tasks = salidafallo::Supervisor::nuevo();
    {
        let stop = stop.clone();
        tasks.lanzar("rest gate".into(), async move {
            gate.run(gate_rx, stop).await.map_err(|e| io::Error::other(format!("rest gate: {e}")).into())
        });
    }
    for (index, route, syms, url, roster_base) in shard_defs {
        let r2_rank: HashMap<String, usize> =
            syms.iter().enumerate().map(|(i, s)| (s.clone(), roster_base + i)).collect();
        let ctx = ShardCtx {
            venue, route, index, total: total_shards, symbols: syms, url,
            emitter: Emitter { queue: queue.clone(), clock, route, shard: Some(index), seq: Arc::new(AtomicU64::new(0)) },
            clock, book: Arc::new(Mutex::new(ShardBook::default())), gate: gate_tx.clone(),
            stop: stop.clone(), fatal: fatal_tx.clone(), rotate_after, connections: Arc::clone(&connections),
            gate_info: info_rx.clone(), r2_max_gap, r2_symbols_total, r2_rank,
        };
        tasks.lanzar(format!("shard {index} {route}"), async move {
            ctx.run().await.map_err(|e| io::Error::other(format!("shard {index}: {e}")).into())
        });
    }
    drop(gate_tx);
    let mut errors = Vec::new();
    let mut exit = 0u8;
    // Una sola creacion del monitor para esta captura, fuera del select! y sin un
    // bucle exterior que lo reconstruya. El select! sondea el mismo future fijado;
    // las esperas y las comprobaciones sucesivas viven dentro de watch_space.
    let space_watch = store::watch_space(&root, min_free_bytes, STORE_SCAN_EVERY);
    tokio::pin!(space_watch);
    let stop_reason = tokio::select! {
        signal = tokio::signal::ctrl_c() => {
            if let Err(e) = signal { errors.push(e.to_string()); }
            "ctrl_c"
        }
        ended = deadline(clock, until) => {
            if let Err(e) = ended { errors.push(e.to_string()); }
            "deadline"
        }
        _ = stop_file(root.join(STOP_FILE)) => "stop_file",
        stopped = low_rx => {
            let event = salidafallo::clasificar_store(stopped, store_exit.leer(), ctl.store.cerrada());
            let (code, message, reason) = event.aplicar(&ctl);
            exit = code;
            errors.push(message);
            reason
        }
        stop = &mut space_watch => {
            exit = stop.code();
            errors.push(stop.message());
            stop.reason()
        }
        reason = fatal_rx.recv() => {
            let reason = reason.unwrap_or_else(|| "task channel closed".into());
            ctl.fatal(Clase::Lector, &reason);
            errors.push(reason);
            "reader_error"
        }
        (name, joined) = tasks.siguiente() => {
            if let Some(e) = salidafallo::Supervisor::evaluar(&ctl, &name, joined, false) {
                errors.push(e);
            }
            "task_error"
        }
        _ = ctl.esperar_fatal() => "fatal",
        _ = &mut writer_done => {
            ctl.fatal(Clase::Escritor, "capture-gzip termino antes de la parada");
            "writer_error"
        }
        _ = &mut sealer_done => {
            ctl.fatal(Clase::Sellador, "capture-seal termino antes de la parada");
            "sealer_error"
        }
    };
    // Parada (sana o fatal). Store deja de admitir trabajo nuevo, como la cancelacion de antes.
    ctl.store.cerrar();
    let _ = stop_tx.send(true);
    errors.extend(tasks.cerrar(&ctl).await);
    drop(fatal_tx);
    while let Ok(e) = fatal_rx.try_recv() {
        ctl.fatal(Clase::Lector, &e);
        errors.push(e);
    }
    drop(queue);
    // P4d: corte sano solo despues de que las conexiones terminaron y el drenaje L2 final entro
    // a la cola. Si ya hubo fatal, el corte fatal (del vigilante) prevalece.
    ctl.admision.cortar(salidafallo::Motivo::Sana);
    ctl.entregar_fin_async(salidafallo::FinAsync { stop_reason, exit, errors });
    Ok(CaptureEnd::Finisher)
}

enum CaptureEnd {
    /// Salida antes de arrancar los hilos de archivo (sin END.json), como antes.
    Early(u8),
    /// El finalizador nativo escribe END.json y fija el codigo.
    Finisher,
}

struct FinishParts {
    ctl: Arc<Control>,
    dir: PathBuf,
    clock: Clock,
    audit: Arc<QueueAudit>,
    connections: Arc<AtomicU64>,
    writer_thread: thread::JoinHandle<ArchiveOutcome>,
    sealer_thread: thread::JoinHandle<SealReport>,
    store_thread: thread::JoinHandle<()>,
    writer_finished: Arc<std::sync::atomic::AtomicBool>,
    sealer_finished: Arc<std::sync::atomic::AtomicBool>,
    store_finished: Arc<std::sync::atomic::AtomicBool>,
}

/// Finalizador unico (P4b). Con fatal no espera al runtime: le basta el fin del escritor y del
/// sellador. No hace join de store si no termino: lo registra y el cierre queda incompleto.
fn finish_capture(p: FinishParts) {
    let ctl = &p.ctl;
    let mut woke_async = false;
    loop {
        let fatal = ctl.es_fatal();
        if fatal && !woke_async {
            woke_async = true;
            ctl.avisar_async();
        }
        let archive_done = p.writer_finished.load(Ordering::SeqCst) && p.sealer_finished.load(Ordering::SeqCst);
        if ctl.async_listo() && (archive_done || !fatal) {
            break;
        }
        if fatal && archive_done {
            break;
        }
        thread::park();
    }
    let fatal_mode = ctl.es_fatal() && !ctl.async_listo();
    let fin = ctl.tomar_fin_async().unwrap_or(salidafallo::FinAsync { stop_reason: "fatal", exit: 1, errors: Vec::new() });
    let mut errors = fin.errors;
    let mut incomplete = Vec::<String>::new();
    let outcome = match p.writer_thread.join() {
        Ok(o) => o,
        Err(_) => {
            errors.push("archive worker panicked".into());
            incomplete.push("escritor sin resultado (panic)".into());
            ArchiveOutcome::default()
        }
    };
    let report = match p.sealer_thread.join() {
        Ok(r) => r,
        Err(_) => {
            errors.push("archive worker panicked".into());
            incomplete.push("sellador sin resultado (panic)".into());
            SealReport::default()
        }
    };
    let store_state = if ctl.es_fatal() && !p.store_finished.load(Ordering::SeqCst) {
        incomplete.push("capture-store no confirmo su parada al escribir END".into());
        "sin_confirmar"
    } else if p.store_thread.join().is_err() {
        errors.push("hilo de archivo termino con panic".into());
        "panic"
    } else {
        "terminado"
    };
    let seal_telemetry = report.telemetry_json();
    if let Some(error) = &report.error {
        errors.push(error.clone());
    }
    for f in &outcome.failures {
        errors.push(format!("escritor: {f}"));
    }
    let counts = &report.counts;
    let enqueued = p.audit.accepted.load(Ordering::Relaxed);
    if counts.rows != enqueued as u64 {
        errors.push(format!("{} rows were enqueued but {} were sealed", enqueued, counts.rows));
    }
    if counts.routes.len() != 2 {
        errors.push("capture has no market frames from at least one required route".into());
    }
    // Contabilidad de admision (P4a): instantanea del corte y estado resuelto.
    let adm = &ctl.admision;
    let cut = adm.corte();
    let resolved = adm.resuelta();
    let confirmed = adm.confirmadas();
    let accounting = if !resolved {
        incomplete.push(format!("admision sin resolver: {} ofertas en vuelo", adm.en_vuelo()));
        json!("N/D: reservas en curso sin resolver; no hay instantanea coherente de filas")
    } else {
        if outcome.rows_written + outcome.rows_unwritten != confirmed {
            incomplete.push(format!("confirmadas {confirmed} != escritas {} + no escritas {}", outcome.rows_written, outcome.rows_unwritten));
        }
        json!({"confirmed": confirmed, "written": outcome.rows_written, "unwritten": outcome.rows_unwritten,
            "sealed": counts.rows})
    };
    if outcome.rows_unwritten > 0 || !report.failures.is_empty() {
        incomplete.push("hay filas o segmentos sin sellar".into());
    }
    let first = ctl.primer_fatal();
    let (secondary, secondary_lost) = ctl.secundarios();
    let status = if first.is_some() || !errors.is_empty() { "error" } else { "stopped" };
    let stop_reason = p.audit.stop_reason(fin.stop_reason, fin.exit);
    let rejected = match p.audit.rejected.lock() {
        Ok(mut rows) => std::mem::take(&mut *rows),
        Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
    };
    let stamp = p.clock.stamp().unwrap_or(Stamp { utc_ms: 0, mono_ns: 0 });
    let end = json!({
        "status":status,"stop_reason":stop_reason,"ended_utc_ms":stamp.utc_ms,"duration_ns":stamp.mono_ns,
        "segments":report.segments,"rows":counts.rows,"kinds":counts.kinds,"streams":counts.streams,
        "frames_by_route":counts.routes,"errors":errors,
        "connections_opened":p.connections.load(Ordering::Relaxed),
        "archive_queue":{
            "enqueued_rows":enqueued,
            "sampled_peak_rows":p.audit.peak_rows.load(Ordering::Relaxed),
            "sampled_peak_charged_bytes":p.audit.peak_charged_bytes.load(Ordering::Relaxed),
            "dequeued_rows":p.audit.dequeued.load(Ordering::Relaxed),
            "max_dequeue_age_ns":p.audit.max_dequeue_age_ns.load(Ordering::Relaxed),
            "failure_reason":p.audit.failure_reason(),
            "rejected_rows":rejected,
        },
        "archive_seal":seal_telemetry,
        "salidafallo":{
            "first_fatal": first.as_ref().map(|(c, d)| json!({"class": c.nombre(), "detail": d})),
            "secondary_errors": secondary, "secondary_lost": secondary_lost,
            "finisher_mode": if fatal_mode { "fatal_sin_runtime" } else { "async_entregado" },
            "admission": {
                "cut": cut.map(|c| json!({"reason": format!("{:?}", c.motivo), "confirmed": c.confirmadas, "in_flight": c.en_vuelo})),
                "resolved": resolved, "accounting": accounting,
                "reservations_cancelled": adm.reservas_canceladas(), "rejected_after_cut": adm.rechazos_tras_corte(),
            },
            "writer": outcome.json(),
            "store": {"state": store_state, "cut": ctl.store.corte().map(|c| json!({"in_progress": c.en_curso, "last_op": format!("{:?}", c.ultima)}))},
            "incomplete_close": !incomplete.is_empty(), "incomplete_reasons": incomplete,
            "deadline_ns": u64::try_from(ctl.plazo().as_nanos()).unwrap_or(u64::MAX),
        },
        "recode":"las horas selladas se re-codifican en el hilo capture-store; lo pendiente se retoma al arrancar",
    });
    let mut incomplete_close = end["salidafallo"]["incomplete_close"] == true && first.is_some();
    if let Err(e) = write_json(&p.dir.join("END.json"), &end) {
        // P4b: si END no puede persistirse se registra fuera (stderr) y no condiciona la salida.
        eprintln!("CAPTURE END NO PERSISTIDO: {e}");
        ctl.fatal(Clase::Cierre, &format!("END.json no persistido: {e}"));
        incomplete_close = true;
    }
    let code = if fin.exit != 0 {
        fin.exit
    } else if errors.is_empty() && first.is_none() {
        0
    } else {
        1
    };
    if code != 0 {
        eprintln!("CAPTURE STOP {}", errors.join(" | "));
    }
    ctl.fin_listo(code, incomplete_close);
}

fn cli_codec(cmd: &str, args: &[String]) -> CaptureResult<()> {
    match (cmd, args) {
        ("encode", [input, output, rest @ ..]) => {
            let p = codec_params(rest)?;
            let t = Instant::now();
            let src: Box<dyn Read> = if input.ends_with(".gz") {
                Box::new(flate2::read::GzDecoder::new(BufReader::new(File::open(input)?)))
            } else {
                Box::new(BufReader::new(File::open(input)?))
            };
            let mut w = BufWriter::new(new_file(Path::new(output))?);
            let r = codec::encode(src, &mut w, p)?;
            w.flush()?;
            drop(w);
            let enc_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let d = codec::decode(BufReader::new(File::open(output)?), io::sink())?;
            let dec_s = t.elapsed().as_secs_f64();
            let in_bytes = fs::metadata(input)?.len();
            let out_bytes = fs::metadata(output)?.len();
            println!("{}", json!({"input":input,"input_bytes":in_bytes,"output":output,"output_bytes":out_bytes,
                "rows":r.lines,"raw_bytes":r.raw_bytes,"raw_sha256":r.raw_sha256,"decoded_sha256":d.raw_sha256,
                "identical":d.raw_sha256 == r.raw_sha256 && d.raw_bytes == r.raw_bytes && d.lines == r.lines,
                "skeletons":r.skeletons,"contexts":r.contexts,"raw_fallback_lines":r.raw_fallback_lines,
                "ratio_vs_input":in_bytes as f64 / out_bytes.max(1) as f64,"encode_s":enc_s,"decode_s":dec_s,
                "zstd_level":p.zstd_level,"window_log":p.window_log,"block_raw_bytes":p.block_raw_bytes}));
            Ok(())
        }
        ("decode", [input, output]) => {
            let w = BufWriter::new(new_file(Path::new(output))?);
            let r = codec::decode(BufReader::new(File::open(input)?), w)?;
            println!("{}", json!({"rows":r.lines,"raw_bytes":r.raw_bytes,"raw_sha256":r.raw_sha256}));
            Ok(())
        }
        ("verify", [input]) => {
            let r = codec::decode(BufReader::new(File::open(input)?), io::sink())?;
            println!("{}", json!({"rows":r.lines,"raw_bytes":r.raw_bytes,"raw_sha256":r.raw_sha256,"ok":true}));
            Ok(())
        }
        _ => Err(io::Error::other("usage: captura_v2 encode <in.jsonl[.gz]> <out.tgll> [zstd_level=N] | decode <in.tgll> <out.jsonl> | verify <in.tgll>").into()),
    }
}

/// Arranque sincrono (SALIDAFALLO P3). Solo el recorrido `capture` arma hook y vigilante antes
/// de construir el runtime; las herramientas encode/decode/verify no cambian de conducta.
fn main() -> std::process::ExitCode {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("capture") => return run_capture(&args[2..]),
        Some(cmd @ ("encode" | "decode" | "verify")) => cli_codec(cmd, &args[2..]).map(|_| 0u8),
        _ => Err(io::Error::other("usage: captura_v2 <capture|encode|decode|verify> ...").into()),
    };
    match result {
        Ok(code) => std::process::ExitCode::from(code),
        Err(e) => {
            eprintln!("CAPTURE ERROR {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run_capture(args: &[String]) -> std::process::ExitCode {
    let plazo = match opt::<u64>(args, "plazo_fatal_ms") {
        Ok(v) => v.map(Duration::from_millis).unwrap_or(salidafallo::PLAZO_FATAL),
        Err(e) => {
            eprintln!("CAPTURE ERROR {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // SALIDAFALLO-4: el hilo de salida ordenada existe antes del vigilante.
    let salida = match salidafallo::SalidaProceso::arrancar() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("CAPTURE ERROR no arranca el hilo de salida: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let ctl = Arc::new(Control::nuevo(
        plazo,
        Box::new(salidafallo::RelojMonotono::nuevo()),
        Box::new(salida),
        Arc::new(salidafallo::CompuertaStore::real(cancel)),
    ));
    ctl.registrar_principal(thread::current());
    let vigilante = {
        let ctl = Arc::clone(&ctl);
        thread::Builder::new().name("capture-vigia".into()).spawn(move || salidafallo::vigilar(&ctl))
    };
    match vigilante {
        Ok(h) => ctl.registrar_vigilante(h.thread().clone()),
        Err(e) => {
            eprintln!("CAPTURE ERROR no arranca el vigilante: {e}");
            return std::process::ExitCode::FAILURE;
        }
    }
    // El hook solo se instala con el vigilante ya vivo (si no, su camino seguro terminaria).
    while !ctl.vigilante_vivo() {
        thread::yield_now();
    }
    salidafallo::instalar_hook(Arc::clone(&ctl));
    let runtime = match tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("CAPTURE ERROR runtime: {e}");
            return salir_sin_finalizador(&ctl, 1);
        }
    };
    let result = runtime.block_on(capture(args, Arc::clone(&ctl)));
    match result {
        Ok(CaptureEnd::Early(code)) => {
            drop(runtime);
            salir_sin_finalizador(&ctl, code)
        }
        Err(e) => {
            eprintln!("CAPTURE ERROR {e}");
            drop(runtime);
            salir_sin_finalizador(&ctl, 1)
        }
        Ok(CaptureEnd::Finisher) => {
            ctl.esperar_fin();
            // El vigilante sigue armado durante la destruccion del runtime.
            drop(runtime);
            if ctl.desarmar().is_err() {
                esperar_al_vigilante()
            }
            std::process::ExitCode::from(ctl.codigo_fin())
        }
    }
}

/// Salidas tempranas (antes de los hilos de archivo): mismo codigo que antes si no hay fatal.
fn salir_sin_finalizador(ctl: &Control, code: u8) -> std::process::ExitCode {
    if ctl.desarmar().is_err() {
        // Con fatal no hay finalizador que avise: el cierre posible ya termino.
        ctl.fin_listo(1, true);
        esperar_al_vigilante()
    }
    std::process::ExitCode::from(code)
}

/// Con fatal, la salida es del vigilante (cierre completado o plazo vencido).
fn esperar_al_vigilante() -> ! {
    loop {
        thread::park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::io::BufRead;

    fn frame(sequence: u64, timestamp: u64, payload: &str) -> Value {
        json!({
            "kind":"frame", "route":"public", "stream":"xrpusdt@trade", "symbol":"XRPUSDT",
            "connection":3, "seq":sequence, "recv_utc_ms":timestamp,
            "recv_mono_ns":sequence, "payload":payload,
        })
    }

    #[test]
    fn queue_full_records_the_first_unarchived_event() -> CaptureResult<()> {
        let (queue, mut rx) = ArchiveQueue::new(1, 4 * ROW_CHARGE_OVERHEAD);
        queue.offer(frame(1, 100, "first"), None)?;
        assert!(queue.offer(frame(2, 101, "rejected"), None).is_err());
        let pending = rx.try_recv()?;
        assert_eq!(pending.row["seq"], 1);
        let rejected = queue.audit.rejected.lock().map_err(|_| io::Error::other("poisoned"))?;
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0]["reason"], "row_capacity");
        Ok(())
    }

    #[test]
    fn byte_budget_stops_admission_after_the_first_rejected_row() -> CaptureResult<()> {
        let (queue, mut rx) = ArchiveQueue::new(8, 2 * ROW_CHARGE_OVERHEAD);
        let other = queue.clone();
        queue.offer(frame(1, 100, "accepted"), None)?;
        let error = other.offer(frame(2, 101, "rejected"), None).unwrap_err();
        assert!(error.to_string().contains("byte_budget"));
        let first = rx.try_recv()?;
        assert_eq!(first.row["seq"], 1);
        drop(first);
        assert!(queue.offer(frame(3, 102, "after drain"), None).is_err());
        assert!(matches!(rx.try_recv(), Err(mpsc::error::TryRecvError::Empty)));
        assert_eq!(queue.audit.failure_reason(), Some("byte_budget"));
        assert_eq!(queue.audit.stop_reason("reader_error", 0), "archive_queue_full");
        assert_eq!(queue.audit.stop_reason("low_disk", 3), "low_disk");
        let rejected = queue.audit.rejected.lock().map_err(|_| io::Error::other("poisoned"))?;
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0]["seq"], 2);
        Ok(())
    }

    #[test]
    fn hour_rotation_seals_and_hands_off_for_recode() -> CaptureResult<()> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let dir = env::temp_dir().join(format!("captura-v2-test-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir)?;
        let (ctl, _salida, _reloj) = salidafallo::tests::control_de_prueba(salidafallo::PLAZO_FATAL);
        let (queue, rx) = ArchiveQueue::with_admission(8, 32 * ROW_CHARGE_OVERHEAD, Arc::clone(&ctl.admision));
        for (sequence, timestamp) in [(1, HOUR_MS - 1), (2, HOUR_MS + 1), (3, HOUR_MS - 2), (4, HOUR_MS + ROTATION_GRACE_MS)] {
            queue.offer(frame(sequence, timestamp, "{}"), Some(Cont::Agg { symbol: "XRPUSDT".into(), a: sequence }))?;
        }
        drop(queue);
        let (seal, pending) = blocking_channel::sync_channel(2);
        let outcome = archive(&RealDisk, &dir, rx, seal, &ctl);
        assert!(outcome.failures.is_empty());
        let (tx, sealed) = blocking_channel::channel();
        let report = seal_segments(&RealDisk, &dir, pending, Some(tx), None);
        assert!(report.error.is_none());
        assert_eq!(report.segments, 2);
        let names: Vec<_> = sealed.try_iter().map(|(_, n)| n).collect();
        assert_eq!(names, ["hour-0-0001", "hour-3600000-0002"]);
        for (file, expected) in [("hour-0-0001", [1, 3]), ("hour-3600000-0002", [2, 4])] {
            let gzip = dir.join(format!("{file}.jsonl.gz"));
            let data = BufReader::new(GzDecoder::new(File::open(&gzip)?));
            let seqs: Vec<u64> = data.lines().map(|l| serde_json::from_str::<Value>(&l.unwrap()).unwrap()["ingest_seq"].as_u64().unwrap()).collect();
            assert_eq!(seqs, expected);
            let m: Value = serde_json::from_reader(File::open(dir.join(format!("{file}.manifest.json")))?)?;
            assert_eq!(m["schema"], 2);
            assert_eq!(m["continuity"]["XRPUSDT"]["last_agg_a"], expected[1]);
            let keys: Vec<String> = m["continuity"]["XRPUSDT"].as_object().unwrap().keys().map(|k| k.to_lowercase()).collect();
            let unique: HashSet<&String> = keys.iter().collect();
            assert_eq!(keys.len(), unique.len(), "claves que solo difieren en mayusculas rompen ConvertFrom-Json");
            // Re-codificacion: verifica y solo entonces borra el .jsonl.gz.
            let p = codec::Params { zstd_level: 3, block_raw_bytes: 1 << 20, window_log: 20 };
            let r = store::recode(&dir, file, p)?;
            assert_eq!(r["identical"], true);
            assert!(!gzip.exists());
            let mut out = Vec::new();
            codec::decode(File::open(dir.join(format!("{file}.tgll")))?, &mut out)?;
            assert_eq!(out.iter().filter(|&&b| b == b'\n').count(), 2);
        }
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn recode_keeps_gzip_when_manifest_does_not_match() -> CaptureResult<()> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let dir = env::temp_dir().join(format!("captura-v2-bad-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir)?;
        let mut seg = Segment::open(&RealDisk, &dir, 0, 1, 1)?;
        seg.append(&frame(1, 1, "{}"), None, 1)?;
        seg.seal(&RealDisk, &dir, &mut PhaseTimes::new()).map_err(|f| io::Error::other(f.error))?;
        let manifest = dir.join("hour-0-0001.manifest.json");
        let mut m: Value = serde_json::from_reader(File::open(&manifest)?)?;
        m["sha256"] = json!("00");
        fs::remove_file(&manifest)?;
        write_json(&manifest, &m)?;
        let p = codec::Params { zstd_level: 3, block_raw_bytes: 1 << 20, window_log: 20 };
        assert!(store::recode(&dir, "hour-0-0001", p).is_err());
        assert!(dir.join("hour-0-0001.jsonl.gz").exists());
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn roster_drops_bvol_indices() -> CaptureResult<()> {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!("roster-{stamp}.csv"));
        fs::write(&path, "simbolo,tipo,ruta,bytes\nBTCBVOLUSDT,BVOLIndex,x,1\nBTCUSDT,klines,x,1\nBTCUSDT,aggTrades,x,1\nADAUSDT,klines,x,1\n")?;
        assert_eq!(roster(&path)?, ["ADAUSDT", "BTCUSDT"]);
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn urls_use_public_and_market_routes() -> CaptureResult<()> {
        let v = Venue::parse("mainnet")?;
        let syms = vec!["BTCUSDT".to_string()];
        assert_eq!(v.url("public", &syms, false), "wss://fstream.binance.com/public/stream?streams=btcusdt@bookTicker/btcusdt@depth@100ms");
        assert_eq!(v.url("market", &syms, true), "wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade/btcusdt@markPrice@1s/!forceOrder@arr");
        Ok(())
    }
}
