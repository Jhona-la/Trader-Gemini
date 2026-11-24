//! SALIDAFALLO-9 (P8): banco del ensayo en subproceso.
//!
//! Prueba `#[ignore]` (no corre en la regresion ni entre las PURAS). Lanza el `captura_v2.exe`
//! indicado por `P8_EXE` (el release del SHA ensayado, con su hash anotado aqui) con el venue
//! `local` contra servidores WebSocket y REST propios en 127.0.0.1, con datos sinteticos y una
//! raiz nueva `P8_RAIZ\<caso>` que debe no existir. No borra nada (no reutiliza `TestDir`, cuyo
//! Drop borra), no toca Binance y solo puede terminar a su propio hijo, identificado por el
//! handle de `Child` con su PID y su hora de creacion. El caso sale de `P8_CASO`.

use super::*;
use flate2::read::GzDecoder;
use std::{
    io::BufRead,
    os::windows::io::AsRawHandle,
    process::{Command, Stdio},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::broadcast,
};

/// Cuatro simbolos: con SHARDS_PER_ROUTE = 4 da la disposicion de produccion (4 public + 4 market,
/// un simbolo por shard). El defecto heredado de r3b con menos simbolos (indices con huecos y
/// total - index fuera de rango en run_market) lo corrige SALIDAFALLO-14 (`shard_defs`).
const SIMBOLOS: [&str; 4] = ["FSTUSDT", "SNDUSDT", "TRDUSDT", "FTHUSDT"];
/// Ritmo del generador sintetico del servidor (parametro del banco, no una cifra del colector).
const PASO_SERVIDOR: Duration = Duration::from_millis(20);
/// Rafaga de aggTrade para agotar la cola de archivo: el doble de su capacidad de filas derivada,
/// floor(B / q_min) (FALLAS-2). Cada trama cobra al menos q_min, asi que basta para agotar B.
const RAFAGA: usize = 2 * (QUEUE_BYTE_BUDGET / ROW_CHARGE_OVERHEAD);
/// Margen del tope del padre sobre la duracion pedida y el plazo (preparacion y arranque). No es
/// parte de la cota del colector: si el padre llega a su tope, el caso es FALLO.
const MARGEN_PADRE: Duration = Duration::from_secs(120);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Modo {
    Normal,
    /// Cierra una vez la primera conexion market y la primera public (reconexion sana).
    CerrarUnaVez,
    /// La primera conexion market recibe una rafaga de RAFAGA tramas.
    Rafaga,
    /// FALLAS-2: la primera conexion market recibe barrera r3b + RAFAGA_EXTRA tramas aggTrade.
    RafagaBarrera,
}

/// FALLAS-2: tramas de la rafaga de barrera_filas por encima de la capacidad antigua (parametro del
/// banco: asegura que la testigo C_antigua + 1 y algunas mas lleguen con el escritor detenido).
const RAFAGA_EXTRA: usize = 4_096;

#[derive(Clone, Copy)]
struct Escenario {
    nombre: &'static str,
    inyectar: Option<&'static str>,
    codigos: &'static [i32],
    /// Override de plazo_fatal_ms (ensayo abreviado); None = PLAZO_FATAL por defecto.
    plazo_ms: Option<u64>,
    until_s: u64,
    modo: Modo,
}

/// Codigos (salidafallo.rs): 1 = fatal con cierre completo; 4 = cierre incompleto (plazo vencido, END no
/// persistido, store sin confirmar o filas/segmentos sin sellar); 5 = fallo del vigilante.
/// Plazo abreviado de ensayo para los casos fatales: parametro del banco, no verifica la cota por
/// defecto (eso lo hace `salida_bloqueada_plazo_defecto`).
const PLAZO_ENSAYO_MS: Option<u64> = Some(20_000);

const ESCENARIOS: [Escenario; 29] = [
    Escenario { nombre: "sana", inyectar: None, codigos: &[0], plazo_ms: None, until_s: 40, modo: Modo::CerrarUnaVez },
    Escenario { nombre: "panic_tarea", inyectar: Some("panic_tarea"), codigos: &[1], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "fin_tarea", inyectar: Some("fin_tarea"), codigos: &[1], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "cancelar_conexion_market", inyectar: Some("cancelar_conexion_market"), codigos: &[1], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "panic_worker_park", inyectar: Some("panic_worker_park"), codigos: &[1, 4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "pausa_reserva", inyectar: Some("pausa_reserva"), codigos: &[1], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "pausa_confirmada", inyectar: Some("pausa_confirmada"), codigos: &[1], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "retener_confirmada", inyectar: Some("retener_confirmada"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "cola_llena", inyectar: Some("cola_llena"), codigos: &[1], plazo_ms: None, until_s: 120, modo: Modo::Rafaga },
    Escenario { nombre: "fallo_append", inyectar: Some("fallo_append"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "fallo_manifiesto", inyectar: Some("fallo_manifiesto"), codigos: &[4], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "sellador_bloqueado", inyectar: Some("sellador_bloqueado"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "store_ocupado", inyectar: Some("store_ocupado"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "fallo_end", inyectar: Some("fallo_end"), codigos: &[4], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "bloqueo_end", inyectar: Some("bloqueo_end"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "fatal_durante_end", inyectar: Some("fatal_durante_end"), codigos: &[1], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "fatal_drop_runtime", inyectar: Some("fatal_drop_runtime"), codigos: &[1], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "fallo_vigilante", inyectar: Some("fallo_vigilante"), codigos: &[5], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "salida_bloqueada", inyectar: Some("salida_bloqueada"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "salida_bloqueada_plazo_defecto", inyectar: Some("salida_bloqueada"), codigos: &[4], plazo_ms: None, until_s: 600, modo: Modo::Normal },
    // SALIDAFALLO-12 (C3 de la N1 v3): parada sana con filas depth/ticker identificables retenidas
    // en el arbitro al parar, y fallos reales de cada fase del archivo en el exe ensayado.
    Escenario { nombre: "retener_l2", inyectar: Some("retener_l2"), codigos: &[0], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "fallo_open", inyectar: Some("fallo_open"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "fallo_handoff", inyectar: Some("fallo_handoff"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    Escenario { nombre: "fallo_gzip_finish", inyectar: Some("fallo_gzip_finish"), codigos: &[4], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "fallo_flush", inyectar: Some("fallo_flush"), codigos: &[4], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "fallo_hash", inyectar: Some("fallo_hash"), codigos: &[4], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "fallo_publish_gzip", inyectar: Some("fallo_publish_gzip"), codigos: &[4], plazo_ms: None, until_s: 30, modo: Modo::Normal },
    Escenario { nombre: "fallo_manifiesto_segundo", inyectar: Some("fallo_manifiesto_segundo"), codigos: &[4], plazo_ms: PLAZO_ENSAYO_MS, until_s: 120, modo: Modo::Normal },
    // FALLAS-2: barrera antigua de filas con presupuesto disponible, en el exe y con parada sana.
    Escenario { nombre: "barrera_filas", inyectar: Some("barrera_filas"), codigos: &[0], plazo_ms: None, until_s: 90, modo: Modo::RafagaBarrera },
];

// --- Servidores locales ----------------------------------------------------------------------

fn trama(simbolo: &str, tail: &str, data: Value) -> (String, String) {
    let stream = format!("{}@{tail}", simbolo.to_ascii_lowercase());
    let texto = json!({"stream": stream, "data": data}).to_string();
    (stream, texto)
}

fn agg(simbolo: &str, a: u64) -> (String, String) {
    trama(simbolo, "aggTrade", json!({"e":"aggTrade","s":simbolo,"a":a,"p":"1.0","q":"1","T":a}))
}

/// Streams pedidos en la URL de la conexion (`...?streams=a/b/c`).
fn streams_de(ruta: &str) -> HashSet<String> {
    ruta.split_once("streams=").map(|(_, s)| s.split('/').map(str::to_owned).collect()).unwrap_or_default()
}

struct Servidor {
    ws: std::net::SocketAddr,
    rest: std::net::SocketAddr,
    conexiones: Arc<Mutex<Vec<String>>>,
    rest_pedidos: Arc<Mutex<Vec<String>>>,
    /// SALIDAFALLO-12 (C3): 0 normal; 1 = pedido el lote final L2; 2 = enviado (public callado).
    lote_l2: Arc<AtomicU64>,
    /// Identidades del lote final, fijadas por el servidor ANTES de enviarlo: (simbolo, flujo, u).
    retenidas: Arc<Mutex<Vec<(String, &'static str, u64)>>>,
    /// FALLAS-2: (simbolo, primer a, ultimo a) de la rafaga enviada, fijados por el servidor.
    rafaga: Arc<Mutex<Option<(String, u64, u64)>>>,
}

async fn servidor(modo: Modo) -> CaptureResult<Servidor> {
    let ws = TcpListener::bind(("127.0.0.1", 0)).await?;
    let http = TcpListener::bind(("127.0.0.1", 0)).await?;
    let (ws_addr, rest_addr) = (ws.local_addr()?, http.local_addr()?);
    let (tx_pub, _) = broadcast::channel::<(String, String)>(8192);
    let (tx_mkt, _) = broadcast::channel::<(String, String)>(8192);
    // Contadores por simbolo: u del libro (depth/bookTicker) y a de aggTrade.
    let u: Arc<Vec<AtomicU64>> = Arc::new(SIMBOLOS.iter().map(|_| AtomicU64::new(1000)).collect());
    let a: Arc<Vec<AtomicU64>> = Arc::new(SIMBOLOS.iter().map(|_| AtomicU64::new(1)).collect());
    let lote_l2 = Arc::new(AtomicU64::new(0));
    let retenidas: Arc<Mutex<Vec<(String, &'static str, u64)>>> = Arc::default();
    let rafaga_rango: Arc<Mutex<Option<(String, u64, u64)>>> = Arc::default();
    {
        let (tx_pub, tx_mkt, u, a) = (tx_pub.clone(), tx_mkt.clone(), Arc::clone(&u), Arc::clone(&a));
        let (lote_l2, retenidas) = (Arc::clone(&lote_l2), Arc::clone(&retenidas));
        tokio::spawn(async move {
            let mut tick = time::interval(PASO_SERVIDOR);
            let mut n = 0u64;
            loop {
                tick.tick().await;
                n += 1;
                let lote = lote_l2.load(Ordering::SeqCst);
                for (i, s) in SIMBOLOS.iter().enumerate() {
                    if lote == 1 {
                        // Lote final: depth con hueco de predecesor (k - 1 nunca se envia), asi que el
                        // arbitro lo retiene, y bookTicker con un u nuevo. Las identidades se anotan antes.
                        let k = u[i].fetch_add(2, Ordering::SeqCst) + 2;
                        if let Ok(mut v) = retenidas.lock() {
                            v.push(((*s).to_owned(), "depth", k));
                            v.push(((*s).to_owned(), "ticker", k));
                        }
                        let _ = tx_pub.send(trama(s, "depth@100ms", json!({"e":"depthUpdate","s":s,"U":k,"u":k,"pu":k - 1,"b":[],"a":[]})));
                        let _ = tx_pub.send(trama(s, "bookTicker", json!({"e":"bookTicker","s":s,"u":k,"b":"1.0","B":"2","a":"1.1","A":"3"})));
                    } else if lote == 0 {
                    let nu = u[i].fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = tx_pub.send(trama(s, "depth@100ms", json!({"e":"depthUpdate","s":s,"U":nu,"u":nu,"pu":nu - 1,"b":[],"a":[]})));
                    let _ = tx_pub.send(trama(s, "bookTicker", json!({"e":"bookTicker","s":s,"u":nu,"b":"1.0","B":"2","a":"1.1","A":"3"})));
                    }
                    let _ = tx_mkt.send(agg(s, a[i].fetch_add(1, Ordering::SeqCst) + 1));
                    if n % (1000 / PASO_SERVIDOR.as_millis() as u64) == 0 {
                        let _ = tx_mkt.send(trama(s, "markPrice@1s", json!({"e":"markPriceUpdate","s":s,"p":"1.0"})));
                    }
                }
                if lote == 1 {
                    lote_l2.store(2, Ordering::SeqCst);
                }
            }
        });
    }
    let conexiones: Arc<Mutex<Vec<String>>> = Arc::default();
    {
        let conexiones = Arc::clone(&conexiones);
        let a = Arc::clone(&a);
        let rafaga_rango = Arc::clone(&rafaga_rango);
        tokio::spawn(async move {
            let (mut cerrada_pub, mut cerrada_mkt, mut rafaga_hecha) = (false, false, false);
            loop {
                let Ok((socket, peer)) = ws.accept().await else { return };
                assert!(peer.ip().is_loopback());
                let ruta = Arc::new(Mutex::new(String::new()));
                let r2 = Arc::clone(&ruta);
                let cb = move |req: &tokio_tungstenite::tungstenite::handshake::server::Request,
                               resp: tokio_tungstenite::tungstenite::handshake::server::Response| {
                    *r2.lock().expect("ruta") = req.uri().to_string();
                    Ok(resp)
                };
                let Ok(mut sock) = tokio_tungstenite::accept_hdr_async(socket, cb).await else { continue };
                let ruta = ruta.lock().expect("ruta").clone();
                conexiones.lock().expect("conexiones").push(ruta.clone());
                let pedidos = streams_de(&ruta);
                let publica = ruta.starts_with("/public/");
                let mut rx = if publica { tx_pub.subscribe() } else { tx_mkt.subscribe() };
                let cerrar = modo == Modo::CerrarUnaVez && if publica { !std::mem::replace(&mut cerrada_pub, true) } else { !std::mem::replace(&mut cerrada_mkt, true) };
                let rafaga = matches!(modo, Modo::Rafaga | Modo::RafagaBarrera) && !publica && !std::mem::replace(&mut rafaga_hecha, true);
                let tramas = if modo == Modo::RafagaBarrera { inyeccion::barrera_r3b_filas() + RAFAGA_EXTRA } else { RAFAGA };
                let a = Arc::clone(&a);
                let rafaga_rango = Arc::clone(&rafaga_rango);
                tokio::spawn(async move {
                    if rafaga {
                        if let Some(i) = SIMBOLOS.iter().position(|s| pedidos.contains(&format!("{}@aggTrade", s.to_ascii_lowercase()))) {
                            let (mut primero, mut ultimo) = (None, 0);
                            for _ in 0..tramas {
                                let id = a[i].fetch_add(1, Ordering::SeqCst) + 1;
                                let (_, t) = agg(SIMBOLOS[i], id);
                                if sock.send(Message::Text(t)).await.is_err() {
                                    return;
                                }
                                primero.get_or_insert(id);
                                ultimo = id;
                            }
                            if let (Ok(mut r), Some(p)) = (rafaga_rango.lock(), primero) {
                                *r = Some((SIMBOLOS[i].to_owned(), p, ultimo));
                            }
                        }
                    }
                    let fin = time::sleep(if cerrar { Duration::from_secs(3) } else { Duration::from_secs(u64::MAX / 4) });
                    tokio::pin!(fin);
                    loop {
                        tokio::select! {
                            _ = &mut fin => {
                                let _ = sock.close(None).await;
                                return;
                            }
                            m = rx.recv() => match m {
                                Ok((stream, t)) => if pedidos.contains(&stream) && sock.send(Message::Text(t)).await.is_err() { return },
                                Err(broadcast::error::RecvError::Lagged(_)) => {}
                                Err(broadcast::error::RecvError::Closed) => return,
                            },
                            entrada = sock.next() => match entrada {
                                Some(Ok(Message::Ping(p))) => { let _ = sock.send(Message::Pong(p)).await; }
                                Some(Ok(_)) => {}
                                _ => return,
                            },
                        }
                    }
                });
            }
        });
    }
    let rest_pedidos: Arc<Mutex<Vec<String>>> = Arc::default();
    {
        let pedidos = Arc::clone(&rest_pedidos);
        let u = Arc::clone(&u);
        tokio::spawn(async move {
            let mut usado = 0u64;
            loop {
                let Ok((mut socket, peer)) = http.accept().await else { return };
                assert!(peer.ip().is_loopback());
                let mut cabecera = Vec::new();
                let mut trozo = [0; 1024];
                while !cabecera.ends_with(b"\r\n\r\n") && cabecera.len() < 8192 {
                    match socket.read(&mut trozo).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => cabecera.extend_from_slice(&trozo[..n]),
                    }
                }
                let linea = String::from_utf8_lossy(&cabecera).lines().next().unwrap_or_default().to_owned();
                pedidos.lock().expect("pedidos").push(linea.clone());
                let simbolo = |l: &str| SIMBOLOS.iter().position(|s| l.contains(&format!("symbol={s}&")));
                let (estado, cuerpo) = if linea.starts_with("GET /fapi/v1/exchangeInfo ") {
                    ("200 OK", json!({
                        "rateLimits":[{"rateLimitType":"REQUEST_WEIGHT","interval":"SECOND","intervalNum":1,"limit":1000}],
                        "symbols": SIMBOLOS.iter().map(|s| json!({"symbol":s,"status":"TRADING"})).collect::<Vec<_>>()}))
                } else if let (true, Some(i)) = (linea.starts_with("GET /fapi/v1/depth?"), simbolo(&linea)) {
                    ("200 OK", json!({"lastUpdateId": u[i].load(Ordering::SeqCst), "bids": [], "asks": []}))
                } else if linea.starts_with("GET /fapi/v1/aggTrades?") {
                    ("200 OK", json!([]))
                } else {
                    ("404 Not Found", json!({}))
                };
                usado += 1;
                let cuerpo = cuerpo.to_string();
                let r = format!("HTTP/1.1 {estado}\r\nContent-Type: application/json\r\nX-MBX-USED-WEIGHT-1S: {usado}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{cuerpo}", cuerpo.len());
                let _ = socket.write_all(r.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
    }
    Ok(Servidor { ws: ws_addr, rest: rest_addr, conexiones, rest_pedidos, lote_l2, retenidas, rafaga: rafaga_rango })
}
// --- Identidad del hijo ------------------------------------------------------------------------

#[repr(C)]
#[derive(Default)]
struct Filetime {
    lo: u32,
    hi: u32,
}

unsafe extern "system" {
    fn GetProcessTimes(h: *mut std::ffi::c_void, c: *mut Filetime, e: *mut Filetime, k: *mut Filetime, u: *mut Filetime) -> i32;
}

/// Hora de creacion del proceso (FILETIME, 100 ns desde 1601) leida del handle del hijo.
fn creacion(child: &std::process::Child) -> u64 {
    let (mut c, mut e, mut k, mut u) = (Filetime::default(), Filetime::default(), Filetime::default(), Filetime::default());
    let ok = unsafe { GetProcessTimes(child.as_raw_handle(), &mut c, &mut e, &mut k, &mut u) };
    assert_ne!(ok, 0, "GetProcessTimes");
    (u64::from(c.hi) << 32) | u64::from(c.lo)
}

/// FALLAS-2 (A10/A37/A45): memoria del hijo leida de su handle (PROCESS_MEMORY_COUNTERS).
#[repr(C)]
#[derive(Default)]
struct MemoriaProceso {
    cb: u32,
    page_faults: u32,
    peak_working_set: usize,
    working_set: usize,
    quota_peak_paged: usize,
    quota_paged: usize,
    quota_peak_nonpaged: usize,
    quota_nonpaged: usize,
    pagefile: usize,
    peak_pagefile: usize,
}

unsafe extern "system" {
    fn K32GetProcessMemoryInfo(h: *mut std::ffi::c_void, m: *mut MemoriaProceso, cb: u32) -> i32;
}

fn memoria(child: &std::process::Child) -> Option<MemoriaProceso> {
    let mut m = MemoriaProceso { cb: std::mem::size_of::<MemoriaProceso>() as u32, ..MemoriaProceso::default() };
    let ok = unsafe { K32GetProcessMemoryInfo(child.as_raw_handle(), &mut m, m.cb) };
    (ok != 0).then_some(m)
}

// --- Validacion externa de archivos ------------------------------------------------------------

fn sha_archivo(p: &Path) -> (u64, String) {
    let b = fs::read(p).expect("leer");
    (b.len() as u64, format!("{:x}", Sha256::digest(&b)))
}

/// Descomprime entero un gzip (final o parcial) y valida filas, hora UTC e ingest_seq.
fn validar_gzip(p: &Path, hora: Option<u64>) -> Value {
    let (bytes, sha) = sha_archivo(p);
    let mut lineas = 0u64;
    let mut fuera_de_hora = 0u64;
    let mut seq_no_creciente = 0u64;
    let mut ultimo = 0u64;
    let mut kinds = BTreeMap::<String, u64>::new();
    let mut error = None;
    let lector = std::io::BufReader::new(GzDecoder::new(File::open(p).expect("abrir")));
    for l in lector.lines() {
        match l {
            Ok(l) => {
                lineas += 1;
                let v: Value = serde_json::from_str(&l).unwrap_or(Value::Null);
                *kinds.entry(v["kind"].as_str().unwrap_or("?").to_owned()).or_default() += 1;
                if let (Some(h), Some(t)) = (hora, v["recv_utc_ms"].as_u64()) {
                    if t < h || t >= h + HOUR_MS {
                        fuera_de_hora += 1;
                    }
                }
                let s = v["ingest_seq"].as_u64().unwrap_or(0);
                if s <= ultimo {
                    seq_no_creciente += 1;
                }
                ultimo = s;
            }
            Err(e) => {
                error = Some(e.to_string());
                break;
            }
        }
    }
    json!({"path": p, "bytes": bytes, "sha256": sha, "rows": lineas, "error": error, "kinds": kinds,
        "rows_outside_utc_hour": fuera_de_hora, "ingest_seq_not_increasing": seq_no_creciente})
}

fn inventario(raiz: &Path) -> (Vec<Value>, Vec<Value>) {
    let mut archivos = Vec::new();
    let mut gz = Vec::new();
    let mut pila = vec![raiz.to_path_buf()];
    while let Some(d) = pila.pop() {
        let Ok(it) = fs::read_dir(&d) else { continue };
        let mut entradas: Vec<_> = it.flatten().map(|e| e.path()).collect();
        entradas.sort();
        for p in entradas {
            if p.is_dir() {
                pila.push(p);
                continue;
            }
            let (bytes, sha) = sha_archivo(&p);
            archivos.push(json!({"path": p, "bytes": bytes, "sha256": sha}));
            let nombre = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let base = nombre.strip_suffix(".jsonl.gz").or_else(|| nombre.strip_suffix(".jsonl.gz.partial"));
            if let Some(base) = base {
                let manifiesto = p.with_file_name(format!("{base}.manifest.json"));
                let m: Option<Value> = fs::read(&manifiesto).ok().and_then(|b| serde_json::from_slice(&b).ok());
                let hora = base.strip_prefix("hour-").and_then(|r| r.split('-').next()).and_then(|h| h.parse().ok());
                let mut v = validar_gzip(&p, hora);
                v["final"] = json!(nombre.ends_with(".jsonl.gz"));
                v["manifest"] = m.clone().unwrap_or(Value::Null);
                if let Some(m) = m {
                    v["manifest_matches"] = json!(m["bytes"] == v["bytes"] && m["sha256"] == v["sha256"] && m["rows"] == v["rows"]
                        && m["utc_hour_ms"].as_u64() == hora);
                }
                gz.push(v);
            }
        }
    }
    (archivos, gz)
}

/// SALIDAFALLO-12 (C3): identidad (simbolo, flujo, u) de una fila depth/bookTicker del gzip, leida
/// del payload original (`data.s`, `data.u`) y del stream.
fn identidad_l2(fila: &Value) -> Option<(String, &'static str, u64)> {
    let stream = fila["stream"].as_str()?;
    let flujo = if stream.contains("@depth") { "depth" } else if stream.contains("@bookTicker") { "ticker" } else { return None };
    let p: Value = serde_json::from_str(fila["payload"].as_str()?).ok()?;
    let d = if p["data"].is_object() { &p["data"] } else { &p };
    Some((d["s"].as_str()?.to_owned(), flujo, d["u"].as_u64()?))
}

/// SALIDAFALLO-12 (C3): oraculo de las filas retenidas. Para cada identidad que el servidor fijo
/// ANTES de enviar el lote final: si el testigo del hijo la vio pendiente en el arbitro al empezar
/// `drain_l2(.., true)` y cuantas filas con esa identidad hay en los gzip decodificados (y de que
/// clase). Lo exigido lo decide `comprobar`.
fn filas_retenidas(srv: &Servidor, gz: &[Value], stdout: &str) -> Value {
    let esperadas = srv.retenidas.lock().map(|v| v.clone()).unwrap_or_default();
    let mut pendientes = HashSet::<(String, &'static str, u64)>::new();
    for l in stdout.lines().filter_map(|l| l.strip_prefix("P8_L2_PENDIENTES ")) {
        for campo in l.split(' ') {
            let (flujo, lista) = match campo.split_once('=') {
                Some(("depth", v)) => ("depth", v),
                Some(("ticker", v)) => ("ticker", v),
                _ => continue,
            };
            for par in lista.split(',').filter(|p| !p.is_empty()) {
                if let Some((s, u)) = par.rsplit_once(':').and_then(|(s, u)| u.parse().ok().map(|u| (s.to_owned(), u))) {
                    pendientes.insert((s, flujo, u));
                }
            }
        }
    }
    let mut vistas = BTreeMap::<(String, &'static str, u64), Vec<String>>::new();
    for g in gz {
        let Some(p) = g["path"].as_str() else { continue };
        let lector = std::io::BufReader::new(GzDecoder::new(File::open(p).expect("abrir")));
        for l in lector.lines().map_while(Result::ok) {
            let fila: Value = serde_json::from_str(&l).unwrap_or(Value::Null);
            if let Some(id) = identidad_l2(&fila) {
                vistas.entry(id).or_default().push(fila["kind"].as_str().unwrap_or("?").to_owned());
            }
        }
    }
    json!(esperadas.iter().map(|id| {
        let clases = vistas.get(id).cloned().unwrap_or_default();
        json!({"symbol": id.0, "flow": id.1, "u": id.2, "pending_at_final_drain": pendientes.contains(id),
            "rows_in_gzip": clases.len(), "kinds": clases})
    }).collect::<Vec<_>>())
}

/// FALLAS-2 (A40/A41/A51): conciliacion por identidad de lo archivado. Identidad del emisor
/// (route, shard, seq), fijada antes de la cola; identidad del servidor (simbolo, a) de cada aggTrade.
/// Para la rafaga, cuantas de las identidades enviadas faltan o se repiten en los gzip.
fn identidades(srv: &Servidor, gz: &[Value]) -> Value {
    let mut emisor = HashMap::<(String, i64, u64), u64>::new();
    let mut agg = HashMap::<(String, u64), u64>::new();
    for g in gz {
        let Some(p) = g["path"].as_str() else { continue };
        let lector = std::io::BufReader::new(GzDecoder::new(File::open(p).expect("abrir")));
        for l in lector.lines().map_while(Result::ok) {
            let f: Value = serde_json::from_str(&l).unwrap_or(Value::Null);
            let clave = (f["route"].as_str().unwrap_or("?").to_owned(), f["shard"].as_i64().unwrap_or(-1), f["seq"].as_u64().unwrap_or(0));
            *emisor.entry(clave).or_default() += 1;
            if f["stream"].as_str().is_some_and(|s| s.ends_with("@aggTrade")) {
                let d: Value = f["payload"].as_str().and_then(|x| serde_json::from_str(x).ok()).unwrap_or(Value::Null);
                if let (Some(s), Some(a)) = (d["data"]["s"].as_str(), d["data"]["a"].as_u64()) {
                    *agg.entry((s.to_owned(), a)).or_default() += 1;
                }
            }
        }
    }
    let rango = srv.rafaga.lock().ok().and_then(|r| r.clone());
    let rafaga = rango.map(|(s, p, u)| {
        let n = |a: u64| agg.get(&(s.clone(), a)).copied().unwrap_or(0);
        json!({"symbol": s, "first_a": p, "last_a": u, "sent": u - p + 1,
            "missing": (p..=u).filter(|a| n(*a) == 0).count(), "duplicated": (p..=u).filter(|a| n(*a) > 1).count()})
    });
    json!({"emitter_identities": emisor.len(), "emitter_duplicated": emisor.values().filter(|c| **c > 1).count(),
        "agg_identities": agg.len(), "agg_duplicated": agg.values().filter(|c| **c > 1).count(), "burst": rafaga})
}

// --- Criterios por caso ------------------------------------------------------------------------

/// SALIDAFALLO-11 (C2): cota superior del tiempo del marcador a la salida observada por el padre.
/// Derivacion (sin cifras a mano, todo de la corrida): el padre ve el marcador como mucho una vuelta
/// despues de escribirse y ve la salida como mucho una vuelta despues de ocurrir; esa vuelta medida
/// (`parent_max_loop_ns`, que incluye el sueno del sondeo en el mismo temporizador del sistema) es la
/// resolucion. La tolerancia del lado del hijo es el retraso del `park_timeout` del vigilante sobre
/// el mismo temporizador, acotado por la misma vuelta medida. Cota = plazo efectivo + resolucion +
/// tolerancia. La salida del proceso tras `TerminateProcess` queda dentro de lo observado; no es
/// una cota universal (contrato de `SalidaProceso`).
fn cota_superior_ns(plazo_ns: u64, vuelta_ns: u64) -> u64 {
    let resolucion = vuelta_ns;
    let tolerancia = vuelta_ns;
    plazo_ns.saturating_add(resolucion).saturating_add(tolerancia)
}

fn comprobar(e: &Escenario, r: &Value) -> Vec<String> {
    let mut f = Vec::new();
    let mut exigir = |ok: bool, que: &str| {
        if !ok {
            f.push(que.to_owned());
        }
    };
    let end = &r["end"];
    let sf = &end["salidafallo"];
    let ff = &sf["first_fatal"];
    let detalle = ff["detail"].as_str().unwrap_or("");
    let gz = r["gzip"].as_array().cloned().unwrap_or_default();
    let finales_validos = gz.iter().filter(|g| g["final"] == true && g["manifest_matches"] == true && g["error"].is_null()
        && g["rows_outside_utc_hour"] == 0 && g["ingest_seq_not_increasing"] == 0).count();
    let contable = || {
        let a = &sf["admission"]["accounting"];
        a.is_object() && a["confirmed"].as_u64() == Some(a["written"].as_u64().unwrap_or(0) + a["unwritten"].as_u64().unwrap_or(0))
    };
    exigir(r["killed_by_parent"] == false, "salida autonoma (el padre no termino al hijo)");
    exigir(e.codigos.iter().any(|c| r["exit_code"].as_i64() == Some(i64::from(*c))), "codigo de salida esperado");
    match e.nombre {
        "sana" => {
            exigir(end["status"] == "stopped" && ff.is_null(), "END sano sin fatal");
            exigir(end["stop_reason"] == "deadline", "parada por until_ms");
            exigir(finales_validos >= 1 && finales_validos == gz.len(), "todos los gzip finales con manifiesto valido");
            let filas: u64 = gz.iter().filter_map(|g| g["rows"].as_u64()).sum();
            exigir(end["rows"].as_u64() == Some(filas) && filas > 0, "filas de END = filas de los gzip");
            exigir(end["frames_by_route"]["public"].as_u64() > Some(0) && end["frames_by_route"]["market"].as_u64() > Some(0), "tramas de las dos rutas");
            exigir(contable(), "contabilidad de admision coherente");
            let gaps: u64 = gz.iter().filter_map(|g| g["kinds"]["socket_gap"].as_u64()).sum();
            exigir(gaps >= 1, "reconexion market recuperable (socket_gap)");
        }
        "panic_tarea" => exigir(ff["class"] == "panic" && detalle.contains("P8 inyectado: panic de tarea"), "primer fatal: panic de tarea"),
        "fin_tarea" => exigir(ff["class"] == "tarea_fin_inesperado" && detalle.contains("inyector p8"), "primer fatal: fin inesperado"),
        "cancelar_conexion_market" => {
            exigir(ff["class"] == "lector_error" && detalle.contains("cancelada"), "primer fatal: JoinError de cancelacion visto por run_market");
            exigir(end["stop_reason"] == "task_error", "fatal durante la captura, antes de la parada ordinaria");
        }
        // Hilo con el nombre de los workers de Tokio 1.53 y ubicacion en el punto on_thread_park.
        "panic_worker_park" => exigir(ff["class"] == "panic" && detalle.contains("tokio-rt-worker") && detalle.contains("on_thread_park"),
            "primer fatal: panic de un worker real fuera de Future::poll"),
        "pausa_reserva" => {
            exigir(sf["admission"]["reservations_cancelled"].as_u64() >= Some(1), "reserva liberada sin confirmar");
            exigir(contable(), "contabilidad de admision coherente");
        }
        "pausa_confirmada" => {
            exigir(sf["admission"]["cut"]["in_flight"].as_u64() >= Some(1), "confirmada en vuelo en el corte");
            exigir(sf["admission"]["resolved"] == true && contable(), "drenada y contabilidad coherente");
        }
        "retener_confirmada" | "sellador_bloqueado" | "bloqueo_end" => exigir(end.is_null(), "sin END (cierre incompleto declarado)"),
        "cola_llena" => {
            // FALLAS-2: con la capacidad derivada el primer rechazo es por agotamiento real del
            // presupuesto de bytes (cargo de la fila + bytes ya cargados > B), no por filas.
            let rech = &end["archive_queue"]["rejected_rows"][0];
            exigir(end["archive_queue"]["failure_reason"] == "byte_budget", "rechazo por agotamiento del presupuesto de bytes");
            exigir(rech["charged_bytes"].as_u64().zip(rech["row_charge_bytes"].as_u64()).zip(rech["byte_budget"].as_u64())
                .is_some_and(|((c, q), b)| c + q > b), "agotamiento real: cargados + cargo de la fila > B");
            exigir(rech["queue_occupied"].as_u64() < rech["queue_capacity_rows"].as_u64(), "quedaban plazas: la barrera de filas no actuo");
            exigir(sf["writer"]["drain_after_cut_ns"].is_u64() && sf["writer"]["rows_after_cut"].as_u64() >= Some(1), "drenaje medido");
            exigir(contable(), "contabilidad de admision coherente");
            exigir(finales_validos >= 1, "gzip final valido con las aceptadas");
        }
        "fallo_append" => {
            let w = &sf["writer"]["failures"][0];
            exigir(w["phase"] == "append" && w["state"] == "jsonl.gz.partial_conservado_sin_sellar", "fallo de append con estado");
            exigir(w["path"].as_str().is_some_and(|p| Path::new(p).exists()), "el parcial se conserva en su ruta");
        }
        "fallo_manifiesto" => {
            let s = &end["archive_seal"]["failures"][0];
            exigir(s["phase"] == "publish_manifest" && s["state"] == "jsonl.gz_final_sin_manifiesto_valido", "manifiesto fallido tras el rename");
            exigir(gz.iter().any(|g| g["final"] == true && g["manifest"].is_null() && g["error"].is_null()), "gzip final integro sin manifiesto");
        }
        "store_ocupado" => {
            exigir(sf["store"]["state"] == "sin_confirmar", "store sin confirmar su parada");
            exigir(sf["store"]["cut"]["in_progress"] == 1 && sf["store"]["cut"]["last_op"] == "Retencion", "operacion admitida identificada");
        }
        "fallo_end" => {
            exigir(end.is_null(), "END no persistido");
            exigir(r["stderr"].as_str().is_some_and(|s| s.contains("CAPTURE END NO PERSISTIDO")), "aviso en stderr");
        }
        "fatal_durante_end" => exigir(ff["class"] == "inyectado_p8" && sf["end_rewrites"] == 1, "END reescrito con el fatal tardio"),
        "fatal_drop_runtime" => exigir(ff["class"] == "inyectado_p8" && detalle.contains("destruccion del runtime")
            && sf["runtime_destroyed_before_end"] == true, "END con el fatal de la destruccion del runtime"),
        "fallo_vigilante" => {}
        "salida_bloqueada" | "salida_bloqueada_plazo_defecto" => {
            exigir(!ff.is_null(), "END escrito antes de la salida bloqueada");
            exigir(r["stdout"].as_str().is_some_and(|s| s.contains("P8_INYECCION salida_bloqueada salida_bloqueada")), "la salida ordenada se bloqueo");
            // El marcador se escribe antes del fatal y el padre lo ve, a lo sumo, una vuelta despues.
            // SALIDAFALLO-11 (C2 de la N1 v3): plazo efectivo en ns (no el truncado a ms del
            // informe) y cota en los dos extremos, con la resolucion medida en la propia corrida.
            let desde = r["parent_ns_from_first_marker_to_exit"].as_u64();
            let plazo_ns = r["deadline_ns_effective"].as_u64();
            let vuelta = r["parent_max_loop_ns"].as_u64();
            exigir(plazo_ns.is_some() && plazo_ns == sf["deadline_ns"].as_u64(), "plazo efectivo en ns = deadline_ns de END");
            let (desde, plazo_ns, vuelta) = (desde.unwrap_or(0), plazo_ns.unwrap_or(u64::MAX), vuelta.unwrap_or(u64::MAX));
            exigir(desde >= plazo_ns.saturating_sub(vuelta), "la salida llego por el plazo, no antes");
            exigir(desde <= cota_superior_ns(plazo_ns, vuelta), "la salida llego dentro de la cota superior desde el marcador");
        }
        // SALIDAFALLO-12 (C3 de la N1 v3).
        "retener_l2" => {
            exigir(end["status"] == "stopped" && ff.is_null() && end["stop_reason"] == "deadline", "parada sana por until_ms sin fatal");
            exigir(finales_validos >= 1 && finales_validos == gz.len(), "todos los gzip finales con manifiesto valido");
            let filas: u64 = gz.iter().filter_map(|g| g["rows"].as_u64()).sum();
            exigir(end["rows"].as_u64() == Some(filas) && filas > 0, "filas de END = filas de los gzip");
            exigir(r["l2_batch_state"] == 2, "el servidor envio el lote final");
            let ret = r["l2_retained"].as_array().cloned().unwrap_or_default();
            let flujos = |f: &str| ret.iter().filter(|x| x["flow"] == f).count();
            exigir(flujos("depth") == SIMBOLOS.len() && flujos("ticker") == SIMBOLOS.len(), "una identidad depth y una ticker por simbolo en el lote");
            exigir(ret.iter().all(|x| x["pending_at_final_drain"] == true), "cada identidad seguia en el arbitro al empezar drain_l2(.., true)");
            exigir(ret.iter().all(|x| x["rows_in_gzip"] == 1 && x["kinds"][0] == "frame"), "cada identidad retenida esta una sola vez en los gzip");
            exigir(contable(), "contabilidad de admision coherente");
        }
        "fallo_open" => {
            let w = &sf["writer"]["failures"][0];
            exigir(w["phase"] == "open" && w["state"] == "partial_no_creado", "fallo real de open con estado");
            exigir(w["path"].as_str().is_some_and(|p| !Path::new(p).exists()), "el parcial no se creo");
            exigir(r["injection_markers"].to_string().contains("fallo_open"), "el fallo actuo en Disk::create");
        }
        "fallo_handoff" => {
            exigir(r["injection_markers"].to_string().contains("sellador_se_retira"), "el sellador se retiro tras el primer segmento");
            let w = sf["writer"]["failures"].as_array().cloned().unwrap_or_default();
            exigir(w.iter().any(|w| w["phase"] == "handoff" && w["state"] == "jsonl.gz.partial_conservado_sin_sellar"
                && w["path"].as_str().is_some_and(|p| Path::new(p).exists())), "handoff fallido de verdad con el parcial conservado");
            exigir(finales_validos >= 1, "el primer segmento quedo sellado y valido");
        }
        "fallo_gzip_finish" | "fallo_flush" | "fallo_hash" | "fallo_publish_gzip" => {
            let fase = match e.nombre { "fallo_gzip_finish" => "gzip_finish", "fallo_flush" => "flush", "fallo_hash" => "sha256", _ => "publish_gzip" };
            let s = &end["archive_seal"]["failures"][0];
            exigir(s["phase"] == fase && s["state"] == "jsonl.gz.partial", "fallo real de la fase del sellado con su estado");
            exigir(s["path"].as_str().is_some_and(|p| p.ends_with(".jsonl.gz.partial") && Path::new(p).exists()), "el parcial se conserva en su ruta");
            exigir(r["injection_markers"].to_string().contains(&format!("fallo_{fase}")), "el fallo actuo en la fase");
            exigir(!gz.iter().any(|g| g["final"] == true), "ningun gzip final publicado con el fallo");
        }
        "fallo_manifiesto_segundo" => {
            let s = end["archive_seal"]["failures"].as_array().cloned().unwrap_or_default();
            exigir(s.len() == 1 && s[0]["phase"] == "publish_manifest", "un solo fallo: el manifiesto del primer segmento");
            let sin = gz.iter().find(|g| g["final"] == true && g["manifest"].is_null() && g["error"].is_null());
            exigir(sin.is_some(), "primer gzip final integro sin manifiesto");
            // Hora del segmento fallido, leida de su nombre (hour-<ms>-<n>). Los rezagados de esa misma
            // hora (filas marcadas antes del desfase que llegan tras la rotacion) no cuentan como el
            // segundo segmento: se exige uno sano de una hora posterior.
            let hora_fallida: Option<u64> = sin.and_then(|a| a["path"].as_str()).and_then(|p| p.rsplit("hour-").next())
                .and_then(|r| r.split('-').next()).and_then(|h| h.parse().ok());
            let sano = gz.iter().find(|g| g["final"] == true && g["manifest_matches"] == true && g["error"].is_null()
                && g["rows_outside_utc_hour"] == 0 && g["ingest_seq_not_increasing"] == 0 && g["rows"].as_u64() > Some(0)
                && g["manifest"]["utc_hour_ms"].as_u64() > hora_fallida && hora_fallida.is_some());
            exigir(sano.is_some(), "segundo segmento sellado sano, de la hora siguiente, tras el fallo del manifiesto");
        }
        // FALLAS-2: escritor detenido hasta superar la capacidad antigua; parada sana y conciliacion.
        "barrera_filas" => {
            let aq = &end["archive_queue"];
            exigir(end["status"] == "stopped" && ff.is_null() && end["stop_reason"] == "deadline", "parada sana por until_ms sin fatal");
            exigir(aq["failure_reason"].is_null() && aq["rejected_rows"].as_array().is_some_and(Vec::is_empty), "sin rechazos de admision");
            exigir(aq["sampled_peak_rows"].as_u64() > Some(inyeccion::barrera_r3b_filas() as u64), "la cola supero la capacidad antigua de filas");
            exigir(r["injection_markers"].to_string().contains("escritor_reanudado"), "el escritor se reanudo tras la barrera");
            exigir(finales_validos >= 1 && finales_validos == gz.len(), "todos los gzip finales con manifiesto valido");
            let filas: u64 = gz.iter().filter_map(|g| g["rows"].as_u64()).sum();
            exigir(end["rows"].as_u64() == Some(filas) && aq["enqueued_rows"].as_u64() == Some(filas), "filas de END = aceptadas = filas de los gzip");
            exigir(contable(), "contabilidad de admision coherente");
            let id = &r["identities"];
            exigir(id["emitter_duplicated"] == 0 && id["agg_duplicated"] == 0, "ninguna identidad archivada repetida");
            exigir(id["burst"]["missing"] == 0 && id["burst"]["sent"].as_u64() > Some(inyeccion::barrera_r3b_filas() as u64), "cada trama de la rafaga archivada una vez");
        }
        _ => exigir(false, "caso desconocido"),
    }
    f
}

/// SALIDAFALLO-11 (C2): control de la cota superior sobre el mismo `comprobar` del banco. Un
/// informe sintetico de `salida_bloqueada` pasa en el borde de la cota y falla una unidad por
/// encima (y una por debajo del extremo inferior), con la vuelta y el plazo de un caso real
/// (recibo v3 `salida_bloqueada_plazo_defecto/harness.json`).
#[test]
fn p8_cota_superior_control() {
    let e = *ESCENARIOS.iter().find(|e| e.nombre == "salida_bloqueada_plazo_defecto").expect("escenario");
    let plazo = u64::try_from(salidafallo::PLAZO_FATAL.as_nanos()).expect("ns");
    let vuelta = 24_724_900_u64;
    let informe = |desde: u64| json!({
        "killed_by_parent": false, "exit_code": 4,
        "end": {"salidafallo": {"first_fatal": {"class": "inyectado_p8", "detail": "x"}, "deadline_ns": plazo}},
        "stdout": "P8_INYECCION salida_bloqueada salida_bloqueada control_ns=1",
        "parent_ns_from_first_marker_to_exit": desde, "deadline_ns_effective": plazo, "parent_max_loop_ns": vuelta,
    });
    let cota = cota_superior_ns(plazo, vuelta);
    assert_eq!(cota, plazo + 2 * vuelta);
    assert_eq!(comprobar(&e, &informe(plazo)), Vec::<String>::new(), "en el plazo: pasa");
    assert_eq!(comprobar(&e, &informe(cota)), Vec::<String>::new(), "en el borde de la cota: pasa");
    assert_eq!(comprobar(&e, &informe(cota + 1)), ["la salida llego dentro de la cota superior desde el marcador"], "por encima: falla");
    assert_eq!(comprobar(&e, &informe(plazo - vuelta - 1)), ["la salida llego por el plazo, no antes"], "por debajo: falla");
    let mut otro = informe(plazo);
    otro["end"]["salidafallo"]["deadline_ns"] = json!(plazo - 1);
    assert_eq!(comprobar(&e, &otro), ["plazo efectivo en ns = deadline_ns de END"], "el plazo del informe debe ser el del hijo");
}

/// SALIDAFALLO-12 (C3): control del oraculo de retenidas y del segundo segmento sobre el mismo
/// `comprobar`. El informe sano pasa; quitar una identidad del gzip, duplicarla o no verla pendiente
/// en el drenaje final falla por su asercion, y un "segundo segmento" de la misma hora no basta.
#[test]
fn p8_criterios_c3_control() {
    let e = *ESCENARIOS.iter().find(|e| e.nombre == "retener_l2").expect("escenario");
    let ids: Vec<Value> = SIMBOLOS.iter().flat_map(|s| ["depth", "ticker"].map(|f| json!({"symbol": s, "flow": f, "u": 7,
        "pending_at_final_drain": true, "rows_in_gzip": 1, "kinds": ["frame"]}))).collect();
    let gz = json!([{"final": true, "manifest_matches": true, "error": null, "rows_outside_utc_hour": 0,
        "ingest_seq_not_increasing": 0, "rows": 10}]);
    let informe = |ids: Vec<Value>| json!({"killed_by_parent": false, "exit_code": 0, "l2_batch_state": 2, "l2_retained": ids, "gzip": gz,
        "end": {"status": "stopped", "stop_reason": "deadline", "rows": 10, "salidafallo": {"first_fatal": null,
            "admission": {"accounting": {"confirmed": 10, "written": 10, "unwritten": 0}}}}});
    assert_eq!(comprobar(&e, &informe(ids.clone())), Vec::<String>::new(), "sano: pasa");
    let mut falta = ids.clone();
    falta[3]["rows_in_gzip"] = json!(0);
    assert_eq!(comprobar(&e, &informe(falta)), ["cada identidad retenida esta una sola vez en los gzip"], "perdida: falla");
    let mut doble = ids.clone();
    doble[0]["rows_in_gzip"] = json!(2);
    assert_eq!(comprobar(&e, &informe(doble)), ["cada identidad retenida esta una sola vez en los gzip"], "duplicada: falla");
    let mut vencida = ids.clone();
    vencida[5]["pending_at_final_drain"] = json!(false);
    assert_eq!(comprobar(&e, &informe(vencida)), ["cada identidad seguia en el arbitro al empezar drain_l2(.., true)"], "no retenida: falla");
    assert_eq!(comprobar(&e, &informe(ids[1..].to_vec())), ["una identidad depth y una ticker por simbolo en el lote"], "lote incompleto: falla");

    let e = *ESCENARIOS.iter().find(|e| e.nombre == "fallo_manifiesto_segundo").expect("escenario");
    let seg = |n: &str, hora: u64, man: bool| json!({"path": format!("x\\run-local-1-2\\hour-{hora}-{n}.jsonl.gz"), "final": true,
        "error": null, "rows": 3, "rows_outside_utc_hour": 0, "ingest_seq_not_increasing": 0,
        "manifest": if man { json!({"utc_hour_ms": hora}) } else { Value::Null }, "manifest_matches": man});
    let informe = |gz: Value| json!({"killed_by_parent": false, "exit_code": 4, "gzip": gz,
        "end": {"archive_seal": {"failures": [{"phase": "publish_manifest"}]}, "salidafallo": {}}});
    let h = HOUR_MS;
    assert_eq!(comprobar(&e, &informe(json!([seg("0001", h, false), seg("0002", 2 * h, true)]))), Vec::<String>::new(), "hora siguiente: pasa");
    assert_eq!(comprobar(&e, &informe(json!([seg("0001", h, false), seg("0003", h, true)]))),
        ["segundo segmento sellado sano, de la hora siguiente, tras el fallo del manifiesto"], "rezagado de la misma hora: falla");
}

// --- El caso ---------------------------------------------------------------------------------

#[test]
#[ignore = "P8: ensayo en subproceso; se corre a mano con P8_EXE, P8_RAIZ y P8_CASO"]
fn p8_caso() -> CaptureResult<()> {
    let exe = PathBuf::from(env::var("P8_EXE")?);
    let nombre = env::var("P8_CASO")?;
    let e = *ESCENARIOS.iter().find(|e| e.nombre == nombre).ok_or_else(|| io::Error::other("P8_CASO desconocido"))?;
    let raiz = PathBuf::from(env::var("P8_RAIZ")?).join(e.nombre);
    assert!(!raiz.exists(), "la raiz de evidencia debe ser nueva: {}", raiz.display());
    fs::create_dir_all(&raiz)?;
    let (exe_bytes, exe_sha) = sha_archivo(&exe);
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    let srv = rt.block_on(servidor(e.modo))?;
    let roster = raiz.join("roster.csv");
    fs::write(&roster, format!("simbolo,tipo\n{}", SIMBOLOS.iter().map(|s| format!("{s},PERPETUAL\n")).collect::<String>()))?;
    let out = raiz.join("out");
    let utc0 = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let plazo_ms = e.plazo_ms.unwrap_or(salidafallo::PLAZO_FATAL.as_millis() as u64);
    // Plazo efectivo del hijo, exacto en ns (el de `plazo_ms` solo sirve para el tope del padre).
    let plazo_efectivo = e.plazo_ms.map_or(salidafallo::PLAZO_FATAL, Duration::from_millis);
    // SALIDAFALLO-12 (C3): los casos que cruzan una hora desplazan la marca UTC del hijo; until_ms
    // se da en esa misma escala para que la parada ordinaria no llegue antes de tiempo.
    let caso_iny = e.inyectar.map(|c| inyeccion::desde_args(&["local".to_owned(), format!("inyectar={c}")]));
    let desfase = match caso_iny {
        Some(Ok(Some(c))) if inyeccion::usa_desfase(c) => inyeccion::DESFASE_CRUCE_MS,
        _ => 0,
    };
    let mut args = vec![
        "capture".to_owned(), "local".to_owned(), out.display().to_string(), roster.display().to_string(),
        format!("until_ms={}", utc0 + e.until_s * 1000 + desfase),
        format!("ws_local={}", srv.ws), format!("rest_local=http://{}", srv.rest),
    ];
    if let Some(c) = e.inyectar {
        args.push(format!("inyectar={c}"));
    }
    if let Some(p) = e.plazo_ms {
        args.push(format!("plazo_fatal_ms={p}"));
    }
    let stdout_p = raiz.join("stdout.log");
    let stderr_p = raiz.join("stderr.log");
    let t0 = Instant::now();
    let mut child = Command::new(&exe)
        .args(&args)
        .current_dir(&raiz)
        .env_remove("HTTP_PROXY").env_remove("HTTPS_PROXY").env_remove("ALL_PROXY")
        .env("NO_PROXY", "127.0.0.1")
        .stdin(Stdio::null())
        .stdout(File::create(&stdout_p)?)
        .stderr(File::create(&stderr_p)?)
        .spawn()?;
    let pid = child.id();
    let creado = creacion(&child);
    println!("P8 {} hijo pid={pid} creacion_filetime={creado} exe_sha256={exe_sha}", e.nombre);
    let tope = Duration::from_secs(e.until_s) + Duration::from_millis(plazo_ms) + MARGEN_PADRE;
    let mut marcador: Option<Duration> = None;
    let mut killed = false;
    let mut vuelta_max = Duration::ZERO;
    let mut vuelta0 = Instant::now();
    let mut muestras_mem = Vec::new();
    let mut ultima_muestra: Option<Instant> = None;
    let estado = loop {
        if ultima_muestra.is_none_or(|m| m.elapsed() >= Duration::from_millis(100)) {
            ultima_muestra = Some(Instant::now());
            if let Some(m) = memoria(&child) {
                muestras_mem.push(json!([t0.elapsed().as_nanos() as u64, m.working_set, m.pagefile]));
            }
        }
        vuelta_max = vuelta_max.max(vuelta0.elapsed());
        vuelta0 = Instant::now();
        if let Some(st) = child.try_wait()? {
            break st;
        }
        let visto = fs::read_to_string(&stdout_p).unwrap_or_default();
        if marcador.is_none() && visto.contains("P8_INYECCION") && visto.contains(" fatal control_ns=") {
            marcador = Some(t0.elapsed());
        }
        // C3: con el arbitro ya congelado, el servidor envia el lote final identificable y calla public.
        if e.nombre == "retener_l2" && srv.lote_l2.load(Ordering::SeqCst) == 0 && visto.contains("P8_INYECCION retener_l2 l2_congelado") {
            srv.lote_l2.store(1, Ordering::SeqCst);
        }
        if t0.elapsed() >= tope {
            // Regla 8: solo el hijo propio, con su identidad comprobada por el mismo handle.
            assert_eq!((child.id(), creacion(&child)), (pid, creado), "identidad del hijo");
            child.kill()?;
            killed = true;
            break child.wait()?;
        }
        thread::sleep(Duration::from_millis(5));
    };
    let fin = t0.elapsed();
    let mem_fin = memoria(&child);
    let stdout = fs::read_to_string(&stdout_p).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_p).unwrap_or_default();
    if marcador.is_none() && stdout.contains(" fatal control_ns=") {
        marcador = Some(fin);
    }
    let (archivos, gz) = inventario(&out);
    let end_p = fs::read_dir(&out).ok().and_then(|it| it.flatten().map(|d| d.path()).find(|p| p.is_dir()))
        .map(|d| d.join("END.json"));
    let end: Value = end_p.as_ref().and_then(|p| fs::read(p).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
    let nanos = |d: Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
    let mut r = json!({
        "case": e.nombre, "inject": e.inyectar, "expected_exit_codes": e.codigos, "server_mode": format!("{:?}", e.modo),
        "exe": exe, "exe_bytes": exe_bytes, "exe_sha256": exe_sha,
        "command": args, "cwd": raiz, "pid": pid, "process_creation_filetime": creado,
        "loopback": {"ws": srv.ws.to_string(), "rest": srv.rest.to_string()},
        "launch_utc_ms": utc0, "deadline_ms": plazo_ms, "deadline_is_default": e.plazo_ms.is_none(),
        "deadline_ns_effective": nanos(plazo_efectivo),
        "upper_bound_ns_from_marker": cota_superior_ns(nanos(plazo_efectivo), nanos(vuelta_max)),
        "parent_cap_ns": nanos(tope), "parent_poll_ms": 5, "parent_max_loop_ns": nanos(vuelta_max),
        "parent_ns_launch_to_exit": nanos(fin),
        "parent_ns_launch_to_first_marker": marcador.map(nanos),
        "parent_ns_from_first_marker_to_exit": marcador.map(|m| nanos(fin.saturating_sub(m))),
        "exit_code": estado.code(), "killed_by_parent": killed,
        "end_path": end_p, "end": end, "files": archivos, "gzip": gz,
        "simulated_deletions": stdout.lines().filter(|l| l.starts_with("P8_BORRADO_SIMULADO")).count(),
        "injection_markers": stdout.lines().filter(|l| l.starts_with("P8_INYECCION")).collect::<Vec<_>>(),
        "ws_connections": srv.conexiones.lock().map(|v| v.clone()).unwrap_or_default(),
        "utc_shift_ms": desfase,
        "l2_batch_state": srv.lote_l2.load(Ordering::SeqCst),
        "l2_retained": filas_retenidas(&srv, &gz, &stdout),
        "rest_requests": srv.rest_pedidos.lock().map(|v| v.len()).unwrap_or_default(),
        "stdout": stdout, "stderr": stderr,
    });
    r["identities"] = identidades(&srv, &gz);
    r["child_memory"] = json!({"peak_working_set_bytes": mem_fin.as_ref().map(|m| m.peak_working_set),
        "peak_pagefile_bytes": mem_fin.as_ref().map(|m| m.peak_pagefile), "sample_every_ms": 100,
        "samples_ns_ws_pagefile": muestras_mem, "source": "K32GetProcessMemoryInfo sobre el handle del hijo"});
    let fallas = comprobar(&e, &r);
    r["checks_failed"] = json!(fallas);
    r["verdict"] = json!(if fallas.is_empty() { "PASS" } else { "FALLO" });
    fs::write(raiz.join("harness.json"), serde_json::to_vec_pretty(&r)?)?;
    println!("P8 {} veredicto={} codigo={:?} fallas={fallas:?}", e.nombre, r["verdict"], estado.code());
    drop(rt);
    assert!(fallas.is_empty(), "{}: {fallas:?}", e.nombre);
    Ok(())
}
