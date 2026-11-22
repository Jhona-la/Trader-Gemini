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
/// un simbolo por shard). Con menos de 4 simbolos los indices de shard tienen huecos y run_market
/// (codigo heredado de r3b) calcula total - index fuera de rango: lo encontro el primer ensayo.
const SIMBOLOS: [&str; 4] = ["FSTUSDT", "SNDUSDT", "TRDUSDT", "FTHUSDT"];
/// Ritmo del generador sintetico del servidor (parametro del banco, no una cifra del colector).
const PASO_SERVIDOR: Duration = Duration::from_millis(20);
/// Rafaga de aggTrade para llenar la cola de archivo: el doble de su capacidad de filas.
const RAFAGA: usize = 2 * QUEUE_CAPACITY;
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
}

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

const ESCENARIOS: [Escenario; 20] = [
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
    {
        let (tx_pub, tx_mkt, u, a) = (tx_pub.clone(), tx_mkt.clone(), Arc::clone(&u), Arc::clone(&a));
        tokio::spawn(async move {
            let mut tick = time::interval(PASO_SERVIDOR);
            let mut n = 0u64;
            loop {
                tick.tick().await;
                n += 1;
                for (i, s) in SIMBOLOS.iter().enumerate() {
                    let nu = u[i].fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = tx_pub.send(trama(s, "depth@100ms", json!({"e":"depthUpdate","s":s,"U":nu,"u":nu,"pu":nu - 1,"b":[],"a":[]})));
                    let _ = tx_pub.send(trama(s, "bookTicker", json!({"e":"bookTicker","s":s,"u":nu,"b":"1.0","B":"2","a":"1.1","A":"3"})));
                    let _ = tx_mkt.send(agg(s, a[i].fetch_add(1, Ordering::SeqCst) + 1));
                    if n % (1000 / PASO_SERVIDOR.as_millis() as u64) == 0 {
                        let _ = tx_mkt.send(trama(s, "markPrice@1s", json!({"e":"markPriceUpdate","s":s,"p":"1.0"})));
                    }
                }
            }
        });
    }
    let conexiones: Arc<Mutex<Vec<String>>> = Arc::default();
    {
        let conexiones = Arc::clone(&conexiones);
        let a = Arc::clone(&a);
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
                let rafaga = modo == Modo::Rafaga && !publica && !std::mem::replace(&mut rafaga_hecha, true);
                let a = Arc::clone(&a);
                tokio::spawn(async move {
                    if rafaga {
                        if let Some(i) = SIMBOLOS.iter().position(|s| pedidos.contains(&format!("{}@aggTrade", s.to_ascii_lowercase()))) {
                            for _ in 0..RAFAGA {
                                let (_, t) = agg(SIMBOLOS[i], a[i].fetch_add(1, Ordering::SeqCst) + 1);
                                if sock.send(Message::Text(t)).await.is_err() {
                                    return;
                                }
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
    Ok(Servidor { ws: ws_addr, rest: rest_addr, conexiones, rest_pedidos })
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

// --- Criterios por caso ------------------------------------------------------------------------

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
            exigir(end["archive_queue"]["failure_reason"] == "row_capacity", "rechazo por capacidad de filas");
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
            let desde = r["parent_ns_from_first_marker_to_exit"].as_u64().unwrap_or(0);
            let plazo_ns = r["deadline_ms"].as_u64().unwrap_or(u64::MAX).saturating_mul(1_000_000);
            let vuelta = r["parent_max_loop_ns"].as_u64().unwrap_or(u64::MAX);
            exigir(desde >= plazo_ns.saturating_sub(vuelta), "la salida llego por el plazo, no antes");
        }
        _ => exigir(false, "caso desconocido"),
    }
    f
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
    let mut args = vec![
        "capture".to_owned(), "local".to_owned(), out.display().to_string(), roster.display().to_string(),
        format!("until_ms={}", utc0 + e.until_s * 1000),
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
    let estado = loop {
        vuelta_max = vuelta_max.max(vuelta0.elapsed());
        vuelta0 = Instant::now();
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if marcador.is_none() && fs::read_to_string(&stdout_p).is_ok_and(|s| s.contains("P8_INYECCION") && s.contains(" fatal control_ns=")) {
            marcador = Some(t0.elapsed());
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
        "parent_cap_ns": nanos(tope), "parent_poll_ms": 5, "parent_max_loop_ns": nanos(vuelta_max),
        "parent_ns_launch_to_exit": nanos(fin),
        "parent_ns_launch_to_first_marker": marcador.map(nanos),
        "parent_ns_from_first_marker_to_exit": marcador.map(|m| nanos(fin.saturating_sub(m))),
        "exit_code": estado.code(), "killed_by_parent": killed,
        "end_path": end_p, "end": end, "files": archivos, "gzip": gz,
        "simulated_deletions": stdout.lines().filter(|l| l.starts_with("P8_BORRADO_SIMULADO")).count(),
        "injection_markers": stdout.lines().filter(|l| l.starts_with("P8_INYECCION")).collect::<Vec<_>>(),
        "ws_connections": srv.conexiones.lock().map(|v| v.clone()).unwrap_or_default(),
        "rest_requests": srv.rest_pedidos.lock().map(|v| v.len()).unwrap_or_default(),
        "stdout": stdout, "stderr": stderr,
    });
    let fallas = comprobar(&e, &r);
    r["checks_failed"] = json!(fallas);
    r["verdict"] = json!(if fallas.is_empty() { "PASS" } else { "FALLO" });
    fs::write(raiz.join("harness.json"), serde_json::to_vec_pretty(&r)?)?;
    println!("P8 {} veredicto={} codigo={:?} fallas={fallas:?}", e.nombre, r["verdict"], estado.code());
    drop(rt);
    assert!(fallas.is_empty(), "{}: {fallas:?}", e.nombre);
    Ok(())
}
