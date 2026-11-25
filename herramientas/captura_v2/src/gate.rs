//! Puerta REST unica de la captura v2. Todas las llamadas REST (exchangeInfo, fotos del libro
//! y relleno de aggTrades) pasan por aqui, una a la vez.
//!
//! - Limite: `rateLimits` REQUEST_WEIGHT de exchangeInfo (limite, intervalo, intervalNum).
//! - Peso usado: cabecera `X-MBX-USED-WEIGHT-<n><letra>` de cada respuesta.
//! - Peso de cada tipo de llamada: aprendido como el salto de la cabecera entre dos respuestas
//!   seguidas de la misma ventana (nos quedamos con el mayor visto). Nada escrito a mano.
//! - Balde de fichas: ritmo = limite / ventana * factor; capacidad = una llamada. Una llamada
//!   de peso w espera w / ritmo desde la anterior (formula de la especificacion de Qoder 2.6.3).
//! - Admision: no se lanza si usado + peso >= limite; se espera al cambio de ventana.
//! - 429: pausa max(Retry-After, fin de ventana) y el factor del balde se parte a la mitad;
//!   tras una ventana entera sin 429 el factor se duplica hasta volver a 1.
//! - 418: puerta cerrada hasta reiniciar el proceso (y hasta Retry-After entre reinicios,
//!   guardado en REST_BAN.json). Fila `rest_ban` y archivo ALERTA_REST_418.txt.

use crate::{CaptureResult, Clock, Emitter};
use serde_json::{Value, json};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap},
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, watch};

/// Con la puerta cerrada (418) el shard reintenta tras el tope de espera del kit v1 (3600 s,
/// main.rs:693); la puerta lo vuelve a rechazar mientras siga cerrada.
const CLOSED_COOLDOWN: Duration = Duration::from_secs(3600);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Kind {
    Info,
    Depth,
    AggFill,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Info => "exchange_info",
            Kind::Depth => "depth",
            Kind::AggFill => "agg_trades",
        }
    }
}

#[derive(Debug)]
pub struct Request {
    pub kind: Kind,
    /// 0 = hueco de profundidad, 1 = primera vista, 2 = relleno de aggTrades (spec 2.6.4),
    /// 3 = snapshot periodico R2 (public_l2.rs).
    pub priority: u8,
    pub shard: usize,
    pub symbol: String,
    pub generation: u64,
    pub trigger_u: u64,
    pub from_id: u64,
    pub limit: u64,
    /// true = snapshot periodico R2 (se registra en la fila para que el lector lo sepa).
    pub periodic: bool,
    pub reply: Option<mpsc::UnboundedSender<Reply>>,
}

#[derive(Debug)]
pub enum Reply {
    Depth {
        symbol: String,
        generation: u64,
        ok: bool,
        cooldown: Duration,
    },
    Agg {
        symbol: String,
        from_id: u64,
        ok: bool,
        last_id: Option<u64>,
    },
}

pub struct GateConfig {
    pub rest: &'static str,
    pub root: PathBuf,
    pub agg_limit_max: u64,
    pub default_cooldown_s: u64,
    pub info_refresh: Duration,
}

struct Limits {
    limit: u64,
    window: Duration,
    header: String,
}

pub struct Gate {
    cfg: GateConfig,
    client: reqwest::Client,
    emitter: Emitter,
    clock: Clock,
    limits: Option<Limits>,
    used: Option<(u64, u64)>,
    learned: HashMap<Kind, u64>,
    factor: f64,
    last_sent: Option<Instant>,
    last_cost: u64,
    paused_until: Option<Instant>,
    last_429_window: Option<u64>,
    closed: bool,
    pub info: watch::Sender<Value>,
}

fn now_ms(clock: Clock) -> u64 {
    clock.stamp().map(|s| s.utc_ms).unwrap_or(0)
}

impl Gate {
    pub fn new(
        cfg: GateConfig,
        client: reqwest::Client,
        emitter: Emitter,
        clock: Clock,
    ) -> (Self, watch::Receiver<Value>) {
        let (info, info_rx) = watch::channel(Value::Null);
        (
            Self {
                cfg,
                client,
                emitter,
                clock,
                limits: None,
                used: None,
                learned: HashMap::new(),
                factor: 1.0,
                last_sent: None,
                last_cost: 0,
                paused_until: None,
                last_429_window: None,
                closed: false,
                info,
            },
            info_rx,
        )
    }

    fn ban_file(&self) -> PathBuf {
        self.cfg.root.join("REST_BAN.json")
    }

    fn window_index(&self, utc_ms: u64) -> u64 {
        match &self.limits {
            Some(l) => utc_ms / (l.window.as_millis() as u64).max(1),
            None => 0,
        }
    }

    fn until_window_end(&self) -> Duration {
        let Some(l) = &self.limits else {
            return Duration::from_secs(self.cfg.default_cooldown_s);
        };
        let w = (l.window.as_millis() as u64).max(1);
        let t = now_ms(self.clock);
        Duration::from_millis(w - t % w)
    }

    /// Mayor peso aprendido de cualquier tipo: cota para un tipo cuyo peso aun no conocemos.
    fn cost(&self, kind: Kind) -> u64 {
        self.learned
            .get(&kind)
            .copied()
            .or_else(|| self.learned.values().copied().max())
            .unwrap_or(1)
    }

    /// Cuanto hay que esperar antes de poder lanzar una llamada de este tipo (None = ya).
    fn wait_for(&mut self, kind: Kind) -> Option<Duration> {
        let now = Instant::now();
        if let Some(p) = self.paused_until {
            if now < p {
                return Some(p - now);
            }
            self.paused_until = None;
        }
        let Some(l) = &self.limits else {
            return None;
        };
        let cost = self.cost(kind);
        // Recuperacion del factor: una ventana completa sin 429.
        if let Some(w) = self.last_429_window
            && self.window_index(now_ms(self.clock)) > w + 1
            && self.factor < 1.0
        {
            self.factor = (self.factor * 2.0).min(1.0);
            self.last_429_window = if self.factor < 1.0 {
                Some(self.window_index(now_ms(self.clock)) - 1)
            } else {
                None
            };
        }
        if let Some((window, used)) = self.used
            && window == self.window_index(now_ms(self.clock))
            && used + cost >= l.limit
        {
            return Some(self.until_window_end() + Duration::from_millis(250));
        }
        if let Some(last) = self.last_sent {
            let rate_per_ms = l.limit as f64 / l.window.as_millis() as f64 * self.factor;
            let spacing =
                Duration::from_secs_f64(self.last_cost.max(1) as f64 / rate_per_ms / 1000.0);
            if now < last + spacing {
                return Some(last + spacing - now);
            }
        }
        None
    }

    fn note_used(&mut self, kind: Kind, headers: &reqwest::header::HeaderMap) -> Option<u64> {
        let name = self
            .limits
            .as_ref()
            .map(|l| l.header.clone())
            .unwrap_or_else(|| "x-mbx-used-weight-1m".into());
        let used = headers
            .get(name.as_str())
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())?;
        let window = self.window_index(now_ms(self.clock));
        if let Some((w, before)) = self.used
            && w == window
            && used > before
        {
            let delta = used - before;
            let e = self.learned.entry(kind).or_insert(0);
            if delta > *e {
                *e = delta;
            }
        }
        self.used = Some((window, used));
        Some(used)
    }

    async fn get(
        &mut self,
        kind: Kind,
        url: String,
    ) -> (
        Option<u16>,
        Option<u64>,
        Option<u64>,
        Option<String>,
        Option<String>,
    ) {
        self.last_sent = Some(Instant::now());
        self.last_cost = self.cost(kind);
        match self.client.get(url).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let retry_after = resp
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse().ok());
                let used = self.note_used(kind, resp.headers());
                match resp.text().await {
                    Ok(t) => (Some(status), retry_after, used, Some(t), None),
                    Err(e) => (Some(status), retry_after, used, None, Some(e.to_string())),
                }
            }
            Err(e) => (None, None, None, None, Some(e.to_string())),
        }
    }

    fn emit(&self, kind: &str, row: Value) -> CaptureResult<()> {
        self.emitter.emit_now(kind, 0, row, None)
    }

    /// Respuesta 429/418 comun a todos los tipos. Devuelve true si la puerta debe dejar de
    /// atender esta llamada (y reencolarla si fue 429).
    fn throttle(
        &mut self,
        status: Option<u16>,
        retry_after: Option<u64>,
        body: &Option<String>,
    ) -> CaptureResult<bool> {
        match status {
            Some(429) => {
                let wait =
                    Duration::from_secs(retry_after.unwrap_or(0)).max(self.until_window_end());
                self.paused_until = Some(Instant::now() + wait);
                self.factor /= 2.0;
                self.last_429_window = Some(self.window_index(now_ms(self.clock)));
                self.emit(
                    "rest_429",
                    json!({"retry_after_s":retry_after,"pause_ms":wait.as_millis() as u64,
                    "rate_factor":self.factor,"payload":body}),
                )?;
                Ok(true)
            }
            Some(418) => {
                self.closed = true;
                let until = retry_after.map(|s| now_ms(self.clock) + s * 1000);
                let record = json!({"banned_at_utc_ms":now_ms(self.clock),"retry_after_s":retry_after,
                    "until_utc_ms":until,"payload":body});
                let _ = fs::write(self.ban_file(), serde_json::to_vec_pretty(&record)?);
                let _ = fs::write(
                    self.cfg.root.join("ALERTA_REST_418.txt"),
                    format!(
                        "HTTP 418 de Binance: REST cerrada en la captura v2. Detalle: {record}\n"
                    ),
                );
                eprintln!("ALERTA REST 418: puerta REST cerrada hasta reiniciar; WebSocket sigue");
                self.emit("rest_ban", record)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    async fn fetch_info(&mut self) -> CaptureResult<bool> {
        let url = format!("{}/fapi/v1/exchangeInfo", self.cfg.rest);
        let (status, retry_after, used, body, error) = self.get(Kind::Info, url).await;
        if self.throttle(status, retry_after, &body)? {
            return Ok(false);
        }
        let parsed: Option<Value> = body.as_deref().and_then(|b| serde_json::from_str(b).ok());
        let mut limit_row = Value::Null;
        if let Some(v) = &parsed {
            for rl in v["rateLimits"].as_array().into_iter().flatten() {
                if rl["rateLimitType"] == "REQUEST_WEIGHT" {
                    let (Some(limit), Some(unit), Some(num)) = (
                        rl["limit"].as_u64(),
                        rl["interval"].as_str(),
                        rl["intervalNum"].as_u64(),
                    ) else {
                        continue;
                    };
                    let (secs, letter) = match unit {
                        "SECOND" => (1, 's'),
                        "MINUTE" => (60, 'm'),
                        "HOUR" => (3600, 'h'),
                        "DAY" => (86400, 'd'),
                        _ => continue,
                    };
                    self.limits = Some(Limits {
                        limit,
                        window: Duration::from_secs(secs * num),
                        header: format!("x-mbx-used-weight-{num}{letter}"),
                    });
                    limit_row = rl.clone();
                    break;
                }
            }
        }
        let ok = self.limits.is_some() && status == Some(200);
        let mut statuses = BTreeMap::new();
        if let Some(v) = &parsed {
            for s in v["symbols"].as_array().into_iter().flatten() {
                if let (Some(sym), Some(st)) = (s["symbol"].as_str(), s["status"].as_str()) {
                    statuses.insert(sym.to_string(), st.to_string());
                }
            }
        }
        let summary = json!({"http_status":status,"request_weight_limit":limit_row,"used_weight":used,
            "symbol_status":statuses,"error":error,"learned_weights":self.learned_json()});
        self.emit("exchange_info", summary.clone())?;
        let _ = self.info.send(summary);
        Ok(ok)
    }

    fn learned_json(&self) -> Value {
        let m: BTreeMap<&str, u64> = self.learned.iter().map(|(k, v)| (k.name(), *v)).collect();
        json!(m)
    }

    async fn serve(&mut self, req: Request) -> CaptureResult<Option<Request>> {
        if self.closed {
            Self::fail(req, CLOSED_COOLDOWN);
            return Ok(None);
        }
        let request = self.clock.stamp()?;
        match req.kind {
            Kind::Info => {
                self.fetch_info().await?;
                Ok(None)
            }
            Kind::Depth => {
                // limit=1000: maximo que admite el endpoint (doc de Binance), igual que el kit v1.
                let url = format!(
                    "{}/fapi/v1/depth?symbol={}&limit=1000",
                    self.cfg.rest, req.symbol
                );
                let (status, retry_after, used, body, error) = self.get(Kind::Depth, url).await;
                if self.throttle(status, retry_after, &body)? {
                    return Ok(if self.closed {
                        Self::fail(req, CLOSED_COOLDOWN);
                        None
                    } else {
                        Some(req)
                    });
                }
                let last_id = if status == Some(200) {
                    body.as_deref()
                        .and_then(|b| serde_json::from_str::<Value>(b).ok())
                        .and_then(|v| v["lastUpdateId"].as_u64())
                } else {
                    None
                };
                let error = error.or_else(|| {
                    last_id.is_none().then(|| {
                        format!("REST snapshot missing valid lastUpdateId (HTTP {status:?})")
                    })
                });
                let kind = if last_id.is_some() {
                    "snapshot"
                } else {
                    "snapshot_error"
                };
                self.emitter.emit_now(kind, 0, json!({"shard_request":req.shard,"symbol":req.symbol,
                    "depth_generation":req.generation,"trigger_u":req.trigger_u,"r2_periodic":req.periodic,
                    "request_utc_ms":request.utc_ms,"request_mono_ns":request.mono_ns,
                    "http_status":status,"last_update_id":last_id,"used_weight":used,
                    "learned_weights":self.learned_json(),"payload":body,"error":error}), None)?;
                let cooldown = Duration::from_secs(
                    retry_after
                        .unwrap_or(self.cfg.default_cooldown_s)
                        .clamp(1, 3600),
                );
                if let Some(tx) = &req.reply {
                    let _ = tx.send(Reply::Depth {
                        symbol: req.symbol.clone(),
                        generation: req.generation,
                        ok: last_id.is_some(),
                        cooldown,
                    });
                }
                Ok(None)
            }
            Kind::AggFill => {
                let url = format!(
                    "{}/fapi/v1/aggTrades?symbol={}&fromId={}&limit={}",
                    self.cfg.rest,
                    req.symbol,
                    req.from_id,
                    req.limit.min(self.cfg.agg_limit_max)
                );
                let (status, retry_after, used, body, error) = self.get(Kind::AggFill, url).await;
                if self.throttle(status, retry_after, &body)? {
                    return Ok(if self.closed {
                        Self::fail(req, CLOSED_COOLDOWN);
                        None
                    } else {
                        Some(req)
                    });
                }
                let parsed: Option<Value> = if status == Some(200) {
                    body.as_deref().and_then(|b| serde_json::from_str(b).ok())
                } else {
                    None
                };
                let last_id = parsed
                    .as_ref()
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.last())
                    .and_then(|t| t["a"].as_u64());
                let ok = parsed.as_ref().is_some_and(Value::is_array);
                let kind = if ok { "agg_fill" } else { "agg_fill_error" };
                self.emitter.emit_now(
                    kind,
                    0,
                    json!({"shard_request":req.shard,"symbol":req.symbol,
                    "from_id":req.from_id,"limit":req.limit,"last_agg_id":last_id,
                    "request_utc_ms":request.utc_ms,"request_mono_ns":request.mono_ns,
                    "http_status":status,"used_weight":used,"learned_weights":self.learned_json(),
                    "payload":body,"error":error}),
                    None,
                )?;
                if let Some(tx) = &req.reply {
                    let _ = tx.send(Reply::Agg {
                        symbol: req.symbol.clone(),
                        from_id: req.from_id,
                        ok,
                        last_id,
                    });
                }
                Ok(None)
            }
        }
    }

    fn fail(req: Request, cooldown: Duration) {
        if let Some(tx) = &req.reply {
            let _ = match req.kind {
                Kind::Depth => tx.send(Reply::Depth {
                    symbol: req.symbol,
                    generation: req.generation,
                    ok: false,
                    cooldown,
                }),
                Kind::AggFill => tx.send(Reply::Agg {
                    symbol: req.symbol,
                    from_id: req.from_id,
                    ok: false,
                    last_id: None,
                }),
                Kind::Info => Ok(()),
            };
        }
    }

    pub async fn run(
        mut self,
        mut rx: mpsc::UnboundedReceiver<Request>,
        mut stop: watch::Receiver<bool>,
    ) -> CaptureResult<()> {
        // Veto 418 de una corrida anterior.
        if let Ok(text) = fs::read_to_string(self.ban_file()) {
            let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            match v["until_utc_ms"].as_u64() {
                Some(until) if until > now_ms(self.clock) => {
                    self.paused_until =
                        Some(Instant::now() + Duration::from_millis(until - now_ms(self.clock)));
                    self.emit("rest_ban_inherited", v)?;
                }
                Some(_) => {}
                None => {
                    self.closed = true;
                    self.emit("rest_ban_inherited", json!({"detail":v,"note":"418 sin Retry-After: REST cerrada hasta que el operador borre REST_BAN.json"}))?;
                }
            }
        }
        let mut heap: BinaryHeap<Reverse<(u8, u64, usize)>> = BinaryHeap::new();
        let mut store: HashMap<usize, Request> = HashMap::new();
        let mut order = 0u64;
        let mut next_id = 0usize;
        let mut pending_depth: HashMap<String, usize> = HashMap::new();
        let mut info_ok = false;
        let mut info_due = Instant::now();
        let mut info_backoff = Duration::from_secs(self.cfg.default_cooldown_s);
        loop {
            if *stop.borrow() {
                return Ok(());
            }
            if !self.closed && Instant::now() >= info_due {
                let wait = self.wait_for(Kind::Info);
                if wait.is_none() {
                    info_ok = self.fetch_info().await? || info_ok;
                    if self.limits.is_some() && info_ok {
                        info_due = Instant::now() + self.cfg.info_refresh;
                        info_backoff = Duration::from_secs(self.cfg.default_cooldown_s);
                    } else {
                        info_due = Instant::now() + info_backoff;
                        info_backoff = (info_backoff * 2).min(Duration::from_secs(300));
                    }
                    continue;
                }
            }
            let mut sleep_for = Duration::from_secs(1);
            if self.limits.is_some()
                && let Some(Reverse((_, _, id))) = heap.peek().copied()
            {
                let kind = store[&id].kind;
                match self.wait_for(kind) {
                    None => {
                        heap.pop();
                        let req = store.remove(&id).expect("request in store");
                        if req.kind == Kind::Depth {
                            pending_depth.remove(&req.symbol);
                        }
                        if let Some(back) = self.serve(req).await? {
                            order += 1;
                            heap.push(Reverse((back.priority, order, id)));
                            if back.kind == Kind::Depth {
                                pending_depth.insert(back.symbol.clone(), id);
                            }
                            store.insert(id, back);
                        }
                        continue;
                    }
                    Some(w) => sleep_for = w.min(Duration::from_secs(1)),
                }
            }
            tokio::select! {
                _ = stop.changed() => {}
                got = rx.recv() => {
                    let Some(req) = got else { return Ok(()); };
                    if self.closed {
                        Self::fail(req, CLOSED_COOLDOWN);
                        continue;
                    }
                    // Una foto pendiente por simbolo: la mas nueva reemplaza a la vieja.
                    if req.kind == Kind::Depth
                        && let Some(old) = pending_depth.get(&req.symbol).copied()
                        && let Some(slot) = store.get_mut(&old)
                    {
                        let prio = slot.priority.min(req.priority);
                        *slot = req;
                        slot.priority = prio;
                        continue;
                    }
                    next_id += 1;
                    order += 1;
                    heap.push(Reverse((req.priority, order, next_id)));
                    if req.kind == Kind::Depth {
                        pending_depth.insert(req.symbol.clone(), next_id);
                    }
                    store.insert(next_id, req);
                }
                _ = tokio::time::sleep(sleep_for) => {}
            }
        }
    }
}
