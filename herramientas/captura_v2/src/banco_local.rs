//! Banco de aceptación. Todos los sockets de prueba usan loopback.
use super::*;
use flate2::read::GzDecoder;
use std::io::BufRead;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::JoinHandle,
};

type TestSocket = WebSocketStream<TcpStream>;
const SYMBOL: &str = "FSTUSDT";
const OTHER: &str = "SNDUSDT";
const DEPTH: &str = "fstusdt@depth@100ms";
const WAIT: Duration = Duration::from_secs(12);
static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);
impl TestDir {
    fn new() -> CaptureResult<Self> {
        let name = format!(
            ".banco-local-{}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        );
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .canonicalize()?
            .join(name);
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let path = &self.0;
        if path.canonicalize().ok().as_ref() == Some(path)
            && path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(".banco-local-"))
        {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn emitter(
    queue: ArchiveQueue,
    clock: Clock,
    route: &'static str,
    shard: Option<usize>,
) -> Emitter {
    Emitter {
        queue,
        clock,
        route,
        shard,
        seq: Arc::new(AtomicU64::new(0)),
    }
}

fn frame(symbol: &str, tail: &str, data: Value) -> String {
    json!({"stream":format!("{}@{tail}",symbol.to_ascii_lowercase()),"data":data}).to_string()
}
fn depth(u: u64, pu: u64) -> String {
    frame(
        SYMBOL,
        "depth@100ms",
        json!({"e":"depthUpdate","s":SYMBOL,"U":u,"u":u,"pu":pu,"b":[],"a":[]}),
    )
}
fn ticker(symbol: &str, u: u64) -> String {
    frame(
        symbol,
        "bookTicker",
        json!({"e":"bookTicker","s":symbol,"u":u,"b":"123.0","B":"2","a":"123.1","A":"3"}),
    )
}

fn exchange_info() -> Value {
    json!({
        "rateLimits":[{"rateLimitType":"REQUEST_WEIGHT","interval":"SECOND","intervalNum":1,"limit":1000}],
        "symbols":[{"symbol":SYMBOL,"status":"TRADING"},{"symbol":OTHER,"status":"TRADING"}]
    })
}

async fn local_rest(
    listener: TcpListener,
    mut respond: impl FnMut(&str) -> (&'static str, Option<u64>, Value),
) -> CaptureResult<()> {
    let mut used = 0;
    loop {
        let (mut socket, peer) = listener.accept().await?;
        assert!(peer.ip().is_loopback());
        let mut header = Vec::new();
        let mut chunk = [0; 1024];
        while !header.ends_with(b"\r\n\r\n") && header.len() < 8192 {
            let n = socket.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            header.extend_from_slice(&chunk[..n]);
        }
        let request = std::str::from_utf8(&header)?
            .lines()
            .next()
            .unwrap_or_default();
        let (status, retry_after, body) = respond(request);
        let body = body.to_string();
        let retry_after = retry_after
            .map(|seconds| format!("Retry-After: {seconds}\r\n"))
            .unwrap_or_default();
        used += 1;
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nX-MBX-USED-WEIGHT-1S: {used}\r\n{retry_after}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await?;
        socket.shutdown().await?;
    }
}

struct Bank {
    ws: TcpListener,
    rest: JoinHandle<CaptureResult<()>>,
    shard: JoinHandle<CaptureResult<()>>,
    gate: JoinHandle<CaptureResult<()>>,
    stop: watch::Sender<bool>,
    rx: ArchiveReceiver,
    fatal: mpsc::UnboundedReceiver<String>,
    book: Arc<Mutex<ShardBook>>,
    rows: Vec<(Value, Option<Cont>)>,
    snapshots: Arc<AtomicU64>,
    _dir: TestDir,
}
impl Drop for Bank {
    fn drop(&mut self) {
        self.shard.abort();
        self.gate.abort();
        self.rest.abort();
    }
}
impl Bank {
    async fn start(rotate_after: Duration) -> CaptureResult<Self> {
        let dir = TestDir::new()?;
        let ws = TcpListener::bind(("127.0.0.1", 0)).await?;
        let http = TcpListener::bind(("127.0.0.1", 0)).await?;
        let rest_url: &'static str =
            Box::leak(format!("http://{}", http.local_addr()?).into_boxed_str());
        let venue = Venue {
            name: "local",
            ws: "127.0.0.1",
            rest: rest_url,
        };
        let clock = Clock(Instant::now());
        let (queue, rx) = ArchiveQueue::new(256, 2 << 20);
        let (stop, stop_rx) = watch::channel(false);
        let (fatal_tx, fatal) = mpsc::unbounded_channel();
        let (gate_tx, gate_rx) = mpsc::unbounded_channel();
        let snapshots = Arc::new(AtomicU64::new(0));
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(WAIT)
            .build()?;
        let (gate, info_rx) = gate::Gate::new(
            gate::GateConfig {
                rest: rest_url,
                root: dir.0.clone(),
                agg_limit_max: AGG_LIMIT_MAX,
                default_cooldown_s: 1,
                info_refresh: Duration::from_secs(60),
            },
            client,
            emitter(queue.clone(), clock, "rest", None),
            clock,
        );
        let book = Arc::new(Mutex::new(ShardBook::default()));
        let ctx = ShardCtx {
            venue,
            route: "public",
            index: 0,
            total: 1,
            symbols: vec![SYMBOL.into(), OTHER.into()],
            url: format!(
                "ws://{}/public/stream?streams=fstusdt@bookTicker/fstusdt@depth@100ms/sndusdt@bookTicker/sndusdt@depth@100ms",
                ws.local_addr()?
            ),
            emitter: emitter(queue, clock, "public", Some(0)),
            clock,
            book: Arc::clone(&book),
            gate: gate_tx,
            stop: stop_rx.clone(),
            fatal: fatal_tx,
            rotate_after,
            connections: Arc::new(AtomicU64::new(0)),
            gate_info: info_rx,
            r2_max_gap: public_l2::R2_MAX_GAP_DEFAULT,
            r2_symbols_total: 2,
            r2_rank: HashMap::from([(SYMBOL.to_owned(), 0), (OTHER.to_owned(), 1)]),
        };
        let rest_snapshots = Arc::clone(&snapshots);
        let rest = tokio::spawn(local_rest(http, move |request| {
            if request.starts_with("GET /fapi/v1/exchangeInfo ") {
                return ("200 OK", None, exchange_info());
            }
            assert!(
                request.starts_with(&format!("GET /fapi/v1/depth?symbol={SYMBOL}&limit=1000 ")),
                "{request}"
            );
            let id = if rest_snapshots.fetch_add(1, Ordering::Relaxed) == 0 {
                100
            } else {
                110
            };
            ("200 OK", None, json!({"lastUpdateId":id,"bids":[],"asks":[]}))
        }));
        let gate = tokio::spawn(gate.run(gate_rx, stop_rx));
        let shard = tokio::spawn(ctx.run());
        Ok(Self {
            ws,
            rest,
            shard,
            gate,
            stop,
            rx,
            fatal,
            book,
            rows: Vec::new(),
            snapshots,
            _dir: dir,
        })
    }

    async fn socket(&mut self, id: u64) -> CaptureResult<TestSocket> {
        let (socket, peer) = time::timeout(WAIT, self.ws.accept()).await??;
        assert!(peer.ip().is_loopback());
        let ws = time::timeout(WAIT, tokio_tungstenite::accept_async(socket)).await??;
        self.until("socket_open", |r| {
            r["kind"] == "socket_open" && r["connection"] == id
        })
        .await?;
        Ok(ws)
    }

    async fn until(
        &mut self,
        description: &str,
        matches: impl Fn(&Value) -> bool,
    ) -> CaptureResult<Value> {
        time::timeout(WAIT, async {
            loop {
                self.drain();
                if let Some((row, _)) = self.rows.iter().find(|(r, _)| matches(r)) {
                    return Ok(row.clone());
                }
                tokio::select! {
                    Some(error) = self.fatal.recv() => return Err(io::Error::other(error).into()),
                    next = self.rx.rx.recv() => match next {
                        Some(QueuedRow { row, cont, .. }) => self.rows.push((row, cont)),
                        None => return Err(io::Error::other("capture queue closed in local test").into()),
                    },
                }
            }
        }).await.map_err(|_| io::Error::other(format!("local test timed out waiting for {description}")))?
    }

    fn drain(&mut self) {
        while let Ok(QueuedRow { row, cont, .. }) = self.rx.try_recv() {
            self.rows.push((row, cont));
        }
    }

    async fn saw_on_backup(&self, stream: &str) -> CaptureResult<()> {
        time::timeout(WAIT, async {
            while !lock(&self.book)
                .seen
                .get(&2)
                .is_some_and(|seen| seen.contains_key(stream))
            {
                time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await?;
        Ok(())
    }

    async fn finish(&mut self) -> CaptureResult<()> {
        self.stop.send(true)?;
        time::timeout(WAIT, &mut self.shard).await???;
        time::timeout(WAIT, &mut self.gate).await???;
        self.drain();
        assert!(self.fatal.try_recv().is_err());
        Ok(())
    }

    fn values(&self) -> impl Iterator<Item = &Value> {
        self.rows.iter().map(|(r, _)| r)
    }

    async fn expect_payload(&mut self, payload: &str) -> CaptureResult<Value> {
        self.until(payload, |r| r["kind"] == "frame" && r["payload"] == payload)
            .await
    }
}

struct RestGateBank {
    dir: TestDir,
    stop: watch::Sender<bool>,
    tx: mpsc::UnboundedSender<gate::Request>,
    rx: ArchiveReceiver,
    gate: JoinHandle<CaptureResult<()>>,
    rest: JoinHandle<CaptureResult<()>>,
    requests: Arc<Mutex<Vec<(String, Instant)>>>,
}

impl Drop for RestGateBank {
    fn drop(&mut self) {
        self.gate.abort();
        self.rest.abort();
    }
}

impl RestGateBank {
    async fn start(first_status: &'static str, retry_after: Duration) -> CaptureResult<Self> {
        let dir = TestDir::new()?;
        let http = TcpListener::bind(("127.0.0.1", 0)).await?;
        let rest_url = Box::leak(format!("http://{}", http.local_addr()?).into_boxed_str());
        let clock = Clock(Instant::now());
        let (queue, rx) = ArchiveQueue::new(32, 128 << 10);
        let (stop, stop_rx) = watch::channel(false);
        let (tx, gate_rx) = mpsc::unbounded_channel();
        let (gate, _info) = gate::Gate::new(
            gate::GateConfig {
                rest: rest_url,
                root: dir.0.clone(),
                agg_limit_max: AGG_LIMIT_MAX,
                default_cooldown_s: 1,
                info_refresh: Duration::from_secs(60),
            },
            reqwest::Client::builder().no_proxy().timeout(WAIT).build()?,
            emitter(queue, clock, "rest", None),
            clock,
        );
        let requests = Arc::new(Mutex::new(Vec::new()));
        let trace = Arc::clone(&requests);
        let mut rejected = false;
        let rest = tokio::spawn(local_rest(http, move |request| {
            let path = request
                .strip_prefix("GET ")
                .and_then(|get| get.split_once(' '))
                .map(|(path, _)| path)
                .expect("expected HTTP GET from local gate");
            trace.lock().unwrap().push((path.to_owned(), Instant::now()));
            if path == "/fapi/v1/exchangeInfo" {
                return ("200 OK", None, exchange_info());
            }
            if !rejected {
                rejected = true;
                return (first_status, Some(retry_after.as_secs()), json!({}));
            }
            ("200 OK", None, json!({"lastUpdateId":100,"bids":[],"asks":[]}))
        }));
        let gate = tokio::spawn(gate.run(gate_rx, stop_rx));
        Ok(Self { dir, stop, tx, rx, gate, rest, requests })
    }

    fn depth(
        &self,
        symbol: &str,
        generation: u64,
    ) -> CaptureResult<mpsc::UnboundedReceiver<gate::Reply>> {
        let (reply, rx) = mpsc::unbounded_channel();
        self.tx.send(gate::Request {
            kind: gate::Kind::Depth,
            priority: 3,
            shard: 0,
            symbol: symbol.to_owned(),
            generation,
            trigger_u: 100,
            from_id: 0,
            limit: 0,
            periodic: true,
            reply: Some(reply),
        })?;
        Ok(rx)
    }

    async fn row(&mut self, kind: &str) -> CaptureResult<Value> {
        time::timeout(WAIT, async {
            while let Some(QueuedRow { row, .. }) = self.rx.rx.recv().await {
                if row["kind"] == kind {
                    return Ok(row);
                }
            }
            Err(io::Error::other(format!("REST test archive closed before {kind}")).into())
        })
        .await?
    }

    async fn finish(&mut self) -> CaptureResult<()> {
        self.stop.send(true)?;
        time::timeout(WAIT, &mut self.gate).await???;
        assert!(!self.rest.is_finished(), "local REST fixture stopped accepting");
        Ok(())
    }
}

async fn depth_reply(
    rx: &mut mpsc::UnboundedReceiver<gate::Reply>,
    symbol: &str,
    generation: u64,
    expected_ok: bool,
) -> CaptureResult<Duration> {
    let Some(gate::Reply::Depth {
        symbol: got_symbol,
        generation: got_generation,
        ok,
        cooldown,
    }) = time::timeout(WAIT, rx.recv()).await?
    else {
        return Err(io::Error::other("expected Depth reply from local gate").into());
    };
    assert_eq!(got_symbol, symbol);
    assert_eq!(got_generation, generation);
    assert_eq!(ok, expected_ok);
    Ok(cooldown)
}

#[tokio::test]
async fn local_r2_gate_429_requeues_without_requests_during_pause() -> CaptureResult<()> {
    let pause = Duration::from_secs(2);
    let mut bank = RestGateBank::start("429 Too Many Requests", pause).await?;
    let mut first = bank.depth(SYMBOL, 1)?;
    let row = bank.row("rest_429").await?;
    assert_eq!(row["retry_after_s"], pause.as_secs());
    assert_eq!(row["pause_ms"], pause.as_millis() as u64);
    assert_eq!(row["rate_factor"].as_f64(), Some(0.5));
    let mut queued = bank.depth(OTHER, 2)?;

    depth_reply(&mut first, SYMBOL, 1, true).await?;
    depth_reply(&mut queued, OTHER, 2, true).await?;
    bank.finish().await?;

    let requests = bank.requests.lock().unwrap();
    let paths: Vec<_> = requests.iter().map(|(path, _)| path.as_str()).collect();
    let first_path = format!("/fapi/v1/depth?symbol={SYMBOL}&limit=1000");
    let queued_path = format!("/fapi/v1/depth?symbol={OTHER}&limit=1000");
    assert_eq!(
        paths,
        [
            "/fapi/v1/exchangeInfo",
            first_path.as_str(),
            first_path.as_str(),
            queued_path.as_str(),
        ]
    );
    // The server timestamps the request before sending the 429.
    let during_pause = requests[2..]
        .iter()
        .filter(|(_, received)| received.duration_since(requests[1].1) < pause)
        .count();
    assert_eq!(during_pause, 0, "REST requests during HTTP 429 pause: {requests:?}");
    println!(
        "local REST 429 requests_during_pause={during_pause} first_retry_ms={} requests_total={}",
        requests[2].1.duration_since(requests[1].1).as_millis(),
        requests.len()
    );
    Ok(())
}

#[tokio::test]
async fn local_r2_gate_418_rejects_requests_without_http() -> CaptureResult<()> {
    let retry_after = Duration::from_secs(120);
    let mut bank = RestGateBank::start("418 I'm a teapot", retry_after).await?;
    let started_utc_ms = Clock(Instant::now()).stamp()?.utc_ms;
    let mut first = bank.depth(SYMBOL, 1)?;
    let row = bank.row("rest_ban").await?;
    assert_eq!(row["retry_after_s"], retry_after.as_secs());
    let until = row["until_utc_ms"].as_u64().unwrap();
    let banned_at = row["banned_at_utc_ms"].as_u64().unwrap();
    let retry_ms = retry_after.as_millis() as u64;
    assert!((started_utc_ms + retry_ms..=banned_at + retry_ms).contains(&until));
    let mut rejected = bank.depth(OTHER, 2)?;
    let cooldown = depth_reply(&mut first, SYMBOL, 1, false).await?;
    assert_eq!(cooldown, Duration::from_secs(3600));
    assert_eq!(depth_reply(&mut rejected, OTHER, 2, false).await?, cooldown);
    bank.finish().await?;

    let requests = bank.requests.lock().unwrap();
    let paths: Vec<_> = requests.iter().map(|(path, _)| path.as_str()).collect();
    let first_path = format!("/fapi/v1/depth?symbol={SYMBOL}&limit=1000");
    assert_eq!(paths, ["/fapi/v1/exchangeInfo", first_path.as_str()]);
    let ban: Value = serde_json::from_reader(File::open(bank.dir.0.join("REST_BAN.json"))?)?;
    assert_eq!(ban["until_utc_ms"], until);
    assert!(fs::read_to_string(bank.dir.0.join("ALERTA_REST_418.txt"))?.contains("HTTP 418"));
    println!(
        "local REST 418 requests_after_ban={} cooldown_s={} requests_total={}",
        requests.len() - 2, cooldown.as_secs(), requests.len()
    );
    Ok(())
}

async fn publish(ws: &mut TestSocket, text: String) -> CaptureResult<()> {
    time::timeout(WAIT, ws.send(Message::Text(text))).await??;
    Ok(())
}

fn payload_sha<'a>(payloads: impl Iterator<Item = &'a str>) -> String {
    let mut digest = Sha256::new();
    for payload in payloads {
        digest.update(payload.as_bytes());
        digest.update(b"\n");
    }
    format!("{:x}", digest.finalize())
}

fn verify_frames<'a>(rows: impl IntoIterator<Item = &'a Value>, expected: &[String]) {
    let mut actual_by_stream = BTreeMap::<String, Vec<&str>>::new();
    let mut expected_by_stream = BTreeMap::<String, Vec<&str>>::new();
    let stream = |payload: &str| {
        serde_json::from_str::<Value>(payload).unwrap()["stream"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    for payload in expected {
        expected_by_stream
            .entry(stream(payload))
            .or_default()
            .push(payload);
    }
    for row in rows {
        if row["kind"] == "frame" {
            let payload = row["payload"].as_str().unwrap();
            actual_by_stream
                .entry(stream(payload))
                .or_default()
                .push(payload);
        }
    }
    assert_eq!(actual_by_stream, expected_by_stream);
    for (stream, expected) in expected_by_stream {
        let actual = &actual_by_stream[&stream];
        let actual_sha = payload_sha(actual.iter().copied());
        assert_eq!(actual_sha, payload_sha(expected.iter().copied()));
        println!(
            "local {stream} frames={} payload_sha256={actual_sha}",
            actual.len()
        );
    }
}

async fn seed(bank: &mut Bank, primary: &mut TestSocket) -> CaptureResult<Vec<String>> {
    let startup = bank.until("R2 startup plan", |r| {
        r["kind"] == "r2_plan" && r["reason"] == "arranque"
    }).await?;
    assert_eq!(startup["symbols"], 2);
    let resolved = bank.until("R2 plan with local exchangeInfo", |r| {
        r["kind"] == "r2_plan" && r["limit"] == 1000
    }).await?;
    assert_eq!(resolved["window_s"], 1);
    assert_eq!(resolved["interval_ms"], 300_000);
    let expected = vec![depth(100, 99), ticker(SYMBOL, 1000), ticker(OTHER, 1000)];
    for payload in &expected {
        publish(primary, payload.clone()).await?;
    }
    bank.until("snapshot generation 1", |r| {
        r["kind"] == "snapshot" && r["depth_generation"] == 1 && r["last_update_id"] == 100
    })
    .await?;
    bank.expect_payload(&expected[1]).await?;
    Ok(expected)
}

fn verify_no_l2_gap(bank: &Bank) {
    assert_eq!(bank.snapshots.load(Ordering::Relaxed), 1);
    assert_eq!(bank.values().filter(|r| r["kind"] == "snapshot").count(), 1);
    assert_eq!(
        bank.values().filter(|r| r["kind"] == "depth_gap").count(),
        0
    );
    assert_eq!(
        bank.values().filter(|r| r["kind"] == "socket_gap").count(),
        0
    );
}

#[tokio::test]
async fn local_resets_alternating_roles_preserve_frames() -> CaptureResult<()> {
    let mut bank = Bank::start(Duration::from_secs(3600)).await?;
    let mut primary = bank.socket(1).await?;
    let mut backup = bank.socket(2).await?;
    let mut expected = seed(&mut bank, &mut primary).await?;

    publish(&mut backup, ticker(SYMBOL, 1000)).await?;
    publish(&mut backup, ticker(OTHER, 1000)).await?;
    let early = depth(102, 101);
    publish(&mut backup, early.clone()).await?;
    bank.saw_on_backup(DEPTH).await?;
    bank.drain();
    assert!(
        !bank
            .values()
            .any(|r| r["kind"] == "frame" && r["payload"] == early)
    );
    let predecessor = depth(101, 100);
    publish(&mut primary, predecessor.clone()).await?;
    expected.extend([predecessor.clone(), early.clone()]);
    bank.expect_payload(&early).await?;
    publish(&mut primary, early.clone()).await?;
    publish(&mut backup, predecessor).await?;

    drop(primary);
    bank.until("promotion of backup after primary reset", |r| {
        r["kind"] == "backup_promoted" && r["connection"] == 2 && r["from_connection"] == 1
    })
    .await?;
    // The promoted socket must deliver without waiting for the next handshake.
    let first_after = depth(103, 102);
    publish(&mut backup, first_after.clone()).await?;
    expected.push(first_after.clone());
    bank.expect_payload(&first_after).await?;
    let replacement = bank.socket(3).await?;

    drop(replacement);
    let mut next_backup = bank.socket(4).await?;
    let second_after = depth(104, 103);
    let b1001 = ticker(SYMBOL, 1001);
    let b1002 = ticker(SYMBOL, 1002);
    for payload in [&second_after, &b1002] {
        publish(&mut next_backup, payload.clone()).await?;
    }
    for payload in [&second_after, &b1001, &b1002] {
        publish(&mut backup, payload.clone()).await?;
    }
    expected.extend([second_after.clone(), b1001, b1002.clone()]);
    bank.expect_payload(&b1002).await?;
    bank.finish().await?;
    verify_no_l2_gap(&bank);
    assert_eq!(
        bank.values()
            .filter(|r| r["kind"] == "backup_promoted")
            .count(),
        1
    );
    assert!(
        bank.values()
            .filter(|r| r["kind"] == "frame" && r["stream"] == DEPTH)
            .all(|r| r["depth_generation"] == 1)
    );
    verify_frames(bank.values(), &expected);
    Ok(())
}

#[tokio::test]
async fn local_dual_reset_reports_gap_and_reanchors() -> CaptureResult<()> {
    let mut bank = Bank::start(Duration::from_secs(3600)).await?;
    let mut primary = bank.socket(1).await?;
    let mut backup = bank.socket(2).await?;
    let mut expected = seed(&mut bank, &mut primary).await?;
    publish(&mut backup, depth(100, 99)).await?;
    bank.saw_on_backup(DEPTH).await?;
    drop(primary);
    drop(backup);
    bank.until("dual socket gap", |r| {
        r["kind"] == "socket_gap" && r["coverage_lost"] == true
    })
    .await?;

    let mut recovered = bank.socket(3).await?;
    let _warm_replacement = bank.socket(4).await?;
    let first = depth(110, 100);
    publish(&mut recovered, first.clone()).await?;
    let gap = bank
        .until("depth gap generation 2", |r| {
            r["kind"] == "depth_gap" && r["depth_generation"] == 2
        })
        .await?;
    assert_eq!(gap["previous_u"], 100);
    assert_eq!(gap["pu"], 100);
    assert_eq!(gap["reason"], "both L2 connections lost");
    bank.until("snapshot generation 2", |r| {
        r["kind"] == "snapshot" && r["depth_generation"] == 2 && r["last_update_id"] == 110
    })
    .await?;
    let next = depth(111, 110);
    publish(&mut recovered, next.clone()).await?;
    let canonical = bank.expect_payload(&next).await?;
    assert_eq!(canonical["depth_generation"], 2);
    expected.extend([first, next]);
    bank.finish().await?;
    assert_eq!(bank.snapshots.load(Ordering::Relaxed), 2);
    assert_eq!(
        bank.values().filter(|r| r["kind"] == "depth_gap").count(),
        1
    );
    verify_frames(bank.values(), &expected);
    Ok(())
}

fn verify_gzip_files(dir: &Path, expected_count: u64) -> CaptureResult<Vec<Value>> {
    let mut archived = Vec::<Value>::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .ends_with(".manifest.json")
        {
            continue;
        }
        let manifest: Value = serde_json::from_reader(File::open(&path)?)?;
        let file = dir.join(manifest["file"].as_str().unwrap());
        let (bytes, sha) = digest(&file)?;
        assert_eq!(manifest["schema"], 2);
        assert_eq!(manifest["bytes"], bytes);
        assert_eq!(manifest["sha256"], sha);
        let reader = BufReader::new(GzDecoder::new(File::open(file)?));
        let before = archived.len();
        for row in reader.lines() {
            let row: Value = serde_json::from_str(&row?)?;
            let timestamp = row["recv_utc_ms"].as_u64().unwrap();
            assert_eq!(manifest["utc_hour_ms"], timestamp / HOUR_MS * HOUR_MS);
            archived.push(row);
        }
        assert_eq!(manifest["rows"], (archived.len() - before) as u64);
    }
    archived.sort_by_key(|row| row["ingest_seq"].as_u64().unwrap());
    assert_eq!(archived.len() as u64, expected_count);
    for (i, row) in archived.iter().enumerate() {
        assert_eq!(row["ingest_seq"], (i + 1) as u64);
    }
    Ok(archived)
}

#[tokio::test]
async fn local_rotation_and_hour_boundary_keep_payload_and_seals() -> CaptureResult<()> {
    let mut bank = Bank::start(Duration::from_secs(2)).await?;
    let mut primary = bank.socket(1).await?;
    let mut backup = bank.socket(2).await?;
    let mut expected = seed(&mut bank, &mut primary).await?;
    for payload in &expected {
        publish(&mut backup, payload.clone()).await?;
    }
    bank.until("planned rotation to warm backup", |r| {
        r["kind"] == "rotation_ready" && r["connection"] == 2
    })
    .await?;
    let _warm_replacement = bank.socket(3).await?;
    let next = depth(101, 100);
    let next_book = ticker(SYMBOL, 1001);
    publish(&mut backup, next.clone()).await?;
    publish(&mut backup, next_book.clone()).await?;
    bank.expect_payload(&next_book).await?;
    expected.extend([next.clone(), next_book.clone()]);
    bank.finish().await?;
    verify_no_l2_gap(&bank);

    // Re-use the rows from the sockets in the real writer, with a controlled UTC clock.
    let output = TestDir::new()?;
    let (ctl, queue, rx) = shared_archive(128, 2 << 20);
    let (seal_tx, seal_rx) = blocking_channel::sync_channel(SEAL_QUEUE_CAPACITY);
    let seal_path = output.0.clone();
    let sealer = ArchiveWorker::spawn_local("local-seal", move || {
        Ok(seal_segments(&RealDisk, &seal_path, seal_rx, None, None))
    })?;
    let write_path = output.0.clone();
    let writer = spawn_shared_writer("local-archive", &ctl, write_path, rx, seal_tx)?;
    let mut after_roll = false;
    for (r, cont) in &bank.rows {
        let mut row = r.clone();
        after_roll |= row["kind"] == "rotation_ready";
        let ts = if row["payload"] == next_book {
            HOUR_MS + ROTATION_GRACE_MS + 1
        } else if after_roll {
            HOUR_MS + 1
        } else {
            HOUR_MS - 1
        };
        row["recv_utc_ms"] = json!(ts);
        queue.offer(row, cont.clone())?;
    }
    drop(queue);
    // Parada sana en el orden de produccion: soltar la cola y cortar la admision.
    ctl.admision.cortar(salidafallo::Motivo::Sana);
    time::timeout(WAIT, writer.finish(false)).await??;
    let report = time::timeout(WAIT, sealer.finish(false)).await??;
    assert_eq!(report.segments, 2);
    assert!(report.error.is_none(), "{:?}", report.error);
    let rows = verify_gzip_files(&output.0, bank.rows.len() as u64)?;
    assert!(
        rows.iter()
            .any(|r| r["payload"] == next && r["recv_utc_ms"] == HOUR_MS + 1)
    );
    assert!(
        rows.iter()
            .any(|r| r["payload"] == expected[0] && r["recv_utc_ms"] == HOUR_MS - 1)
    );
    verify_frames(&rows, &expected);
    println!(
        "local rollover seals={} rows={}",
        report.segments,
        rows.len()
    );
    Ok(())
}

fn sample_row(ordinal: u64, time_ms: u64) -> Value {
    json!({"kind":"sample","route":"local","seq":ordinal,"recv_utc_ms":time_ms})
}

/// D29: cola y escritor reales con UNA sola `Admision`, la del `Control` de prueba, como en
/// produccion (`ArchiveQueue::derivada` con `Arc::clone(&ctl.admision)`). Asi `offer` (al salir
/// de `Vuelo`) y el corte despiertan al mismo escritor que se registro en `archive`.
fn shared_archive(cap: usize, byte_limit: usize) -> (Arc<Control>, ArchiveQueue, ArchiveReceiver) {
    let (ctl, _, _) = salidafallo::tests::control_de_prueba(salidafallo::PLAZO_FATAL);
    let (queue, rx) = ArchiveQueue::with_admission(cap, byte_limit, Arc::clone(&ctl.admision));
    (ctl, queue, rx)
}

/// Escritor real (`archive`) con el mismo `Control` que la cola de `shared_archive`.
fn spawn_shared_writer(
    name: &str,
    ctl: &Arc<Control>,
    dir: PathBuf,
    rx: ArchiveReceiver,
    seal: blocking_channel::SyncSender<Segment<File>>,
) -> CaptureResult<ArchiveWorker<CaptureResult<()>>> {
    let writer_ctl = Arc::clone(ctl);
    ArchiveWorker::spawn(
        name,
        Arc::clone(ctl),
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        move || {
            let outcome = archive(&RealDisk, &dir, rx, seal, &writer_ctl);
            match outcome.failures.first() {
                None => Ok(()),
                Some(f) => Err(io::Error::other(f.to_string()).into()),
            }
        },
    )
}

async fn stall_sealer(overload: bool) -> CaptureResult<()> {
    let dir = TestDir::new()?;
    let cap = if overload { 3 } else { 8 };
    let (ctl, queue, rx) = shared_archive(cap, 256 << 10);
    let audit = Arc::clone(&queue.audit);
    // Fill the seal handoff through the real writer while the sealer is paused.
    let (seal_tx, seal_rx) = blocking_channel::sync_channel(1);
    let (release, delayed) = blocking_channel::channel::<()>();
    let seal_path = dir.0.clone();
    let sealer = ArchiveWorker::spawn_local("local-seal-delayed", move || {
        delayed.recv()?;
        Ok(seal_segments(&RealDisk, &seal_path, seal_rx, None, None))
    })?;
    let write_path = dir.0.clone();
    let writer = spawn_shared_writer("local-writer-delayed", &ctl, write_path, rx, seal_tx)?;
    for ordinal in 1..=3 {
        let ts = (ordinal - 1) * HOUR_MS + ROTATION_GRACE_MS + 1;
        queue.offer(sample_row(ordinal, ts), None)?;
        time::timeout(WAIT, async {
            while audit.dequeued.load(Ordering::Relaxed) < ordinal as usize {
                time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await?;
    }
    for ordinal in 4..=6 {
        queue.offer(
            sample_row(ordinal, 2 * HOUR_MS + ROTATION_GRACE_MS + ordinal),
            None,
        )?;
    }
    assert_eq!(audit.peak_rows.load(Ordering::Relaxed), 3);
    if overload {
        let err = queue
            .offer(sample_row(7, 2 * HOUR_MS + ROTATION_GRACE_MS + 7), None)
            .unwrap_err();
        assert!(err.to_string().contains("row_capacity"), "{err}");
        assert_eq!(audit.failure_reason(), Some("row_capacity"));
        assert_eq!(audit.stop_reason("reader_error", 0), "archive_queue_full");
        // Como el cierre fatal de produccion: cortar la admision que comparte el escritor.
        ctl.admision.cortar(salidafallo::Motivo::Fatal);
    } else {
        assert_eq!(audit.failure_reason(), None);
    }
    time::sleep(Duration::from_millis(40)).await;
    release.send(())?;
    if overload {
        time::timeout(WAIT, async {
            while audit.dequeued.load(Ordering::Relaxed) < 6 {
                time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await?;
        assert!(queue.offer(sample_row(8, 2 * HOUR_MS + 1), None).is_err());
        let rejected = audit.rejected.lock().unwrap();
        assert_eq!(rejected[0]["seq"], 7);
        assert_eq!(rejected[0]["reason"], "row_capacity");
    }
    drop(queue);
    if !overload {
        // Parada sana en el orden de produccion: soltar la cola y cortar la admision.
        ctl.admision.cortar(salidafallo::Motivo::Sana);
    }
    time::timeout(WAIT, writer.finish(false)).await??;
    let report = time::timeout(WAIT, sealer.finish(false)).await??;
    assert!(report.error.is_none(), "{:?}", report.error);
    assert_eq!(report.segments, 3);
    let rows = verify_gzip_files(&dir.0, 6)?;
    let accepted: Vec<_> = rows.iter().map(|r| r["seq"].as_u64().unwrap()).collect();
    assert_eq!(accepted, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(audit.accepted.load(Ordering::Relaxed), 6);
    let age = audit.max_dequeue_age_ns.load(Ordering::Relaxed);
    assert!(age >= 10_000_000, "queue age {age}");
    let metrics = report.telemetry_json();
    for phase in [
        "gzip_finish_ns",
        "flush_ns",
        "sync_all_ns",
        "file_close_ns",
        "sha256_ns",
        "publish_gzip_ns",
        "publish_manifest_ns",
        "handoff_to_seal_ns",
    ] {
        assert!(
            metrics["phase_max_ns"][phase].as_u64().is_some(),
            "missing {phase}: {metrics}"
        );
    }
    println!(
        "local slow seal overload={overload} peak_rows=3 max_age_ns={age} accepted=6 phases={}",
        metrics["phase_max_ns"]
    );
    Ok(())
}

#[tokio::test]
async fn local_slow_seal_drains_accepted_rows() -> CaptureResult<()> {
    stall_sealer(false).await
}

#[tokio::test]
async fn local_sustained_overload_fails_closed_with_cause() -> CaptureResult<()> {
    stall_sealer(true).await
}

#[tokio::test]
async fn local_missing_predecessor_expires_with_both_sockets_open() -> CaptureResult<()> {
    let mut bank = Bank::start(Duration::from_secs(3600)).await?;
    let mut primary = bank.socket(1).await?;
    let mut backup = bank.socket(2).await?;
    let mut expected = seed(&mut bank, &mut primary).await?;

    let orphan = depth(110, 105);
    let sent = Instant::now();
    publish(&mut backup, orphan.clone()).await?;
    bank.saw_on_backup(DEPTH).await?;
    bank.drain();
    assert!(
        !bank
            .values()
            .any(|r| r["kind"] == "frame" && r["payload"] == orphan)
    );
    let gap = bank
        .until("missing predecessor expiry", |r| {
            r["kind"] == "depth_gap" && r["depth_generation"] == 2
        })
        .await?;
    assert_eq!(gap["previous_u"], 100);
    assert_eq!(gap["pu"], 105);
    let elapsed = sent.elapsed();
    assert!(
        elapsed >= Duration::from_millis(160),
        "arbitration expired too early: {elapsed:?}"
    );
    bank.until("re-anchor snapshot", |r| {
        r["kind"] == "snapshot" && r["depth_generation"] == 2 && r["last_update_id"] == 110
    })
    .await?;
    let anchor = depth(111, 110);
    publish(&mut primary, anchor.clone()).await?;
    bank.expect_payload(&anchor).await?;
    expected.extend([orphan, anchor]);
    bank.finish().await?;
    assert_eq!(bank.snapshots.load(Ordering::Relaxed), 2);
    assert_eq!(
        bank.values().filter(|r| r["kind"] == "depth_gap").count(),
        1
    );
    assert_eq!(
        bank.values().filter(|r| r["kind"] == "socket_gap").count(),
        0
    );
    verify_frames(bank.values(), &expected);
    println!("local depth predecessor wait_ms={}", elapsed.as_millis());
    Ok(())
}
