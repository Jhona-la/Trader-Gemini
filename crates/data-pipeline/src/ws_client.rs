use crate::parser::BookTickerEvent;
use futures_util::StreamExt;
use hickory_resolver::config::*;
use hickory_resolver::AsyncResolver;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_tungstenite::client_async_tls;
use url::Url;

/// C-01 (INFORME 14, FASE 0): ticks anómalos CONSECUTIVOS tras los que un
/// movimiento extremo (>20% vs `last_valid`) se acepta como nivel legítimo.
/// A ~10 ticks/s son ~1s de confirmación continua: suficiente para distinguir
/// un glitch de 1-2 ticks de un crash real. Sin este escape, el
/// `is_extreme_glitch` rechazaba TODOS los ticks para siempre en un crash
/// real (los ticks rechazados jamás entran al ring ⇒ `last_valid` congelado
/// en el precio pre-crash) y el motor operaba contra un precio muerto.
const EXTREME_GLITCH_CONFIRM_TICKS: usize = 10;

pub struct BinanceStreamer {
    pub coin_id: usize,
    pub symbol: String,
    pub arena: Arc<quantum_arena::GlobalArena>,
    pub is_testnet: bool,
}

impl BinanceStreamer {
    pub fn new(
        coin_id: usize,
        symbol: &str,
        arena: Arc<quantum_arena::GlobalArena>,
        is_testnet: bool,
    ) -> Self {
        Self {
            coin_id,
            symbol: symbol.to_lowercase(),
            arena,
            is_testnet,
        }
    }

    pub async fn start<F>(&self, callback: F)
    where
        F: FnMut(BookTickerEvent) + Send + 'static,
    {
        self.start_with_trade_handler(callback, |_| {}).await;
    }

    pub async fn start_with_trade_handler<F, T>(&self, mut callback: F, mut trade_callback: T)
    where
        F: FnMut(BookTickerEvent) + Send + 'static,
        T: FnMut(crate::parser::AggTradeEvent) + Send + 'static,
    {
        // Latency Accelerator: Endpoint Pool (AWS, Tokyo routing emulation)
        let endpoints = if self.is_testnet {
            vec!["stream.binancefuture.com"]
        } else {
            vec![
                "fstream.binance.com",
                "fstream-auth.binance.com",
                "fstream.binance.com", // Double check GeoDNS round-robin
            ]
        };

        let mut best_host = endpoints[0].to_string();
        let mut best_latency = u128::MAX;

        for ep in endpoints {
            let start = std::time::Instant::now();
            if tokio::time::timeout(
                std::time::Duration::from_millis(1500),
                tokio::net::TcpStream::connect((ep, 443)),
            )
            .await
            .is_ok()
            {
                let latency = start.elapsed().as_millis();
                if latency < best_latency {
                    best_latency = latency;
                    best_host = ep.to_string();
                }
            }
        }
        println!(
            "🚀 [LATENCY ACCELERATOR] Selected {} with {}ms latency for {}",
            best_host, best_latency, self.symbol
        );

        // F2.5 — ZONA RAW (opt-in TG_RAW_ARCHIVE=1): archivo append-only de
        // ticks validados con checksum por segmento. Fuente de verdad para
        // re-derivación de features/backtests; sin impacto en hot-path.
        let mut raw_archive = if std::env::var("TG_RAW_ARCHIVE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            match storage_engine::raw_zone::RawTickArchive::open("data/raw_ticks", &self.symbol) {
                Ok(a) => {
                    println!(
                        "🗄️  [RAW-ZONE] Archivo vivo para {} en data/raw_ticks/",
                        self.symbol
                    );
                    Some(a)
                }
                Err(e) => {
                    println!(
                        "⚠️ [RAW-ZONE] Archivo NO disponible ({}): continuando sin él",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Multiplexing stream: bookTicker + aggTrade + depth10
        let stream_url = format!(
            "wss://{}/stream?streams={}@bookTicker/{}@aggTrade/{}@depth10@100ms",
            best_host, self.symbol, self.symbol, self.symbol
        );
        let url = Url::parse(&stream_url).expect("Bad WS URL");

        let mut backoff_ms = 100;
        let max_backoff_ms = 5000;

        let host = url.host_str().unwrap_or("fstream.binance.com");
        let port = url.port_or_known_default().unwrap_or(443);

        // ── F2.2: IP PINNING CON INVALIDACIÓN ─────────────────────────────────
        // El pinning original era PARA SIEMPRE: si Binance rota la IP (lo hace),
        // el stream reconectaba infinitamente a una IP muerta. Nueva política:
        // la IP se cachea mientras la conexión sea sana; tras fallos TCP/TLS
        // consecutivos se RE-RESUELVE DNS (GeoDNS entrega IPs frescas).
        // Jamás .expect(): DNS caído NO mata el motor (panic=abort).
        let resolver = AsyncResolver::tokio(ResolverConfig::cloudflare(), ResolverOpts::default());
        let mut dns_resolution_count: u64 = 0;
        let mut target_cache: Option<std::net::SocketAddr> = None;
        let mut consecutive_failures: u32 = 0;

        let resolve_target = |host: &str| {
            let resolver = &resolver;
            let host = host.to_string();
            async move {
                // 1) Cloudflare 1.1.1.1
                if let Ok(Ok(response)) = tokio::time::timeout(
                    std::time::Duration::from_millis(1500),
                    resolver.lookup_ip(host.clone()),
                )
                .await
                {
                    if let Some(ip) = response.into_iter().next() {
                        return Some(std::net::SocketAddr::new(ip, port));
                    }
                }
                // 2) Fallback DNS del SO
                if let Ok(Ok(mut addrs)) = tokio::time::timeout(
                    std::time::Duration::from_millis(1500),
                    tokio::net::lookup_host((host, port)),
                )
                .await
                {
                    if let Some(addr) = addrs.next() {
                        return Some(addr);
                    }
                }
                None
            }
        };

        loop {
            let addr_now = match target_cache {
                Some(a) => a,
                None => {
                    dns_resolution_count += 1;
                    match resolve_target(host).await {
                        Some(addr) => {
                            println!(
                                "🟢 [DNS] Resolución #{} para {}: {} ({})",
                                dns_resolution_count,
                                host,
                                addr.ip(),
                                if dns_resolution_count == 1 {
                                    "inicial"
                                } else {
                                    "re-resolución tras fallos"
                                }
                            );
                            if dns_resolution_count > 1 {
                                backoff_ms = backoff_ms.min(500);
                            }
                            consecutive_failures = 0;
                            target_cache = Some(addr);
                            addr
                        }
                        None => {
                            println!(
                                "⚠️ [DNS] Resolución fallida para {} (intento #{}). Reintentando en {}ms...",
                                host, dns_resolution_count, backoff_ms
                            );
                            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                            backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                            continue;
                        }
                    }
                }
            };

            // FASE 22: Conexión directa TCP con Zero-Nagle para latencia nula
            match tokio::net::TcpStream::connect(addr_now).await {
                Ok(tcp_stream) => {
                    let _ = tcp_stream.set_nodelay(true);

                    let std_stream = match tcp_stream.into_std() {
                        Ok(s) => s,
                        Err(e) => {
                            println!("⚠️ [WS] Falló conversión a std::net::TcpStream: {}. Reintentando...", e);
                            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                            continue;
                        }
                    };
                    let socket = socket2::Socket::from(std_stream);
                    let _ = socket.set_recv_buffer_size(65536);
                    let _ = socket.set_send_buffer_size(65536);

                    let keepalive = socket2::TcpKeepalive::new()
                        .with_time(std::time::Duration::from_secs(30))
                        .with_interval(std::time::Duration::from_secs(5));
                    let _ = socket.set_tcp_keepalive(&keepalive);
                    let tcp_stream = match tokio::net::TcpStream::from_std(socket.into()) {
                        Ok(s) => s,
                        Err(e) => {
                            println!("⚠️ [WS] Falló reconversión a tokio::net::TcpStream: {}. Reintentando...", e);
                            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                            continue;
                        }
                    };

                    match client_async_tls(url.clone(), tcp_stream).await {
                        Ok((ws_stream, _)) => {
                            let connected_at = std::time::Instant::now();
                            let mut consecutive_anomalies: usize = 0;
                            let mut reconnect_fast_recovery_ticks: usize = 3; // FIX #1003: Calibrar de inmediato tras reconexión
                            let (_, mut read) = ws_stream.split();

                            loop {
                                let dynamic_timeout_ms = self
                                    .arena
                                    .config
                                    .latency_ms_panic_threshold
                                    .load(Ordering::Relaxed)
                                    as u64;
                                let timeout_res = tokio::time::timeout(
                                    std::time::Duration::from_millis(dynamic_timeout_ms),
                                    read.next(),
                                )
                                .await;

                                let msg = match timeout_res {
                                    Ok(Some(m)) => m,
                                    Ok(None) => break, // Stream closed
                                    Err(_) => {
                                        // Timeout: no tick received in dynamic_timeout_ms
                                        self.arena
                                            .last_ws_latency_ms
                                            .store(dynamic_timeout_ms, Ordering::Relaxed);
                                        // The GodEngine will pick this up and activate kill switch
                                        // F2.2: la ruta puede estar muerta — invalidar IP y re-resolver.
                                        target_cache = None;
                                        break;
                                    }
                                };

                                // Note: Real latency is measured per-event later.

                                // Si llevamos conectados más de 10 segundos sin errores, la conexión es estable
                                if connected_at.elapsed().as_secs() > 10 {
                                    backoff_ms = 100;
                                }

                                match msg {
                                    Ok(msg) => {
                                        let bytes = msg.into_data();

                                        // Intentar parsear como BookTicker
                                        if let Some(mut event) =
                                            BookTickerEvent::parse_from_json(&bytes)
                                        {
                                            // F2.1: aduana de datos — lo corrupto NUNCA
                                            // entra al motor ni al tensor (directriz).
                                            if let Err(reason) =
                                                crate::validation::validate_book_ticker(&event)
                                            {
                                                crate::validation::count_reject(reason);
                                                continue;
                                            }

                                            // FASE 9: Bayesian Anomaly Rejection (Estasis Probabilística Adaptativa)
                                            // Protege al motor de glitches aislados sin congelar la ingesta ante breakouts reales
                                            let current_price =
                                                (event.bid_price + event.ask_price) * 0.5;

                                            // FIX #1003: Durante los primeros ticks de reconexión, aceptar el nuevo nivel de precio de inmediato
                                            if reconnect_fast_recovery_ticks > 0 {
                                                reconnect_fast_recovery_ticks -= 1;
                                                consecutive_anomalies = 0;
                                            } else {
                                                let mut recent_ticks =
                                                    [quantum_arena::state::CompactTick::default();
                                                        10];
                                                let count_ticks = self.arena.coins[self.coin_id]
                                                    .tick_ring
                                                    .snapshot_recent_into(10, &mut recent_ticks);
                                                let recent_slice = &recent_ticks[..count_ticks];

                                                if !recent_slice.is_empty() {
                                                    let mut sum = 0.0;
                                                    let mut sum_sq = 0.0;
                                                    let mut count = 0.0;
                                                    let mut last_valid = 0.0;

                                                    for t in recent_slice {
                                                        let p = (t.bid_price + t.ask_price) * 0.5;
                                                        if p > 0.0 {
                                                            sum += p;
                                                            sum_sq += p * p;
                                                            count += 1.0;
                                                            last_valid = p;
                                                        }
                                                    }

                                                    if count > 2.0 && last_valid > 0.0 {
                                                        let mean = sum / count;
                                                        let variance =
                                                            (sum_sq / count) - (mean * mean);
                                                        let std_dev = variance.max(0.0).sqrt();

                                                        // Bayesian Stasis: Umbral adaptativo con piso del 0.8% (para permitir volatilidad real)
                                                        let dynamic_threshold =
                                                            (std_dev * 6.0).max(last_valid * 0.008);

                                                        let is_outlier =
                                                            (current_price - last_valid).abs()
                                                                > dynamic_threshold;
                                                        // Glitch extremo (> 20% en 1 tick)
                                                        let is_extreme_glitch =
                                                            (current_price - last_valid).abs()
                                                                > (last_valid * 0.20);

                                                        if is_outlier {
                                                            consecutive_anomalies += 1;
                                                            // C-01 (INFORME 14, FASE 0): el escape de
                                                            // FIX #385 (3 ticks) NO se aplicaba a los
                                                            // glitches extremos, y como los ticks
                                                            // rechazados nunca entran al ring,
                                                            // `last_valid` quedaba congelado en el
                                                            // precio pre-crash: en un crash REAL (>20%)
                                                            // el filtro rechazaba TODOS los ticks PARA
                                                            // SIEMPRE. Escape: tras
                                                            // EXTREME_GLITCH_CONFIRM_TICKS ticks
                                                            // anómalos CONSECUTIVOS (~1s de
                                                            // confirmación continua a ~10 ticks/s) se
                                                            // ACEPTA el tick — entra al ring y
                                                            // `last_valid` recalibra al nuevo nivel.
                                                            // Un crash real no debe congelar el motor.
                                                            if is_extreme_glitch
                                                                && consecutive_anomalies
                                                                    < EXTREME_GLITCH_CONFIRM_TICKS
                                                            {
                                                                // Glitch extremo aislado, descartar
                                                                continue;
                                                            }
                                                            // FIX #385: si llegan 3 ticks consecutivos
                                                            // en el nuevo nivel, es un movimiento de
                                                            // mercado legítimo
                                                            if consecutive_anomalies < 3 {
                                                                continue; // Glitch aislado, descartar
                                                            }
                                                            // Transición de régimen de precio
                                                            // confirmada: resetear contador y aceptar
                                                            consecutive_anomalies = 0;
                                                        } else {
                                                            consecutive_anomalies = 0;
                                                        }
                                                    }
                                                }
                                            }

                                            event.coin_id = self.coin_id;
                                            // F2.5 — ZONA RAW: tick VALIDADO (aduana
                                            // F2.1 superada) al archivo append-only
                                            // con checksum. BufWriter 1MB: un syscall
                                            // cada ~26k ticks; el append es copy de
                                            // 40 bytes en RAM — sin impacto medible
                                            // en el hot-path de lectura.
                                            if let Some(archive) = raw_archive.as_mut() {
                                                let _ = archive.append(
                                                    storage_engine::raw_zone::RawTick {
                                                        ts_ms: event.event_time,
                                                        bid: event.bid_price,
                                                        ask: event.ask_price,
                                                        bid_qty: event.bid_qty,
                                                        ask_qty: event.ask_qty,
                                                    },
                                                );
                                            }
                                            self.arena.update_market_data(
                                                self.coin_id,
                                                event.bid_price,
                                                event.ask_price,
                                                event.bid_qty,
                                                event.ask_qty,
                                                event.event_time,
                                            );
                                            self.arena.increment_tick();

                                            // Medir latencia de red real
                                            if event.event_time > 0 {
                                                let current_time = std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                    .unwrap_or_default()
                                                    .as_millis()
                                                    as i64;

                                                let offset = self
                                                    .arena
                                                    .server_time_offset_ms
                                                    .load(Ordering::Relaxed);
                                                let synced_time = current_time + offset;

                                                if synced_time >= event.event_time as i64 {
                                                    let latency = (synced_time
                                                        - event.event_time as i64)
                                                        as u64;
                                                    self.arena
                                                        .last_ws_latency_ms
                                                        .store(latency, Ordering::Relaxed);
                                                }
                                            }

                                            callback(event);
                                        }
                                        // Si no es BookTicker, intentar parsear como AggTrade
                                        else if let Some(agg_event) =
                                            crate::parser::AggTradeEvent::parse_from_json(&bytes)
                                        {
                                            self.arena.update_agg_trade(
                                                self.coin_id,
                                                agg_event.is_buyer_maker,
                                                agg_event.qty,
                                            );
                                            self.arena.increment_tick();
                                            // D-421: Notificar callback ante transacciones AggTrade
                                            trade_callback(agg_event);
                                        }
                                        // Finalmente, intentar parsear como DepthEvent
                                        else if let Some(depth_event) =
                                            crate::parser::DepthEvent::parse_from_json(&bytes)
                                        {
                                            self.arena.update_l2_depth(
                                                self.coin_id,
                                                depth_event.bid_wall,
                                                depth_event.ask_wall,
                                            );
                                            self.arena.increment_tick();
                                        }
                                    }
                                    Err(_) => {
                                        // Error de red interno (stream corrupto): invalidar IP y reconectar
                                        target_cache = None;
                                        break;
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Fallo TLS: la IP cacheada puede estar muerta (F2.2).
                            consecutive_failures += 1;
                            if consecutive_failures >= 3 {
                                println!(
                                    "🔁 [WS] {} fallos TLS consecutivos a {} — re-resolviendo DNS",
                                    consecutive_failures, addr_now
                                );
                                target_cache = None;
                                consecutive_failures = 0;
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                            backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                        }
                    }
                }
                Err(_) => {
                    // Fallo TCP: Binance rota IPs — la pineada puede haber muerto.
                    consecutive_failures += 1;
                    if consecutive_failures >= 2 {
                        println!(
                            "🔁 [WS] TCP a {} falló {} veces — re-resolviendo DNS (rotación de IP de Binance)",
                            addr_now, consecutive_failures
                        );
                        target_cache = None;
                        consecutive_failures = 0;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binance_streamer_initialization() {
        let arena = Arc::new(quantum_arena::GlobalArena::new(13.0));
        let streamer = BinanceStreamer::new(0, "BTCUSDT", arena.clone(), true);

        assert_eq!(streamer.coin_id, 0);
        assert_eq!(streamer.symbol, "btcusdt");
        assert!(streamer.is_testnet);
    }

    #[test]
    fn test_binance_streamer_mainnet_symbol_normalization() {
        let arena = Arc::new(quantum_arena::GlobalArena::new(13.0));
        let streamer = BinanceStreamer::new(5, "SolUsdt", arena.clone(), false);

        assert_eq!(streamer.coin_id, 5);
        assert_eq!(streamer.symbol, "solusdt");
        assert!(!streamer.is_testnet);
    }
}
