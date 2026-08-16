use crate::parser::BookTickerEvent;
use futures_util::StreamExt;
use hickory_resolver::config::*;
use hickory_resolver::AsyncResolver;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_tungstenite::client_async_tls;
use url::Url;

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

    pub async fn start<F>(&self, mut callback: F)
    where
        F: FnMut(BookTickerEvent) + Send + 'static,
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

        // FASE 14: Acelerador DNS Cuántico (Cloudflare 1.1.1.1) + IP Pinning Institucional
        // Resolvemos la IP EXACTAMENTE UNA VEZ. La fijamos en memoria RAM (IP Pinning)
        // para que en caso de caída del Websocket, la reconexión tome microsegundos en lugar de milisegundos DNS.
        let target_addr_cache = {
            let mut resolved_addr = None;
            let resolver =
                AsyncResolver::tokio(ResolverConfig::cloudflare(), ResolverOpts::default());
            println!(
                "⚡ [DNS QUANTUM] Iniciando Resolución Cloudflare 1.1.1.1 para {}",
                host
            );

            if let Ok(Ok(response)) = tokio::time::timeout(
                std::time::Duration::from_millis(1500),
                resolver.lookup_ip(host.to_string()),
            )
            .await
            {
                if let Some(ip) = response.into_iter().next() {
                    resolved_addr = Some(std::net::SocketAddr::new(ip, port));
                    println!("🟢 [IP PINNING] Rutas ancladas a Binance: {}", ip);
                }
            }

            if resolved_addr.is_none() {
                // Fallback local OS
                if let Ok(Ok(mut addrs)) = tokio::time::timeout(
                    std::time::Duration::from_millis(1500),
                    tokio::net::lookup_host((host, port)),
                )
                .await
                {
                    if let Some(addr) = addrs.next() {
                        resolved_addr = Some(addr);
                        println!("⚠️ [IP PINNING FALLBACK] DNS Local usado: {}", addr.ip());
                    }
                }
            }
            resolved_addr.expect(
                "❌ [CRÍTICO] Imposible resolver DNS de Binance. Imposible iniciar el motor HFT.",
            )
        };

        loop {
            let target_addr = target_addr_cache;

            // FASE 22: Conexión directa TCP con Zero-Nagle para latencia nula
            match tokio::net::TcpStream::connect(target_addr).await {
                Ok(tcp_stream) => {
                    let _ = tcp_stream.set_nodelay(true);

                    // FASE 19: Configuración Extrema de Latencia Cuántica (Windows OS)
                    let socket = socket2::Socket::from(tcp_stream.into_std().unwrap());
                    // Reducir buffers dramáticamente para forzar a Winsock a entregar el frame a user-space
                    // inmediatamente, sin esperar a llenar buffers grandes. Priorizamos LATENCIA sobre THROUGHPUT.
                    let _ = socket.set_recv_buffer_size(65536); // 64KB Recv Buffer (Baja latencia)
                    let _ = socket.set_send_buffer_size(65536); // 64KB Send Buffer

                    // FASE 27: TCP Keep-Alive para prevenir desconexiones silenciosas del Firewall/AWS
                    let keepalive = socket2::TcpKeepalive::new()
                        .with_time(std::time::Duration::from_secs(30))
                        .with_interval(std::time::Duration::from_secs(5));
                    let _ = socket.set_tcp_keepalive(&keepalive);
                    let tcp_stream = tokio::net::TcpStream::from_std(socket.into()).unwrap();

                    match client_async_tls(url.clone(), tcp_stream).await {
                        Ok((ws_stream, _)) => {
                            let connected_at = std::time::Instant::now();
                            // Si logramos conectar, reseteamos el backoff después de probar que es estable
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
                                            // FASE 9: Bayesian Anomaly Rejection (Estasis Probabilística)
                                            // Protege al motor de glitches usando micro-volatilidad histórica real
                                            let current_price =
                                                (event.bid_price + event.ask_price) * 0.5;
                                            let mut recent_ticks =
                                                [quantum_arena::state::CompactTick::default(); 10];
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
                                                    let variance = (sum_sq / count) - (mean * mean);
                                                    let std_dev = variance.max(0.0).sqrt();

                                                    // Bayesian Stasis: El precio no debe desviar más de 5 desviaciones estándar en 100ms
                                                    // Si la volatilidad es muy baja, usamos un límite piso del 0.2% para evitar falsos positivos
                                                    let dynamic_threshold =
                                                        (std_dev * 5.0).max(last_valid * 0.002);

                                                    if (current_price - last_valid).abs()
                                                        > dynamic_threshold
                                                    {
                                                        println!("🛡️ [BAYESIAN STASIS] Anomalía Cuántica bloqueada en {}: Precio {}, Media {:.4}, Umbral Dinámico {:.4}", self.symbol, current_price, mean, dynamic_threshold);
                                                        continue; // Corrupt data, do not pollute the tensor
                                                    }
                                                }
                                            }

                                            event.coin_id = self.coin_id;
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
                                                    .unwrap()
                                                    .as_millis()
                                                    as i64;

                                                let offset = self
                                                    .arena
                                                    .server_time_offset_ms
                                                    .load(Ordering::Relaxed);
                                                let synced_time = current_time - offset;

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
                                        }
                                    }
                                    Err(_) => {
                                        // Error de red interno (stream corrupto/desconexión Binance), romper el bucle interno y reconectar
                                        break;
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Fallo de conexión TLS
                            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                            backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                        }
                    }
                }
                Err(_) => {
                    // Fallo de conexión TCP
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                }
            }
        }
    }
}
