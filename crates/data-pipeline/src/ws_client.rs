use crate::parser::BookTickerEvent;
use futures_util::StreamExt;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_tungstenite::connect_async;
use url::Url;

/// URL del stream de Binance para la mejor oferta y demanda en tiempo real.
const BINANCE_WS_URL: &str = "wss://fstream.binance.com/ws/";

pub struct BinanceStreamer {
    pub coin_id: usize,
    pub symbol: String,
    pub arena: Arc<quantum_arena::GlobalArena>,
}

impl BinanceStreamer {
    pub fn new(coin_id: usize, symbol: &str, arena: Arc<quantum_arena::GlobalArena>) -> Self {
        Self {
            coin_id,
            symbol: symbol.to_lowercase(),
            arena,
        }
    }

    pub async fn start<F>(&self, mut callback: F)
    where
        F: FnMut(BookTickerEvent) + Send + 'static,
    {
        // Multiplexing stream: bookTicker + aggTrade + depth10
        let stream_url = format!("wss://fstream.binance.com/stream?streams={}@bookTicker/{}@aggTrade/{}@depth10@100ms", self.symbol, self.symbol, self.symbol);
        let url = Url::parse(&stream_url).expect("Bad WS URL");
        
        let mut backoff_ms = 100;
        let max_backoff_ms = 5000;

        loop {
            // Intentar conectar
            match connect_async(url.clone()).await {
                Ok((ws_stream, _)) => {
                    let connected_at = std::time::Instant::now();
                    // Si logramos conectar, reseteamos el backoff después de probar que es estable
                    let (_, mut read) = ws_stream.split();

                    let mut last_tick_time = std::time::Instant::now();

                    loop {
                        let timeout_res = tokio::time::timeout(
                            std::time::Duration::from_millis(1500),
                            read.next()
                        ).await;

                        let msg = match timeout_res {
                            Ok(Some(m)) => m,
                            Ok(None) => break, // Stream closed
                            Err(_) => {
                                // Timeout: no tick received in 1500ms
                                self.arena.last_ws_latency_ms.store(1500, Ordering::Relaxed);
                                // The GodEngine will pick this up and activate kill switch
                                break;
                            }
                        };
                        
                        let latency = last_tick_time.elapsed().as_millis() as u64;
                        self.arena.last_ws_latency_ms.store(latency, Ordering::Relaxed);
                        last_tick_time = std::time::Instant::now();

                        // Si llevamos conectados más de 10 segundos sin errores, la conexión es estable
                        if connected_at.elapsed().as_secs() > 10 {
                            backoff_ms = 100;
                        }

                        match msg {
                            Ok(msg) => {
                                let bytes = msg.into_data();
                                
                                // Intentar parsear como BookTicker
                                if let Some(mut event) = BookTickerEvent::parse_from_json(&bytes) {
                                    event.coin_id = self.coin_id;
                                    self.arena.update_market_data(
                                        self.coin_id, 
                                        event.bid_price, 
                                        event.ask_price, 
                                        event.bid_qty, 
                                        event.ask_qty
                                    );
                                    self.arena.increment_tick();
                                    callback(event);
                                } 
                                // Si no es BookTicker, intentar parsear como AggTrade
                                else if let Some(agg_event) = crate::parser::AggTradeEvent::parse_from_json(&bytes) {
                                    self.arena.update_agg_trade(
                                        self.coin_id,
                                        agg_event.is_buyer_maker,
                                        agg_event.qty
                                    );
                                }
                                // Finalmente, intentar parsear como DepthEvent
                                else if let Some(depth_event) = crate::parser::DepthEvent::parse_from_json(&bytes) {
                                    self.arena.update_l2_depth(
                                        self.coin_id,
                                        depth_event.bid_wall,
                                        depth_event.ask_wall
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
                    // Fallo de conexión directa, esperar el backoff actual antes de reintentar
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    // Aumentar backoff exponencialmente (clamp a max_backoff_ms)
                    backoff_ms = (backoff_ms * 2).min(max_backoff_ms);
                }
            }
        }
    }
}
