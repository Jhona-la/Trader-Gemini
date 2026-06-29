use crate::parser::BookTickerEvent;
use futures_util::StreamExt;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_tungstenite::connect_async;
use url::Url;

/// URL del stream de Binance para la mejor oferta y demanda en tiempo real.
const BINANCE_WS_URL: &str = "wss://stream.binance.com:9443/ws/";

pub struct BinanceStreamer {
    pub symbol: String,
    // Haremos uso del QuantumArena más tarde, por ahora pasamos un callback o canal.
}

impl BinanceStreamer {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_lowercase(),
        }
    }

    /// Inicia el bucle asíncrono para escuchar a Binance.
    /// Toma un closure que se ejecutará en cada tick válido.
    pub async fn start<F>(&self, mut on_tick: F)
    where
        F: FnMut(BookTickerEvent) + Send + 'static,
    {
        // stream: <symbol>@bookTicker
        let stream_url = format!("{}{}@bookTicker", BINANCE_WS_URL, self.symbol);
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

                    while let Some(msg) = read.next().await {
                        // Si llevamos conectados más de 10 segundos sin errores, la conexión es estable
                        if connected_at.elapsed().as_secs() > 10 {
                            backoff_ms = 100;
                        }

                        match msg {
                            Ok(msg) => {
                                if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                                    if let Some(event) = BookTickerEvent::parse_from_json(&text) {
                                        // Inyectar evento de vuelta a la lógica del bot
                                        on_tick(event);
                                    }
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
