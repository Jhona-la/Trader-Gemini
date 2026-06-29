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

        loop {
            // Intentar conectar
            match connect_async(url.clone()).await {
                Ok((ws_stream, _)) => {
                    // split() divide en escritura y lectura, por ahora solo leemos.
                    let (_, mut read) = ws_stream.split();

                    while let Some(msg) = read.next().await {
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
                                // Error de red interno, romper el bucle interno y reconectar
                                break;
                            }
                        }
                    }
                }
                Err(_) => {
                    // Fallo de conexión, esperar 1 segundo antes de reconectar
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
            }
        }
    }
}
