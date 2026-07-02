use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Kline {
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub close_time: u64,
}

pub struct HistoricalLoader {
    client: Client,
    base_url: String,
}

impl Default for HistoricalLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoricalLoader {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            base_url: "https://fapi.binance.com/fapi/v1/klines".to_string(),
        }
    }

    /// Fetch Klines from Binance REST API asíncronamente
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: Option<u64>,
        end_time: Option<u64>,
        limit: u32,
    ) -> Result<Vec<Kline>, String> {
        let mut url = format!(
            "{}?symbol={}&interval={}&limit={}",
            self.base_url,
            symbol.to_uppercase(),
            interval,
            limit
        );

        if let Some(st) = start_time {
            url.push_str(&format!("&startTime={}", st));
        }
        if let Some(et) = end_time {
            url.push_str(&format!("&endTime={}", et));
        }

        let resp = self.client.get(&url).send().await.map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!("Binance API Error: {}", resp.status()));
        }

        let data: Vec<Value> = resp.json().await.map_err(|e| e.to_string())?;
        
        let mut klines = Vec::with_capacity(data.len());
        for row in data {
            if let Some(arr) = row.as_array() {
                if arr.len() >= 7 {
                    let kline = Kline {
                        open_time: arr[0].as_u64().unwrap_or(0),
                        open: arr[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        high: arr[2].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        low: arr[3].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        close: arr[4].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        volume: arr[5].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        close_time: arr[6].as_u64().unwrap_or(0),
                    };
                    klines.push(kline);
                }
            }
        }

        Ok(klines)
    }

    /// Fetch AggTrades from Binance REST API asíncronamente (Max 1 hora de diferencia por request)
    pub async fn fetch_agg_trades(
        &self,
        symbol: &str,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<quantum_arena::TickEvent>, String> {
        let mut all_ticks = Vec::new();
        let mut current_start = start_time;
        
        // El límite de la API de Binance es 1 hora entre start y end para aggTrades
        let max_window = 60 * 60 * 1000; 

        while current_start < end_time {
            let mut current_end = current_start + max_window;
            if current_end > end_time {
                current_end = end_time;
            }

            let url = format!(
                "{}?symbol={}&startTime={}&endTime={}&limit=1000",
                "https://fapi.binance.com/fapi/v1/aggTrades",
                symbol.to_uppercase(),
                current_start,
                current_end
            );

            let resp = self.client.get(&url).send().await.map_err(|e| e.to_string())?;

            if !resp.status().is_success() {
                // Rate limit o error temporal, esperamos y reintentamos
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                continue;
            }

            let data: Vec<Value> = resp.json().await.map_err(|e| e.to_string())?;
            if data.is_empty() {
                current_start = current_end + 1;
                continue;
            }

            let mut last_time = current_start;
            
            // Acumular AggTrades en micro-batches de 100ms para generar OBI realista
            // En vez de generar 1 tick por trade (OBI siempre ±1.0),
            // agrupamos trades del mismo intervalo de 100ms y sumamos buy/sell volume.
            let batch_interval_ms = 100; // 100ms micro-batches
            let mut batch_start = 0u64;
            let mut batch_buy_vol = 0.0_f64;
            let mut batch_sell_vol = 0.0_f64;
            let mut batch_last_price = 0.0_f64;
            let mut batch_count = 0u32;
            
            for row in data {
                if let Some(obj) = row.as_object() {
                    let price: f64 = obj.get("p").and_then(|v| v.as_str()).unwrap_or("0").parse().unwrap_or(0.0);
                    let qty: f64 = obj.get("q").and_then(|v| v.as_str()).unwrap_or("0").parse().unwrap_or(0.0);
                    let timestamp: u64 = obj.get("T").and_then(|v| v.as_u64()).unwrap_or(0);
                    let is_buyer_maker = obj.get("m").and_then(|v| v.as_bool()).unwrap_or(false);
                    
                    last_time = timestamp;
                    
                    // Inicializar primer batch
                    if batch_count == 0 {
                        batch_start = timestamp;
                    }
                    
                    // Si seguimos dentro del mismo micro-batch, acumulamos
                    if timestamp - batch_start < batch_interval_ms {
                        if is_buyer_maker {
                            batch_sell_vol += qty;
                        } else {
                            batch_buy_vol += qty;
                        }
                        batch_last_price = price;
                        batch_count += 1;
                    } else {
                        // Flush del batch anterior si tiene datos
                        if batch_count > 0 && batch_last_price > 0.0 {
                            let spread = batch_last_price * 0.0001; // 0.01% spread realista
                            all_ticks.push(quantum_arena::TickEvent {
                                coin_id: 0,
                                timestamp: batch_start,
                                bid_price: batch_last_price - spread / 2.0,
                                ask_price: batch_last_price + spread / 2.0,
                                bid_qty: batch_buy_vol.max(0.001), // Evitar division por cero
                                ask_qty: batch_sell_vol.max(0.001),
                            });
                        }
                        
                        // Iniciar nuevo batch con el trade actual
                        batch_start = timestamp;
                        batch_buy_vol = 0.0;
                        batch_sell_vol = 0.0;
                        if is_buyer_maker {
                            batch_sell_vol = qty;
                        } else {
                            batch_buy_vol = qty;
                        }
                        batch_last_price = price;
                        batch_count = 1;
                    }
                }
            }
            // Flush del último batch
            if batch_count > 0 && batch_last_price > 0.0 {
                let spread = batch_last_price * 0.0001;
                all_ticks.push(quantum_arena::TickEvent {
                    coin_id: 0,
                    timestamp: batch_start,
                    bid_price: batch_last_price - spread / 2.0,
                    ask_price: batch_last_price + spread / 2.0,
                    bid_qty: batch_buy_vol.max(0.001),
                    ask_qty: batch_sell_vol.max(0.001),
                });
            }
            
            // Avanzamos al último timestamp recibido + 1 ms para evitar duplicados
            // O avanzamos la ventana si no hubo datos
            current_start = last_time + 1;
            
            // Be kind to the API rate limits (1200 weight / min)
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        Ok(all_ticks)
    }
}
