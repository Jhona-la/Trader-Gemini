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
        let is_testnet = std::env::var("USE_TESTNET")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            == "true";
        let base_url = if is_testnet {
            "https://testnet.binancefuture.com/fapi/v1/klines"
        } else {
            "https://fapi.binance.com/fapi/v1/klines"
        };
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            base_url: base_url.to_string(),
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

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!("Binance API Error: {}", resp.status()));
        }

        let data: Vec<Value> = resp.json().await.map_err(|e| e.to_string())?;

        let mut klines = Vec::with_capacity(data.len());
        for row in data {
            if let Some(arr) = row.as_array() {
                let open_time = arr[0].as_u64().unwrap_or(0);
                let open: f64 = arr[1].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                let high: f64 = arr[2].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                let low: f64 = arr[3].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                let close: f64 = arr[4].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                let volume: f64 = arr[5].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                let close_time = arr[6].as_u64().unwrap_or(0);

                if open > 0.0
                    && high > 0.0
                    && low > 0.0
                    && close > 0.0
                    && open.is_finite()
                    && high.is_finite()
                    && low.is_finite()
                    && close.is_finite()
                    && volume.is_finite()
                    && volume >= 0.0
                {
                    let kline = Kline {
                        open_time,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        close_time,
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
        let coin_id = quantum_arena::symbol_registry::try_index(symbol).unwrap_or(0);
        let mut all_ticks = Vec::with_capacity(100_000); // 100k limit to protect RAM
        let mut current_start = start_time;

        let max_time_span = 7 * 24 * 60 * 60 * 1000; // Max 7 days
                                                     // FIX #691: Prevenir underflow si start_time > end_time
        let end_time = if end_time.saturating_sub(start_time) > max_time_span {
            start_time.saturating_add(max_time_span)
        } else {
            end_time.max(start_time)
        };

        // El límite de la API de Binance es 1 hora entre start y end para aggTrades
        let max_window = 60 * 60 * 1000;

        let is_testnet = std::env::var("USE_TESTNET")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            == "true";
        let base_url = if is_testnet {
            "https://testnet.binancefuture.com/fapi/v1/aggTrades"
        } else {
            "https://fapi.binance.com/fapi/v1/aggTrades"
        };

        while current_start < end_time {
            let mut current_end = current_start + max_window;
            if current_end > end_time {
                current_end = end_time;
            }

            let url = format!(
                "{}?symbol={}&startTime={}&endTime={}&limit=1000",
                base_url,
                symbol.to_uppercase(),
                current_start,
                current_end
            );

            let mut retries = 0;
            let data: Vec<Value> = loop {
                let resp_res = self.client.get(&url).send().await;
                match resp_res {
                    Ok(resp) => {
                        let status = resp.status();
                        if status.is_success() {
                            match resp.json::<Vec<Value>>().await {
                                Ok(d) => break d,
                                Err(e) => {
                                    retries += 1;
                                    if retries >= 3 {
                                        return Err(format!(
                                            "JSON decode error for {}: {}",
                                            symbol, e
                                        ));
                                    }
                                    tokio::time::sleep(tokio::time::Duration::from_millis(
                                        500 * retries,
                                    ))
                                    .await;
                                }
                            }
                        } else if status.as_u16() == 429 || status.is_server_error() {
                            retries += 1;
                            if retries >= 5 {
                                return Err(format!(
                                    "Binance API rate limit / server error {} after retries for {}",
                                    status, symbol
                                ));
                            }
                            tokio::time::sleep(tokio::time::Duration::from_millis(
                                500 * (1 << retries),
                            ))
                            .await;
                        } else {
                            return Err(format!(
                                "Binance API client error: {} for {}",
                                status, symbol
                            ));
                        }
                    }
                    Err(e) => {
                        retries += 1;
                        if retries >= 4 {
                            return Err(format!("Network connection error for {}: {}", symbol, e));
                        }
                        tokio::time::sleep(tokio::time::Duration::from_millis(500 * retries)).await;
                    }
                }
            };
            if data.is_empty() {
                current_start = current_end + 1;
                continue;
            }

            let mut last_time = current_start;

            // Acumular AggTrades en micro-batches de 100ms para generar OBI realista
            // En vez de generar 1 tick por trade (OBI siempre ±1.0),
            // agrupamos trades del mismo intervalo de 100ms y sumamos buy/sell volume.
            let batch_interval_ms = 100; // 100ms micro-batches
            let mut batch_start: u64 = 0;
            let mut batch_buy_vol = 0.0;
            let mut batch_sell_vol = 0.0;
            let mut batch_last_price = 0.0;
            let mut batch_count = 0;

            for item in data {
                let timestamp = item["T"].as_u64().unwrap_or(0);
                let price: f64 = item["p"]
                    .as_str()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or_default();
                let qty: f64 = item["q"]
                    .as_str()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or_default();
                let is_buyer_maker = item["m"].as_bool().unwrap_or(false);

                if timestamp > 0 && price > 0.0 {
                    last_time = last_time.max(timestamp);

                    if batch_start == 0 {
                        batch_start = timestamp;
                    }

                    if timestamp < batch_start + batch_interval_ms {
                        // Mismo batch: acumular volumen por lado
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
                            let tick_size = quantum_arena::symbol_registry::try_spec(coin_id)
                                .map(|s| s.tick_size)
                                .unwrap_or(0.0001);
                            let spread = (batch_last_price * 0.00015).max(tick_size * 2.0);
                            // Modelar liquidez resting L2: volumen basal continuo para evitar OBI binario +-1.0
                            let base_depth = ((batch_buy_vol + batch_sell_vol) * 0.25).max(1.0);
                            let b_qty = batch_buy_vol + base_depth;
                            let a_qty = batch_sell_vol + base_depth;
                            all_ticks.push(quantum_arena::TickEvent {
                                coin_id,
                                timestamp: batch_start,
                                bid_price: batch_last_price - spread / 2.0,
                                ask_price: batch_last_price + spread / 2.0,
                                bid_qty: b_qty,
                                ask_qty: a_qty,
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
                let tick_size = quantum_arena::symbol_registry::try_spec(coin_id)
                    .map(|s| s.tick_size)
                    .unwrap_or(0.0001);
                let spread = (batch_last_price * 0.00015).max(tick_size * 2.0);
                let base_depth = ((batch_buy_vol + batch_sell_vol) * 0.25).max(1.0);
                let b_qty = batch_buy_vol + base_depth;
                let a_qty = batch_sell_vol + base_depth;
                all_ticks.push(quantum_arena::TickEvent {
                    coin_id,
                    timestamp: batch_start,
                    bid_price: batch_last_price - spread / 2.0,
                    ask_price: batch_last_price + spread / 2.0,
                    bid_qty: b_qty,
                    ask_qty: a_qty,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_loader_instantiation() {
        let loader = HistoricalLoader::new();
        let default_loader = HistoricalLoader::default();
        let _ = loader;
        let _ = default_loader;
    }

    #[test]
    fn test_kline_serialization_and_finiteness() {
        let kline = Kline {
            open_time: 1672531140000,
            open: 50000.0,
            high: 50100.0,
            low: 49950.0,
            close: 50050.0,
            volume: 120.5,
            close_time: 1672531199999,
        };

        let json_str = serde_json::to_string(&kline).unwrap();
        let decoded: Kline = serde_json::from_str(&json_str).unwrap();

        assert_eq!(decoded.open_time, 1672531140000);
        assert_eq!(decoded.close, 50050.0);
        assert!(decoded.open.is_finite() && decoded.high >= decoded.low);
    }
}
