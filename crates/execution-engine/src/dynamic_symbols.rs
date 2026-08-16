use reqwest::blocking::Client;
use serde_json::Value;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Define la métrica cuántica de liquidez y volatilidad para un activo
#[derive(Debug, Clone)]
pub struct SymbolScore {
    pub symbol: String,
    pub volume_usd: f64,
    pub price_change_pct: f64,
    pub score: f64,
}

// Para usar BinaryHeap de mayor a menor
impl PartialEq for SymbolScore {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for SymbolScore {}
impl PartialOrd for SymbolScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Rust's BinaryHeap is a max-heap by default.
        // We want highest score at the top, so we order by score.
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for SymbolScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Selector Dinámico Cuántico de Símbolos
/// Consume el endpoint de 24h Ticker de Binance Futures para extraer
/// la matemática del volumen y la volatilidad, escogiendo el Top 10 algorítmicamente.
pub struct DynamicSymbolSelector {
    client: Client,
    pub active_top_symbols: Vec<String>,
}

impl DynamicSymbolSelector {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            active_top_symbols: Vec::new(),
        }
    }

    /// Filtra y clasifica todo el universo de Binance Futures
    pub fn refresh_universe(&mut self) -> Result<(Vec<String>, Vec<String>), String> {
        let is_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
        let base_url = if is_testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        };
        let url = format!("{}/fapi/v1/ticker/24hr", base_url);
        let res = self.client.get(&url).send().map_err(|e| e.to_string())?;
        
        if !res.status().is_success() {
            return Err(format!("Binance API Error: {}", res.status()));
        }

        let body = res.text().map_err(|e| e.to_string())?;
        let items: Vec<Value> = serde_json::from_str(&body).unwrap_or_default();
        
        let mut heap = BinaryHeap::new();

        for item in items {
            if let (Some(symbol), Some(volume_str), Some(price_change_str), Some(_status)) = (
                item.get("symbol").and_then(|v| v.as_str()),
                item.get("quoteVolume").and_then(|v| v.as_str()),
                item.get("priceChangePercent").and_then(|v| v.as_str()),
                item.get("lastPrice").and_then(|v| v.as_str()), // We just ensure it has a price
            ) {
                // Must be acapital baseT pair
                if !symbol.ends_with("USDT") {
                    continue;
                }

                let vol: f64 = volume_str.parse().unwrap_or(0.0);
                let change_pct: f64 = price_change_str.parse::<f64>().unwrap_or(0.0).abs();
                
                // Filtro Anti-Riesgo: Mínimo 10 Millonescapital base de volumen 24h
                if vol < 10_000_000.0 {
                    continue;
                }

                // Ecuación Cuántica (Volatilidad * Log(Volumen))
                let score = change_pct * vol.log10();
                
                heap.push(SymbolScore {
                    symbol: symbol.to_string(),
                    volume_usd: vol,
                    price_change_pct: change_pct,
                    score,
                });
            }
        }

        let mut top_64 = Vec::new();
        let mut top_10 = Vec::new();
        
        // BTCUSDT siempre debe estar por decreto de ancla
        top_64.push("BTCUSDT".to_string());
        top_10.push("BTCUSDT".to_string());
        
        while let Some(top) = heap.pop() {
            if top.symbol != "BTCUSDT" {
                if top_64.len() < 64 {
                    top_64.push(top.symbol.clone());
                }
                if top_10.len() < 10 {
                    top_10.push(top.symbol.clone());
                }
            }
            if top_64.len() >= 64 {
                break;
            }
        }

        self.active_top_symbols = top_10.clone();
        println!("🌌 [DYNAMIC-SYMBOLS] Top 10 Universo Seleccionado: {:?}", self.active_top_symbols);
        Ok((top_64, top_10))
    }
}
