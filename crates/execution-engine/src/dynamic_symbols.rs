use reqwest::Client;
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
    pub async fn refresh_universe(&mut self) -> Result<(Vec<String>, Vec<String>), String> {
        let is_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
        let base_url = if is_testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        };
        let url = format!("{}/fapi/v1/ticker/24hr", base_url);
        let res = self.client.get(&url).send().await.map_err(|e| e.to_string())?;

        if !res.status().is_success() {
            return Err(format!("Binance API Error: {}", res.status()));
        }

        let body = res.text().await.map_err(|e| e.to_string())?;
        let items: Vec<Value> = serde_json::from_str(&body).unwrap_or_default();
        
        let (top_64, top_10) = Self::parse_and_rank_json_tickers(&items, is_testnet);
        self.active_top_symbols = top_10.clone();
        println!("🌌 [DYNAMIC-SYMBOLS] Top 10 Universo Seleccionado: {:?}", self.active_top_symbols);
        Ok((top_64, top_10))
    }

    /// Función pura desacoplada para análisis, filtrado y ranking algorítmico de tickers JSON
    pub fn parse_and_rank_json_tickers(items: &[Value], is_testnet: bool) -> (Vec<String>, Vec<String>) {
        let mut heap = BinaryHeap::new();

        for item in items {
            if let (Some(symbol), Some(volume_str), Some(price_change_str), Some(_status)) = (
                item.get("symbol").and_then(|v| v.as_str()),
                item.get("quoteVolume").and_then(|v| v.as_str()),
                item.get("priceChangePercent").and_then(|v| v.as_str()),
                item.get("lastPrice").and_then(|v| v.as_str()), // We just ensure it has a price
            ) {
                // Must be a USDT pair
                if !symbol.ends_with("USDT") {
                    continue;
                }

                let vol: f64 = volume_str.parse().unwrap_or(0.0);
                let change_pct: f64 = price_change_str.parse::<f64>().unwrap_or(0.0).abs();
                // FIX #1498: Umbral adaptativo para Testnet vs Producción
                let min_vol = if is_testnet { 10_000.0 } else { 10_000_000.0 };
                if vol < min_vol {
                    continue;
                }

                // Ecuación Cuántica (Volatilidad * Log(Volumen))
                let score = change_pct * vol.max(10.0).log10();
                if score.is_finite() && score > 0.0 {
                    heap.push(SymbolScore {
                        symbol: symbol.to_string(),
                        volume_usd: vol,
                        price_change_pct: change_pct,
                        score,
                    });
                }
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

        (top_64, top_10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_score_ordering_and_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(SymbolScore {
            symbol: "ETHUSDT".to_string(),
            volume_usd: 50_000_000.0,
            price_change_pct: 2.0,
            score: 15.0,
        });
        heap.push(SymbolScore {
            symbol: "SOLUSDT".to_string(),
            volume_usd: 80_000_000.0,
            price_change_pct: 5.0,
            score: 35.0,
        });
        heap.push(SymbolScore {
            symbol: "DOGEUSDT".to_string(),
            volume_usd: 10_000_000.0,
            price_change_pct: 1.0,
            score: 7.0,
        });

        // Top score should be SOLUSDT with score 35.0
        let top = heap.pop().unwrap();
        assert_eq!(top.symbol, "SOLUSDT");
        assert_eq!(top.score, 35.0);

        let second = heap.pop().unwrap();
        assert_eq!(second.symbol, "ETHUSDT");
    }

    #[test]
    fn test_symbol_score_equality() {
        let s1 = SymbolScore {
            symbol: "BTCUSDT".to_string(),
            volume_usd: 100.0,
            price_change_pct: 1.0,
            score: 10.0,
        };
        let s2 = SymbolScore {
            symbol: "ETHUSDT".to_string(),
            volume_usd: 200.0,
            price_change_pct: 2.0,
            score: 10.0,
        };
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_dynamic_symbol_selector_instantiation() {
        let selector = DynamicSymbolSelector::new();
        assert!(selector.active_top_symbols.is_empty());
    }

    #[test]
    fn test_symbol_score_nan_cmp_defense() {
        let s_nan = SymbolScore {
            symbol: "NAN_COIN".to_string(),
            volume_usd: 100.0,
            price_change_pct: 1.0,
            score: f64::NAN,
        };
        let s_valid = SymbolScore {
            symbol: "VALID_COIN".to_string(),
            volume_usd: 100.0,
            price_change_pct: 1.0,
            score: 10.0,
        };
        // Verify cmp fallback does not panic
        assert_eq!(s_nan.cmp(&s_valid), Ordering::Equal);
    }

    #[test]
    fn test_dynamic_symbol_selector_parse_and_rank_json() {
        let items: Vec<Value> = serde_json::from_str(r#"[
            {"symbol": "SOLUSDT", "quoteVolume": "15000000.0", "priceChangePercent": "5.0", "lastPrice": "150.0"},
            {"symbol": "ETHUSDT", "quoteVolume": "50000000.0", "priceChangePercent": "2.0", "lastPrice": "3000.0"},
            {"symbol": "LOWVOLUSDT", "quoteVolume": "5000.0", "priceChangePercent": "10.0", "lastPrice": "1.0"},
            {"symbol": "BTCBUSD", "quoteVolume": "100000000.0", "priceChangePercent": "3.0", "lastPrice": "60000.0"}
        ]"#).unwrap();

        let (top_64, top_10) = DynamicSymbolSelector::parse_and_rank_json_tickers(&items, false);
        assert!(top_10.contains(&"BTCUSDT".to_string()));
        assert!(top_10.contains(&"SOLUSDT".to_string()));
        assert!(top_10.contains(&"ETHUSDT".to_string()));
        // LOWVOLUSDT (5000 < 10M) and BTCBUSD (not USDT) must be excluded
        assert!(!top_10.contains(&"LOWVOLUSDT".to_string()));
        assert!(!top_10.contains(&"BTCBUSD".to_string()));
        assert!(!top_64.is_empty());
    }

    #[test]
    fn test_dynamic_symbol_selector_testnet_adaptive_volume() {
        let items: Vec<Value> = serde_json::from_str(r#"[
            {"symbol": "AVAXUSDT", "quoteVolume": "25000.0", "priceChangePercent": "4.0", "lastPrice": "30.0"}
        ]"#).unwrap();

        // In production (>10M), 25K is filtered out
        let (_, top_10_prod) = DynamicSymbolSelector::parse_and_rank_json_tickers(&items, false);
        assert!(!top_10_prod.contains(&"AVAXUSDT".to_string()));

        // In testnet (>10K), 25K is accepted
        let (_, top_10_testnet) = DynamicSymbolSelector::parse_and_rank_json_tickers(&items, true);
        assert!(top_10_testnet.contains(&"AVAXUSDT".to_string()));
    }
}


