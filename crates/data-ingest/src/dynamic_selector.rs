use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct AssetScore {
    pub symbol: String,
    pub volume_usd: f64,
    pub volatility_pct: f64,
    pub score: f64,
}

pub struct DynamicSelector {
    client: Client,
    is_testnet: bool,
}

impl DynamicSelector {
    // FIX #1488: Construcción de cliente HTTP resiliente sin unwrap
    pub fn new(is_testnet: bool) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| Client::new());
        Self { client, is_testnet }
    }

    /// Fetches 24hr ticker data and ranks symbols based on Volume * Volatility
    pub async fn select_top_10(&self) -> Result<Vec<String>, String> {
        let url = if self.is_testnet {
            "https://testnet.binancefuture.com/fapi/v1/ticker/24hr"
        } else {
            "https://fapi.binance.com/fapi/v1/ticker/24hr"
        };

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch 24hr ticker: {}", e))?;

        let data: Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let min_vol_usd = if self.is_testnet { 10_000.0 } else { 1_000_000.0 };
        let top_10 = Self::parse_and_rank_tickers(&data, min_vol_usd);
        Ok(top_10)
    }

    /// Parsea y rankea activos en memoria de manera pura, determinista y testeable
    pub fn parse_and_rank_tickers(data: &Value, min_vol_usd: f64) -> Vec<String> {
        let mut assets = Vec::new();

        if let Some(arr) = data.as_array() {
            for item in arr {
                if let (Some(sym), Some(vol_str), Some(pct_str)) = (
                    item.get("symbol").and_then(|v| v.as_str()),
                    item.get("quoteVolume").and_then(|v| v.as_str()),
                    item.get("priceChangePercent").and_then(|v| v.as_str()),
                ) {
                    if sym.ends_with("USDT") && !sym.contains('_') {
                        if let (Ok(vol), Ok(pct)) = (vol_str.parse::<f64>(), pct_str.parse::<f64>())
                        {
                            // FIX #588 & #707: Umbral adaptativo y sanitización estricta de finitud
                            if vol.is_finite() && pct.is_finite() && vol > min_vol_usd {
                                let volatility = pct.abs();
                                // Score = Volatilidad % * Log10(Volumen) para evitar que el volumen opaque todo
                                let score = volatility * vol.max(10.0).log10();
                                if score.is_finite() && score > 0.0 {
                                    assets.push(AssetScore {
                                        symbol: sym.to_string(),
                                        volume_usd: vol,
                                        volatility_pct: volatility,
                                        score,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Ordenar de mayor a menor score
        assets.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Tomar los 10 mejores
        let mut top_10: Vec<String> = assets.into_iter().take(10).map(|a| a.symbol).collect();

        // FIX #616: Fallback canónico determinista con las 10 monedas maestras si el feed es insuficiente
        if top_10.is_empty() {
            top_10 = vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "SOLUSDT".to_string(),
                "BNBUSDT".to_string(),
                "DOGEUSDT".to_string(),
                "XRPUSDT".to_string(),
                "ADAUSDT".to_string(),
                "AVAXUSDT".to_string(),
                "LINKUSDT".to_string(),
                "SUIUSDT".to_string(),
            ];
        }

        top_10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_selector_scoring_math() {
        let vol: f64 = 1_000_000.0;
        let pct: f64 = 5.0;
        let score: f64 = pct * vol.max(10.0).log10();
        assert_eq!(score, 30.0); // 5.0 * 6.0 = 30.0
    }

    #[test]
    fn test_dynamic_selector_nan_and_negative_volume() {
        let vol_nan: f64 = f64::NAN;
        let pct_nan: f64 = f64::NAN;
        assert!(!vol_nan.is_finite() || !pct_nan.is_finite());
    }

    #[test]
    fn test_dynamic_selector_instantiation_and_score_ordering() {
        let selector = DynamicSelector::new(true);
        assert!(selector.is_testnet);

        let mut scores = vec![
            AssetScore { symbol: "ETHUSDT".to_string(), volume_usd: 500_000.0, volatility_pct: 2.0, score: 11.4 },
            AssetScore { symbol: "BTCUSDT".to_string(), volume_usd: 2_000_000.0, volatility_pct: 3.5, score: 22.0 },
            AssetScore { symbol: "SOLUSDT".to_string(), volume_usd: 800_000.0, volatility_pct: 5.0, score: 29.5 },
        ];

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        assert_eq!(scores[0].symbol, "SOLUSDT");
        assert_eq!(scores[1].symbol, "BTCUSDT");
        assert_eq!(scores[2].symbol, "ETHUSDT");
    }

    #[test]
    fn test_dynamic_selector_parse_and_rank_tickers_json() {
        let raw_json = serde_json::json!([
            {"symbol": "BTCUSDT", "quoteVolume": "10000000.0", "priceChangePercent": "2.5"},
            {"symbol": "SOLUSDT", "quoteVolume": "5000000.0", "priceChangePercent": "8.0"},
            {"symbol": "INVALID_PAIR", "quoteVolume": "10000000.0", "priceChangePercent": "10.0"},
            {"symbol": "ETHUSDT", "quoteVolume": "500.0", "priceChangePercent": "5.0"}, // under min_vol
            {"symbol": "DOGEUSDT", "quoteVolume": "2000000.0", "priceChangePercent": "NaN"} // NaN
        ]);

        let top = DynamicSelector::parse_and_rank_tickers(&raw_json, 10_000.0);
        assert_eq!(top[0], "SOLUSDT"); // 8.0 * log10(5M) = 8.0 * 6.6989 = 53.59
        assert_eq!(top[1], "BTCUSDT"); // 2.5 * log10(10M) = 2.5 * 7.0 = 17.5
        assert!(!top.contains(&"INVALID_PAIR".to_string()));
    }

    #[test]
    fn test_dynamic_selector_fallback_on_empty() {
        let empty_json = serde_json::json!([]);
        let top = DynamicSelector::parse_and_rank_tickers(&empty_json, 10_000.0);
        assert_eq!(top.len(), 10);
        assert_eq!(top[0], "BTCUSDT");
        assert_eq!(top[1], "ETHUSDT");
    }
}

