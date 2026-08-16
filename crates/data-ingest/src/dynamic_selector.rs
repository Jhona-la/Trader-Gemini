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
    pub fn new(is_testnet: bool) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
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

        let mut assets = Vec::new();

        if let Some(arr) = data.as_array() {
            for item in arr {
                if let (Some(sym), Some(vol_str), Some(pct_str)) = (
                    item.get("symbol").and_then(|v| v.as_str()),
                    item.get("quoteVolume").and_then(|v| v.as_str()),
                    item.get("priceChangePercent").and_then(|v| v.as_str()),
                ) {
                    if sym.ends_with("USDT") && !sym.contains("_") {
                        if let (Ok(vol), Ok(pct)) = (vol_str.parse::<f64>(), pct_str.parse::<f64>()) {
                            // Ignorar activos con nulo volumen (delisted o pausados)
                            if vol > 1_000_000.0 {
                                let volatility = pct.abs();
                                // Score = Volatilidad % * Log10(Volumen) para evitar que el volumen opaque todo
                                let score = volatility * vol.log10();
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

        // Ordenar de mayor a menor score
        assets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Tomar los 10 mejores
        let top_10: Vec<String> = assets.into_iter().take(10).map(|a| a.symbol).collect();
        
        if top_10.is_empty() {
            Err("No se encontraron activos que cumplan el criterio".to_string())
        } else {
            Ok(top_10)
        }
    }
}
