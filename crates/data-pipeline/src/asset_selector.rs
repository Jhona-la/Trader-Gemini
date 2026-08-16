use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct Ticker24h {
    symbol: String,
    #[serde(rename = "quoteVolume")]
    quote_volume: String,
    #[serde(rename = "priceChangePercent")]
    price_change_percent: String,
}

#[derive(Debug, Clone)]
pub struct SelectedAsset {
    pub symbol: String,
    pub volume: f64,
    pub volatility: f64,
}

/// Consulta Binance API (Futures) para extraer los Top N símbolos más líquidos y volátiles.
pub async fn fetch_top_dynamic_assets(limit: usize, is_testnet: bool) -> Result<Vec<SelectedAsset>, String> {
    let base_url = if is_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };

    let url = format!("{}/fapi/v1/ticker/24hr", base_url);
    
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;

    let res = client.get(&url).send().await.map_err(|e| e.to_string())?;
    
    if !res.status().is_success() {
        return Err(format!("Binance API falló con status: {}", res.status()));
    }

    let tickers: Vec<Ticker24h> = res.json().await.map_err(|e| e.to_string())?;

    let mut valid_assets: Vec<SelectedAsset> = tickers
        .into_iter()
        .filter(|t| t.symbol.ends_with("USDT"))
        .filter_map(|t| {
            let vol = t.quote_volume.parse::<f64>().unwrap_or(0.0);
            let pct = t.price_change_percent.parse::<f64>().unwrap_or(0.0).abs();
            if vol > 10_000_000.0 { // Filtrar ilíquidos extremos (menos de $10M al día)
                Some(SelectedAsset {
                    symbol: t.symbol,
                    volume: vol,
                    volatility: pct,
                })
            } else {
                None
            }
        })
        .collect();

    // Ordenar por una métrica combinada cuántica (Volumen * Volatilidad) para HFT puro
    // Queremos los pares donde más dinero se mueve Y que más se mueven
    valid_assets.sort_by(|a, b| {
        let score_a = a.volume * (1.0 + a.volatility / 100.0);
        let score_b = b.volume * (1.0 + b.volatility / 100.0);
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    valid_assets.truncate(limit);
    Ok(valid_assets)
}
