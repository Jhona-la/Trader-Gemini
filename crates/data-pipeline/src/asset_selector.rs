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

    let stablecoins = [
        "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "EURUSDT", "DAIUSDT", "AEURUSDT",
    ];

    let min_vol = if is_testnet { 0.0 } else { 1_000_000.0 };
    let min_pct = if is_testnet { 0.0 } else { 0.1 };

    let mut valid_assets: Vec<SelectedAsset> = tickers
        .into_iter()
        .filter(|t| t.symbol.ends_with("USDT") && !stablecoins.contains(&t.symbol.as_str()))
        .filter_map(|t| {
            let vol = t.quote_volume.parse::<f64>().unwrap_or(0.0);
            let pct = t.price_change_percent.parse::<f64>().unwrap_or(0.0).abs();
            // FIX #1422: Umbral sensible al entorno (Testnet vs Mainnet)
            if vol >= min_vol && pct >= min_pct {
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

    // FIX #1422: Ranking unificado con log-volume y banda de volatilidad exponencial
    let score = |a: &SelectedAsset| -> f64 {
        let pct = if a.volatility.is_finite() && a.volatility >= 0.0 { a.volatility } else { 0.0 };
        let vol = if a.volume.is_finite() && a.volume >= 0.0 { a.volume } else { 0.0 };
        let s = (1.0 + vol).ln() * pct * (-pct / 15.0_f64).exp();
        if s.is_finite() { s } else { 0.0 }
    };
    valid_assets.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    valid_assets.truncate(limit);
    Ok(valid_assets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_ranking_score_monotonicity() {
        let a1 = SelectedAsset {
            symbol: "BTCUSDT".to_string(),
            volume: 10_000_000.0,
            volatility: 3.5,
        };
        let a2 = SelectedAsset {
            symbol: "SHIBUSDT".to_string(),
            volume: 1_000_000.0,
            volatility: 1.0,
        };

        let score = |a: &SelectedAsset| -> f64 {
            let pct = if a.volatility.is_finite() && a.volatility >= 0.0 { a.volatility } else { 0.0 };
            let vol = if a.volume.is_finite() && a.volume >= 0.0 { a.volume } else { 0.0 };
            let s = (1.0 + vol).ln() * pct * (-pct / 15.0_f64).exp();
            if s.is_finite() { s } else { 0.0 }
        };

        assert!(score(&a1) > score(&a2));
    }

    #[test]
    fn test_asset_ranking_nan_and_negative_immunity() {
        let bad = SelectedAsset {
            symbol: "BADUSDT".to_string(),
            volume: f64::NAN,
            volatility: -5.0,
        };

        let score = |a: &SelectedAsset| -> f64 {
            let pct = if a.volatility.is_finite() && a.volatility >= 0.0 { a.volatility } else { 0.0 };
            let vol = if a.volume.is_finite() && a.volume >= 0.0 { a.volume } else { 0.0 };
            let s = (1.0 + vol).ln() * pct * (-pct / 15.0_f64).exp();
            if s.is_finite() { s } else { 0.0 }
        };

        assert_eq!(score(&bad), 0.0);
    }
}
