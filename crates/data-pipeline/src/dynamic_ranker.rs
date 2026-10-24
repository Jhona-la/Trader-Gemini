use quantum_arena::symbol_registry::SymbolSpec;
use quantum_arena::symbols::{calculate_spectral_universe_score, is_sane_binance_symbol};
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

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ExchangeInfo {
    symbols: Vec<ExchangeSymbol>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ExchangeSymbol {
    symbol: String,
    status: String,
    filters: Vec<Filter>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
#[serde(tag = "filterType")]
enum Filter {
    #[serde(rename = "PRICE_FILTER")]
    Price {
        #[serde(rename = "tickSize")]
        tick_size: String,
    },
    #[serde(rename = "LOT_SIZE")]
    LotSize {
        #[serde(rename = "stepSize")]
        step_size: String,
        #[serde(rename = "minQty")]
        min_qty: String,
    },
    #[serde(rename = "MIN_NOTIONAL")]
    MinNotional {
        #[serde(default)]
        notional: Option<String>,
        #[serde(rename = "minNotional", default)]
        min_notional: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone)]
pub struct SelectedAsset {
    pub symbol: String,
    pub volume: f64,
    pub volatility: f64,
}

pub async fn fetch_dynamic_universe(
    limit: usize,
    is_testnet: bool,
) -> Result<Vec<SymbolSpec>, String> {
    let base_url = if is_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };

    let client = Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;

    // 1. Fetch Tickers for Ranking
    let ticker_url = format!("{}/fapi/v1/ticker/24hr", base_url);
    let ticker_res = client
        .get(&ticker_url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !ticker_res.status().is_success() {
        return Err(format!(
            "Ticker API falló con status: {}",
            ticker_res.status()
        ));
    }
    let tickers: Vec<Ticker24h> = ticker_res.json().await.map_err(|e| e.to_string())?;

    let min_vol = if is_testnet { 0.0 } else { 1_000_000.0 };

    let mut valid_assets: Vec<SelectedAsset> = tickers
        .into_iter()
        .filter(|t| is_sane_binance_symbol(&t.symbol, is_testnet))
        .filter_map(|t| {
            // F1-091 — misma fuente única que `asset_selector`: estadística
            // ilegible ⇒ símbolo excluido, nunca volumen inventado.
            let (vol, pct) =
                crate::cero_dato::estadistica_legible(&t.quote_volume, &t.price_change_percent)?;
            if vol >= min_vol {
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

    // F4.6 / Ola Ω215 — RANKING ESPECTRAL CANÓNICO UNIFICADO:
    // score = calculate_spectral_universe_score(volume, volatility, None, None)
    // Combina ln(1+vol) con la campana unimodal pct*exp(-pct/15) sin duplicidad de fórmulas.
    valid_assets.sort_by(|a, b| {
        let score_b = calculate_spectral_universe_score(b.volume, b.volatility, None, None);
        let score_a = calculate_spectral_universe_score(a.volume, a.volatility, None, None);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    valid_assets.truncate(limit);

    // 2. Fetch Exchange Info to build SymbolSpec
    let info_url = format!("{}/fapi/v1/exchangeInfo", base_url);
    let info_res = client
        .get(&info_url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !info_res.status().is_success() {
        return Err(format!(
            "ExchangeInfo API falló con status: {}",
            info_res.status()
        ));
    }
    let exchange_info: ExchangeInfo = info_res.json().await.map_err(|e| e.to_string())?;

    let mut specs = Vec::new();
    for (idx, asset) in valid_assets.into_iter().enumerate() {
        if let Some(ex_sym) = exchange_info
            .symbols
            .iter()
            .find(|s| s.symbol == asset.symbol && s.status == "TRADING")
        {
            let mut step_size = 0.001;
            let mut tick_size = 0.01;
            let mut min_qty = 0.001;
            let mut min_notional = 5.0; // By default binance futures uses 5.0 USD min notional

            for filter in &ex_sym.filters {
                match filter {
                    Filter::Price { tick_size: ts } => {
                        tick_size = ts.parse().unwrap_or(tick_size);
                    }
                    Filter::LotSize {
                        step_size: ss,
                        min_qty: mq,
                    } => {
                        step_size = ss.parse().unwrap_or(step_size);
                        min_qty = mq.parse().unwrap_or(min_qty);
                    }
                    Filter::MinNotional {
                        notional,
                        min_notional: mn,
                    } => {
                        if let Some(n) = notional {
                            min_notional = n.parse().unwrap_or(min_notional);
                        } else if let Some(m) = mn {
                            min_notional = m.parse().unwrap_or(min_notional);
                        }
                    }
                    _ => {}
                }
            }

            let max_leverage = match asset.symbol.as_str() {
                "BTCUSDT" | "ETHUSDT" => 125,
                "SOLUSDT" | "BNBUSDT" | "DOGEUSDT" | "XRPUSDT" | "ADAUSDT" | "AVAXUSDT" => 75,
                _ => 50,
            };

            specs.push(SymbolSpec {
                symbol: asset.symbol.clone(),
                step_size,
                tick_size,
                min_qty,
                min_notional,
                max_leverage,
                maker_fee: 0.0002, // Default VIP0
                taker_fee: 0.0005, // Default VIP0
                is_shadow: idx >= 10,
            });
        }
    }

    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// QO-U1a / Ola Ω215 — el saneamiento canónico mata la basura de testnet:
    /// mojibake (牛来USDT), triviales (4USDT/AUSDT), stablecoins y blacklist.
    #[test]
    fn qo_u1a_sanitizer_rechaza_basura_testnet() {
        // Basura REAL rechazada mediante predicado canónico:
        assert!(!is_sane_binance_symbol("牛来USDT", false), "mojibake debe morir");
        assert!(!is_sane_binance_symbol("4USDT", false), "base trivial de 1 carácter");
        assert!(!is_sane_binance_symbol("AUSDT", false), "base trivial de 1 carácter");
        assert!(!is_sane_binance_symbol("USDCUSDT", false), "stablecoin");
        assert!(!is_sane_binance_symbol("FDUSDUSDT", false), "stablecoin");
        assert!(!is_sane_binance_symbol("USELESSUSDT", true), "blacklist testnet");

        // Legítimos sobreviven:
        assert!(is_sane_binance_symbol("BTCUSDT", false));
        assert!(is_sane_binance_symbol("NEARUSDT", false));
        assert!(is_sane_binance_symbol("1000PEPEUSDT", false), "bases con dígitos son legítimas");
    }

    #[test]
    fn test_dynamic_ranker_scoring_function() {
        let score_15 = calculate_spectral_universe_score(10_000_000.0, 15.0, None, None);
        let score_50 = calculate_spectral_universe_score(10_000_000.0, 50.0, None, None);

        // Volatilidad moderada (15%) debe tener mayor puntaje que extrema (50%)
        assert!(
            score_15 > score_50,
            "15% vol debe puntuar más alto que 50% vol"
        );
    }

    #[test]
    fn test_dynamic_ranker_nan_immunity() {
        assert_eq!(calculate_spectral_universe_score(f64::NAN, 15.0, None, None), 0.0);
        assert_eq!(calculate_spectral_universe_score(1000.0, f64::NAN, None, None), 0.0);
        assert_eq!(calculate_spectral_universe_score(f64::NAN, f64::NAN, None, None), 0.0);
    }
}
