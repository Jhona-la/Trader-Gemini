use quantum_arena::symbol_registry::SymbolSpec;
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
        .filter(|t| t.symbol.ends_with("USDT"))
        .filter_map(|t| {
            let vol = t.quote_volume.parse::<f64>().unwrap_or(0.0);
            let pct = t.price_change_percent.parse::<f64>().unwrap_or(0.0).abs();
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

    // Ordenar por métrica combinada
    valid_assets.sort_by(|a, b| {
        let score_a = a.volume * (1.0 + a.volatility / 100.0);
        let score_b = b.volume * (1.0 + b.volatility / 100.0);
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

            specs.push(SymbolSpec {
                symbol: asset.symbol.clone(),
                step_size,
                tick_size,
                min_qty,
                min_notional,
                max_leverage: 20,  // Default safe max leverage
                maker_fee: 0.0002, // Default VIP0
                taker_fee: 0.0005, // Default VIP0
                is_shadow: idx >= 10,
            });
        }
    }

    Ok(specs)
}
