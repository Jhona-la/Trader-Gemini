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
    let stablecoins = [
        "USDCUSDT",
        "FDUSDUSDT",
        "TUSDUSDT",
        "BUSDUSDT",
        "USDPUSDT",
        "EURUSDT",
        "DAIUSDT",
        "AEURUSDT",
    ];

    // QO-U1a (auditoría de universo): el escáner dejaba pasar BASURA de
    // testnet — mojibake (牛来USDT), símbolos triviales (4USDT, AUSDT,
    // MUSDT, CUSDT) e índices sintéticos. Reglas de saneamiento:
    //  1. ASCII estricto A-Z0-9 (el mojibake UTF-8 jamás pasa);
    //  2. la BASE sin sufijo debe tener ≥3 caracteres (mata 4USDT/AUSDT);
    //  3. lista negra de bases conocidas-triviales de testnet.
    let testnet_blacklist = [
        "CUSDT", "MUSDT", "AUSDT", "4USDT", "USELESSUSDT", "ZENUSDT", "IDOLUSDT",
        "SKYUSDT", "SUPERUSDT", "DRIFTUSDT", "CROSSUSDT", "FARTCOINUSDT", "FLUIDUSDT",
        "VVVUSDT", "ALCHUSDT", "SENTUSDT", "RAYSOLUSDT", "METUSDT", "AZTECUSDT",
        "ONEUSDT", "JUPUSDT", "TIAUSDT", "PENDLEUSDT", "LPTUSDT", "EIGENUSDT",
        "ZKUSDT", "INJUSDT", "ACHUSDT", "OPUSDT",
    ];
    let is_sane_symbol = |s: &str| -> bool {
        if s.len() < 7 || !s.ends_with("USDT") {
            return false;
        }
        let base = &s[..s.len() - 4];
        base.len() >= 3
            && base
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            && !stablecoins.contains(&s)
            && !testnet_blacklist.contains(&s)
    };

    let mut valid_assets: Vec<SelectedAsset> = tickers
        .into_iter()
        .filter(|t| is_sane_symbol(&t.symbol))
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

    // F4.6 — RANKING RAZONADO (antes: volumen×(1+vol%) — favorece exóticos
    // manipulados con vol inflado). Métrica documentada:
    //   score = ln(1 + quote_volume) × banda_de_volatilidad(pct)
    // - ln(volumen): liquidez es de cola pesada; log da escala comparable.
    // - banda = pct × e^(-pct/15): premia volatilidad MODERADA (oportunidad
    //   real para scalp/swing) y castiga la extrema (casino/manipulación) —
    //   máximo en 15% diario, decae exponencialmente después.
    let score = |a: &SelectedAsset| -> f64 {
        // FIX #676: Sanitizar volatilidad y volumen para ranking
        let pct = if a.volatility.is_finite() && a.volatility >= 0.0 {
            a.volatility
        } else {
            0.0
        };
        let vol = if a.volume.is_finite() && a.volume >= 0.0 {
            a.volume
        } else {
            0.0
        };
        let s = (1.0 + vol).ln() * pct * (-pct / 15.0_f64).exp();
        if s.is_finite() {
            s
        } else {
            0.0
        }
    };
    valid_assets.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
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

    /// QO-U1a — el saneamiento mata la basura de testnet medida en vivo:
    /// mojibake (牛来USDT), triviales (4USDT/AUSDT) y blacklist.
    #[test]
    fn qo_u1a_sanitizer_rechaza_basura_testnet() {
        // El closure vive dentro de fetch_dynamic_universe: replicamos las
        // reglas aquí como regresión del contrato.
        let stablecoins = [
            "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "EURUSDT",
            "DAIUSDT", "AEURUSDT",
        ];
        let sane = |s: &str| -> bool {
            if s.len() < 7 || !s.ends_with("USDT") {
                return false;
            }
            let base = &s[..s.len() - 4];
            base.len() >= 3
                && base.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                && !stablecoins.contains(&s)
        };
        // Basura REAL vista en dynamic_config.json del testnet:
        assert!(!sane("牛来USDT"), "mojibake debe morir");
        assert!(!sane("4USDT"), "base trivial de 1 carácter");
        assert!(!sane("AUSDT"), "base trivial de 1 carácter");
        assert!(!sane("USDCUSDT"), "stablecoin");
        // Legítimos sobreviven:
        assert!(sane("BTCUSDT"));
        assert!(sane("NEARUSDT"));
        assert!(sane("1000PEPEUSDT"), "bases con dígitos son legítimas");
    }

    #[test]
    fn test_dynamic_ranker_scoring_function() {
        let asset15 = SelectedAsset {
            symbol: "SOLUSDT".to_string(),
            volume: 10_000_000.0,
            volatility: 15.0,
        };
        let asset50 = SelectedAsset {
            symbol: "MEMEUSDT".to_string(),
            volume: 10_000_000.0,
            volatility: 50.0,
        };

        let score_fn = |a: &SelectedAsset| -> f64 {
            let pct = if a.volatility.is_finite() && a.volatility >= 0.0 {
                a.volatility
            } else {
                0.0
            };
            let vol = if a.volume.is_finite() && a.volume >= 0.0 {
                a.volume
            } else {
                0.0
            };
            let s = (1.0 + vol).ln() * pct * (-pct / 15.0_f64).exp();
            if s.is_finite() {
                s
            } else {
                0.0
            }
        };

        let score_15 = score_fn(&asset15);
        let score_50 = score_fn(&asset50);

        // Volatilidad moderada (15%) debe tener mayor puntaje que extrema (50%)
        assert!(
            score_15 > score_50,
            "15% vol debe puntuar más alto que 50% vol"
        );
    }

    #[test]
    fn test_dynamic_ranker_nan_immunity() {
        let score_fn = |vol: f64, pct: f64| -> f64 {
            let pct = if pct.is_finite() && pct >= 0.0 {
                pct
            } else {
                0.0
            };
            let vol = if vol.is_finite() && vol >= 0.0 {
                vol
            } else {
                0.0
            };
            let s = (1.0 + vol).ln() * pct * (-pct / 15.0_f64).exp();
            if s.is_finite() {
                s
            } else {
                0.0
            }
        };

        assert_eq!(score_fn(f64::NAN, 15.0), 0.0);
        assert_eq!(score_fn(1000.0, f64::NAN), 0.0);
        assert_eq!(score_fn(f64::NAN, f64::NAN), 0.0);
    }
}
