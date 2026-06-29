use serde::Deserialize;

/// Estructura de mapeo ultra-ligera para el evento "bookTicker" de Binance.
/// Binance manda strings para los precios y volúmenes, nosotros los pasaremos a f64 crudo.
#[derive(Debug, Deserialize)]
pub struct RawBookTickerEvent {
    #[serde(rename = "u")]
    pub update_id: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bid_price: String,
    #[serde(rename = "B")]
    pub bid_qty: String,
    #[serde(rename = "a")]
    pub ask_price: String,
    #[serde(rename = "A")]
    pub ask_qty: String,
}

#[derive(Debug, Clone)]
pub struct BookTickerEvent {
    pub update_id: u64,
    pub symbol: String,
    pub bid_price: f64,
    pub bid_qty: f64,
    pub ask_price: f64,
    pub ask_qty: f64,
}

impl BookTickerEvent {
    #[inline(always)]
    pub fn parse_from_json(json_str: &str) -> Option<Self> {
        let raw: RawBookTickerEvent = match serde_json::from_str(json_str) {
            Ok(event) => event,
            Err(_) => return None,
        };

        // Parseo de los strings a floats
        let bid_price = raw.bid_price.parse::<f64>().unwrap_or(0.0);
        let bid_qty = raw.bid_qty.parse::<f64>().unwrap_or(0.0);
        let ask_price = raw.ask_price.parse::<f64>().unwrap_or(0.0);
        let ask_qty = raw.ask_qty.parse::<f64>().unwrap_or(0.0);

        if bid_price == 0.0 || ask_price == 0.0 {
            return None;
        }

        Some(Self {
            update_id: raw.update_id,
            symbol: raw.symbol,
            bid_price,
            bid_qty,
            ask_price,
            ask_qty,
        })
    }
}
