use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration};

/// Estructura para almacenar métricas Macroeconómicas y TradFi
pub struct MacroState {
    pub dxy_index: AtomicU64,
    pub nasdaq_index: AtomicU64,
    pub sp500_index: AtomicU64,
    pub btc_dominance: AtomicU64,
}

impl Default for MacroState {
    fn default() -> Self {
        Self::new()
    }
}

impl MacroState {
    pub fn new() -> Self {
        Self {
            dxy_index: AtomicU64::new(104.5_f64.to_bits()),
            nasdaq_index: AtomicU64::new(18000.0_f64.to_bits()),
            sp500_index: AtomicU64::new(5100.0_f64.to_bits()),
            btc_dominance: AtomicU64::new(52.0_f64.to_bits()),
        }
    }

    pub fn update(&self, dxy: f64, ndx: f64, spy: f64, dom: f64) {
        if dxy > 0.0 && dxy.is_finite() {
            self.dxy_index.store(dxy.to_bits(), Ordering::Relaxed);
        }
        if ndx > 0.0 && ndx.is_finite() {
            self.nasdaq_index.store(ndx.to_bits(), Ordering::Relaxed);
        }
        if spy > 0.0 && spy.is_finite() {
            self.sp500_index.store(spy.to_bits(), Ordering::Relaxed);
        }
        if dom > 0.0 && dom.is_finite() {
            self.btc_dominance.store(dom.to_bits(), Ordering::Relaxed);
        }
    }

    pub fn get_dxy(&self) -> f64 {
        f64::from_bits(self.dxy_index.load(Ordering::Relaxed))
    }
    pub fn get_nasdaq(&self) -> f64 {
        f64::from_bits(self.nasdaq_index.load(Ordering::Relaxed))
    }
    pub fn get_sp500(&self) -> f64 {
        f64::from_bits(self.sp500_index.load(Ordering::Relaxed))
    }
    /// FIX #1424: Getter expuesto para BTC Dominance
    pub fn get_btc_dominance(&self) -> f64 {
        f64::from_bits(self.btc_dominance.load(Ordering::Relaxed))
    }
}

/// Tarea asíncrona que extrae datos TradFi REALES de FRED y Binance cada 60 segundos
pub async fn run_macro_feed_poller(state: Arc<MacroState>) {
    let mut ticker = interval(Duration::from_secs(60));
    let client = reqwest::Client::builder()
        .user_agent("TraderGemini/5.0 (Windows NT 10.0; Win64; x64)")
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let fred_series: [(&str, &str); 3] = [
        ("SP500", "SP500"),
        ("NASDAQCOM", "NASDAQ"),
        ("DTWEXBGS", "DXY"),
    ];

    loop {
        ticker.tick().await;

        let mut dxy = f64::from_bits(state.dxy_index.load(Ordering::Relaxed));
        let mut ndx = f64::from_bits(state.nasdaq_index.load(Ordering::Relaxed));
        let mut spy = f64::from_bits(state.sp500_index.load(Ordering::Relaxed));

        for (series, tag) in &fred_series {
            let url = format!(
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}",
                series
            );
            if let Ok(res) = client.get(&url).send().await {
                if let Ok(csv) = res.text().await {
                    if let Some((_, val)) = csv
                        .lines()
                        .skip(1)
                        .filter_map(|l| {
                            let mut parts = l.split(',');
                            let d = parts.next()?.trim();
                            let v = parts.next()?.trim();
                            let f = v.parse::<f64>().ok()?;
                            Some((d, f))
                        })
                        .last()
                    {
                        match *tag {
                            "DXY" => dxy = val,
                            "NASDAQ" => ndx = val,
                            "SP500" => spy = val,
                            _ => {}
                        }
                    }
                }
            }
        }

        // Fetch BTC dominance proxy via CoinGecko or DefiLlama global market cap if accessible
        let mut btc_dom = f64::from_bits(state.btc_dominance.load(Ordering::Relaxed));
        if let Ok(res) = client.get("https://api.coingecko.com/api/v3/global").send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(dom) = json.get("data")
                    .and_then(|d| d.get("market_cap_percentage"))
                    .and_then(|m| m.get("btc"))
                    .and_then(|v| v.as_f64())
                {
                    btc_dom = dom;
                }
            }
        }

        state.update(dxy, ndx, spy, btc_dom);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_state_initial_values_and_update() {
        let state = MacroState::new();
        assert!((state.get_dxy() - 104.5).abs() < 1e-4);
        assert!((state.get_nasdaq() - 18000.0).abs() < 1e-4);
        assert!((state.get_sp500() - 5100.0).abs() < 1e-4);
        assert!((state.get_btc_dominance() - 52.0).abs() < 1e-4);

        state.update(105.2, 18500.0, 5200.0, 54.5);
        assert_eq!(state.get_dxy(), 105.2);
        assert_eq!(state.get_nasdaq(), 18500.0);
        assert_eq!(state.get_sp500(), 5200.0);
        assert_eq!(state.get_btc_dominance(), 54.5);
    }

    #[test]
    fn test_macro_state_nan_and_negative_immunity() {
        let state = MacroState::new();
        let old_dxy = state.get_dxy();
        let old_ndx = state.get_nasdaq();
        let old_spy = state.get_sp500();
        let old_dom = state.get_btc_dominance();

        // Feed NaN and negatives
        state.update(f64::NAN, -100.0, f64::INFINITY, 0.0);
        assert_eq!(state.get_dxy(), old_dxy);
        assert_eq!(state.get_nasdaq(), old_ndx);
        assert_eq!(state.get_sp500(), old_spy);
        assert_eq!(state.get_btc_dominance(), old_dom);
    }
}

