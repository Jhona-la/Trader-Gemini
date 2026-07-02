use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use serde_json::Value;

/// Estructura para almacenar métricas Macroeconómicas y TradFi
pub struct MacroState {
    pub dxy_index: AtomicU64,
    pub nasdaq_index: AtomicU64,
    pub sp500_index: AtomicU64,
    pub btc_dominance: AtomicU64,
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
        if dxy > 0.0 { self.dxy_index.store(dxy.to_bits(), Ordering::Relaxed); }
        if ndx > 0.0 { self.nasdaq_index.store(ndx.to_bits(), Ordering::Relaxed); }
        if spy > 0.0 { self.sp500_index.store(spy.to_bits(), Ordering::Relaxed); }
        if dom > 0.0 { self.btc_dominance.store(dom.to_bits(), Ordering::Relaxed); }
    }
    
    pub fn get_dxy(&self) -> f64 { f64::from_bits(self.dxy_index.load(Ordering::Relaxed)) }
    pub fn get_nasdaq(&self) -> f64 { f64::from_bits(self.nasdaq_index.load(Ordering::Relaxed)) }
    pub fn get_sp500(&self) -> f64 { f64::from_bits(self.sp500_index.load(Ordering::Relaxed)) }
}

/// Tarea asíncrona que extrae datos TradFi REALES cada 60 segundos
pub async fn run_macro_feed_poller(state: Arc<MacroState>) {
    let mut ticker = interval(Duration::from_secs(60));
    let client = reqwest::Client::new();
    let url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=DX-Y.NYB,^GSPC,^NDX,BTC-USD";

    loop {
        ticker.tick().await;
        
        match client.get(url).send().await {
            Ok(res) => {
                if let Ok(json) = res.json::<Value>().await {
                    if let Some(results) = json["quoteResponse"]["result"].as_array() {
                        let mut dxy = 0.0;
                        let mut ndx = 0.0;
                        let mut spy = 0.0;
                        let mut btc = 0.0;

                        for item in results {
                            let symbol = item["symbol"].as_str().unwrap_or("");
                            let price = item["regularMarketPrice"].as_f64().unwrap_or(0.0);
                            
                            match symbol {
                                "DX-Y.NYB" => dxy = price,
                                "^NDX" => ndx = price,
                                "^GSPC" => spy = price,
                                "BTC-USD" => btc = price, // Placeholder para calcular dominancia si tuviéramos altcap
                                _ => {}
                            }
                        }

                        state.update(dxy, ndx, spy, 52.0);
                        // println!("🌐 [REAL MacroFeed] TradFi Updated: DXY={:.2} | NASDAQ={:.2} | SP500={:.2}", dxy, ndx, spy);
                    }
                }
            },
            Err(e) => {
                eprintln!("⚠️ Error obteniendo datos Macro TradFi: {}", e);
            }
        }
    }
}
