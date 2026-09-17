use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration};

pub struct OnChainState {
    pub btc_funding_rate: AtomicU64,
    pub btc_open_interest: AtomicU64,
    pub alt_funding_rates: [AtomicU64; 32],
    pub alt_open_interests: [AtomicU64; 32],
}

impl Default for OnChainState {
    fn default() -> Self {
        Self::new()
    }
}

impl OnChainState {
    pub fn new() -> Self {
        // Inicializar 32 slots de atomic u64 en 0.0
        const ZERO_BITS: u64 = 0;
        Self {
            btc_funding_rate: AtomicU64::new(ZERO_BITS),
            btc_open_interest: AtomicU64::new(ZERO_BITS),
            alt_funding_rates: [
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
            ],
            alt_open_interests: [
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
                AtomicU64::new(ZERO_BITS),
            ],
        }
    }

    pub fn get_funding(&self) -> f64 {
        f64::from_bits(self.btc_funding_rate.load(Ordering::Relaxed))
    }

    pub fn get_oi(&self) -> f64 {
        f64::from_bits(self.btc_open_interest.load(Ordering::Relaxed))
    }

    pub fn get_symbol_funding(&self, coin_id: usize) -> f64 {
        if coin_id < 32 {
            f64::from_bits(self.alt_funding_rates[coin_id].load(Ordering::Relaxed))
        } else {
            self.get_funding()
        }
    }

    pub fn set_symbol_funding(&self, coin_id: usize, funding: f64) {
        if coin_id < 32 {
            // FIX #1539: Clamping de tasa de fondeo de altcoins a [-1.0, 1.0]
            let safe_funding = if funding.is_finite() {
                funding.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            self.alt_funding_rates[coin_id].store(safe_funding.to_bits(), Ordering::Relaxed);
        }
    }

    pub fn get_symbol_oi(&self, coin_id: usize) -> f64 {
        if coin_id < 32 {
            f64::from_bits(self.alt_open_interests[coin_id].load(Ordering::Relaxed))
        } else {
            self.get_oi()
        }
    }

    pub fn set_symbol_oi(&self, coin_id: usize, oi: f64) {
        if coin_id < 32 {
            let safe_oi = if oi.is_finite() && oi >= 0.0 { oi } else { 0.0 };
            self.alt_open_interests[coin_id].store(safe_oi.to_bits(), Ordering::Relaxed);
        }
    }
}

/// Extrae Funding Rate y Open Interest de Binance Futures para todos los activos
pub async fn run_onchain_feed(state: Arc<OnChainState>) {
    let mut ticker = interval(Duration::from_secs(30));
    // FIX #1425: Timeout de 10s para prevenir bloqueos por indisponibilidad de red
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let is_testnet = std::env::var("USE_TESTNET")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        == "true"
        || std::env::var("SHADOW_MODE")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            == "true";
    let base_url = if is_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };
    let url_funding_global = format!("{}/fapi/v1/premiumIndex", base_url);
    let url_oi = format!("{}/fapi/v1/openInterest?symbol=BTCUSDT", base_url);

    loop {
        ticker.tick().await;

        // Fetch Global Funding Rates (todos los símbolos del universo activo)
        if let Ok(res) = client.get(&url_funding_global).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(arr) = json.as_array() {
                    for item in arr {
                        let sym = item.get("symbol").and_then(|v| v.as_str()).unwrap_or("");
                        if let Some(funding_str) =
                            item.get("lastFundingRate").and_then(|v| v.as_str())
                        {
                            if let Ok(funding) = funding_str.parse::<f64>() {
                                if funding.is_finite() {
                                    // FIX #1539: Clamping de tasa de fondeo a [-1.0, 1.0]
                                    let safe_funding = funding.clamp(-1.0, 1.0);
                                    if sym == "BTCUSDT" {
                                        state
                                            .btc_funding_rate
                                            .store(safe_funding.to_bits(), Ordering::Relaxed);
                                    }
                                    if let Some(coin_id) = quantum_arena::symbols::get_coin_id(sym)
                                    {
                                        state.set_symbol_funding(coin_id, safe_funding);
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(funding_str) =
                    json.get("lastFundingRate").and_then(|v| v.as_str())
                {
                    if let Ok(funding) = funding_str.parse::<f64>() {
                        if funding.is_finite() {
                            let safe_funding = funding.clamp(-1.0, 1.0);
                            state
                                .btc_funding_rate
                                .store(safe_funding.to_bits(), Ordering::Relaxed);
                        }
                    }
                }
            }
        }

        // Fetch Open Interest
        if let Ok(res) = client.get(&url_oi).send().await {
            if let Ok(json) = res.json::<Value>().await {
                if let Some(oi_str) = json.get("openInterest").and_then(|v| v.as_str()) {
                    if let Ok(oi) = oi_str.parse::<f64>() {
                        if oi.is_finite() && oi >= 0.0 {
                            state
                                .btc_open_interest
                                .store(oi.to_bits(), Ordering::Relaxed);
                            state.set_symbol_oi(0, oi);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onchain_state_get_and_set() {
        let state = OnChainState::new();
        state
            .btc_funding_rate
            .store(0.0001f64.to_bits(), Ordering::Relaxed);
        state
            .btc_open_interest
            .store(50000.0f64.to_bits(), Ordering::Relaxed);

        assert_eq!(state.get_funding(), 0.0001);
        assert_eq!(state.get_oi(), 50000.0);

        state.set_symbol_funding(5, 0.0005);
        state.set_symbol_oi(5, 12000.0);
        assert_eq!(state.get_symbol_funding(5), 0.0005);
        assert_eq!(state.get_symbol_oi(5), 12000.0);
    }

    #[test]
    fn test_onchain_state_nan_and_negative_oi_immunity() {
        let state = OnChainState::new();
        state.set_symbol_funding(2, f64::NAN);
        assert_eq!(state.get_symbol_funding(2), 0.0);

        state.set_symbol_oi(2, -100.0);
        assert_eq!(state.get_symbol_oi(2), 0.0);
    }
}
