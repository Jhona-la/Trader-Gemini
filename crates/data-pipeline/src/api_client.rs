use hmac::{Hmac, Mac};

use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Resuelve credenciales Binance con nombres canónicos y alias históricos.
/// Canónico: BINANCE_API_KEY / BINANCE_SECRET_KEY (y *_TESTNET_* para demo).
/// Alias aceptados: BINANCE_API_SECRET, BINANCE_DEMO_SECRET_KEY (compatibilidad con .env viejos).
/// F1 centralizará esto en un único proveedor de credenciales del workspace.
fn resolve_credentials(is_testnet: bool) -> Result<(String, String), String> {
    let (key, secret) = if is_testnet {
        (
            env_first(&["BINANCE_TESTNET_API_KEY", "BINANCE_DEMO_API_KEY"]),
            env_first(&["BINANCE_TESTNET_SECRET_KEY", "BINANCE_DEMO_SECRET_KEY"]),
        )
    } else {
        (
            env_first(&["BINANCE_API_KEY", "MAINNET_API_KEY"]),
            env_first(&["BINANCE_SECRET_KEY", "BINANCE_API_SECRET", "MAINNET_SECRET_KEY"]),
        )
    };
    if key.is_empty() || secret.is_empty() {
        return Err("Faltan credenciales (canónico: BINANCE_API_KEY / BINANCE_SECRET_KEY)".to_string());
    }
    Ok((key, secret))
}

fn env_first(names: &[&str]) -> String {
    for n in names {
        if let Ok(v) = std::env::var(n) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return v;
            }
        }
    }
    String::new()
}

pub async fn get_real_wallet_balance() -> Result<f64, String> {
    let is_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
    let (api_key, api_secret) = resolve_credentials(is_testnet)?;

    let base_url = if is_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let query_string = format!("timestamp={}", timestamp);

    let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
        .map_err(|e| format!("HMAC Error: {}", e))?;
    mac.update(query_string.as_bytes());
    let signature = hex::encode(mac.finalize().into_bytes());

    let url = format!("{}/fapi/v2/balance?{}&signature={}", base_url, query_string, signature);

    let client = quantum_arena::net_multiplexer::get_global_client();
    let res = client
        .get(&url)
        .header("X-MBX-APIKEY", api_key)
        .send()
        .await
        .map_err(|e| format!("HTTP Error: {}", e))?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(format!("Binance API Error [{}]: {}", status, body));
    }

    let json: serde_json::Value = res.json().await.map_err(|e| format!("JSON Error: {}", e))?;
    
    let mut usdt_balance = 0.0;

    if let Some(array) = json.as_array() {
        for asset in array {
            if let Some(asset_name) = asset.get("asset").and_then(|a| a.as_str()) {
                if asset_name == "USDT" {
                    if let Some(balance_str) = asset.get("balance").and_then(|b| b.as_str()) {
                        usdt_balance = balance_str.parse::<f64>().unwrap_or(0.0);
                    }
                    break;
                }
            }
        }
    }

    Ok(usdt_balance)
}

/// Extrae las comisiones exactas de tu cuenta para un símbolo particular en Futures.
/// Retorna `(maker_commission_rate, taker_commission_rate)` como proporciones decimales (ej: 0.0002 para 0.02%).
pub async fn get_commission_rate(symbol: &str) -> Result<(f64, f64), String> {
    let is_testnet = std::env::var("USE_TESTNET").unwrap_or_default().trim().to_lowercase() == "true";
    let (api_key, api_secret) = resolve_credentials(is_testnet)?;

    let base_url = if is_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let query_string = format!("symbol={}&timestamp={}", symbol, timestamp);

    let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
        .map_err(|e| format!("HMAC Error: {}", e))?;
    mac.update(query_string.as_bytes());
    let signature = hex::encode(mac.finalize().into_bytes());

    let url = format!("{}/fapi/v1/commissionRate?{}&signature={}", base_url, query_string, signature);

    let client = quantum_arena::net_multiplexer::get_global_client();
    let res = client
        .get(&url)
        .header("X-MBX-APIKEY", api_key)
        .send()
        .await
        .map_err(|e| format!("HTTP Error: {}", e))?;

    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(format!("Binance API Error [{}]: {}", status, body));
    }

    let json: serde_json::Value = res.json().await.map_err(|e| format!("JSON Error: {}", e))?;

    let maker_rate = json.get("makerCommissionRate")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0002); // Fallback: 0.02% (Base Binance Futures)

    let taker_rate = json.get("takerCommissionRate")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0004); // Fallback: 0.04%

    Ok((maker_rate, taker_rate))
}
