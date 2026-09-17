use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn get_http_client() -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

/// Resuelve credenciales Binance con nombres canónicos y alias históricos.
/// Canónico: BINANCE_API_KEY / BINANCE_SECRET_KEY (y *_TESTNET_* para demo).
/// Alias aceptados: BINANCE_API_SECRET, BINANCE_DEMO_SECRET_KEY (compatibilidad con .env viejos).
/// F1 centralizará esto en un único proveedor de credenciales del workspace.
pub fn resolve_credentials_from_lookup<F>(
    is_testnet: bool,
    lookup: F,
) -> Result<(String, String), String>
where
    F: Fn(&str) -> Option<String>,
{
    let (key, secret) = if is_testnet {
        (
            lookup_first(
                &["BINANCE_TESTNET_API_KEY", "BINANCE_DEMO_API_KEY"],
                &lookup,
            ),
            lookup_first(
                &["BINANCE_TESTNET_SECRET_KEY", "BINANCE_DEMO_SECRET_KEY"],
                &lookup,
            ),
        )
    } else {
        (
            lookup_first(&["BINANCE_API_KEY", "MAINNET_API_KEY"], &lookup),
            lookup_first(
                &[
                    "BINANCE_SECRET_KEY",
                    "BINANCE_API_SECRET",
                    "MAINNET_SECRET_KEY",
                ],
                &lookup,
            ),
        )
    };
    if key.is_empty() || secret.is_empty() {
        return Err(
            "Faltan credenciales (canónico: BINANCE_API_KEY / BINANCE_SECRET_KEY)".to_string(),
        );
    }
    Ok((key, secret))
}

pub fn lookup_first<F>(names: &[&str], lookup: &F) -> String
where
    F: Fn(&str) -> Option<String>,
{
    for n in names {
        if let Some(v) = lookup(n) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return v;
            }
        }
    }
    String::new()
}

fn resolve_credentials(is_testnet: bool) -> Result<(String, String), String> {
    resolve_credentials_from_lookup(is_testnet, |k| std::env::var(k).ok())
}

#[allow(dead_code)]
fn env_first(names: &[&str]) -> String {
    lookup_first(names, &|k| std::env::var(k).ok())
}

pub async fn get_real_wallet_balance() -> Result<f64, String> {
    let is_testnet = std::env::var("USE_TESTNET")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        == "true";
    let (api_key, api_secret) = resolve_credentials(is_testnet)?;

    let base_url = if is_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let query_string = format!("timestamp={}", timestamp);

    let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
        .map_err(|e| format!("HMAC Error: {}", e))?;
    mac.update(query_string.as_bytes());
    let signature = hex::encode(mac.finalize().into_bytes());

    let url = format!(
        "{}/fapi/v2/balance?{}&signature={}",
        base_url, query_string, signature
    );

    let client = get_http_client();
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
                        let parsed = balance_str.parse::<f64>().unwrap_or(0.0);
                        // FIX #674: Validar no-negatividad y finitud
                        usdt_balance = if parsed.is_finite() && parsed >= 0.0 {
                            parsed
                        } else {
                            0.0
                        };
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
    let is_testnet = std::env::var("USE_TESTNET")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        == "true";
    let (api_key, api_secret) = resolve_credentials(is_testnet)?;

    let base_url = if is_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let query_string = format!("symbol={}&timestamp={}", symbol, timestamp);

    let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
        .map_err(|e| format!("HMAC Error: {}", e))?;
    mac.update(query_string.as_bytes());
    let signature = hex::encode(mac.finalize().into_bytes());

    let url = format!(
        "{}/fapi/v1/commissionRate?{}&signature={}",
        base_url, query_string, signature
    );

    let client = get_http_client();
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

    let raw_maker = json
        .get("makerCommissionRate")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0002); // Fallback: 0.02% (Base Binance Futures)

    let raw_taker = json
        .get("takerCommissionRate")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0004); // Fallback: 0.04%

    let maker_rate = if raw_maker.is_finite() && raw_maker >= 0.0 {
        raw_maker
    } else {
        0.0002
    };
    let taker_rate = if raw_taker.is_finite() && raw_taker >= 0.0 {
        raw_taker
    } else {
        0.0004
    };

    Ok((maker_rate, taker_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_env_first_precedence() {
        let mut map = HashMap::new();
        map.insert("TEST_VAR_A", "value_a".to_string());
        map.insert("TEST_VAR_B", "value_b".to_string());

        let val = lookup_first(&["TEST_NON_EXISTENT", "TEST_VAR_A", "TEST_VAR_B"], &|k| {
            map.get(k).cloned()
        });
        assert_eq!(val, "value_a");
    }

    #[test]
    fn test_resolve_credentials_missing() {
        let map: HashMap<&str, String> = HashMap::new();
        let res = resolve_credentials_from_lookup(true, |k| map.get(k).cloned());
        assert!(res.is_err());
    }

    #[test]
    fn test_resolve_credentials_present() {
        let mut map = HashMap::new();
        map.insert("BINANCE_TESTNET_API_KEY", "test_key_123".to_string());
        map.insert("BINANCE_TESTNET_SECRET_KEY", "test_secret_456".to_string());

        let res = resolve_credentials_from_lookup(true, |k| map.get(k).cloned());
        assert!(res.is_ok());
        let (k, s) = res.unwrap();
        assert_eq!(k, "test_key_123");
        assert_eq!(s, "test_secret_456");

        // Mainnet test
        let mut mainnet_map = HashMap::new();
        mainnet_map.insert("BINANCE_API_KEY", "main_key".to_string());
        mainnet_map.insert("BINANCE_SECRET_KEY", "main_secret".to_string());
        let main_res = resolve_credentials_from_lookup(false, |k| mainnet_map.get(k).cloned());
        assert!(main_res.is_ok());
        let (mk, ms) = main_res.unwrap();
        assert_eq!(mk, "main_key");
        assert_eq!(ms, "main_secret");
    }
}
