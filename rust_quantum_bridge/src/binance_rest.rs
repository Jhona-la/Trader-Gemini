use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use reqwest::header;
use hmac::{Hmac, Mac, digest::KeyInit};
use sha2::Sha256;
use hex;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

fn get_timestamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

fn generate_signature(secret_key: &str, query_string: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret_key.as_bytes())
        .expect("HMAC can take key of any size");
    mac.update(query_string.as_bytes());
    let result = mac.finalize();
    hex::encode(result.into_bytes())
}

#[pyfunction]
#[pyo3(signature = (
    api_key,
    secret_key,
    symbol,
    side,
    order_type,
    quantity,
    price = None,
    is_testnet = false,
    is_futures = true,
    time_in_force = None,
    reduce_only = None,
    position_side = None
))]
pub fn place_order_sync(
    api_key: String,
    secret_key: String,
    symbol: String,
    side: String,
    order_type: String,
    quantity: f64,
    price: Option<f64>,
    is_testnet: bool,
    is_futures: bool,
    time_in_force: Option<String>,
    reduce_only: Option<bool>,
    position_side: Option<String>
) -> PyResult<String> {
    let base_url = if is_testnet {
        if is_futures { "https://testnet.binancefuture.com" } else { "https://testnet.binance.vision" }
    } else {
        if is_futures { "https://fapi.binance.com" } else { "https://api.binance.com" }
    };
    
    let path = if is_futures { "/fapi/v1/order" } else { "/api/v3/order" };

    let timestamp = get_timestamp();
    
    let mut params = vec![
        format!("quantity={}", quantity),
        format!("side={}", side),
        format!("symbol={}", symbol),
        format!("timestamp={}", timestamp),
        format!("type={}", order_type),
    ];
    
    if let Some(p) = price {
        params.push(format!("price={}", p));
    }
    if let Some(tif) = time_in_force {
        params.push(format!("timeInForce={}", tif));
    }
    if let Some(ro) = reduce_only {
        params.push(format!("reduceOnly={}", ro));
    }
    if let Some(ps) = position_side {
        if is_futures {
            params.push(format!("positionSide={}", ps));
        }
    }
    
    let query_string = params.join("&");
    let signature = generate_signature(&secret_key, &query_string);
    let final_url = format!("{}{}?{}&signature={}", base_url, path, query_string, signature);
    
    let mut headers = header::HeaderMap::new();
    headers.insert("X-MBX-APIKEY", header::HeaderValue::from_str(&api_key).unwrap());
    
    // Synchronous execution using reqwest blocking — nanosecond-optimized single-shot client
    let client = reqwest::blocking::Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Reqwest client build error: {}", e)))?;
        
    let res = client.post(&final_url)
        .send()
        .map_err(|e| PyRuntimeError::new_err(format!("Reqwest send error: {}", e)))?;
        
    let status = res.status();
    let text = res.text().unwrap_or_else(|_| "".to_string());
    
    if !status.is_success() {
        return Err(PyRuntimeError::new_err(format!("Binance API Error {}: {}", status, text)));
    }
    
    Ok(text)
}
