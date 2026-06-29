use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;
use reqwest::{Client, Method, RequestBuilder};
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

pub struct BinanceApi {
    client: Client,
    api_key: String,
    secret_key: String,
    base_url: String,
}

impl BinanceApi {
    pub fn new(api_key: String, secret_key: String, testnet: bool) -> Self {
        let base_url = if testnet {
            "https://testnet.binancefuture.com"
        } else {
            "https://fapi.binance.com"
        }.to_string();

        Self {
            client: Client::new(),
            api_key,
            secret_key,
            base_url,
        }
    }

    fn get_timestamp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
    }

    fn sign(&self, query_string: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query_string.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    pub async fn create_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        price: Option<f64>,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let timestamp = Self::get_timestamp();
        let mut query = format!(
            "symbol={}&side={}&type={}&quantity={}&timestamp={}",
            symbol.to_uppercase(),
            side.to_uppercase(),
            if price.is_some() { "LIMIT" } else { "MARKET" },
            quantity,
            timestamp
        );

        if let Some(p) = price {
            query.push_str(&format!("&price={}&timeInForce=GTC", p));
        }

        let signature = self.sign(&query);
        query.push_str(&format!("&signature={}", signature));

        let url = format!("{}/fapi/v1/order?{}", self.base_url, query);

        let req = self.client
            .request(Method::POST, &url)
            .header("X-MBX-APIKEY", &self.api_key);

        let res = req.send().await?;
        let status = res.status();
        let body: Value = res.json().await?;

        if status.is_success() {
            Ok(body)
        } else {
            Err(format!("API Error: {}", body).into())
        }
    }
}
