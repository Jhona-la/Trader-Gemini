use reqwest::{Client, header};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

pub struct BinanceRestExecutor {
    api_key: String,
    secret_key: String,
    base_url: String,
    client: Client,
}

impl BinanceRestExecutor {
    pub fn new(api_key: String, secret_key: String, is_testnet: bool) -> Self {
        let base_url = if is_testnet {
            "https://testnet.binance.vision".to_string()
        } else {
            "https://api.binance.com".to_string()
        };

        let mut headers = header::HeaderMap::new();
        headers.insert("X-MBX-APIKEY", header::HeaderValue::from_str(&api_key).unwrap());

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .unwrap();

        Self {
            api_key,
            secret_key,
            base_url,
            client,
        }
    }

    fn generate_signature(&self, query_string: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query_string.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    fn get_timestamp() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    }

    pub async fn create_order(
        &self,
        symbol: &str,
        side: &str,
        order_type: &str,
        quantity: f64,
        price: Option<f64>,
    ) -> Result<Value, reqwest::Error> {
        let timestamp = Self::get_timestamp();
        
        let mut query = format!(
            "symbol={}&side={}&type={}&quantity={}&timestamp={}",
            symbol, side, order_type, quantity, timestamp
        );

        if let Some(p) = price {
            query.push_str(&format!("&price={}&timeInForce=GTC", p));
        }

        let signature = self.generate_signature(&query);
        let url = format!("{}/api/v3/order?{}&signature={}", self.base_url, query, signature);

        let res = self.client.post(&url).send().await?;
        let json = res.json::<Value>().await?;
        Ok(json)
    }
}
