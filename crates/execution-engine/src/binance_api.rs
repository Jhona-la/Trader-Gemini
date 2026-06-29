use hmac::{Hmac, Mac};
use sha2::Sha256;

// Create alias for HMAC-SHA256
type HmacSha256 = Hmac<Sha256>;

pub const ORDER_TYPE_MARKET: &str = "MARKET";
pub const TIME_IN_FORCE_IOC: &str = "IOC";
pub const SIDE_BUY: &str = "BUY";
pub const SIDE_SELL: &str = "SELL";

/// Firma criptográfica O(1) de Binance.
/// Genera el string hex HMAC-SHA256 necesario para autenticar la orden.
pub fn sign_payload(query_string: &str, secret_key: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret_key.as_bytes())
        .expect("HMAC can take key of any size");
    
    mac.update(query_string.as_bytes());
    let result = mac.finalize();
    
    hex::encode(result.into_bytes())
}
