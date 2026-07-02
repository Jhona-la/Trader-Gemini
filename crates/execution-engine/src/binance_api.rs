use hmac::{Hmac, Mac};
use sha2::Sha256;

// Create alias for HMAC-SHA256
type HmacSha256 = Hmac<Sha256>;

pub const ORDER_TYPE_MARKET: &str = "MARKET";
pub const ORDER_TYPE_LIMIT: &str = "LIMIT";
pub const TIME_IN_FORCE_IOC: &str = "IOC";
pub const TIME_IN_FORCE_GTC: &str = "GTC";
pub const SIDE_BUY: &str = "BUY";
pub const SIDE_SELL: &str = "SELL";

/// Firma criptográfica O(1) de Binance sin alojamientos (Zero-Allocation).
/// Escribe el HMAC-SHA256 directamente en un buffer preasignado.
#[inline(always)]
pub fn sign_payload_to_buffer(query_string: &str, secret_key: &str, out_buf: &mut [u8; 64]) {
    let mut mac = HmacSha256::new_from_slice(secret_key.as_bytes())
        .expect("HMAC can take key of any size");
    
    mac.update(query_string.as_bytes());
    let result = mac.finalize();
    
    // Codificar directamente al buffer en stack en lugar de alojar un String
    hex::encode_to_slice(result.into_bytes(), out_buf).expect("Buffer must be 64 bytes");
}
