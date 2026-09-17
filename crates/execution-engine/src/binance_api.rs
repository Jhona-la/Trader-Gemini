use hmac::{Hmac, Mac};
use sha2::Sha256;

// Create alias for HMAC-SHA256
type HmacSha256 = Hmac<Sha256>;

pub const ORDER_TYPE_MARKET: &str = "MARKET";
pub const ORDER_TYPE_LIMIT: &str = "LIMIT";
pub const TIME_IN_FORCE_IOC: &str = "IOC";
pub const TIME_IN_FORCE_GTC: &str = "GTC";
pub const TIME_IN_FORCE_GTX: &str = "GTX"; // Post Only (Maker)
pub const SIDE_BUY: &str = "BUY";
pub const SIDE_SELL: &str = "SELL";

/// Firma criptográfica O(1) de Binance sin alojamientos (Zero-Allocation).
/// Escribe el HMAC-SHA256 directamente en un buffer preasignado.
// FIX #1496: Firma criptográfica resiliente sin expect ni pánicos
#[inline(always)]
pub fn sign_payload_to_buffer(query_string: &str, secret_key: &str, out_buf: &mut [u8; 64]) {
    if let Ok(mut mac) = HmacSha256::new_from_slice(secret_key.as_bytes()) {
        mac.update(query_string.as_bytes());
        let result = mac.finalize();
        let _ = hex::encode_to_slice(result.into_bytes(), out_buf);
    } else {
        out_buf.fill(b'0');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_payload_to_buffer_deterministic() {
        let mut buf1 = [0u8; 64];
        let mut buf2 = [0u8; 64];
        sign_payload_to_buffer(
            "symbol=BTCUSDT&timestamp=1700000000000",
            "secret_key_123",
            &mut buf1,
        );
        sign_payload_to_buffer(
            "symbol=BTCUSDT&timestamp=1700000000000",
            "secret_key_123",
            &mut buf2,
        );
        assert_eq!(buf1, buf2);

        let sig_str = std::str::from_utf8(&buf1).unwrap();
        assert_eq!(sig_str.len(), 64);
        assert!(sig_str.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_sign_payload_different_queries_differ() {
        let mut buf1 = [0u8; 64];
        let mut buf2 = [0u8; 64];
        sign_payload_to_buffer("symbol=BTCUSDT", "secret", &mut buf1);
        sign_payload_to_buffer("symbol=ETHUSDT", "secret", &mut buf2);
        assert_ne!(buf1, buf2);
    }
}
