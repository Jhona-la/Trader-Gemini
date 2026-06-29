use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;

type HmacSha256 = Hmac<Sha256>;

/// Generates an HMAC-SHA256 signature for a given payload and secret.
/// Returns a hex-encoded String.
pub fn sign_binance_payload(secret: &str, payload: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC can take key of any size");
    
    mac.update(payload.as_bytes());
    
    let result = mac.finalize();
    hex::encode(result.into_bytes())
}
