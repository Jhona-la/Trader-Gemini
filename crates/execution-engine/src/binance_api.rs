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
/// USD-M POST /fapi/v1/leverage schema; instrument/account tiers can be lower.
pub const MAX_INITIAL_LEVERAGE: u32 = 125;

/// The documented USD-M POST /fapi/v1/order schema has no icebergQty parameter.
/// Do not silently fall back to a fully visible order or claim paper support.
pub const SUPPORTS_NATIVE_ICEBERG: bool = false;

/// LG-12 (Línea G) — LÍMITES DE PETICIONES DEL EXCHANGE (USD-M Futures,
/// `exchangeInfo.rateLimits`): REQUEST_WEIGHT 2 400/min, ORDERS 300/10 s y
/// 1 200/min. Son infraestructura del exchange, no parámetros de estrategia:
/// antes el host los tomaba del genoma (mutaban en [1 000; 5 000] y
/// [100; 500]); con un gen por encima del límite real el freno local al 80 %
/// no actuaba antes de un 429/418, y ningún backtest puede medirlos. El
/// margen para las peticiones en vuelo lo pone el headroom del ejecutor.
pub const LIMITE_PESO_1M: usize = 2_400;
pub const LIMITE_ORDENES_10S: usize = 300;
pub const LIMITE_ORDENES_1M: usize = 1_200;

/// N5 (CL-177): ventanas de esos tres contadores, con la misma fuente
/// (`exchangeInfo.rateLimits`: REQUEST_WEIGHT 1 MINUTE, ORDERS 10 SECOND y
/// 1 MINUTE). Cada cabecera `X-MBX-USED-WEIGHT-1M` / `X-MBX-ORDER-COUNT-10S/1M`
/// vale dentro de la suya.
pub const VENTANA_PESO_MS: u64 = 60_000;
pub const VENTANA_ORDENES_10S_MS: u64 = 10_000;
pub const VENTANA_ORDENES_1M_MS: u64 = 60_000;

/// N5 (CL-179): lo que una petición suma a esos contadores: `peso` al de
/// REQUEST_WEIGHT y `ordenes` a cada uno de los dos de ORDERS (10 s y 1 min).
/// Fuente: docs/BINANCE_API_MAP.md §1, comprobado el 2026-11-01 contra la
/// documentación oficial de USDⓈ-M Futures (páginas Trade y Account). N22
/// (CL-218): cada coste es el documentado. Subestimar deja pasar una petición
/// a un 429, y sobrestimar tampoco es gratis en el tramo crítico: frena una
/// pierna de protección que el exchange admitiría.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CosteBinance {
    pub peso: usize,
    pub ordenes: usize,
}

impl CosteBinance {
    pub const NULO: CosteBinance = CosteBinance { peso: 0, ordenes: 0 };

    /// El coste de `n` peticiones iguales.
    pub const fn por(self, n: usize) -> CosteBinance {
        CosteBinance {
            peso: self.peso.saturating_mul(n),
            ordenes: self.ordenes.saturating_mul(n),
        }
    }
}

/// POST /fapi/v1/order: peso 0 y una orden en cada contador ORDERS.
pub const COSTE_ORDEN: CosteBinance = CosteBinance { peso: 0, ordenes: 1 };
/// POST /fapi/v1/algoOrder (pierna TP/SL o trailing): peso 0 y una orden en
/// cada contador ORDERS, como /fapi/v1/order. CL-218: era peso 1 supuesto, y
/// con el peso en el límite el SL o el TP se frenaba aquí aunque el exchange
/// lo admitía.
pub const COSTE_PIERNA_ALGO: CosteBinance = CosteBinance {
    peso: 0,
    ordenes: 1,
};
/// DELETE /fapi/v1/order, /fapi/v1/allOpenOrders, /fapi/v1/algoOrder y
/// /fapi/v1/algoOpenOrders: peso 1 cada una, sin órdenes.
pub const COSTE_CANCELACION: CosteBinance = CosteBinance { peso: 1, ordenes: 0 };
/// GET /fapi/v1/order: peso 1.
pub const COSTE_CONSULTA_ORDEN: CosteBinance = CosteBinance { peso: 1, ordenes: 0 };
/// GET /fapi/v2/positionRisk: peso 5.
pub const COSTE_POSITION_RISK: CosteBinance = CosteBinance { peso: 5, ordenes: 0 };
/// GET /fapi/v1/openAlgoOrders con símbolo: peso 1 (40 sin él), como
/// /fapi/v1/openOrders.
pub const COSTE_ALGO_ABIERTAS_SIMBOLO: CosteBinance = CosteBinance { peso: 1, ordenes: 0 };

/// A successful HTTP status alone does not confirm the requested configuration.
/// This validates identity and leverage only, not account margin tiers/capacity.
pub fn validate_leverage_confirmation(
    body: &str,
    symbol: &str,
    leverage: u32,
) -> Result<(), String> {
    #[derive(serde::Deserialize)]
    struct Confirmation {
        symbol: String,
        leverage: u32,
    }
    if symbol.is_empty() || !(1..=MAX_INITIAL_LEVERAGE).contains(&leverage) {
        return Err("invalid leverage confirmation request".into());
    }
    let confirmation: Confirmation =
        serde_json::from_str(body).map_err(|_| "malformed leverage confirmation".to_string())?;
    if confirmation.symbol != symbol || confirmation.leverage != leverage {
        return Err("leverage confirmation does not match requested symbol/value".into());
    }
    Ok(())
}

/// Firma HMAC-SHA256 sin alojamientos adicionales para el resultado.
/// El trabajo escala con los bytes del mensaje y la normalización de la clave;
/// que el digest tenga longitud fija no hace constante el coste de procesarlos.
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
