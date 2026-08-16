//! TIPOS DE ORDEN BINANCE (F1.1) — Respuestas tipadas del ciclo de vida de órdenes.
//!
//! QUÉ: structs serde que parsean la respuesta REAL de /fapi/v1/order (POST/GET/DELETE).
//! POR QUÉ: sin esto no hay reconciliación posible: fills parciales, avgPrice,
//!          rechazos y estados quedaban invisibles (audit F1).
//! CÓMO: Binance Futures envía números como strings ("executedQty":"0.0");
//!       `string_or_f64` los tolera y también acepta números nativos.

use serde::Deserialize;

/// Deserializa campos que Binance envía como string O como número.
pub fn string_or_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Val {
        S(String),
        F(f64),
        N(serde_json::Number),
    }
    match Val::deserialize(deserializer)? {
        Val::F(f) => Ok(f),
        Val::N(n) => n
            .as_f64()
            .ok_or_else(|| serde::de::Error::custom("número fuera de rango f64")),
        Val::S(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| serde::de::Error::custom(format!("string no numérica: {}", s))),
    }
}

/// Fill individual dentro de una orden ejecutada.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct Fill {
    #[serde(deserialize_with = "string_or_f64", default)]
    pub price: f64,
    #[serde(deserialize_with = "string_or_f64", default)]
    pub qty: f64,
    #[serde(deserialize_with = "string_or_f64", default)]
    pub commission: f64,
    #[serde(rename = "commissionAsset", default)]
    pub commission_asset: String,
    #[serde(rename = "tradeId", default)]
    pub trade_id: u64,
}

/// Reconocimiento completo de una orden (POST ack, GET query, cancel response).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct OrderAck {
    #[serde(rename = "orderId", default)]
    pub order_id: u64,
    #[serde(default)]
    pub symbol: String,
    /// NEW | PARTIALLY_FILLED | FILLED | CANCELED | EXPIRED | PENDING_CANCEL | REJECTED
    #[serde(default)]
    pub status: String,
    #[serde(rename = "clientOrderId", default)]
    pub client_order_id: String,
    #[serde(rename = "origQty", deserialize_with = "string_or_f64", default)]
    pub orig_qty: f64,
    #[serde(rename = "executedQty", deserialize_with = "string_or_f64", default)]
    pub executed_qty: f64,
    #[serde(rename = "cumQuote", deserialize_with = "string_or_f64", default)]
    pub cum_quote: f64,
    #[serde(rename = "avgPrice", deserialize_with = "string_or_f64", default)]
    pub avg_price: f64,
    #[serde(deserialize_with = "string_or_f64", default)]
    pub price: f64,
    #[serde(default)]
    pub side: String,
    #[serde(rename = "type", default)]
    pub order_type: String,
    #[serde(rename = "updateTime", default)]
    pub update_time: u64,
    #[serde(default)]
    pub fills: Vec<Fill>,
}

impl OrderAck {
    #[inline(always)]
    pub fn is_filled(&self) -> bool {
        self.status == "FILLED"
    }

    #[inline(always)]
    pub fn is_active(&self) -> bool {
        matches!(self.status.as_str(), "NEW" | "PARTIALLY_FILLED")
    }

    /// Cantidad restante por ejecutar (>= 0).
    #[inline(always)]
    pub fn remaining_qty(&self) -> f64 {
        (self.orig_qty - self.executed_qty).max(0.0)
    }

    /// Comisión total pagada en los fills reportados.
    pub fn total_commission(&self) -> f64 {
        self.fills.iter().map(|f| f.commission).sum()
    }
}

/// Error estructurado de la API Binance ({ "code": -2010, "msg": "..." }).
#[derive(Debug, Clone, Deserialize)]
pub struct BinanceApiError {
    pub code: i64,
    pub msg: String,
}

pub fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Parsea el body 2xx de una orden a OrderAck.
pub fn parse_order_body(body: &str) -> Result<OrderAck, String> {
    serde_json::from_str(body)
        .map_err(|e| format!("ACK_PARSE_ERROR: {} body={}", e, truncate(body, 200)))
}

/// Convierte un body de rechazo (4xx) en un Err descriptivo y NO ambiguo.
pub fn parse_reject_body(body: &str, http_status: u16) -> String {
    if let Ok(e) = serde_json::from_str::<BinanceApiError>(body) {
        format!("REJECTED code={} msg={}", e.code, truncate(&e.msg, 200))
    } else {
        format!("REJECTED http={} body={}", http_status, truncate(body, 200))
    }
}

/// Códigos que exigen reacción específica del executor (mapa mínimo F1; crece en F1.8).
pub mod error_codes {
    pub const DUPLICATED_ORDER: i64 = -4116; // newClientOrderId repetido activo
    pub const NEW_ORDER_REJECTED: i64 = -2010; // saldo/posición insuficiente, etc.
    pub const SIGNATURE_INVALID: i64 = -1022;
    pub const PARAM_REPEAT: i64 = -1105; // parámetro duplicado (bug histórico positionSide)
    pub const RATE_LIMIT_BAN: i64 = -4164; // VP límite
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ack_with_string_numbers() {
        let body = r#"{
            "orderId": 123, "symbol": "BTCUSDT", "status": "PARTIALLY_FILLED",
            "clientOrderId": "abc", "origQty": "2.0", "executedQty": "0.5",
            "cumQuote": "30.0", "avgPrice": "60.0", "price": "0", "side": "BUY",
            "type": "MARKET", "updateTime": 1700000000000,
            "fills": [{"price": "60.0", "qty": "0.5", "commission": "0.0002", "commissionAsset": "BNB", "tradeId": 1}]
        }"#;
        let ack: OrderAck = serde_json::from_str(body).expect("parse ack");
        assert_eq!(ack.order_id, 123);
        assert!((ack.executed_qty - 0.5).abs() < 1e-12);
        assert!((ack.remaining_qty() - 1.5).abs() < 1e-12);
        assert!(ack.is_active() && !ack.is_filled());
        assert!((ack.total_commission() - 0.0002).abs() < 1e-12);
    }

    #[test]
    fn parse_error_body() {
        let body = r#"{"code": -2010, "msg": "Account has insufficient balance"}"#;
        let e: BinanceApiError = serde_json::from_str(body).expect("parse error");
        assert_eq!(e.code, error_codes::NEW_ORDER_REJECTED);
    }
}
