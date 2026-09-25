//! Exchange observations are not interchangeable with fills or local reservations.
//! These contracts validate quantity-based USD-M query responses, not profitability,
//! snapshot freshness, or attribution of aggregate exposure to a particular intent.
use crate::{executor::ActivePosition, OrderAck, OrderResolution, OrderStatus};
use serde::Deserialize;
use std::collections::HashMap;

// Required wire fields: unlike the legacy internal DTO, absent positionAmt must
// not default to zero. Additional venue fields are intentionally allowed.
#[derive(Deserialize)]
struct PositionRow {
    symbol: String,
    #[serde(
        rename = "positionAmt",
        deserialize_with = "crate::order_types::string_or_f64"
    )]
    amount: f64,
    #[serde(
        rename = "entryPrice",
        deserialize_with = "crate::order_types::string_or_f64"
    )]
    entry_price: f64,
    #[serde(deserialize_with = "crate::order_types::string_or_f64")]
    leverage: f64,
    #[serde(rename = "positionSide")]
    position_side: String,
}

/// Parse the whole snapshot or fail; never return a silently filtered subset.
/// Only exact zero denotes no exposure. A small quantity may be untradeable but
/// is still exposure; order lot filters belong to a different contract.
pub fn parse_active_positions(text: &str) -> Result<Vec<ActivePosition>, String> {
    let rows: Vec<PositionRow> = serde_json::from_str(text)
        .map_err(|e| format!("POSITION_EVIDENCE_INVALID: schema: {e}"))?;
    let mut legs = HashMap::<String, u8>::new();
    let mut positions = Vec::new();
    for (index, row) in rows.into_iter().enumerate() {
        let invalid = || format!("POSITION_EVIDENCE_INVALID: row {index} domain/leg");
        let bit = match row.position_side.as_str() {
            "BOTH" => 1,
            "LONG" if row.amount >= 0.0 => 2,
            "SHORT" if row.amount <= 0.0 => 4,
            _ => return Err(invalid()),
        };
        if row.symbol.is_empty()
            || row.symbol.trim() != row.symbol
            || !row.amount.is_finite()
            || !row.entry_price.is_finite()
            || row.entry_price < 0.0
            || !row.leverage.is_finite()
            || row.leverage < 1.0
            || row.leverage.fract() != 0.0
            || (row.amount != 0.0 && row.entry_price <= 0.0)
        {
            return Err(invalid());
        }
        let mask = legs.entry(row.symbol.clone()).or_default();
        if (*mask & bit) != 0 || (*mask != 0 && (bit == 1 || *mask == 1)) {
            return Err(invalid()); // Duplicate leg or mixed one-way/hedge modes.
        }
        *mask |= bit;
        if row.amount != 0.0 {
            positions.push(ActivePosition {
                symbol: row.symbol,
                qty: row.amount.abs(),
                entry_price: row.entry_price,
                leverage: row.leverage,
                is_long: row.amount > 0.0,
            });
        }
    }
    Ok(positions)
}

/// Accepted means order evidence, not necessarily a fill (NEW is accepted).
/// Every query error, including -2013, is inconclusive: failure to retrieve an
/// order is not a certificate that no execution ever took place.
pub fn classify_query_result(
    symbol: &str,
    client_order_id: &str,
    result: Result<&OrderAck, &str>,
) -> OrderResolution {
    let Ok(ack) = result else {
        return OrderResolution::Timeout;
    };
    if symbol.is_empty()
        || client_order_id.is_empty()
        || ack.symbol != symbol
        || ack.client_order_id != client_order_id
        || ack.order_id == 0
        || !matches!(ack.side.as_str(), "BUY" | "SELL")
        || !ack.orig_qty.is_finite()
        || ack.orig_qty <= 0.0
        || !ack.executed_qty.is_finite()
        || ack.executed_qty < 0.0
        || ack.executed_qty > ack.orig_qty
    {
        return OrderResolution::Timeout;
    }
    match OrderStatus::parse(&ack.status) {
        OrderStatus::New if ack.executed_qty == 0.0 => OrderResolution::Accepted,
        OrderStatus::PartiallyFilled
            if ack.executed_qty > 0.0 && ack.executed_qty < ack.orig_qty =>
        {
            OrderResolution::Accepted
        }
        OrderStatus::Filled if ack.executed_qty == ack.orig_qty => OrderResolution::Accepted,
        OrderStatus::Rejected | OrderStatus::Expired | OrderStatus::Canceled => {
            if ack.executed_qty > 0.0 {
                OrderResolution::Accepted
            } else {
                OrderResolution::Rejected
            }
        }
        _ => OrderResolution::Timeout,
    }
}

/// Strict GET query boundary. Keep permissive OrderAck for legacy/internal POST
/// fixtures, but require the evidence used by this particular consumer on wire.
pub fn parse_query_order_response(body: &str, symbol: &str, id: &str) -> Result<OrderAck, String> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("ORDER_EVIDENCE_INVALID: schema: {e}"))?;
    for field in [
        "symbol",
        "clientOrderId",
        "orderId",
        "side",
        "status",
        "origQty",
        "executedQty",
    ] {
        if value.get(field).is_none() {
            return Err(format!("ORDER_EVIDENCE_INVALID: missing {field}"));
        }
    }
    let ack: OrderAck = serde_json::from_value(value)
        .map_err(|e| format!("ORDER_EVIDENCE_INVALID: schema: {e}"))?;
    if classify_query_result(symbol, id, Ok(&ack)) == OrderResolution::Timeout {
        return Err("ORDER_EVIDENCE_INVALID: identity, quantity or status inconsistent".into());
    }
    Ok(ack)
}

/// A replacement order is safe only after the maker cannot execute any more.
/// A successful GET returning NEW/PARTIALLY_FILLED does NOT confirm cancel.
pub fn terminal_maker_executed_quantity(
    symbol: &str,
    id: &str,
    ack: &OrderAck,
) -> Result<f64, String> {
    if classify_query_result(symbol, id, Ok(ack)) == OrderResolution::Timeout
        || !matches!(
            OrderStatus::parse(&ack.status),
            OrderStatus::Filled
                | OrderStatus::Canceled
                | OrderStatus::Expired
                | OrderStatus::Rejected
        )
    {
        return Err(
            "MAKER_CHASE_UNVERIFIED: maker identity/fills/terminal state not established".into(),
        );
    }
    Ok(ack.executed_qty)
}
