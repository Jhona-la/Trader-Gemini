//! Checked USD-M forceOrder snapshots. This sampled feed is NOT an execution ledger.
//! Deliberately uses JSON structure; braces/keys inside strings are not event boundaries.
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidationSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LiquidationSnapshot {
    pub symbol: String,
    pub side: LiquidationSide,
    pub event_time_ms: u64,
    pub trade_time_ms: u64,
    pub order_quantity: f64,
    pub order_price: f64,
    pub average_price: f64,
    pub filled_quantity: f64,
    pub last_filled_quantity: f64,
    pub status: String,
    /// No st on legacy USD-M endpoint; not a generic assumption about other venues.
    pub legacy_um_contract: bool,
    pub reported_filled_notional: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiquidationParseError {
    MalformedJson,
    InvalidEnvelope,
    InvalidField(&'static str),
    UnsupportedContractType,
}

#[derive(Debug, Default)]
pub struct LiquidationBatch {
    pub snapshots: Vec<LiquidationSnapshot>,
    /// Record-local errors do not erase valid records from a mixed UM/CM array.
    pub rejected: Vec<(usize, LiquidationParseError)>,
}

fn number(o: &Value, key: &'static str) -> Result<f64, LiquidationParseError> {
    let n = match &o[key] {
        Value::String(s) => s.parse::<f64>().ok(),
        value => value.as_f64(),
    }
    .filter(|n| n.is_finite() && *n >= 0.0);
    n.ok_or(LiquidationParseError::InvalidField(key))
}

fn timestamp(o: &Value, key: &'static str) -> Result<u64, LiquidationParseError> {
    o[key]
        .as_u64()
        .filter(|t| *t > 0)
        .ok_or(LiquidationParseError::InvalidField(key))
}

fn parse_snapshot(v: &Value) -> Result<LiquidationSnapshot, LiquidationParseError> {
    use LiquidationParseError::InvalidField;
    if v["e"].as_str() != Some("forceOrder") {
        return Err(InvalidField("e"));
    }
    let o = v
        .get("o")
        .filter(|o| o.is_object())
        .ok_or(InvalidField("o"))?;
    let symbol = o["s"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or(InvalidField("s"))?;
    let side = match o["S"].as_str() {
        Some("BUY") => LiquidationSide::Buy,
        Some("SELL") => LiquidationSide::Sell,
        _ => return Err(InvalidField("S")),
    };
    if let (Some(a), Some(b)) = (o.get("st"), v.get("st")) {
        if a != b {
            return Err(InvalidField("st"));
        }
    }
    let st = o.get("st").or_else(|| v.get("st"));
    if st.is_some_and(|s| s.as_u64() != Some(1)) {
        return Err(LiquidationParseError::UnsupportedContractType);
    }
    let event_time_ms = timestamp(v, "E")?;
    let trade_time_ms = timestamp(o, "T")?;
    if trade_time_ms > event_time_ms {
        return Err(InvalidField("T > E"));
    }
    let order_quantity = number(o, "q")?;
    let order_price = number(o, "p")?;
    let average_price = number(o, "ap")?;
    let filled_quantity = number(o, "z")?;
    let last_filled_quantity = number(o, "l")?;
    if order_quantity <= 0.0
        || filled_quantity > order_quantity
        || last_filled_quantity > filled_quantity
    {
        return Err(InvalidField("0 <= l <= z <= q; q > 0"));
    }
    if filled_quantity > 0.0 && average_price <= 0.0 {
        return Err(InvalidField("ap"));
    }
    let status = o["X"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or(InvalidField("X"))?;
    let reported_filled_notional = average_price * filled_quantity;
    if !reported_filled_notional.is_finite() {
        return Err(InvalidField("ap*z overflow"));
    }
    Ok(LiquidationSnapshot {
        symbol: symbol.into(),
        side,
        event_time_ms,
        trade_time_ms,
        order_quantity,
        order_price,
        average_price,
        filled_quantity,
        last_filled_quantity,
        status: status.into(),
        legacy_um_contract: st.is_none(),
        reported_filled_notional,
    })
}

/// Decode raw object/array or combined {stream,data}. A malformed JSON frame
/// cannot emit partial callbacks; semantically bad records are diagnosed individually.
pub fn decode_liquidation_snapshots(
    payload: &[u8],
) -> Result<LiquidationBatch, LiquidationParseError> {
    let root: Value =
        serde_json::from_slice(payload).map_err(|_| LiquidationParseError::MalformedJson)?;
    let data = if let Some(data) = root.get("data") {
        if !root["stream"].is_string() || root.get("e").is_some() {
            return Err(LiquidationParseError::InvalidEnvelope);
        }
        data
    } else {
        &root
    };
    let values: Vec<&Value> = match data {
        Value::Array(items) => items.iter().collect(),
        Value::Object(_) => vec![data],
        _ => return Err(LiquidationParseError::InvalidEnvelope),
    };
    let mut batch = LiquidationBatch::default();
    for (index, value) in values.into_iter().enumerate() {
        match parse_snapshot(value) {
            Ok(event) => batch.snapshots.push(event),
            Err(reason) => batch.rejected.push((index, reason)),
        }
    }
    Ok(batch)
}
