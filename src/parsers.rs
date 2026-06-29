use serde_json::Value;

/// Parses a Binance DepthUpdate (Orderbook) JSON string instantly.
/// Returns (Event_time, last_update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty)
pub fn parse_binance_depth(json_str: &str) -> Option<(i64, i64, f64, f64, f64, f64)> {
    let parsed: Value = serde_json::from_str(json_str).ok()?;
    let data = parsed.get("data").unwrap_or(&parsed);
    
    let e = data.get("E")?.as_i64()?;
    let last_update_id = data.get("u")?.as_i64()?;
    
    let bids = data.get("b")?.as_array()?;
    let asks = data.get("a")?.as_array()?;
    
    if bids.is_empty() || asks.is_empty() {
        return None;
    }
    
    let best_bid = bids[0].as_array()?;
    let best_ask = asks[0].as_array()?;
    
    let bp = best_bid[0].as_str()?.parse::<f64>().ok()?;
    let bq = best_bid[1].as_str()?.parse::<f64>().ok()?;
    
    let ap = best_ask[0].as_str()?.parse::<f64>().ok()?;
    let aq = best_ask[1].as_str()?.parse::<f64>().ok()?;
    
    Some((e, last_update_id, bp, bq, ap, aq))
}

/// Parses a Binance Trade JSON string.
/// Returns (Event_time, trade_time, price, qty, is_buyer_maker, symbol)
pub fn parse_binance_trade(json_str: &str) -> Option<(i64, i64, f64, f64, bool, String)> {
    let parsed: Value = serde_json::from_str(json_str).ok()?;
    let data = parsed.get("data").unwrap_or(&parsed);
    
    let e = data.get("E")?.as_i64()?;
    let t = data.get("T")?.as_i64()?;
    let p = data.get("p")?.as_str()?.parse::<f64>().ok()?;
    let q = data.get("q")?.as_str()?.parse::<f64>().ok()?;
    let m = data.get("m")?.as_bool()?;
    let s = data.get("s")?.as_str()?.to_string();
    
    Some((e, t, p, q, m, s))
}

/// Parses a Binance Kline/Candlestick JSON string.
/// Returns (Event_time, symbol, open, high, low, close, volume, is_kline_closed)
pub fn parse_binance_kline(json_str: &str) -> Option<(i64, String, f64, f64, f64, f64, f64, bool)> {
    let parsed: Value = serde_json::from_str(json_str).ok()?;
    let data = parsed.get("data").unwrap_or(&parsed);
    
    // Check if it's a kline event
    if data.get("e")?.as_str()? != "kline" {
        return None;
    }
    
    let e = data.get("E")?.as_i64()?;
    let s = data.get("s")?.as_str()?.to_string();
    
    let k = data.get("k")?;
    let open = k.get("o")?.as_str()?.parse::<f64>().ok()?;
    let high = k.get("h")?.as_str()?.parse::<f64>().ok()?;
    let low = k.get("l")?.as_str()?.parse::<f64>().ok()?;
    let close = k.get("c")?.as_str()?.parse::<f64>().ok()?;
    let volume = k.get("v")?.as_str()?.parse::<f64>().ok()?;
    let is_closed = k.get("x")?.as_bool()?;
    
    Some((e, s, open, high, low, close, volume, is_closed))
}
