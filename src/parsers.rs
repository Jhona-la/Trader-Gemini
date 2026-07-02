use simd_json::prelude::*;
use simd_json::prelude::ValueAsScalar;
use simd_json::prelude::ValueObjectAccess;

/// Parses a Binance DepthUpdate (Orderbook) JSON string instantly.
/// Returns (Event_time, symbol, last_update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty)
pub fn parse_binance_depth<'a>(json_str: &'a mut str) -> Option<(i64, &'a str, i64, f64, f64, f64, f64)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    
    // Depth stream structure can be {"stream": "...", "data": { ... }}
    // Or just the raw object if not combined stream
    let data = if let Some(d) = parsed.get("data") { d } else { &parsed };
    
    let e = data.get("E")?.as_i64()?;
    let s_temp = data.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };
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
    
    Some((e, s, last_update_id, bp, bq, ap, aq))
}

/// Parses a Binance Trade JSON string.
/// Returns (Event_time, trade_time, price, qty, is_buyer_maker, symbol)
pub fn parse_binance_trade<'a>(json_str: &'a mut str) -> Option<(i64, i64, f64, f64, bool, &'a str)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") { d } else { &parsed };
    
    let e = data.get("E")?.as_i64()?;
    let t = data.get("T")?.as_i64()?;
    let p = data.get("p")?.as_str()?.parse::<f64>().ok()?;
    let q = data.get("q")?.as_str()?.parse::<f64>().ok()?;
    let m = data.get("m")?.as_bool()?;
    let s_temp = data.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };
    
    Some((e, t, p, q, m, s))
}

/// Parses a Binance Kline/Candlestick JSON string.
/// Returns (Event_time, symbol, open, high, low, close, volume, is_kline_closed)
pub fn parse_binance_kline<'a>(json_str: &'a mut str) -> Option<(i64, &'a str, f64, f64, f64, f64, f64, bool)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") { d } else { &parsed };
    
    // Check if it's a kline event
    if data.get("e")?.as_str()? != "kline" {
        return None;
    }
    
    let e = data.get("E")?.as_i64()?;
    let s_temp = data.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };
    
    let k = data.get("k")?;
    let open = k.get("o")?.as_str()?.parse::<f64>().ok()?;
    let high = k.get("h")?.as_str()?.parse::<f64>().ok()?;
    let low = k.get("l")?.as_str()?.parse::<f64>().ok()?;
    let close = k.get("c")?.as_str()?.parse::<f64>().ok()?;
    let volume = k.get("v")?.as_str()?.parse::<f64>().ok()?;
    let is_closed = k.get("x")?.as_bool()?;
    
    Some((e, s, open, high, low, close, volume, is_closed))
}
