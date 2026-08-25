use simd_json::prelude::ValueAsScalar;
use simd_json::prelude::ValueObjectAccess;
use simd_json::prelude::*;

/// Parses a Binance DepthUpdate (Orderbook) JSON string instantly.
/// Returns (Event_time, symbol, last_update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty)
pub fn parse_binance_depth<'a>(
    json_str: &'a mut str,
) -> Option<(i64, &'a str, i64, f64, f64, f64, f64)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;

    // Depth stream structure can be {"stream": "...", "data": { ... }}
    // Or just the raw object if not combined stream
    let data = if let Some(d) = parsed.get("data") {
        d
    } else {
        &parsed
    };

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

    // FIX #1417 / #1472: Data Integrity & Normalization Validation (Descartar precios inválidos, NaNs y libros cruzados bp >= ap)
    if bp <= 0.0 || ap <= 0.0 || bq < 0.0 || aq < 0.0 || bp >= ap || !bp.is_finite() || !ap.is_finite() || !bq.is_finite() || !aq.is_finite() {
        return None;
    }

    Some((e, s, last_update_id, bp, bq, ap, aq))
}

/// Parses a Binance Trade JSON string.
/// Returns (Event_time, trade_time, price, qty, is_buyer_maker, symbol)
pub fn parse_binance_trade<'a>(
    json_str: &'a mut str,
) -> Option<(i64, i64, f64, f64, bool, &'a str)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") {
        d
    } else {
        &parsed
    };

    let e = data.get("E")?.as_i64()?;
    let t = data.get("T")?.as_i64()?;
    let p = data.get("p")?.as_str()?.parse::<f64>().ok()?;
    let q = data.get("q")?.as_str()?.parse::<f64>().ok()?;
    let m = data.get("m")?.as_bool()?;
    let s_temp = data.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };

    // FIX #1472: Data Integrity Validation & Finitude
    if p <= 0.0 || q < 0.0 || !p.is_finite() || !q.is_finite() {
        return None;
    }

    Some((e, t, p, q, m, s))
}

pub type KlineData<'a> = (i64, &'a str, f64, f64, f64, f64, f64, bool);

/// Parses a Binance Kline/Candlestick JSON string.
/// Returns (Event_time, symbol, open, high, low, close, volume, is_kline_closed)
pub fn parse_binance_kline<'a>(json_str: &'a mut str) -> Option<KlineData<'a>> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") {
        d
    } else {
        &parsed
    };

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

    // FIX #1472: Data Integrity Validation & Finitude
    if open <= 0.0 || high < low || low <= 0.0 || close <= 0.0 || volume < 0.0 || !open.is_finite() || !high.is_finite() || !low.is_finite() || !close.is_finite() || !volume.is_finite() {
        return None; // Protect the engine from absurd or manipulated data
    }

    Some((e, s, open, high, low, close, volume, is_closed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_binance_depth_valid_and_crossed_book_rejection() {
        let mut valid_depth = r#"{"e":"depthUpdate","E":1672531200000,"s":"BTCUSDT","u":12345,"b":[["60000.0","1.5"]],"a":[["60001.0","2.0"]]}"#.to_string();
        let res = parse_binance_depth(&mut valid_depth);
        assert!(res.is_some());
        let (e, s, u, bp, bq, ap, aq) = res.unwrap();
        assert_eq!(e, 1672531200000);
        assert_eq!(s, "BTCUSDT");
        assert_eq!(u, 12345);
        assert_eq!(bp, 60000.0);
        assert_eq!(bq, 1.5);
        assert_eq!(ap, 60001.0);
        assert_eq!(aq, 2.0);

        // Crossed book rejection (bid >= ask)
        let mut crossed_depth = r#"{"e":"depthUpdate","E":1672531200000,"s":"BTCUSDT","u":12345,"b":[["60005.0","1.5"]],"a":[["60001.0","2.0"]]}"#.to_string();
        assert!(parse_binance_depth(&mut crossed_depth).is_none());
    }

    #[test]
    fn test_parse_binance_trade_valid_and_negative_rejection() {
        let mut valid_trade = r#"{"e":"trade","E":1672531200000,"s":"BTCUSDT","t":100,"p":"60000.5","q":"0.1","T":1672531200000,"m":true}"#.to_string();
        let res = parse_binance_trade(&mut valid_trade);
        assert!(res.is_some());
        let (e, t, p, q, m, s) = res.unwrap();
        assert_eq!(e, 1672531200000);
        assert_eq!(t, 1672531200000);
        assert_eq!(p, 60000.5);
        assert_eq!(q, 0.1);
        assert!(m);
        assert_eq!(s, "BTCUSDT");

        // Negative price rejection
        let mut invalid_trade = r#"{"e":"trade","E":1672531200000,"s":"BTCUSDT","t":100,"p":"-10.0","q":"0.1","T":1672531200000,"m":true}"#.to_string();
        assert!(parse_binance_trade(&mut invalid_trade).is_none());
    }

    #[test]
    fn test_parse_binance_kline_valid_and_integrity() {
        let mut valid_kline = r#"{"e":"kline","E":1672531200000,"s":"BTCUSDT","k":{"o":"60000.0","h":"60100.0","l":"59900.0","c":"60050.0","v":"10.5","x":true}}"#.to_string();
        let res = parse_binance_kline(&mut valid_kline);
        assert!(res.is_some());
        let (e, s, open, high, low, close, vol, is_closed) = res.unwrap();
        assert_eq!(e, 1672531200000);
        assert_eq!(s, "BTCUSDT");
        assert_eq!(open, 60000.0);
        assert_eq!(high, 60100.0);
        assert_eq!(low, 59900.0);
        assert_eq!(close, 60050.0);
        assert_eq!(vol, 10.5);
        assert!(is_closed);

        // Invalid OHLC (high < low)
        let mut invalid_kline = r#"{"e":"kline","E":1672531200000,"s":"BTCUSDT","k":{"o":"60000.0","h":"59800.0","l":"59900.0","c":"60050.0","v":"10.5","x":true}}"#.to_string();
        assert!(parse_binance_kline(&mut invalid_kline).is_none());
    }
}
