import os

filepath = 'core/rust_engine/src/parsers.rs'
content = '''use serde_json::Value;

/// Parses a Binance DepthUpdate (Orderbook) JSON string instantly.
/// Returns (last_update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty)
pub fn parse_binance_depth(json_str: &str) -> Option<(i64, f64, f64, f64, f64)> {
    let parsed: Value = serde_json::from_str(json_str).ok()?;
    
    let last_update_id = parsed.get("u")?.as_i64()?;
    
    let bids = parsed.get("b")?.as_array()?;
    let asks = parsed.get("a")?.as_array()?;
    
    if bids.is_empty() || asks.is_empty() {
        return None;
    }
    
    let best_bid = bids[0].as_array()?;
    let best_ask = asks[0].as_array()?;
    
    let bp = best_bid[0].as_str()?.parse::<f64>().ok()?;
    let bq = best_bid[1].as_str()?.parse::<f64>().ok()?;
    
    let ap = best_ask[0].as_str()?.parse::<f64>().ok()?;
    let aq = best_ask[1].as_str()?.parse::<f64>().ok()?;
    
    Some((last_update_id, bp, bq, ap, aq))
}

/// Parses a Binance Trade JSON string.
/// Returns (trade_time, price, qty, is_buyer_maker)
pub fn parse_binance_trade(json_str: &str) -> Option<(i64, f64, f64, bool)> {
    let parsed: Value = serde_json::from_str(json_str).ok()?;
    
    let t = parsed.get("T")?.as_i64()?;
    let p = parsed.get("p")?.as_str()?.parse::<f64>().ok()?;
    let q = parsed.get("q")?.as_str()?.parse::<f64>().ok()?;
    let m = parsed.get("m")?.as_bool()?;
    
    Some((t, p, q, m))
}
'''
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

lib_path = 'core/rust_engine/src/lib.rs'
with open(lib_path, 'r', encoding='utf-8') as f:
    lib_content = f.read()

lib_content = 'pub mod parsers;\n' + lib_content

ffi_append = '''
// =====================================================================
// FFI C-ABI EXPORTS: WEBSOCKET JSON PARSERS
// =====================================================================

use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub unsafe extern "C" fn ffi_parse_binance_depth(
    json_ptr: *const c_char,
    out_ptr: *mut f64
) -> bool {
    if json_ptr.is_null() || out_ptr.is_null() { return false; }
    
    let c_str = CStr::from_ptr(json_ptr);
    if let Ok(json_str) = c_str.to_str() {
        if let Some((u, bp, bq, ap, aq)) = parsers::parse_binance_depth(json_str) {
            let out = std::slice::from_raw_parts_mut(out_ptr, 5);
            out[0] = u as f64;
            out[1] = bp;
            out[2] = bq;
            out[3] = ap;
            out[4] = aq;
            return true;
        }
    }
    false
}

#[no_mangle]
pub unsafe extern "C" fn ffi_parse_binance_trade(
    json_ptr: *const c_char,
    out_ptr: *mut f64
) -> bool {
    if json_ptr.is_null() || out_ptr.is_null() { return false; }
    
    let c_str = CStr::from_ptr(json_ptr);
    if let Ok(json_str) = c_str.to_str() {
        if let Some((t, p, q, m)) = parsers::parse_binance_trade(json_str) {
            let out = std::slice::from_raw_parts_mut(out_ptr, 4);
            out[0] = t as f64;
            out[1] = p;
            out[2] = q;
            out[3] = if m { 1.0 } else { 0.0 };
            return true;
        }
    }
    false
}
'''

with open(lib_path, 'a', encoding='utf-8') as f:
    f.write(ffi_append)
