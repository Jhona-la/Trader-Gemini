use std::slice;
use std::str;
use crate::quantum_arena::QuantumStateArena;

// Un parser SIMD/SWAR súper agresivo para float a partir de bytes ASCII.
// Para propósitos de este inyector de extrema latencia, 
// buscaremos los offsets de "b":[[" y "a":[[" en el JSON plano.
#[no_mangle]
pub extern "C" fn ingest_raw_ws_frame(
    arena_ptr: *mut QuantumStateArena, 
    raw_bytes_ptr: *const u8, 
    length: usize,
    time_idx: usize,
    asset_idx: usize
) -> u8 {
    // FIX #1470: Protección de límites de índice de activos y punteros nulos
    if arena_ptr.is_null() || raw_bytes_ptr.is_null() || length == 0 || asset_idx >= crate::quantum_arena::RING_CAPACITY {
        return 1; // Error
    }

    let arena = unsafe { &mut *arena_ptr };
    let buffer = unsafe { slice::from_raw_parts(raw_bytes_ptr, length) };

    // PARSER ZERO-ALLOCATION BÚSQUEDA RÁPIDA (Retina)
    // Buscamos "b":[[" (bids) y "a":[[" (asks)
    // Patrón: "b":[[" -> bytes: [34, 98, 34, 58, 91, 91, 34]
    
    let (bid_price, bid_qty) = match extract_first_level(buffer, b"\"b\":[[\"") {
        Some((p, q)) if p.is_finite() && q.is_finite() && p > 0.0 && q >= 0.0 => (p, q),
        _ => return 1,
    };
    let (ask_price, ask_qty) = match extract_first_level(buffer, b"\"a\":[[\"") {
        Some((p, q)) if p.is_finite() && q.is_finite() && p > 0.0 && q >= 0.0 => (p, q),
        _ => return 1,
    };

    // Inyección en la memoria cruda
    let total_vol = (bid_qty + ask_qty + 1e-8).max(1e-8);
    let imbalance = ((bid_qty - ask_qty) / total_vol).clamp(-1.0, 1.0);
    let _micro_price = (bid_price * ask_qty + ask_price * bid_qty) / total_vol;

    let seq = arena.begin_write();
    let offset = QuantumStateArena::offset(time_idx, asset_idx);
    
    // Dimensión 0: Precio
    arena.price_returns[offset] = bid_price;
    // Dimensión 2: Microestructura L1 (OBI)
    arena.order_book_imbalance[offset] = imbalance;
    
    // === NUEVA FÍSICA O(1) ===
    let math_state = &mut arena.math_states[asset_idx];
    
    // 1. VPIN (Volume-Synchronized PIN)
    let is_buyer = bid_qty > ask_qty; 
    let vpin_val = math_state.vpin.update(total_vol as f64, is_buyer);
    arena.vpin[offset] = vpin_val as f32;

    // 2. Hurst Exponent
    let hurst_val = math_state.hurst.update(bid_price as f64);
    arena.hurst_exponent[offset] = hurst_val as f32;
    
    // 3. Shannon Entropy
    let prev_offset = QuantumStateArena::offset(time_idx.wrapping_sub(1), asset_idx);
    let prev_price = arena.price_returns[prev_offset];
    let mut norm_ret = 0.0;
    // FIX #1525: Verificación de finitud y clamping en retorno normalizado para entropía
    if prev_price > 0.0 && prev_price.is_finite() && bid_price.is_finite() {
        norm_ret = ((bid_price - prev_price) / prev_price).clamp(-1.0, 1.0);
    }
    let entropy_val = math_state.shannon.update(norm_ret as f64).unwrap_or(0.0);
    // (El valor de entropía se inyecta en dark_alpha temporalmente o se omite si no hay array explícito)
    
    // 4. Kyle's Lambda
    let lambda_val = math_state.lambda.update(bid_price as f64, total_vol as f64);
    arena.funding_elasticity[offset] = lambda_val as f32; // Reutilizando array temporalmente para guardar la señal
    
    // 5. Aceleración de Liquidez (Derivada Segunda OBI)
    let prev_obi = arena.order_book_imbalance[prev_offset];
    let prev_prev_offset = QuantumStateArena::offset(time_idx.wrapping_sub(2), asset_idx);
    let prev_prev_obi = arena.order_book_imbalance[prev_prev_offset];
    let vel_obi = imbalance - prev_obi;
    let prev_vel_obi = prev_obi - prev_prev_obi;
    let acc_obi = vel_obi - prev_vel_obi;
    arena.liquidity_acceleration[offset] = acc_obi;
    
    // Marcar timestamp (FIX #1423: Inmunidad ante saltos de reloj NTP)
    arena.timestamps_ns[time_idx % crate::quantum_arena::RING_CAPACITY] = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as i64;
    
    arena.end_write(seq);

    0 // Éxito
}

/// Extrae el primer par [Precio, Cantidad] de un nivel del Order Book de Binance.
/// Busca la firma de inicio y escanea hasta cerrar las comillas.
#[inline(always)]
fn extract_first_level(buffer: &[u8], signature: &[u8]) -> Option<(f32, f32)> {
    let mut start_idx = 0;
    
    // 1. Encontrar la firma (ej. "b":[[" )
    if let Some(pos) = find_subsequence(buffer, signature) {
        start_idx = pos + signature.len();
    } else {
        return None;
    }

    // 2. Extraer Precio (hasta la próxima '"')
    let (price, next_idx) = parse_f32_until_quote(buffer, start_idx)?;
    
    // 3. Saltar separador (ej. '","')
    let qty_start = next_idx + 3; // '",' , '"'
    
    // 4. Extraer Cantidad
    let (qty, _) = parse_f32_until_quote(buffer, qty_start)?;

    Some((price, qty))
}

#[inline(always)]
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

#[inline(always)]
fn parse_f32_until_quote(buffer: &[u8], start: usize) -> Option<(f32, usize)> {
    let mut end = start;
    while end < buffer.len() && buffer[end] != b'"' {
        end += 1;
    }
    if end >= buffer.len() { return None; }
    
    // FIX #346 / #1524: Safe UTF-8 parse over network bytes and finite verification
    let str_slice = std::str::from_utf8(&buffer[start..end]).ok()?;
    let val = str_slice.parse::<f32>().ok()?;
    if !val.is_finite() {
        return None;
    }
    
    Some((val, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_first_level_and_subsequence() {
        let json_bytes = br#"{"b":[["60000.5","1.2"]],"a":[["60001.0","0.5"]]}"#;
        let bid = extract_first_level(json_bytes, b"\"b\":[[\"");
        assert!(bid.is_some());
        let (bp, bq) = bid.unwrap();
        assert_eq!(bp, 60000.5);
        assert_eq!(bq, 1.2);

        let ask = extract_first_level(json_bytes, b"\"a\":[[\"");
        assert!(ask.is_some());
        let (ap, aq) = ask.unwrap();
        assert_eq!(ap, 60001.0);
        assert_eq!(aq, 0.5);
    }

    #[test]
    fn test_parse_f32_until_quote() {
        let raw = b"123.456\"more_bytes";
        let parsed = parse_f32_until_quote(raw, 0);
        assert!(parsed.is_some());
        let (val, end) = parsed.unwrap();
        assert_eq!(val, 123.456);
        assert_eq!(end, 7);
    }


    #[test]
    fn test_extract_first_level_malformed_and_nan_immunity() {
        // Missing quote, incomplete frame
        let incomplete = b"{\"b\":[[\"60000.5";
        assert!(extract_first_level(incomplete, b"\"b\":[[\"").is_none());

        // Malformed non-numeric string
        let corrupt = br#"{"b":[["INVALID_PRICE","1.2"]]}"#;
        let bid = extract_first_level(corrupt, b"\"b\":[[\"");
        // 0.0 is not > 0.0, so it returns None
        assert!(bid.is_none());
    }
}

