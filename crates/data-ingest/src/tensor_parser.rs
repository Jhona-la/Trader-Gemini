/// 🛡️ ALGORITMO #158: TENSOR PARSER (ZERO-ALLOCATION INGEST)
/// Decodificador ultrarrápido para Websockets que asume posiciones fijas o busca
/// patrones de bytes en C. Evita deserializar objetos JSON completos (Serde),
/// extrayendo puramente la semántica matemática de los Ticks (Precio/Volumen).

pub struct TensorParser;

impl TensorParser {
    /// Extrae un f64 desde un arreglo de bytes crudo (ASCII numérico).
    /// Peligro: Asume que la entrada es un número válido. `unsafe` es intencional por HFT.
    #[inline(always)]
    pub fn fast_parse_f64(bytes: &[u8]) -> f64 {
        if bytes.is_empty() {
            return 0.0;
        }
        let mut i = 0;
        let mut sign = 1.0;
        if bytes[0] == b'-' {
            sign = -1.0;
            i += 1;
        } else if bytes[0] == b'+' {
            i += 1;
        }

        let mut int_part = 0.0;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            int_part = int_part * 10.0 + (bytes[i] - b'0') as f64;
            i += 1;
        }

        let mut frac_part = 0.0;
        let mut frac_scale = 1.0;
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            let mut frac_digits = 0;
            while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
                if frac_digits < 18 {
                    frac_part = frac_part * 10.0 + (bytes[i] - b'0') as f64;
                    frac_scale *= 10.0;
                    frac_digits += 1;
                }
                i += 1;
            }
        }

        // FIX #636: Retorno estrictamente finito
        let res = sign * (int_part + frac_part / frac_scale);
        if res.is_finite() {
            res
        } else {
            0.0
        }
    }

    /// Busca la llave `"p":` (Precio) o `"q":` (Cantidad) en un flujo crudo de Binance
    /// sin instanciar un objeto JSON en memoria (Zero-allocation string matching).
    /// FIX #820: Soporta tanto valores con comillas ("123.45") como sin comillas (123.45)
    #[inline]
    pub fn extract_tensor_feature(payload: &[u8], key: &[u8]) -> Option<f64> {
        let mut i = 0;
        let len = payload.len();
        let key_len = key.len();

        while i + key_len < len {
            if &payload[i..i + key_len] == key {
                i += key_len;
                if i < len && payload[i] == b'"' {
                    i += 1;
                    let val_start = i;
                    while i < len && payload[i] != b'"' {
                        i += 1;
                    }
                    return Some(Self::fast_parse_f64(&payload[val_start..i]));
                } else {
                    let val_start = i;
                    while i < len
                        && payload[i] != b','
                        && payload[i] != b'}'
                        && payload[i] != b']'
                        && payload[i] != b' '
                    {
                        i += 1;
                    }
                    return Some(Self::fast_parse_f64(&payload[val_start..i]));
                }
            }
            i += 1;
        }
        None
    }

    /// Analiza un Tick crudo de Binance L1 y extrae Precio y Cantidad directamente a un Tensor
    #[inline]
    pub fn decode_binance_trade_to_tensor(payload: &[u8]) -> [f64; 2] {
        let price = Self::extract_tensor_feature(payload, b"\"p\":").unwrap_or(0.0);
        let qty = Self::extract_tensor_feature(payload, b"\"q\":").unwrap_or(0.0);

        [price, qty]
    }
    /// Decodifica de forma zero-allocation todas las liquidaciones (p, q) en un payload crudo de Binance
    /// sin instanciar objetos JSON en el heap (Zero-allocation Byte-Scanner).
    #[inline]
    pub fn parse_force_orders<F: FnMut(f64, f64)>(payload: &[u8], mut callback: F) {
        if payload.is_empty() || !payload.contains(&b'{') {
            return;
        }
        let mut depth: u32 = 0;
        let mut obj_start: Option<usize> = None;
        let len = payload.len();

        for i in 0..len {
            let b = payload[i];
            if b == b'{' {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            } else if b == b'}' {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start) = obj_start {
                            let block = &payload[start..=i];
                            // Extraer p (precio) y q (cantidad) dentro del bloque del evento
                            let p_opt = Self::extract_tensor_feature(block, b"\"p\":");
                            let q_opt = Self::extract_tensor_feature(block, b"\"q\":");
                            if let (Some(p), Some(q)) = (p_opt, q_opt) {
                                if p.is_finite() && q.is_finite() && p > 0.0 && q > 0.0 {
                                    callback(p, q);
                                }
                            }
                        }
                        obj_start = None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_parse_f64_integer_and_fractional() {
        assert_eq!(TensorParser::fast_parse_f64(b"123.456"), 123.456);
        assert_eq!(TensorParser::fast_parse_f64(b"-50.25"), -50.25);
        assert_eq!(TensorParser::fast_parse_f64(b"+100"), 100.0);
        assert_eq!(TensorParser::fast_parse_f64(b""), 0.0);
    }

    #[test]
    fn test_extract_tensor_feature_and_decode_binance_trade() {
        let payload = br#"{"e":"trade","E":1672531199000,"s":"BTCUSDT","t":12345,"p":"60050.25","q":"1.500","b":88,"a":89,"T":1672531199000,"m":true}"#;
        let tensor = TensorParser::decode_binance_trade_to_tensor(payload);
        assert_eq!(tensor[0], 60050.25);
        assert_eq!(tensor[1], 1.500);
    }

    #[test]
    fn test_tensor_parser_unquoted_and_malformed_bytes() {
        // Unquoted numeric values
        let unquoted_payload = br#"{"p":60050.25,"q":2.5,"status":"FILLED"}"#;
        let p = TensorParser::extract_tensor_feature(unquoted_payload, b"\"p\":");
        let q = TensorParser::extract_tensor_feature(unquoted_payload, b"\"q\":");
        assert_eq!(p, Some(60050.25));
        assert_eq!(q, Some(2.5));

        // Missing key & malformed
        let missing = TensorParser::extract_tensor_feature(br#"{"x":100}"#, b"\"p\":");
        assert_eq!(missing, None);

        let malformed_f64 = TensorParser::fast_parse_f64(b"abc.def");
        assert_eq!(malformed_f64, 0.0);
    }

    #[test]
    fn test_tensor_parser_large_precision_and_trailing_characters() {
        let precise = TensorParser::fast_parse_f64(b"0.1234567890123456");
        assert!((precise - 0.1234567890123456).abs() < 1e-12);

        // Terminating with closing bracket or comma
        let json_segment = br#"{"price":1234.56, "qty":0.5}"#;
        let price = TensorParser::extract_tensor_feature(json_segment, b"\"price\":");
        assert_eq!(price, Some(1234.56));
    }

    #[test]
    fn test_tensor_parser_parse_force_orders() {
        // 1. Single nested forceOrder
        let single_payload = br#"{"e":"forceOrder","E":1568014460893,"o":{"s":"BTCUSDT","S":"SELL","q":"0.014","p":"9910.50"}}"#;
        let mut results = Vec::new();
        TensorParser::parse_force_orders(single_payload, |p, q| {
            results.push((p, q));
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (9910.50, 0.014));

        // 2. Array of force orders
        let array_payload = br#"[
            {"e":"forceOrder","o":{"s":"ETHUSDT","p":"3100.25","q":"2.50"}},
            {"e":"forceOrder","o":{"s":"SOLUSDT","p":"145.00","q":"50.0"}}
        ]"#;
        let mut arr_results = Vec::new();
        TensorParser::parse_force_orders(array_payload, |p, q| {
            arr_results.push((p, q));
        });
        assert_eq!(arr_results.len(), 2);
        assert_eq!(arr_results[0], (3100.25, 2.50));
        assert_eq!(arr_results[1], (145.00, 50.0));

        // 3. Malformed/empty payload
        let mut empty_results = Vec::new();
        TensorParser::parse_force_orders(b"[]", |p, q| {
            empty_results.push((p, q));
        });
        assert_eq!(empty_results.len(), 0);
    }
}
