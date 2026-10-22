/// 🛡️ ALGORITMO #158: TENSOR PARSER (ZERO-ALLOCATION INGEST)
/// Decodificador ultrarrápido para Websockets que asume posiciones fijas o busca
/// patrones de bytes en C. Evita deserializar objetos JSON completos (Serde),
/// extrayendo puramente la semántica matemática de los Ticks (Precio/Volumen).
///
/// # ESTADO ESTRUCTURAL (censo F1-092, 2026-10-22) — LÉELO ANTES DE USAR
///
/// Este módulo es una **API de compatibilidad SIN productor operativo**: sus
/// funciones sólo las consumen sus propios tests y
/// `tests/liquidation_contract.rs` (que ya lo dice: *"retained but removed from
/// the operational producer"*). La ruta VIVA de ingesta de trades es
/// `data_pipeline::parser::AggTradeEvent::parse_from_json`, que usa
/// `ws_client.rs` y que **sí es fail-closed** (exige `price > 0`, `qty >= 0`,
/// finitos, y devuelve `None` ante cualquier campo ilegible).
///
/// Es decir: hay dos implementaciones del mismo trabajo y sólo una está
/// conectada. Queda aquí como deuda declarada con su condición de reactivación
/// —conectar esta ruta exigiría demostrar que alimenta una decisión real, no
/// duplicar `parser.rs`— y, entretanto, **se corrige para que no pueda fabricar
/// un trade a precio 0**: un huérfano que devuelve ceros silenciosos es una
/// trampa armada para el día que alguien lo revive.
pub struct TensorParser;

impl TensorParser {
    /// Extrae un f64 desde un arreglo de bytes crudo (ASCII numérico).
    /// Peligro: Asume que la entrada es un número válido. `unsafe` es intencional por HFT.
    ///
    /// F1-092 — ATENCIÓN: el `0.0` de esta función es **AMBIGUO**. Vale tanto
    /// para `"0"` (un cero real) como para `""`, `"abc"` o `"1e400"` (ilegible
    /// o no finito). En el camino de dinero eso es inaceptable: quien necesite
    /// distinguir «es cero» de «no pude leer» debe usar [`Self::parse_estricto`],
    /// que devuelve `Option`. Se conserva la firma rápida porque
    /// `liquidation_contract` la ejercita, pero ninguna decisión debe tomarse
    /// sobre su resultado sin esa distinción.
    #[inline(always)]
    pub fn fast_parse_f64(bytes: &[u8]) -> f64 {
        Self::escanear(bytes).0
    }

    /// Parse ESTRICTO (F1-092): `Some(v)` sólo si se consumió **al menos un
    /// dígito de mantisa** y el resultado es finito. `None` significa ilegible:
    /// vacío, sin dígitos (`"abc"`, `"-"`, `".e5"`) o no finito (`"1e400"`).
    ///
    /// Es la misma exploración de bytes que [`Self::fast_parse_f64`] —no hay
    /// dos parsers que puedan divergir—, con la bandera de validez que la ruta
    /// rápida no puede expresar por su tipo de retorno.
    #[inline(always)]
    fn parse_estricto(bytes: &[u8]) -> Option<f64> {
        let (valor, digitos) = Self::escanear(bytes);
        if digitos && valor.is_finite() {
            Some(valor)
        } else {
            None
        }
    }

    /// Exploración única de bytes numéricos ASCII. Devuelve `(valor, digitos)`
    /// donde `digitos` indica si vio al menos un dígito de la mantisa: es lo
    /// que separa un cero legitimo de una entrada ilegible.
    #[inline(always)]
    fn escanear(bytes: &[u8]) -> (f64, bool) {
        if bytes.is_empty() {
            return (0.0, false);
        }
        let mut i = 0;
        let mut sign = 1.0;
        if bytes[0] == b'-' {
            sign = -1.0;
            i += 1;
        } else if bytes[0] == b'+' {
            i += 1;
        }

        let mut digitos = false;
        let mut int_part = 0.0;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            int_part = int_part * 10.0 + (bytes[i] - b'0') as f64;
            digitos = true;
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
                }
                frac_digits += 1;
                digitos = true;
                i += 1;
            }
        }

        // AGY-AUD-P11: Soporte de notación científica (e.g. 1e-5, 2.45e-4, 5E+3)
        // Crítico en Binance para monedas de bajo satoshi y cantidades fraccionarias.
        let mut exp_sign = 1.0;
        let mut exp_val = 0.0;
        if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
            i += 1;
            if i < bytes.len() {
                if bytes[i] == b'-' {
                    exp_sign = -1.0;
                    i += 1;
                } else if bytes[i] == b'+' {
                    i += 1;
                }
            }
            while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
                exp_val = exp_val * 10.0 + (bytes[i] - b'0') as f64;
                i += 1;
            }
        }

        let base = sign * (int_part + frac_part / frac_scale);
        let res = if exp_val > 0.0 {
            base * 10.0_f64.powf(exp_sign * exp_val)
        } else {
            base
        };

        // FIX #636: Retorno estrictamente finito. La banda de validez la
        // consume `parse_estricto`; aquí sólo se garantiza el invariante.
        if res.is_finite() {
            (res, digitos)
        } else {
            (0.0, false)
        }
    }

    /// Busca la llave `"p":` (Precio) o `"q":` (Cantidad) en un flujo crudo de Binance
    /// sin instanciar un objeto JSON en memoria (Zero-allocation string matching).
    /// FIX #820: Soporta tanto valores con comillas ("123.45") como sin comillas (123.45)
    ///
    /// F1-092 — `Some(v)` significa AHORA «leí un número real». Antes el valor
    /// pasaba por `fast_parse_f64`, que devuelve `0.0` también para basura, así
    /// que un campo presente pero ilegible entregaba `Some(0.0)`: el cero
    /// silencioso que el mandato prohíbe.
    #[inline]
    pub fn extract_tensor_feature(payload: &[u8], key: &[u8]) -> Option<f64> {
        let mut i = 0;
        let len = payload.len();
        let key_len = key.len();

        while i + key_len <= len {
            if &payload[i..i + key_len] == key {
                i += key_len;
                if i < len && payload[i] == b'"' {
                    i += 1;
                    let val_start = i;
                    while i < len && payload[i] != b'"' {
                        i += 1;
                    }
                    return Self::parse_estricto(&payload[val_start..i]);
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
                    return Self::parse_estricto(&payload[val_start..i]);
                }
            }
            i += 1;
        }
        None
    }

    /// Analiza un Tick crudo de Binance L1 y extrae Precio y Cantidad directamente a un Tensor.
    ///
    /// F1-092 — devuelve `None` cuando el tick no se puede leer COMPLETO, o
    /// cuando el precio o la cantidad no son estrictamente positivos. Antes
    /// devolvía `[0.0, 0.0]`: un **trade a precio cero** entrando al tensor, que
    /// cualquier media, retorno o volatilidad corriente abajo convierte en
    /// basura. El criterio es el mismo que ya aplica `parse_force_orders` más
    /// abajo y la ruta viva `AggTradeEvent::parse_from_json`: quien reciba
    /// `None` descarta el tick, no lo adivina.
    #[inline]
    pub fn decode_binance_trade_to_tensor(payload: &[u8]) -> Option<[f64; 2]> {
        let price = Self::extract_tensor_feature(payload, b"\"p\":")?;
        let qty = Self::extract_tensor_feature(payload, b"\"q\":")?;
        if price > 0.0 && qty > 0.0 {
            Some([price, qty])
        } else {
            None
        }
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
        assert_eq!(tensor, Some([60050.25, 1.500]));
    }

    #[test]
    fn test_f1_092_ningun_cero_fabricado_en_el_camino_del_tick() {
        // F1-092: un tick ilegible se DESCARTA, no se convierte en [0.0, 0.0].
        let base = br#"{"e":"trade","s":"BTCUSDT","p":"60050.25","q":"1.500"}"#;
        assert_eq!(
            TensorParser::decode_binance_trade_to_tensor(base),
            Some([60050.25, 1.500])
        );

        // Precio ausente, cantidad ausente, ambos ausentes.
        assert_eq!(
            TensorParser::decode_binance_trade_to_tensor(br#"{"e":"trade","q":"1.5"}"#),
            None
        );
        assert_eq!(
            TensorParser::decode_binance_trade_to_tensor(br#"{"e":"trade","p":"100.0"}"#),
            None
        );
        assert_eq!(TensorParser::decode_binance_trade_to_tensor(br#"{"e":"trade"}"#), None);

        // Valor PRESENTE pero ilegible: antes entregaba Some(0.0).
        for payload in [
            &br#"{"p":"","q":"1.5"}"#[..],
            &br#"{"p":"abc","q":"1.5"}"#[..],
            &br#"{"p":"NaN","q":"1.5"}"#[..],
            &br#"{"p":"inf","q":"1.5"}"#[..],
            &br#"{"p":"1e400","q":"1.5"}"#[..],
        ] {
            assert_eq!(
                TensorParser::decode_binance_trade_to_tensor(payload),
                None,
                "precio ilegible en {}",
                String::from_utf8_lossy(payload)
            );
        }

        // Cero y negativos: un trade a precio 0 o cantidad 0 no es un trade.
        assert_eq!(
            TensorParser::decode_binance_trade_to_tensor(br#"{"p":"0","q":"1.5"}"#),
            None
        );
        assert_eq!(
            TensorParser::decode_binance_trade_to_tensor(br#"{"p":"-100.0","q":"1.5"}"#),
            None
        );
        assert_eq!(
            TensorParser::decode_binance_trade_to_tensor(br#"{"p":"100.0","q":"0"}"#),
            None
        );
        // El cero REAL sigue siendo un cero: la bandera de validez distingue.
        assert_eq!(TensorParser::fast_parse_f64(b"0"), 0.0);
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

        // F1-092: `fast_parse_f64` sigue devolviendo 0.0 (firma rapida, contrato
        // historico), pero ese cero es AMBIGUO. Lo que cambia es que
        // `extract_tensor_feature` ya NO lo propaga como dato.
        let malformed_f64 = TensorParser::fast_parse_f64(b"abc.def");
        assert_eq!(malformed_f64, 0.0, "la ruta rapida conserva su contrato historico");
        assert_eq!(
            TensorParser::extract_tensor_feature(br#"{"p":"abc.def"}"#, b"\"p\":"),
            None,
            "pero el extractor ya no entrega el cero como si fuera un dato"
        );
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

    #[test]
    fn test_tensor_parser_scientific_notation() {
        // Validación de potencias negativas (altcoins de bajo precio / fracciones)
        assert_eq!(TensorParser::fast_parse_f64(b"1e-5"), 0.00001);
        assert!((TensorParser::fast_parse_f64(b"2.45e-4") - 0.000245).abs() < 1e-12);
        assert!((TensorParser::fast_parse_f64(b"1.23E-06") - 0.00000123).abs() < 1e-14);

        // Potencias positivas
        assert_eq!(TensorParser::fast_parse_f64(b"5e3"), 5000.0);
        assert_eq!(TensorParser::fast_parse_f64(b"+2.5E+2"), 250.0);

        // Signos negativos
        assert_eq!(TensorParser::fast_parse_f64(b"-3.5e-2"), -0.035);

        // Integrado en JSON sin comillas y con comillas
        let json_exp_unquoted = br#"{"p":1.5e-3,"q":2e4}"#;
        assert_eq!(TensorParser::extract_tensor_feature(json_exp_unquoted, b"\"p\":"), Some(0.0015));
        assert_eq!(TensorParser::extract_tensor_feature(json_exp_unquoted, b"\"q\":"), Some(20000.0));

        let json_exp_quoted = br#"{"p":"4.56e-5","q":"1.0e2"}"#;
        assert!((TensorParser::extract_tensor_feature(json_exp_quoted, b"\"p\":").unwrap() - 0.0000456).abs() < 1e-11);
        assert_eq!(TensorParser::extract_tensor_feature(json_exp_quoted, b"\"q\":"), Some(100.0));
    }
}
