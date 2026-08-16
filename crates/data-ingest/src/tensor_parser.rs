use std::str::from_utf8_unchecked;

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
        // En producción HFT, usamos rutinas seguras.
        // Reemplazado from_utf8_unchecked por from_utf8 para evitar UB por Segfaults de red.
        if let Ok(s) = std::str::from_utf8(bytes) {
            s.parse::<f64>().unwrap_or(0.0)
        } else {
            0.0
        }
    }

    /// Busca la llave `"p":"` (Precio) o `"q":"` (Cantidad) en un flujo crudo de Binance
    /// sin instanciar un objeto JSON en memoria (Zero-allocation string matching).
    #[inline]
    pub fn extract_tensor_feature(payload: &[u8], key: &[u8]) -> Option<f64> {
        let mut i = 0;
        let len = payload.len();
        let key_len = key.len();

        while i + key_len < len {
            // Coincidencia de prefijo (ej: `"p":"`)
            if &payload[i..i + key_len] == key {
                i += key_len;
                let start = i;
                // Buscar la comilla de cierre
                while i < len && payload[i] != b'"' {
                    i += 1;
                }
                let val_bytes = &payload[start..i];
                return Some(Self::fast_parse_f64(val_bytes));
            }
            i += 1;
        }
        None
    }

    /// Analiza un Tick crudo de Binance L1 y extrae Precio y Cantidad directamente a un Tensor
    #[inline]
    pub fn decode_binance_trade_to_tensor(payload: &[u8]) -> [f64; 2] {
        // Buscar `"p":"`
        let price = Self::extract_tensor_feature(payload, b"\"p\":\"").unwrap_or(0.0);
        // Buscar `"q":"`
        let qty = Self::extract_tensor_feature(payload, b"\"q\":\"").unwrap_or(0.0);
        
        [price, qty]
    }
}
