#[derive(Debug, Clone, Default)]
pub struct BookTickerEvent {
    pub bid_price: f64,
    pub bid_qty: f64,
    pub ask_price: f64,
    pub ask_qty: f64,
    pub coin_id: usize,
    pub event_time: u64,
}

impl BookTickerEvent {
    /// Zero-allocation JSON parser para nanosegundos (Axioma V5).
    /// FIX #821: Cada campo se busca desde offset 0 (tolerante a reordenación de claves JSON).
    /// Binance puede reordenar claves en payloads proxied/batched.
    #[inline(always)]
    pub fn parse_from_json(bytes: &[u8]) -> Option<Self> {
        // FIX #821: Búsqueda independiente por campo — tolerante a cualquier orden de claves JSON
        let (bid_price, _) = Self::extract_f64_from(bytes, 0, b"\"b\":\"")?;
        let (bid_qty, _) = Self::extract_f64_from(bytes, 0, b"\"B\":\"")?;
        let (ask_price, _) = Self::extract_f64_from(bytes, 0, b"\"a\":\"")?;
        let (ask_qty, _) = Self::extract_f64_from(bytes, 0, b"\"A\":\"")?;

        if !bid_price.is_finite()
            || !bid_qty.is_finite()
            || !ask_price.is_finite()
            || !ask_qty.is_finite()
            || bid_price <= 0.0
            || ask_price <= 0.0
            || bid_qty < 0.0
            || ask_qty < 0.0
        {
            return None;
        }

        // Extract Event Time (E) or Transaction Time (T) from the entire byte slice (offset 0)
        let event_time = if let Some((t, _)) = Self::extract_u64_from(bytes, 0, b"\"E\":") {
            t
        } else if let Some((t, _)) = Self::extract_u64_from(bytes, 0, b"\"T\":") {
            t
        } else {
            0
        };

        Some(Self {
            bid_price,
            bid_qty,
            ask_price,
            ask_qty,
            coin_id: 0,
            event_time,
        })
    }

    #[inline(always)]
    pub fn extract_f64_from(bytes: &[u8], i: usize, key: &[u8]) -> Option<(f64, usize)> {
        // memmem::find es acelerado por hardware (SIMD)
        let found_idx = memchr::memmem::find(&bytes[i..], key)?;
        let start = i + found_idx + key.len();

        // Find the closing quote using memchr (also SIMD)
        let end_offset = memchr::memchr(b'"', &bytes[start..])?;
        let end = start + end_offset;

        // Zero-copy, zero-UTF8 validation parse using fast_float
        let val = fast_float::parse(&bytes[start..end]).ok()?;
        Some((val, end + 1))
    }

    #[inline(always)]
    pub fn extract_u64_from(bytes: &[u8], i: usize, key: &[u8]) -> Option<(u64, usize)> {
        let found_idx = memchr::memmem::find(&bytes[i..], key)?;
        let start = i + found_idx + key.len();

        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }

        let mut val = 0u64;
        for &b in &bytes[start..end] {
            val = val * 10 + (b - b'0') as u64;
        }
        Some((val, end))
    }
}

#[derive(Debug, Clone, Default)]
pub struct AggTradeEvent {
    pub price: f64,
    pub qty: f64,
    pub is_buyer_maker: bool,
}

impl AggTradeEvent {
    #[inline(always)]
    pub fn parse_from_json(bytes: &[u8]) -> Option<Self> {
        // En un aggTrade, extraemos p (price), q (qty) y m (is_buyer_maker)
        let (price, i) = BookTickerEvent::extract_f64_from(bytes, 0, b"\"p\":\"")?;
        let (qty, i) = BookTickerEvent::extract_f64_from(bytes, i, b"\"q\":\"")?;

        if !price.is_finite() || !qty.is_finite() || price <= 0.0 || qty < 0.0 {
            return None;
        }

        let m_idx = memchr::memmem::find(&bytes[i..], b"\"m\":")?;
        let after_m = &bytes[i + m_idx + 4..];
        let first_non_ws = after_m.iter().position(|&b| b != b' ' && b != b'\t')?;
        let is_buyer_maker = after_m.get(first_non_ws) == Some(&b't'); // "t"rue or "f"alse

        Some(Self {
            price,
            qty,
            is_buyer_maker,
        })
    }
}

pub struct DepthEvent {
    pub bid_wall: f64,
    pub ask_wall: f64,
}

impl DepthEvent {
    pub fn parse_from_json(bytes: &[u8]) -> Option<Self> {
        // Para depth@100ms usamos serde_json porque solo llega 10 veces por segundo,
        // a diferencia del tick que llega miles de veces por segundo.
        // Binance @depth10 stream no manda "e":"depthUpdate", manda "bids" y "asks"
        if memchr::memmem::find(bytes, b"\"bids\"").is_none()
            || memchr::memmem::find(bytes, b"\"asks\"").is_none()
        {
            return None;
        }

        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(bytes) {
            let data = if json.get("data").is_some() && !json["data"].is_null() {
                &json["data"]
            } else {
                &json
            };
            let mut bid_wall = 0.0;
            let mut ask_wall = 0.0;

            if let Some(bids) = data["bids"].as_array() {
                for bid in bids {
                    if let Some(qty_val) = bid.get(1) {
                        if let Some(qty_str) = qty_val.as_str() {
                            if let Ok(qty) = qty_str.parse::<f64>() {
                                if qty.is_finite() && qty > 0.0 {
                                    bid_wall += qty;
                                }
                            }
                        }
                    }
                }
            }
            if let Some(asks) = data["asks"].as_array() {
                for ask in asks {
                    if let Some(qty_val) = ask.get(1) {
                        if let Some(qty_str) = qty_val.as_str() {
                            if let Ok(qty) = qty_str.parse::<f64>() {
                                if qty.is_finite() && qty > 0.0 {
                                    ask_wall += qty;
                                }
                            }
                        }
                    }
                }
            }

            return Some(Self { bid_wall, ask_wall });
        }
        None
    }
}

/// 🧠 Normalización Cuántica (Teleonómica)
/// Mantiene estadísticas en tiempo real (Welford's Online Algorithm) para normalizar
/// datos crudos (ej. precios) a Z-Scores o rangos Min-Max sin acumular arrays masivos.
#[derive(Debug, Clone)]
pub struct OnlineNormalizer {
    pub count: u64,
    pub mean: f64,
    pub m2: f64, // Sum of squares of differences from the current mean
    pub min: f64,
    pub max: f64,
}

impl Default for OnlineNormalizer {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }
}

impl OnlineNormalizer {
    #[inline(always)]
    pub fn update(&mut self, value: f64) {
        // FIX #694: Protección contra valores no finitos para no corromper la media/varianza
        if !value.is_finite() {
            return;
        }

        self.count += 1;

        let delta = value - self.mean;
        self.mean += delta / (self.count as f64);

        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    #[inline(always)]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count as f64 - 1.0)
        }
    }

    #[inline(always)]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    #[inline(always)]
    pub fn z_score(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std == 0.0 {
            0.0
        } else {
            (value - self.mean) / std
        }
    }

    #[inline(always)]
    pub fn min_max(&self, value: f64) -> f64 {
        if self.max == self.min {
            0.0
        } else {
            (value - self.min) / (self.max - self.min)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_ticker_event_parse_from_json_nominal() {
        let json = br#"{"u":400900217,"s":"BNBUSDT","b":"25.35190000","B":"31.21000000","a":"25.36520000","A":"40.66000000","T":1564014576000,"E":1564014576001}"#;
        let event = BookTickerEvent::parse_from_json(json).expect("Debe parsear");
        assert_eq!(event.bid_price, 25.3519);
        assert_eq!(event.bid_qty, 31.21);
        assert_eq!(event.ask_price, 25.3652);
        assert_eq!(event.ask_qty, 40.66);
        assert_eq!(event.event_time, 1564014576001);
    }

    #[test]
    fn test_book_ticker_event_parse_from_json_reordered_keys() {
        // Claves en orden inverso
        let json =
            br#"{"E":1672531200000,"A":"10.0","a":"100.5","B":"5.0","b":"100.0","s":"BTCUSDT"}"#;
        let event =
            BookTickerEvent::parse_from_json(json).expect("Debe parsear sin importar el orden");
        assert_eq!(event.bid_price, 100.0);
        assert_eq!(event.ask_price, 100.5);
        assert_eq!(event.bid_qty, 5.0);
        assert_eq!(event.ask_qty, 10.0);
        assert_eq!(event.event_time, 1672531200000);
    }

    #[test]
    fn test_book_ticker_event_parse_from_json_nan_and_negative_rejection() {
        let json_negative = br#"{"b":"-10.0","B":"1.0","a":"10.0","A":"1.0","E":100}"#;
        assert!(BookTickerEvent::parse_from_json(json_negative).is_none());

        let json_zero_price = br#"{"b":"0.0","B":"1.0","a":"10.0","A":"1.0","E":100}"#;
        assert!(BookTickerEvent::parse_from_json(json_zero_price).is_none());
    }

    #[test]
    fn test_agg_trade_event_parse_from_json() {
        let json = br#"{"e":"aggTrade","E":123456789,"s":"BNBUSDT","a":12345,"p":"0.001","q":"100","f":100,"l":105,"T":123456785,"m":true}"#;
        let event = AggTradeEvent::parse_from_json(json).expect("Debe parsear");
        assert_eq!(event.price, 0.001);
        assert_eq!(event.qty, 100.0);
        assert!(event.is_buyer_maker);
    }

    #[test]
    fn test_agg_trade_event_parse_from_json_with_whitespace() {
        // Formato Binance con espacios estándar
        let json_true = br#"{"e":"aggTrade","E":123456789,"s":"BTCUSDT","a":99,"p":"60000.5","q":"0.25","f":1,"l":2,"T":12345,"m": true}"#;
        let event_true =
            AggTradeEvent::parse_from_json(json_true).expect("Debe parsear con espacios");
        assert_eq!(event_true.price, 60000.5);
        assert_eq!(event_true.qty, 0.25);
        assert!(event_true.is_buyer_maker);

        let json_false = br#"{"e":"aggTrade","E":123456789,"s":"BTCUSDT","a":100,"p":"60001.0","q":"1.5","f":3,"l":4,"T":12346,"m":  false}"#;
        let event_false =
            AggTradeEvent::parse_from_json(json_false).expect("Debe parsear con tabs/espacios");
        assert_eq!(event_false.price, 60001.0);
        assert_eq!(event_false.qty, 1.5);
        assert!(!event_false.is_buyer_maker);
    }

    #[test]
    fn test_depth_event_parse_from_json() {
        let json = br#"{"bids":[["100.0","5.0"],["99.0","10.0"]],"asks":[["101.0","8.0"],["102.0","12.0"]]}"#;
        let event = DepthEvent::parse_from_json(json).expect("Debe parsear");
        assert_eq!(event.bid_wall, 15.0);
        assert_eq!(event.ask_wall, 20.0);
    }

    #[test]
    fn test_online_normalizer_welford_variance_and_nan_immunity() {
        let mut normalizer = OnlineNormalizer::default();
        normalizer.update(10.0);
        normalizer.update(20.0);
        normalizer.update(30.0);
        normalizer.update(f64::NAN); // Inmune

        assert_eq!(normalizer.count, 3);
        assert_eq!(normalizer.mean, 20.0);
        assert_eq!(normalizer.variance(), 100.0);
        assert_eq!(normalizer.std_dev(), 10.0);
        assert_eq!(normalizer.z_score(20.0), 0.0);
        assert_eq!(normalizer.z_score(30.0), 1.0);
        assert_eq!(normalizer.min_max(10.0), 0.0);
        assert_eq!(normalizer.min_max(30.0), 1.0);
        assert_eq!(normalizer.min_max(20.0), 0.5);
    }
}
