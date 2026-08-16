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
    /// Asume el formato ordenado de Binance: {"u":...,"s":"...","b":"...","B":"...","a":"...","A":"..."}
    #[inline(always)]
    pub fn parse_from_json(bytes: &[u8]) -> Option<Self> {
        let (bid_price, i) = Self::extract_f64_from(bytes, 0, b"\"b\":\"")?;
        let (bid_qty, i) = Self::extract_f64_from(bytes, i, b"\"B\":\"")?;
        let (ask_price, i) = Self::extract_f64_from(bytes, i, b"\"a\":\"")?;
        let (ask_qty, i) = Self::extract_f64_from(bytes, i, b"\"A\":\"")?;

        // Extract Event Time (E) or Transaction Time (T)
        // Note: they are not in quotes: "E":1656093845014
        let event_time = if let Some((t, _)) = Self::extract_u64_from(bytes, i, b"\"E\":") {
            t
        } else if let Some((t, _)) = Self::extract_u64_from(bytes, i, b"\"T\":") {
            t
        } else {
            0
        };

        // El symbol lo dejamos hardcodeado por ahora o no lo parseamos dinámicamente si no se usa
        // En Producción unificada, the symbol is known by the Streamer.

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

        let m_idx = memchr::memmem::find(&bytes[i..], b"\"m\":")?;
        let m_start = i + m_idx + 4;
        let is_buyer_maker = bytes.get(m_start) == Some(&b't'); // "t"rue or "f"alse

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
            let data = &json["data"];
            let mut bid_wall = 0.0;
            let mut ask_wall = 0.0;

            if let Some(bids) = data["bids"].as_array() {
                for bid in bids {
                    if let Some(qty_str) = bid[1].as_str() {
                        bid_wall += qty_str.parse::<f64>().unwrap_or(0.0);
                    }
                }
            }
            if let Some(asks) = data["asks"].as_array() {
                for ask in asks {
                    if let Some(qty_str) = ask[1].as_str() {
                        ask_wall += qty_str.parse::<f64>().unwrap_or(0.0);
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
