#[derive(Debug, Clone, Default)]
pub struct BookTickerEvent {
    pub update_id: u64,
    pub symbol_bytes: [u8; 16], // 100% stack allocated, zero heap
    pub symbol_len: u8,
    pub bid_price: f64,
    pub bid_qty: f64,
    pub ask_price: f64,
    pub ask_qty: f64,
    pub coin_id: usize,
}

impl BookTickerEvent {
    /// Zero-allocation JSON parser para nanosegundos (Axioma V5).
    /// Asume el formato ordenado de Binance: {"u":...,"s":"...","b":"...","B":"...","a":"...","A":"..."}
    #[inline(always)]
    pub fn parse_from_json(bytes: &[u8]) -> Option<Self> {
        
        let (bid_price, i) = Self::extract_f64_from(bytes, 0, b"\"b\":\"")?;
        let (bid_qty, i) = Self::extract_f64_from(bytes, i, b"\"B\":\"")?;
        let (ask_price, i) = Self::extract_f64_from(bytes, i, b"\"a\":\"")?;
        let (ask_qty, _) = Self::extract_f64_from(bytes, i, b"\"A\":\"")?;
        
        // El symbol lo dejamos hardcodeado por ahora o no lo parseamos dinámicamente si no se usa
        // En Producción unificada, the symbol is known by the Streamer.
        
        Some(Self {
            update_id: 0, // Ignoramos update_id para ahorrar ciclos de CPU (no lo usamos)
            symbol_bytes: [0; 16], 
            symbol_len: 0,
            bid_price,
            bid_qty,
            ask_price,
            ask_qty,
            coin_id: 0,
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
        if memchr::memmem::find(bytes, b"\"depthUpdate\"").is_none() {
            return None;
        }
        
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(bytes) {
            let data = &json["data"];
            let mut bid_wall = 0.0;
            let mut ask_wall = 0.0;
            
            if let Some(bids) = data["b"].as_array() {
                for bid in bids {
                    if let Some(qty_str) = bid[1].as_str() {
                        bid_wall += qty_str.parse::<f64>().unwrap_or(0.0);
                    }
                }
            }
            if let Some(asks) = data["a"].as_array() {
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

