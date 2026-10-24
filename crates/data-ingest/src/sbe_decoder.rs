//! # Binance Simple Binary Encoding (SBE) Zero-Copy Ultra-Fast Decoder
//!
//! ## Paradigma de Silicio y Ultra-Baja Latencia
//! El protocolo binario Simple Binary Encoding (SBE) de Binance procesa mensajes directamente
//! como estructuras nativas de bytes en memoria (little-endian), eliminando el costo de parsing
//! JSON y allocations en heap:
//! - JSON Parsing: ~500 - 2,500 ns por mensaje, allocations dinámicas de Strings y Values.
//! - SBE Parsing: ~5 - 15 ns por mensaje, 0 allocations, acceso directo O(1) a campos numéricos.
//!
//! ## Formato de Cabecera SBE (8 bytes)
//! ```text
//! +------------------+------------------+------------------+------------------+
//! | blockLength: u16 |  templateId: u16 |   schemaId: u16  |   version: u16   |
//! +------------------+------------------+------------------+------------------+
//! ```
//!
//! Todos los métodos operan sobre rebanadas de bytes `&[u8]` sin `unwrap()` y con validación
//! estricta de límites contra desbordamientos de búfer.

/// Longitud fija de la cabecera del mensaje SBE en bytes.
pub const SBE_HEADER_LEN: usize = 8;

/// Template IDs canónicos de Binance SBE.
pub const SBE_TEMPLATE_BOOK_TICKER: u16 = 101;
pub const SBE_TEMPLATE_TRADE: u16 = 102;
pub const SBE_TEMPLATE_DEPTH5: u16 = 103;

/// Errores tipados de decodificación SBE a prueba de pánico.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbeDecodeError {
    /// El búfer proporcionado es menor que el tamaño mínimo requerido.
    BufferTooShort { expected: usize, actual: usize },
    /// El templateId no coincide con la estructura solicitada.
    TemplateMismatch { expected: u16, actual: u16 },
    /// El esquema SBE es incompatible.
    UnsupportedSchema { schema_id: u16, version: u16 },
    /// Carga útil corrupta o valor numérico inválido (NaN / Infinito).
    CorruptedPayload,
}

/// Cabecera de 8 bytes de un mensaje SBE de Binance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SbeHeader {
    /// Longitud del bloque raíz sin incluir grupos repetitivos.
    pub block_length: u16,
    /// Identificador del template de mensaje (101: bookTicker, 102: trade, 103: depth5).
    pub template_id: u16,
    /// Identificador de esquema SBE.
    pub schema_id: u16,
    /// Versión del esquema SBE.
    pub version: u16,
}

impl SbeHeader {
    /// Decodifica la cabecera SBE de forma zero-alloc.
    #[inline]
    pub fn decode(buf: &[u8]) -> Result<Self, SbeDecodeError> {
        if buf.len() < SBE_HEADER_LEN {
            return Err(SbeDecodeError::BufferTooShort {
                expected: SBE_HEADER_LEN,
                actual: buf.len(),
            });
        }

        let block_length = u16::from_le_bytes([buf[0], buf[1]]);
        let template_id = u16::from_le_bytes([buf[2], buf[3]]);
        let schema_id = u16::from_le_bytes([buf[4], buf[5]]);
        let version = u16::from_le_bytes([buf[6], buf[7]]);

        Ok(Self {
            block_length,
            template_id,
            schema_id,
            version,
        })
    }

    /// Codifica la cabecera SBE en un búfer mutable de al menos 8 bytes.
    #[inline]
    pub fn encode(&self, out: &mut [u8]) -> Result<(), SbeDecodeError> {
        if out.len() < SBE_HEADER_LEN {
            return Err(SbeDecodeError::BufferTooShort {
                expected: SBE_HEADER_LEN,
                actual: out.len(),
            });
        }
        out[0..2].copy_from_slice(&self.block_length.to_le_bytes());
        out[2..4].copy_from_slice(&self.template_id.to_le_bytes());
        out[4..6].copy_from_slice(&self.schema_id.to_le_bytes());
        out[6..8].copy_from_slice(&self.version.to_le_bytes());
        Ok(())
    }
}

/// Mensaje binario decodificado de mejor postura de libro (`@bookTicker`, Template 101).
/// Tamaño del bloque raíz: 44 bytes (Payload total con header: 52 bytes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbeBookTicker {
    /// Timestamp del evento en microsegundos Epoch UNIX.
    pub event_time_us: u64,
    /// ID numérico mapeado del símbolo (ej. 1 para BTCUSDT, 2 para ETHUSDT).
    pub symbol_id: u32,
    /// Mejor precio de compra (Best Bid).
    pub best_bid_price: f64,
    /// Mejor cantidad de compra.
    pub best_bid_qty: f64,
    /// Mejor precio de venta (Best Ask).
    pub best_ask_price: f64,
    /// Mejor cantidad de venta.
    pub best_ask_qty: f64,
}

impl SbeBookTicker {
    pub const PAYLOAD_LEN: usize = 44;
    pub const TOTAL_LEN: usize = SBE_HEADER_LEN + Self::PAYLOAD_LEN;

    /// Decodifica directamente desde los bytes de red.
    pub fn decode(buf: &[u8]) -> Result<Self, SbeDecodeError> {
        if buf.len() < Self::TOTAL_LEN {
            return Err(SbeDecodeError::BufferTooShort {
                expected: Self::TOTAL_LEN,
                actual: buf.len(),
            });
        }

        let header = SbeHeader::decode(buf)?;
        if header.template_id != SBE_TEMPLATE_BOOK_TICKER {
            return Err(SbeDecodeError::TemplateMismatch {
                expected: SBE_TEMPLATE_BOOK_TICKER,
                actual: header.template_id,
            });
        }

        let payload = &buf[SBE_HEADER_LEN..Self::TOTAL_LEN];
        let event_time_us = u64::from_le_bytes([
            payload[0], payload[1], payload[2], payload[3],
            payload[4], payload[5], payload[6], payload[7],
        ]);
        let symbol_id = u32::from_le_bytes([
            payload[8], payload[9], payload[10], payload[11],
        ]);

        let best_bid_price = f64::from_bits(u64::from_le_bytes([
            payload[12], payload[13], payload[14], payload[15],
            payload[16], payload[17], payload[18], payload[19],
        ]));
        let best_bid_qty = f64::from_bits(u64::from_le_bytes([
            payload[20], payload[21], payload[22], payload[23],
            payload[24], payload[25], payload[26], payload[27],
        ]));
        let best_ask_price = f64::from_bits(u64::from_le_bytes([
            payload[28], payload[29], payload[30], payload[31],
            payload[32], payload[33], payload[34], payload[35],
        ]));
        let best_ask_qty = f64::from_bits(u64::from_le_bytes([
            payload[36], payload[37], payload[38], payload[39],
            payload[40], payload[41], payload[42], payload[43],
        ]));

        if !best_bid_price.is_finite() || !best_bid_qty.is_finite()
            || !best_ask_price.is_finite() || !best_ask_qty.is_finite() {
            return Err(SbeDecodeError::CorruptedPayload);
        }

        Ok(Self {
            event_time_us,
            symbol_id,
            best_bid_price,
            best_bid_qty,
            best_ask_price,
            best_ask_qty,
        })
    }

    /// Codifica en un búfer de al menos 48 bytes.
    pub fn encode(&self, out: &mut [u8]) -> Result<(), SbeDecodeError> {
        if out.len() < Self::TOTAL_LEN {
            return Err(SbeDecodeError::BufferTooShort {
                expected: Self::TOTAL_LEN,
                actual: out.len(),
            });
        }

        let header = SbeHeader {
            block_length: Self::PAYLOAD_LEN as u16,
            template_id: SBE_TEMPLATE_BOOK_TICKER,
            schema_id: 1,
            version: 1,
        };
        header.encode(out)?;

        let payload = &mut out[SBE_HEADER_LEN..Self::TOTAL_LEN];
        payload[0..8].copy_from_slice(&self.event_time_us.to_le_bytes());
        payload[8..12].copy_from_slice(&self.symbol_id.to_le_bytes());
        payload[12..20].copy_from_slice(&self.best_bid_price.to_bits().to_le_bytes());
        payload[20..28].copy_from_slice(&self.best_bid_qty.to_bits().to_le_bytes());
        payload[28..36].copy_from_slice(&self.best_ask_price.to_bits().to_le_bytes());
        payload[36..44].copy_from_slice(&self.best_ask_qty.to_bits().to_le_bytes());

        Ok(())
    }
}

/// Mensaje binario decodificado de ejecución de operación individual (`@trade`, Template 102).
/// Tamaño del bloque raíz: 37 bytes (Payload total con header: 45 bytes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbeTrade {
    /// Timestamp del evento en microsegundos Epoch UNIX.
    pub event_time_us: u64,
    /// ID numérico del símbolo.
    pub symbol_id: u32,
    /// Identificador único de trade en Binance.
    pub trade_id: u64,
    /// Precio de ejecución.
    pub price: f64,
    /// Cantidad operada.
    pub qty: f64,
    /// Si el comprador fue Maker (1: true, 0: false).
    pub is_buyer_maker: bool,
}

impl SbeTrade {
    pub const PAYLOAD_LEN: usize = 37;
    pub const TOTAL_LEN: usize = SBE_HEADER_LEN + Self::PAYLOAD_LEN;

    /// Decodifica directamente desde la secuencia de bytes.
    pub fn decode(buf: &[u8]) -> Result<Self, SbeDecodeError> {
        if buf.len() < Self::TOTAL_LEN {
            return Err(SbeDecodeError::BufferTooShort {
                expected: Self::TOTAL_LEN,
                actual: buf.len(),
            });
        }

        let header = SbeHeader::decode(buf)?;
        if header.template_id != SBE_TEMPLATE_TRADE {
            return Err(SbeDecodeError::TemplateMismatch {
                expected: SBE_TEMPLATE_TRADE,
                actual: header.template_id,
            });
        }

        let payload = &buf[SBE_HEADER_LEN..Self::TOTAL_LEN];
        let event_time_us = u64::from_le_bytes([
            payload[0], payload[1], payload[2], payload[3],
            payload[4], payload[5], payload[6], payload[7],
        ]);
        let symbol_id = u32::from_le_bytes([
            payload[8], payload[9], payload[10], payload[11],
        ]);
        let trade_id = u64::from_le_bytes([
            payload[12], payload[13], payload[14], payload[15],
            payload[16], payload[17], payload[18], payload[19],
        ]);
        let price = f64::from_bits(u64::from_le_bytes([
            payload[20], payload[21], payload[22], payload[23],
            payload[24], payload[25], payload[26], payload[27],
        ]));
        let qty = f64::from_bits(u64::from_le_bytes([
            payload[28], payload[29], payload[30], payload[31],
            payload[32], payload[33], payload[34], payload[35],
        ]));
        let is_buyer_maker = payload[36] != 0;

        if !price.is_finite() || !qty.is_finite() {
            return Err(SbeDecodeError::CorruptedPayload);
        }

        Ok(Self {
            event_time_us,
            symbol_id,
            trade_id,
            price,
            qty,
            is_buyer_maker,
        })
    }

    /// Codifica en un búfer de al menos 44 bytes.
    pub fn encode(&self, out: &mut [u8]) -> Result<(), SbeDecodeError> {
        if out.len() < Self::TOTAL_LEN {
            return Err(SbeDecodeError::BufferTooShort {
                expected: Self::TOTAL_LEN,
                actual: out.len(),
            });
        }

        let header = SbeHeader {
            block_length: Self::PAYLOAD_LEN as u16,
            template_id: SBE_TEMPLATE_TRADE,
            schema_id: 1,
            version: 1,
        };
        header.encode(out)?;

        let payload = &mut out[SBE_HEADER_LEN..Self::TOTAL_LEN];
        payload[0..8].copy_from_slice(&self.event_time_us.to_le_bytes());
        payload[8..12].copy_from_slice(&self.symbol_id.to_le_bytes());
        payload[12..20].copy_from_slice(&self.trade_id.to_le_bytes());
        payload[20..28].copy_from_slice(&self.price.to_bits().to_le_bytes());
        payload[28..36].copy_from_slice(&self.qty.to_bits().to_le_bytes());
        payload[36] = if self.is_buyer_maker { 1 } else { 0 };

        Ok(())
    }
}

/// Nivel individual de profundidad de libro.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SbePriceLevel {
    pub price: f64,
    pub qty: f64,
}

/// Mensaje binario decodificado de los mejores 5 niveles de profundidad (`@depth5`, Template 103).
/// Tamaño del bloque raíz: 20 bytes + 5 bids (80B) + 5 asks (80B) = 180 bytes. Total con header: 188 bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbeDepth5Snapshot {
    pub event_time_us: u64,
    pub symbol_id: u32,
    pub first_update_id: u64,
    pub bids: [SbePriceLevel; 5],
    pub asks: [SbePriceLevel; 5],
}

impl SbeDepth5Snapshot {
    pub const PAYLOAD_LEN: usize = 180;
    pub const TOTAL_LEN: usize = SBE_HEADER_LEN + Self::PAYLOAD_LEN;

    /// Decodifica de forma zero-alloc desde el búfer de red.
    pub fn decode(buf: &[u8]) -> Result<Self, SbeDecodeError> {
        if buf.len() < Self::TOTAL_LEN {
            return Err(SbeDecodeError::BufferTooShort {
                expected: Self::TOTAL_LEN,
                actual: buf.len(),
            });
        }

        let header = SbeHeader::decode(buf)?;
        if header.template_id != SBE_TEMPLATE_DEPTH5 {
            return Err(SbeDecodeError::TemplateMismatch {
                expected: SBE_TEMPLATE_DEPTH5,
                actual: header.template_id,
            });
        }

        let payload = &buf[SBE_HEADER_LEN..Self::TOTAL_LEN];
        let event_time_us = u64::from_le_bytes([
            payload[0], payload[1], payload[2], payload[3],
            payload[4], payload[5], payload[6], payload[7],
        ]);
        let symbol_id = u32::from_le_bytes([
            payload[8], payload[9], payload[10], payload[11],
        ]);
        let first_update_id = u64::from_le_bytes([
            payload[12], payload[13], payload[14], payload[15],
            payload[16], payload[17], payload[18], payload[19],
        ]);

        let mut bids = [SbePriceLevel::default(); 5];
        let mut asks = [SbePriceLevel::default(); 5];

        let mut offset = 20;
        for i in 0..5 {
            let p = f64::from_bits(u64::from_le_bytes([
                payload[offset], payload[offset + 1], payload[offset + 2], payload[offset + 3],
                payload[offset + 4], payload[offset + 5], payload[offset + 6], payload[offset + 7],
            ]));
            let q = f64::from_bits(u64::from_le_bytes([
                payload[offset + 8], payload[offset + 9], payload[offset + 10], payload[offset + 11],
                payload[offset + 12], payload[offset + 13], payload[offset + 14], payload[offset + 15],
            ]));
            if !p.is_finite() || !q.is_finite() {
                return Err(SbeDecodeError::CorruptedPayload);
            }
            bids[i] = SbePriceLevel { price: p, qty: q };
            offset += 16;
        }

        for i in 0..5 {
            let p = f64::from_bits(u64::from_le_bytes([
                payload[offset], payload[offset + 1], payload[offset + 2], payload[offset + 3],
                payload[offset + 4], payload[offset + 5], payload[offset + 6], payload[offset + 7],
            ]));
            let q = f64::from_bits(u64::from_le_bytes([
                payload[offset + 8], payload[offset + 9], payload[offset + 10], payload[offset + 11],
                payload[offset + 12], payload[offset + 13], payload[offset + 14], payload[offset + 15],
            ]));
            if !p.is_finite() || !q.is_finite() {
                return Err(SbeDecodeError::CorruptedPayload);
            }
            asks[i] = SbePriceLevel { price: p, qty: q };
            offset += 16;
        }

        Ok(Self {
            event_time_us,
            symbol_id,
            first_update_id,
            bids,
            asks,
        })
    }

    /// Codifica en un búfer de al menos 184 bytes.
    pub fn encode(&self, out: &mut [u8]) -> Result<(), SbeDecodeError> {
        if out.len() < Self::TOTAL_LEN {
            return Err(SbeDecodeError::BufferTooShort {
                expected: Self::TOTAL_LEN,
                actual: out.len(),
            });
        }

        let header = SbeHeader {
            block_length: Self::PAYLOAD_LEN as u16,
            template_id: SBE_TEMPLATE_DEPTH5,
            schema_id: 1,
            version: 1,
        };
        header.encode(out)?;

        let payload = &mut out[SBE_HEADER_LEN..Self::TOTAL_LEN];
        payload[0..8].copy_from_slice(&self.event_time_us.to_le_bytes());
        payload[8..12].copy_from_slice(&self.symbol_id.to_le_bytes());
        payload[12..20].copy_from_slice(&self.first_update_id.to_le_bytes());

        let mut offset = 20;
        for i in 0..5 {
            payload[offset..offset + 8].copy_from_slice(&self.bids[i].price.to_bits().to_le_bytes());
            payload[offset + 8..offset + 16].copy_from_slice(&self.bids[i].qty.to_bits().to_le_bytes());
            offset += 16;
        }

        for i in 0..5 {
            payload[offset..offset + 8].copy_from_slice(&self.asks[i].price.to_bits().to_le_bytes());
            payload[offset + 8..offset + 16].copy_from_slice(&self.asks[i].qty.to_bits().to_le_bytes());
            offset += 16;
        }

        Ok(())
    }
}
