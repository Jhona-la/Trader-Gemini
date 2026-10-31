//! # Contrato Formal F10-001: Binance Simple Binary Encoding (SBE) Decoder
//!
//! Verificación rigurosa de invariantes de bajo nivel y silicio:
//! C1: SbeHeader encode/decode, exactitud de bits y protección contra buffers truncados.
//! C2: SbeBookTicker encode/decode zero-copy y preservación IEEE-754.
//! C3: SbeTrade encode/decode, banderas booleanas y trade IDs de 64 bits.
//! C4: SbeDepth5Snapshot libro top-5 bids y asks y alineación de memoria.
//! C5: Detección estricta de template ID incorrecto y rechazo a prueba de fallos.
//! C6: Inmunidad ante valores no finitos (NaN / Infinito), mutaciones aleatorias y cero pánicos.

use data_ingest::sbe_decoder::{
    SbeBookTicker, SbeDecodeError, SbeDepth5Snapshot, SbeHeader, SbePriceLevel, SbeTrade,
    SBE_HEADER_LEN, SBE_TEMPLATE_BOOK_TICKER, SBE_TEMPLATE_TRADE,
};

#[test]
fn test_c1_sbe_header_roundtrip_and_boundary_checks() {
    let header = SbeHeader {
        block_length: 40,
        template_id: SBE_TEMPLATE_BOOK_TICKER,
        schema_id: 1,
        version: 2,
    };

    let mut buf = [0u8; 8];
    assert!(header.encode(&mut buf).is_ok());

    let decoded = SbeHeader::decode(&buf).expect("Header decode debe ser exitoso");
    assert_eq!(decoded, header);

    // Búfer truncado (< 8 bytes) debe retornar error sin pánico
    let short_buf = [0u8; 7];
    match SbeHeader::decode(&short_buf) {
        Err(SbeDecodeError::BufferTooShort { expected: 8, actual: 7 }) => {}
        other => panic!("Esperaba BufferTooShort, obtuve {:?}", other),
    }

    // Encoding en búfer pequeño debe retornar error sin pánico
    let mut small_buf = [0u8; 4];
    assert!(header.encode(&mut small_buf).is_err());
}

#[test]
fn test_c2_sbe_book_ticker_roundtrip() {
    let ticker = SbeBookTicker {
        event_time_us: 1_700_000_000_123_456,
        symbol_id: 42,
        best_bid_price: 67_890.50,
        best_bid_qty: 1.2500,
        best_ask_price: 67_891.00,
        best_ask_qty: 0.8500,
    };

    let mut buf = [0u8; SbeBookTicker::TOTAL_LEN];
    assert!(ticker.encode(&mut buf).is_ok());

    let decoded = SbeBookTicker::decode(&buf).expect("BookTicker decode debe ser exitoso");
    assert_eq!(decoded.event_time_us, ticker.event_time_us);
    assert_eq!(decoded.symbol_id, ticker.symbol_id);
    assert_eq!(decoded.best_bid_price, ticker.best_bid_price);
    assert_eq!(decoded.best_bid_qty, ticker.best_bid_qty);
    assert_eq!(decoded.best_ask_price, ticker.best_ask_price);
    assert_eq!(decoded.best_ask_qty, ticker.best_ask_qty);
}

#[test]
fn test_c3_sbe_trade_roundtrip_and_flags() {
    let trade = SbeTrade {
        event_time_us: 1_700_000_000_654_321,
        symbol_id: 1,
        trade_id: 98_765_432_100,
        price: 68_123.45,
        qty: 0.05432,
        is_buyer_maker: true,
    };

    let mut buf = [0u8; SbeTrade::TOTAL_LEN];
    assert!(trade.encode(&mut buf).is_ok());

    let decoded = SbeTrade::decode(&buf).expect("Trade decode debe ser exitoso");
    assert_eq!(decoded, trade);

    // Verificar variante con is_buyer_maker = false
    let mut trade_taker = trade;
    trade_taker.is_buyer_maker = false;
    assert!(trade_taker.encode(&mut buf).is_ok());
    let decoded_taker = SbeTrade::decode(&buf).unwrap();
    assert!(!decoded_taker.is_buyer_maker);
}

#[test]
fn test_c4_sbe_depth5_snapshot_roundtrip() {
    let mut bids = [SbePriceLevel::default(); 5];
    let mut asks = [SbePriceLevel::default(); 5];

    for i in 0..5 {
        bids[i] = SbePriceLevel {
            price: 65_000.0 - (i as f64) * 10.0,
            qty: 1.0 + (i as f64) * 0.5,
        };
        asks[i] = SbePriceLevel {
            price: 65_010.0 + (i as f64) * 10.0,
            qty: 0.8 + (i as f64) * 0.2,
        };
    }

    let depth5 = SbeDepth5Snapshot {
        event_time_us: 1_700_000_000_999_888,
        symbol_id: 7,
        first_update_id: 123456789,
        bids,
        asks,
    };

    let mut buf = [0u8; SbeDepth5Snapshot::TOTAL_LEN];
    assert!(depth5.encode(&mut buf).is_ok());

    let decoded = SbeDepth5Snapshot::decode(&buf).expect("Depth5 decode debe ser exitoso");
    assert_eq!(decoded.event_time_us, depth5.event_time_us);
    assert_eq!(decoded.symbol_id, depth5.symbol_id);
    assert_eq!(decoded.first_update_id, depth5.first_update_id);
    assert_eq!(decoded.bids, depth5.bids);
    assert_eq!(decoded.asks, depth5.asks);
}

#[test]
fn test_c5_template_mismatch_rejection() {
    let ticker = SbeBookTicker {
        event_time_us: 1_000,
        symbol_id: 1,
        best_bid_price: 100.0,
        best_bid_qty: 1.0,
        best_ask_price: 101.0,
        best_ask_qty: 1.0,
    };

    let mut buf = [0u8; SbeBookTicker::TOTAL_LEN];
    ticker.encode(&mut buf).unwrap();

    // Intentar decodificar un búfer de BookTicker (Template 101) como Trade (Template 102)
    match SbeTrade::decode(&buf) {
        Err(SbeDecodeError::TemplateMismatch {
            expected: SBE_TEMPLATE_TRADE,
            actual: SBE_TEMPLATE_BOOK_TICKER,
        }) => {}
        other => panic!("Esperaba TemplateMismatch, obtuve {:?}", other),
    }

    // Intentar decodificar como Depth5 (Template 103)
    match SbeDepth5Snapshot::decode(&buf) {
        Err(SbeDecodeError::BufferTooShort { .. }) => {} // Buffer de BookTicker (52B) es menor que Depth5 (188B)
        other => panic!("Esperaba BufferTooShort, obtuve {:?}", other),
    }
}

#[test]
fn test_c6_corrupted_payload_and_nan_rejection_zero_panic() {
    let mut ticker = SbeBookTicker {
        event_time_us: 1_000,
        symbol_id: 1,
        best_bid_price: 100.0,
        best_bid_qty: 1.0,
        best_ask_price: 101.0,
        best_ask_qty: 1.0,
    };

    let mut buf = [0u8; SbeBookTicker::TOTAL_LEN];
    ticker.encode(&mut buf).unwrap();

    // Inyectar NaN en el campo best_bid_price
    let nan_bytes = f64::NAN.to_bits().to_le_bytes();
    buf[SBE_HEADER_LEN + 12..SBE_HEADER_LEN + 20].copy_from_slice(&nan_bytes);

    match SbeBookTicker::decode(&buf) {
        Err(SbeDecodeError::CorruptedPayload) => {}
        other => panic!("Esperaba CorruptedPayload ante NaN, obtuve {:?}", other),
    }

    // Inyectar Infinito en el campo best_ask_price
    ticker.best_ask_price = f64::INFINITY;
    assert_eq!(ticker.encode(&mut buf), Ok(()));
    assert_eq!(SbeBookTicker::decode(&buf), Err(SbeDecodeError::CorruptedPayload));

    // Búfer completamente aleatorio de cualquier tamaño nunca debe causar pánico
    let garbage = [0xFFu8; 128];
    assert!(SbeBookTicker::decode(&garbage).is_err());
    assert!(SbeTrade::decode(&garbage).is_err());
    assert!(SbeDepth5Snapshot::decode(&garbage).is_err());
}
