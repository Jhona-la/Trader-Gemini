//! CONTRATO FORMAL — DEFECTO C-02 (Ola Ω57): Feed Spot-Futuro en Vivo y Activación de StatArb.
//!
//! Verifica que:
//! 1. Los parsers REST y WebSocket de Binance Spot operan sin fallos, filtran números no finitos
//!    y preservan precisión.
//! 2. `apply_spot_tick` actualiza de forma atómica y lock-free `arena.coins[coin_id].spot_bid`,
//!    `spot_ask`, el registro omnisciente per-símbolo y el tensor macro `omni_state.binance_spot`.
//! 3. El spread de arbitraje spot-futuros en `omni_state.spot_futures_arb_spread` se calcula en tiempo real.
//! 4. La alimentación continua da a `statarb_ou_engines` en el núcleo los pares de su prueba
//!    de reversión de la base (CL-97; el contrato está en
//!    `god-engine-core/tests/statarb_live_physics_contract.rs`).

use data_pipeline::omni_multiplexer::OmniState;
use data_pipeline::spot_feed::{
    aplicar_tick_de_spot, apply_spot_tick, par_spot, parse_rest_book_tickers, parse_ws_book_ticker,
};
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::atomic::Ordering;

#[test]
fn c02_test_parser_rest_book_ticker_resilience() {
    let raw_valid = r#"[
        {"symbol":"BTCUSDT","bidPrice":"82100.50","bidQty":"5.123","askPrice":"82100.60","askQty":"3.456"},
        {"symbol":"ETHUSDT","bidPrice":"2450.10","bidQty":"25.0","askPrice":"2450.20","askQty":"40.0"}
    ]"#;
    let parsed = parse_rest_book_tickers(raw_valid);
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].0, "BTCUSDT");
    assert!((parsed[0].1 - 82100.50).abs() < 1e-4);
    assert!((parsed[0].2 - 82100.60).abs() < 1e-4);
    assert!((parsed[0].3 - 5.123).abs() < 1e-4);
    assert!((parsed[0].4 - 3.456).abs() < 1e-4);

    // Payload malformado o con valores inválidos
    let raw_invalid = r#"[
        {"symbol":"BTCUSDT","bidPrice":"NaN","bidQty":"1.0","askPrice":"82100.0","askQty":"1.0"},
        {"symbol":"ETHUSDT","bidPrice":"-10.0","bidQty":"1.0","askPrice":"2450.0","askQty":"1.0"}
    ]"#;
    let parsed_inv = parse_rest_book_tickers(raw_invalid);
    assert_eq!(parsed_inv.len(), 0, "Precios NaN o negativos deben descartarse");
}

#[test]
fn c02_test_parser_ws_book_ticker_combined_stream() {
    let ws_msg = r#"{
        "stream": "solusdt@bookTicker",
        "data": {
            "u": 12345678,
            "s": "SOLUSDT",
            "b": "109.85",
            "B": "500.0",
            "a": "109.86",
            "A": "320.0"
        }
    }"#;
    let res = parse_ws_book_ticker(ws_msg);
    assert!(res.is_some());
    let (sym, bid, ask, bq, aq) = res.unwrap();
    assert_eq!(sym, "SOLUSDT");
    assert!((bid - 109.85).abs() < 1e-4);
    assert!((ask - 109.86).abs() < 1e-4);
    assert!((bq - 500.0).abs() < 1e-4);
    assert!((aq - 320.0).abs() < 1e-4);
}

#[test]
fn c02_test_apply_spot_tick_populates_arena_registry_and_macro_tensor() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "BTCUSDT",
    )]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let omni = OmniState::new();

    // Inicializar futuros en OmniState para probar el cálculo de spread basis
    let fut_price: f64 = 50_100.0;
    omni.binance_futures
        .store(fut_price.to_bits(), Ordering::Relaxed);

    apply_spot_tick(
        &arena,
        &omni,
        0,
        "BTCUSDT",
        50_000.0,
        50_002.0,
        10.5,
        12.3,
    );

    // 1. Verificación en CoinArena
    assert_eq!(arena.coins[0].spot_bid.load(Ordering::Relaxed), 50_000.0);
    assert_eq!(arena.coins[0].spot_ask.load(Ordering::Relaxed), 50_002.0);
    assert_eq!(arena.coins[0].spot_bid_qty.load(Ordering::Relaxed), 10.5);
    assert_eq!(arena.coins[0].spot_ask_qty.load(Ordering::Relaxed), 12.3);

    // 2. Verificación en Registry
    let spot_mid = arena
        .registry
        .get_scoped_parameter(Some("BTCUSDT"), Some(0), "spot_mid", "test")
        .expect("spot_mid debe existir en el registro")
        .get_value();
    assert!((spot_mid - 50_001.0).abs() < 1e-6);

    // 3. Verificación en OmniState
    let macro_spot = f64::from_bits(omni.binance_spot.load(Ordering::Relaxed));
    assert!((macro_spot - 50_001.0).abs() < 1e-6);

    let arb_spread = f64::from_bits(omni.spot_futures_arb_spread.load(Ordering::Relaxed));
    let expected_spread = (fut_price - 50_001.0) / 50_001.0;
    assert!((arb_spread - expected_spread).abs() < 1e-6);
}

/// CL-123: un contrato `1000X` (1000SHIBUSDT) se consulta en spot como `X`
/// (SHIBUSDT) y su tick se aplica a la moneda del contrato en su escala:
/// precios × 1 000 y cantidades ÷ 1 000. Antes el feed pedía el símbolo de
/// futuros también en spot, y uno que no existe hace fallar la consulta
/// agrupada entera.
#[test]
fn cl123_un_contrato_de_mil_unidades_se_lee_en_spot_y_se_escala() {
    assert_eq!(par_spot("1000SHIBUSDT"), ("SHIBUSDT".to_string(), 1000.0));
    assert_eq!(par_spot("1000pepeusdt"), ("PEPEUSDT".to_string(), 1000.0));
    assert_eq!(par_spot("BTCUSDT"), ("BTCUSDT".to_string(), 1.0));
    symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "1000SHIBUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("BTCUSDT"),
        symbol_registry::get_official_binance_spec("1000SHIBUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let omni = OmniState::new();
    let (_, escala) = par_spot("1000SHIBUSDT");
    let destino = (1usize, escala, "1000SHIBUSDT".to_string());
    aplicar_tick_de_spot(&arena, &omni, &destino, 0.000012, 0.000013, 4.0e9, 5.0e9);
    let bid = arena.coins[1].spot_bid.load(Ordering::Relaxed);
    let qty = arena.coins[1].spot_bid_qty.load(Ordering::Relaxed);
    assert!((bid - 0.012).abs() < 1e-12, "{bid}");
    assert!((qty - 4.0e6).abs() < 1e-3, "{qty}");
    let mid = arena
        .registry
        .get_scoped_parameter(Some("1000SHIBUSDT"), Some(1), "spot_mid", "test")
        .expect("spot_mid con el símbolo de futuros")
        .get_value();
    assert!((mid - 0.0125).abs() < 1e-12, "{mid}");
}
