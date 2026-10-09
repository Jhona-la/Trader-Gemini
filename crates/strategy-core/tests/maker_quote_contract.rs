use strategy_core::maker::MakerEngine;

#[test]
fn valid_tiny_price_book_does_not_cross_due_to_absolute_floor() {
    let mut engine = MakerEngine::new(5.0);
    let quote = engine.generate_quote(1e-12, 2e-12, 1.0, 1.0, 0.0, 0.0, 0.001, 0.5, 1.0, 1.0);
    assert!(quote.bid_price > 0.0 && quote.bid_price <= 1e-12);
    assert!(quote.ask_price >= 2e-12 && quote.ask_price.is_finite());
}

// Characterizations of OPEN debt: success here documents a defect, not a fix.
#[test]
fn open_debt_invalid_book_fallback_can_be_crossed() {
    let mut engine = MakerEngine::new(5.0);
    let quote = engine.generate_quote(101.0, 100.0, 1.0, 1.0, 0.0, 0.0, 0.001, 0.5, 1.0, 1.0);
    assert!(quote.bid_price > quote.ask_price);
}

#[test]
fn open_debt_constructor_spread_has_no_effect() {
    let mut narrow = MakerEngine::new(1.0);
    let mut wide = MakerEngine::new(100.0);
    let a = narrow.generate_quote(100.0, 101.0, 1.0, 1.0, 0.0, 0.0, 0.01, 0.5, 1.0, 1.0);
    let b = wide.generate_quote(100.0, 101.0, 1.0, 1.0, 0.0, 0.0, 0.01, 0.5, 1.0, 1.0);
    assert_eq!((a.bid_price, a.ask_price), (b.bid_price, b.ask_price));
}

#[test]
fn maker_obi_skew_es_continuo_c1_smoothstep_sin_quiebre_de_pendiente() {
    let mut engine = MakerEngine::new(5.0);
    // Evaluamos la respuesta de cotización alrededor del umbral th = 0.50
    // OBI = (bid_qty - ask_qty)/(bid_qty + ask_qty). Con ask_qty = 100, variamos bid_qty.
    // Para OBI = 0.50, bid_qty = 300, ask_qty = 100 => (300-100)/400 = 0.50.
    // Con OBI < 0.50, el skew debe ser exactamente 0.
    let q_below = engine.generate_quote(100.0, 100.10, 290.0, 100.0, 0.0, 0.0, 0.001, 0.50, 1.0, 1.0);
    let q_at = engine.generate_quote(100.0, 100.10, 300.0, 100.0, 0.0, 0.0, 0.001, 0.50, 1.0, 1.0);
    let q_just_above = engine.generate_quote(100.0, 100.10, 301.0, 100.0, 0.0, 0.0, 0.001, 0.50, 1.0, 1.0);

    // En el umbral y por debajo, bid_price y ask_price no sufren salto discontinuo
    assert!((q_below.bid_price - q_at.bid_price).abs() < 1e-12);
    // Inmediatamente arriba del umbral, la primera derivada es cero por S'(0) = 0:
    // La tasa de cambio del precio con delta de 1.0 de cantidad es cuadráticamente pequeña (< 1e-4)
    let delta_price = (q_just_above.bid_price - q_at.bid_price).abs();
    assert!(
        delta_price < 1e-3,
        "La respuesta de cotización debe ser suave C^1 sin quiebre lineal brusco, got {delta_price}"
    );
}
