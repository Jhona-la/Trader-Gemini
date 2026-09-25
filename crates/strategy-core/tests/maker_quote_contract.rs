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
