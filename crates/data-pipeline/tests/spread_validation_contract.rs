use data_pipeline::{
    parser::BookTickerEvent,
    validation::{validate_book_ticker, RejectReason},
};

fn event(bid: f64, ask: f64) -> BookTickerEvent {
    BookTickerEvent {
        coin_id: 0,
        event_time: 1000,
        bid_price: bid,
        ask_price: ask,
        bid_qty: 1.0,
        ask_qty: 1.0,
    }
}

#[test]
fn finite_extreme_prices_do_not_bypass_spread_veto_by_midpoint_overflow() {
    assert_eq!(
        validate_book_ticker(&event(6e307, 1.6e308)),
        Err(RejectReason::AbsurdSpread)
    );
}

#[test]
fn spread_classification_is_scale_invariant_for_positive_finite_books() {
    for scale in [1e-300, 1e-100, 1.0, 1e100, 1e307] {
        assert_eq!(
            validate_book_ticker(&event(6.0 * scale, 16.0 * scale)),
            Err(RejectReason::AbsurdSpread)
        );
        assert_eq!(
            validate_book_ticker(&event(9.0 * scale, 10.0 * scale)),
            Ok(())
        );
    }
}

#[test]
fn exact_legacy_policy_boundary_is_preserved_and_zero_spread_is_valid() {
    assert_eq!(validate_book_ticker(&event(3.0, 5.0)), Ok(()));
    assert_eq!(validate_book_ticker(&event(f64::MAX, f64::MAX)), Ok(()));
    assert_eq!(
        validate_book_ticker(&event(f64::from_bits(1), f64::from_bits(1))),
        Ok(())
    );
}
