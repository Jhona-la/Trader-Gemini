//! Pure payload construction; fake credentials, no sockets or exchange requests.
use execution_engine::executor::{ExecutionProvider, OrderExecutor};
use risk_engine::ValidatedOrder;
use signal_engine::SignalType;

fn order() -> ValidatedOrder {
    ValidatedOrder {
        signal: SignalType::Long,
        volume_usd: 10.0,
        leverage: 4.0,
        maker_only: false,
        tp_target: 102.0,
        sl_target: 99.0,
        fee_buffer_multiplier: 1.1,
        tau_ms: 60_000.0,
    }
}

#[test]
fn valid_integer_order_builds_the_budgeted_quantity() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    let p = executor
        .build_payload(&order(), "BTCUSDT", 100.0, 0.001, 0.01)
        .unwrap();
    assert_eq!(p.quantity, 0.4);
    assert_eq!(p.side, "BUY");
    assert!(p.signed_query.contains("quantity=0.4&"));
}

#[test]
fn fractional_or_invalid_leverage_cannot_reach_a_signed_payload() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    for bad in [4.121212121212122, 0.0, -4.0, f64::NAN, f64::INFINITY, 126.0] {
        let mut o = order();
        o.leverage = bad;
        assert!(
            executor
                .build_payload(&o, "BTCUSDT", 100.0, 0.001, 0.01)
                .is_none(),
            "leverage={bad}"
        );
    }
}

#[test]
fn invalid_tick_price_and_margin_never_produce_a_payload() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    let mut maker_order = order();
    maker_order.maker_only = true;
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        assert!(
            executor
                .build_payload(&maker_order, "BTCUSDT", 100.0, 0.001, bad)
                .is_none(),
            "tick={bad}"
        );
        assert!(
            executor
                .build_payload(&order(), "BTCUSDT", bad, 0.001, 0.01)
                .is_none(),
            "price={bad}"
        );
        let mut o = order();
        o.volume_usd = bad;
        assert!(
            executor
                .build_payload(&o, "BTCUSDT", 100.0, 0.001, 0.01)
                .is_none(),
            "margin={bad}"
        );
    }
}

#[test]
fn market_payload_does_not_require_an_unused_limit_price_tick() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    for unused_tick in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let p = executor
            .build_payload(&order(), "BTCUSDT", 100.0, 0.001, unused_tick)
            .unwrap();
        assert_eq!(p.quantity, 0.4);
        assert_eq!(p.price, None);
        assert_eq!(p.order_type, "MARKET");
    }
}

#[test]
fn unrepresentable_lot_arithmetic_cannot_produce_nan_quantity() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    assert!(executor
        .build_payload(&order(), "BTCUSDT", 100.0, f64::from_bits(1), 0.01)
        .is_none());
}

#[test]
fn minimum_tick_fallback_must_not_move_a_maker_buy_above_reference() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    let mut o = order();
    o.maker_only = true;
    assert!(executor
        .build_payload(&o, "BTCUSDT", 0.005, 0.001, 0.01)
        .is_none());
}

#[test]
fn valid_maker_orders_keep_their_side_of_the_reference() {
    let executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    for signal in [SignalType::Long, SignalType::Short] {
        let mut o = order();
        o.signal = signal;
        o.maker_only = true;
        let p = executor
            .build_payload(&o, "BTCUSDT", 100.0, 0.001, 0.01)
            .unwrap();
        let price = p.price.unwrap();
        assert!(price > 0.0);
        assert!(if signal == SignalType::Long {
            price <= 100.0
        } else {
            price >= 100.0
        });
        assert_eq!(p.time_in_force, "GTX");
    }
}

#[test]
fn integer_leverage_conversion_does_not_truncate_or_saturate() {
    let mut o = order();
    for valid in [1.0, 125.0, u32::MAX as f64] {
        o.leverage = valid;
        assert_eq!(o.integer_leverage(), Some(valid as u32));
    }
    for invalid in [
        0.0,
        -1.0,
        1.5,
        f64::NAN,
        f64::INFINITY,
        u32::MAX as f64 + 1.0,
    ] {
        o.leverage = invalid;
        assert_eq!(o.integer_leverage(), None);
    }
}

#[tokio::test]
async fn paper_success_does_not_bypass_the_basic_admission_contract() {
    let mut executor = OrderExecutor::new(String::new(), "offline-test".into(), true);
    executor.set_paper_trading(true);
    assert!(executor
        .execute_order(&order(), "BTCUSDT", 100.0, 0.001)
        .await
        .is_ok());
    for leverage in [1.5, f64::NAN, 0.0, 126.0] {
        let mut o = order();
        o.leverage = leverage;
        assert!(executor
            .execute_order(&o, "BTCUSDT", 100.0, 0.001)
            .await
            .is_err());
    }
    let mut o = order();
    o.volume_usd = f64::MAX;
    assert!(executor
        .execute_order(&o, "BTCUSDT", 100.0, 0.001)
        .await
        .is_err());
    o = order();
    o.signal = SignalType::Flat;
    assert!(executor
        .execute_order(&o, "BTCUSDT", 100.0, 0.001)
        .await
        .is_err());
    for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(executor
            .execute_order(&order(), "BTCUSDT", invalid, 0.001)
            .await
            .is_err());
        assert!(executor
            .execute_order(&order(), "BTCUSDT", 100.0, invalid)
            .await
            .is_err());
    }
}
