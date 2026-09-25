//! OPEN: these passes expose stub behaviour, not an execution simulator certificate.
use execution_engine::{executor::ExecutionProvider, ShadowExecutor};
use risk_engine::ValidatedOrder;
use signal_engine::SignalType;

fn order() -> ValidatedOrder {
    ValidatedOrder {
        signal: SignalType::Long,
        volume_usd: 5.0,
        leverage: 5.0,
        maker_only: false,
        tp_target: 51_000.0,
        sl_target: 49_000.0,
        fee_buffer_multiplier: 1.01,
    }
}

#[tokio::test]
async fn open_shadow_accepts_nan_and_infinite_prices() {
    let s = ShadowExecutor::new(100.0);
    for p in [f64::NAN, f64::INFINITY] {
        assert!(s
            .execute_order(&order(), "TESTUSDT", p, 0.001)
            .await
            .is_ok());
    }
}

#[tokio::test]
async fn open_shadow_accepts_quantity_rounded_to_zero() {
    let s = ShadowExecutor::new(100.0);
    // (5*5)/50000=0.0005, floored to the 0.001 grid => zero.
    assert!(s
        .execute_order(&order(), "TESTUSDT", 50_000.0, 0.001)
        .await
        .is_ok());
}

#[tokio::test]
async fn open_shadow_kill_switch_does_not_block_new_entry() {
    let s = ShadowExecutor::new(100.0);
    s.trigger_kill_switch();
    assert!(s.execute_raw_qty("TESTUSDT", true, 1.0, 0.1).await.is_ok());
}

#[tokio::test]
async fn open_shadow_ack_has_no_fill_position_or_capital_effect() {
    let s = ShadowExecutor::new(100.0);
    s.execute_raw_qty("TESTUSDT", true, 1.0, 0.1).await.unwrap();
    assert!(s.fetch_open_positions().await.unwrap().is_empty());
    assert_eq!(s.fetch_account_balance().await.unwrap(), 100.0);
    assert!(s.query_order("TESTUSDT", "any").await.is_err());
}

#[tokio::test]
async fn open_shadow_limit_path_accepts_invalid_price_and_quantity() {
    let s = ShadowExecutor::new(100.0);
    assert!(s
        .execute_limit_order("TESTUSDT", true, -1.0, f64::NAN, 0.1, 0.1, "x")
        .await
        .is_ok());
}
