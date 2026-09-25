//! OPEN integration debt: a passing diagnostic reproduces a defect, not a repair.
//! Pure payload construction with fake credentials; no exchange IO.
use execution_engine::executor::OrderExecutor;
use risk_engine::ValidatedOrder;
use signal_engine::SignalType;

fn order() -> ValidatedOrder {
    ValidatedOrder {
        signal: SignalType::Long,
        volume_usd: 10.0,
        leverage: 4.0,
        maker_only: false,
        tp_target: 120.0,
        sl_target: 80.0,
        fee_buffer_multiplier: 1.1,
    }
}

#[test]
fn open_lot_rounding_can_drop_below_the_risk_minimum_notional() {
    let ex = OrderExecutor::new(String::new(), "offline-test".into(), true);
    let mut o = order();
    o.volume_usd = 1.2756375;
    assert!(o.volume_usd * o.leverage >= 5.1);
    let payload = ex
        .build_payload(&o, "AUDITUSDT", 100.0, 0.03, 0.01)
        .unwrap();
    assert!((payload.quantity - 0.03).abs() < 1e-15);
    assert!(payload.quantity * 100.0 < 5.0);
    println!(
        "OPEN FMT-175: risk_notional={} payload_notional={}",
        o.volume_usd * o.leverage,
        payload.quantity * 100.0
    );
}

#[test]
fn open_snap_floor_can_increase_the_requested_quantity() {
    let ex = OrderExecutor::new(String::new(), "offline-test".into(), true);
    let mut o = order();
    o.volume_usd = 9.9999999975;
    let requested = o.volume_usd * o.leverage / 100.0;
    let payload = ex.build_payload(&o, "AUDITUSDT", 100.0, 0.1, 0.01).unwrap();
    assert_eq!(payload.quantity, 0.4);
    assert!(payload.quantity > requested);
    println!(
        "OPEN FMT-175: requested={requested} signed_quantity={}",
        payload.quantity
    );
}

#[test]
fn open_maker_price_revalues_notional_without_rebudgeting_quantity() {
    let ex = OrderExecutor::new(String::new(), "offline-test".into(), true);
    let mut o = order();
    o.signal = SignalType::Short;
    o.maker_only = true;
    let payload = ex
        .build_payload(&o, "AUDITUSDT", 100.0, 0.001, 10.0)
        .unwrap();
    assert_eq!(payload.price, Some(110.0));
    assert_eq!(payload.quantity, 0.4);
    let final_notional = payload.quantity * payload.price.unwrap();
    assert_eq!(final_notional, 44.0);
    assert!(final_notional > o.volume_usd * o.leverage);
    println!(
        "OPEN FMT-219: reference_notional={} limit_notional={final_notional}",
        o.volume_usd * o.leverage
    );
}
