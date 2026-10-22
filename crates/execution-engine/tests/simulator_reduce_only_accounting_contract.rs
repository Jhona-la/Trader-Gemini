//! Public simulated-execution API contracts; this adapter is not used by the
//! production host or GodEngineCore replay on the audited b9cd773 snapshot.
//!
//! Independent accounting oracle: Bybit's USDT P&L example closes 0.3 from
//! a 0.4 short opened at 6000, leaving 0.1; each closing fee uses filled qty.
//! https://www.bybit.com/en/help-center/article/Profit-Loss-calculations-USDT-Contract
//! Binance also rejects reduce-only orders in the position's own direction:
//! https://www.binance.com/en-TR/support/faq/detail/360039707291
use execution_engine::executor::{ActivePosition, ExecutionProvider};
use execution_engine::simulator::SimulatedExecutor;

const SYMBOL: &str = "AUDITUSDT";
const INITIAL_CAPITAL: f64 = 1000.0;

fn seeded_position(is_long: bool, entry_price: f64) -> SimulatedExecutor {
    let mut sim = SimulatedExecutor::new(INITIAL_CAPITAL);
    sim.average_latency_ms = 0;
    // Isolate accounting at the existing model's 1bp minimum. For these
    // small notionals, size impact alone remains below that minimum.
    sim.base_slippage_bps = 0.0;
    sim.open_positions.write().unwrap().insert(
        SYMBOL.into(),
        ActivePosition {
            symbol: SYMBOL.into(),
            qty: 0.4,
            entry_price,
            leverage: 5.0,
            update_time: 123,
            is_long,
        },
    );
    sim.last_mids.write().unwrap().insert(SYMBOL.into(), 5000.0);
    sim
}

fn assert_unchanged(sim: &SimulatedExecutor, is_long: bool, entry_price: f64) {
    assert_eq!(*sim.simulated_capital.read().unwrap(), INITIAL_CAPITAL);
    let positions = sim.open_positions.read().unwrap();
    assert_eq!(positions.len(), 1);
    let pos = positions.get(SYMBOL).expect("rejected close must preserve position");
    assert_eq!(pos.qty, 0.4);
    assert_eq!(pos.entry_price, entry_price);
    assert_eq!(pos.is_long, is_long);
    assert_eq!(pos.leverage, 5.0);
    assert_eq!(pos.update_time, 123);
}

#[tokio::test]
async fn partial_short_close_preserves_residual_and_accounts_only_the_fill() {
    let sim = seeded_position(false, 6000.0);
    let taker = sim.fetch_commission_rate(SYMBOL).await.unwrap().1;

    sim.execute_reduce_only_market(SYMBOL, false, 0.3, 0.001)
        .await
        .unwrap();

    let positions = sim.fetch_open_positions().await.unwrap();
    assert_eq!(positions.len(), 1, "closing 0.3 of 0.4 leaves exposure");
    assert!((positions[0].qty - 0.1).abs() < 1e-12);
    assert_eq!(positions[0].entry_price, 6000.0);
    assert!(!positions[0].is_long);
    // Short market exit pays 1bp: 5000.5. Independent linear accounting:
    // 0.3*(6000-5000.5)=299.85; one closing fee on 1500.15 USDT.
    let expected_capital = INITIAL_CAPITAL + 299.85 - 1500.15 * taker;
    assert!((sim.fetch_account_balance().await.unwrap() - expected_capital).abs() < 1e-10);
}

#[tokio::test]
async fn oversize_reduce_only_matches_exact_position_size_including_impact_and_fee() {
    let exact = seeded_position(true, 4000.0);
    let oversize = seeded_position(true, 4000.0);
    let taker = exact.fetch_commission_rate(SYMBOL).await.unwrap().1;

    exact.execute_reduce_only_market(SYMBOL, true, 0.4, 0.001)
        .await
        .unwrap();
    // Requested 1000 would also move size impact above the 1bp floor;
    // the unfilled excess must affect neither execution price nor ledger.
    oversize.execute_reduce_only_market(SYMBOL, true, 1000.0, 0.001)
        .await
        .unwrap();

    assert!(exact.fetch_open_positions().await.unwrap().is_empty());
    assert!(oversize.fetch_open_positions().await.unwrap().is_empty());
    let exact_capital = exact.fetch_account_balance().await.unwrap();
    let oversize_capital = oversize.fetch_account_balance().await.unwrap();
    assert_eq!(oversize_capital, exact_capital, "requested excess is never filled");
    // Long market exit receives 4999.5: 0.4*(4999.5-4000)=399.8.
    let expected_capital = INITIAL_CAPITAL + 399.8 - 1999.8 * taker;
    assert!((oversize_capital - expected_capital).abs() < 1e-10);
}

#[tokio::test]
async fn reduce_only_rejects_the_same_side_for_both_position_directions() {
    for is_long in [false, true] {
        let sim = seeded_position(is_long, 6000.0);
        assert!(sim
            .execute_reduce_only_market(SYMBOL, !is_long, 0.3, 0.001)
            .await
            .is_err());
        assert_unchanged(&sim, is_long, 6000.0);
    }
}

#[tokio::test]
async fn invalid_close_quantities_preserve_capital_and_position() {
    for quantity in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0, -0.0] {
        let sim = seeded_position(true, 6000.0);
        assert!(sim
            .execute_reduce_only_market(SYMBOL, true, quantity, 0.001)
            .await
            .is_err(), "invalid quantity: {quantity:?}");
        assert_unchanged(&sim, true, 6000.0);
    }
}

#[tokio::test]
async fn close_without_position_is_not_a_successful_fill() {
    let sim = SimulatedExecutor::new(INITIAL_CAPITAL);
    assert!(sim
        .execute_reduce_only_market(SYMBOL, true, 0.3, 0.001)
        .await
        .is_err());
    assert!(sim.fetch_open_positions().await.unwrap().is_empty());
    assert_eq!(sim.fetch_account_balance().await.unwrap(), INITIAL_CAPITAL);
}
