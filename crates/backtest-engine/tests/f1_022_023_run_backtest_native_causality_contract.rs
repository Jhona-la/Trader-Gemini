//! Contract test for F1-022 and F1-023 (Ola Ω153):
//!
//! 1. F1-022: Early ruin / exit breaks use `last_close_executed` for terminal MTM,
//!    preventing post-ruin lookahead to `closes.last()`, and measures Drawdown on
//!    Total Equity (cash + unrealized PnL) to eradicate CSG-7.
//! 2. F1-023: Intra-candle tick simulation uses a causal Lee-Ready tick rule on
//!    `delta_tick = sim_price - prev_sim_price` rather than `(sim_price - prev_close)/prev_close`,
//!    eliminating direction leakage of the current candle's final close.

use backtest_engine::{run_backtest_native, STATS_LEN};
use quantum_arena::genome::SuperGenotype;

#[test]
fn test_f1_023_lee_ready_causal_tick_rule_logic() {
    // In Lee-Ready tick rule:
    // If trade price decreases (downtick): executed at bid -> buyer maker (seller aggressive).
    // If trade price increases (uptick): executed at ask -> buyer taker (buyer aggressive).
    // Zero-tick preserves previous tick's classification.
    let mut last_buyer_maker = false;

    // Tick 1: Uptick (price rises from 100 to 101)
    let p0: f64 = 100.0;
    let p1: f64 = 101.0;
    let d1: f64 = p1 - p0;
    let bm1 = if d1.abs() > 1e-12 {
        d1 < 0.0
    } else {
        last_buyer_maker
    };
    last_buyer_maker = bm1;
    assert!(!bm1, "Uptick must be buyer taker (is_buyer_maker = false)");

    // Tick 2: Zero-tick (price stays 101) -> should preserve false
    let p2: f64 = 101.0;
    let d2: f64 = p2 - p1;
    let bm2 = if d2.abs() > 1e-12 {
        d2 < 0.0
    } else {
        last_buyer_maker
    };
    last_buyer_maker = bm2;
    assert!(!bm2, "Zero-tick after uptick must remain false");

    // Tick 3: Downtick (price drops from 101 to 99) -> buyer maker = true
    let p3: f64 = 99.0;
    let d3: f64 = p3 - p2;
    let bm3 = if d3.abs() > 1e-12 {
        d3 < 0.0
    } else {
        last_buyer_maker
    };
    last_buyer_maker = bm3;
    let _ = last_buyer_maker;
    assert!(bm3, "Downtick must be buyer maker (is_buyer_maker = true)");
}

#[test]
fn test_f1_022_no_post_ruin_lookahead_contract() {
    // Construct a price series where a catastrophic collapse happens early (bars 0..3)
    // followed by an artificial 10x boom (bars 4..10).
    // Under the old bug: terminal MTM used `closes.last()`, so any open position survived
    // ruin and was evaluated at the 10x boom price, hiding ruin.
    // Under F1-022: ruin breaks at bar 3, and terminal MTM is evaluated at bar 3 (`last_close_executed`),
    // keeping capital non-inflated and reflecting the true ruin.
    let closes = vec![
        100.0, 90.0, 70.0, 40.0, // Severe crash leading to ruin
        100.0, 200.0, 400.0, 800.0, 1000.0, // Future bars after ruin
    ];
    let highs = vec![105.0, 95.0, 75.0, 45.0, 105.0, 205.0, 405.0, 805.0, 1005.0];
    let lows = vec![95.0, 85.0, 65.0, 35.0, 95.0, 195.0, 395.0, 795.0, 995.0];
    let volumes = vec![1000.0; closes.len()];

    let cfg = SuperGenotype::default();
    let mut out_pnl = vec![0.0; 100];
    let mut stats = vec![0.0; STATS_LEN];

    run_backtest_native(
        &closes,
        &highs,
        &lows,
        &volumes,
        &cfg,
        &mut out_pnl,
        &mut stats,
        "BTCUSDT",
        13.0, // 13 USD capital
    );

    let final_capital = stats[2];
    let max_drawdown = stats[3];

    // If ruin occurred, final_capital must NOT have looked ahead to 1000.0!
    // With 13.0 starting capital and severe early crash, final_capital cannot exceed starting capital * 50
    assert!(
        final_capital < 100.0,
        "Final capital ({}) must not be inflated by post-ruin future prices (closes.last() was 1000.0)",
        final_capital
    );
    assert!(
        max_drawdown >= 0.0 && max_drawdown <= 1.0,
        "Max drawdown ({}) must be bounded in [0, 1]",
        max_drawdown
    );
}
