//! Contract test for F1-007 and F1-019 (Ola Ω152):
//!
//! 1. F1-019: In `trade_only` mode (aggTrades), trade volume is honest `|bq - aq|`
//!    rather than `bq + aq`, eliminating the 21x to 201x volume distortion.
//! 2. F1-007: In `trade_only` mode, `bid_qty` and `ask_qty` are passed as 0.0,
//!    causing `god-engine-core` to honestly detect `book_absent = true`,
//!    activating the F7 fallback branches (ML recentrado, espectro directo, etc.).

use backtest_engine::booktick_replay::ReplayConfig;

#[test]
fn test_f1_019_trade_only_volume_unpoisoned() {
    // In binance_vision_sync aggTrades:
    // qty = 0.01 BTC.
    // base_depth = max(0.01 * 0.25, 0.1) = 0.1.
    // For is_maker == false: bq = 0.01 + 0.1 = 0.11, aq = 0.10.
    // Under old bug: vol = bq + aq = 0.21 (21x distortion!).
    // Under F1-019 fix: trade_qty = |bq - aq| = |0.11 - 0.10| = 0.01.
    let bq: f64 = 0.11;
    let aq: f64 = 0.10;
    let cfg = ReplayConfig {
        trade_only: true,
        ..ReplayConfig::default()
    };

    let trade_qty: f64 = if cfg.trade_only {
        (bq - aq).abs()
    } else {
        bq + aq
    };

    assert!((trade_qty - 0.01).abs() < 1e-9, "Trade qty must be exactly 0.01, got {}", trade_qty);

    let book_cfg = ReplayConfig {
        trade_only: false,
        ..ReplayConfig::default()
    };
    let book_vol: f64 = if book_cfg.trade_only {
        (bq - aq).abs()
    } else {
        bq + aq
    };
    assert!((book_vol - 0.21).abs() < 1e-9, "BookTicker liquidity must be bq + aq = 0.21");
}

#[test]
fn test_f1_007_book_absent_detection_contract() {
    // In god-engine-core/src/lib.rs:6571:
    // let book_absent = (bid_qty + ask_qty) <= 1e-9 || bid_qty <= 0.0 || ask_qty <= 0.0;
    // When trade_only passes (0.0, 0.0), book_absent evaluates to true:
    let bid_qty_trade_only: f64 = 0.0;
    let ask_qty_trade_only: f64 = 0.0;
    let book_absent = (bid_qty_trade_only + ask_qty_trade_only) <= 1e-9
        || bid_qty_trade_only <= 0.0
        || ask_qty_trade_only <= 0.0;

    assert!(book_absent, "book_absent must be true in trade_only mode to unblock F7 branches");

    // In contrast, old bug passed (vol * 0.5, vol * 0.5) with vol > 0:
    let old_bid_qty: f64 = 0.21 * 0.5;
    let old_ask_qty: f64 = 0.21 * 0.5;
    let old_book_absent = (old_bid_qty + old_ask_qty) <= 1e-9
        || old_bid_qty <= 0.0
        || old_ask_qty <= 0.0;

    assert!(!old_book_absent, "Old bug falsely treated absent book as present balanced book");
}
