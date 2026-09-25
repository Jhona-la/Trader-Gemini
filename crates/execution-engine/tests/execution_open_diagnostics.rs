//! OPEN defects plus the limited FMT-184 containment regression from XIX.
//! Passing does not certify a complete position ledger or safe trading.
use execution_engine::{
    dynamic_symbols::{DynamicSymbolSelector, SymbolScore},
    order_types::{parse_order_body, Fill, OrderAck},
    reconciliation::{reconcile_arena, PositionRiskEntry},
};
use std::sync::atomic::Ordering;

#[test]
fn open_debt_missing_ack_identity_is_accepted() {
    let ack = parse_order_body("{}").unwrap();
    assert_eq!(ack.order_id, 0);
    assert!(ack.symbol.is_empty() && ack.client_order_id.is_empty());
}

#[test]
fn open_debt_commissions_mix_currencies_and_discard_rebates() {
    let ack = OrderAck {
        fills: vec![
            Fill {
                commission: 1.0,
                commission_asset: "BNB".into(),
                ..Default::default()
            },
            Fill {
                commission: 1.0,
                commission_asset: "USDT".into(),
                ..Default::default()
            },
            Fill {
                commission: -0.1,
                commission_asset: "USDT".into(),
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    // 1 BNB + 0.9 USDT is not the scalar 2 in any defined numeraire.
    assert_eq!(ack.total_commission(), 2.0);
}

#[test]
fn open_debt_second_selector_fabricates_btc_without_observations() {
    let (large, small) = DynamicSymbolSelector::parse_and_rank_json_tickers(&[], false);
    assert_eq!(large, vec!["BTCUSDT"]);
    assert_eq!(small, vec!["BTCUSDT"]);
}

#[test]
fn open_debt_second_selector_accepts_invalid_price_and_duplicates() {
    let row = serde_json::json!({"symbol":"AUDITUSDT","quoteVolume":"20000000","priceChangePercent":"2","lastPrice":"NaN"});
    let (_, small) = DynamicSymbolSelector::parse_and_rank_json_tickers(&[row.clone(), row], false);
    assert_eq!(small.iter().filter(|s| *s == "AUDITUSDT").count(), 2);
}

#[test]
fn open_debt_symbol_score_total_order_contract_is_inconsistent() {
    let score = SymbolScore {
        symbol: "AUDITUSDT".into(),
        volume_usd: 1.0,
        price_change_pct: 1.0,
        score: f64::NAN,
    };
    assert_ne!(score, score);
    assert_eq!(score.cmp(&score), std::cmp::Ordering::Equal);
    assert_eq!(score.partial_cmp(&score), None);
}

#[test]
fn regression_ambiguous_hedge_and_reversal_preserve_local_state() {
    use quantum_arena::{position::PositionHorizon, symbol_registry, symbols, GlobalArena};
    // This integration binary has only one test touching the global universe.
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "AUDITUSDT",
    )]);
    symbols::update_dynamic_universe(vec!["AUDITUSDT".into()]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let slot = &arena.coins[0].positions.position;
    let remote = |qty, side: &str, price| PositionRiskEntry {
        symbol: "AUDITUSDT".into(),
        position_amt: qty,
        position_side: side.into(),
        entry_price: price,
        leverage: 10.0,
        ..Default::default()
    };
    assert!(slot.open_with_horizon(
        true,
        100.0,
        1.0,
        10.0,
        1,
        110.0,
        90.0,
        PositionHorizon::Continuous
    ));
    arena.used_margin.store(10.0, Ordering::Relaxed);
    reconcile_arena(
        &[remote(1.0, "LONG", 100.0), remote(-1.0, "SHORT", 120.0)],
        &arena,
        2,
    );
    assert!(
        slot.is_open(),
        "XIX must not clear a known hedge as net zero"
    );
    slot.close();

    assert!(slot.open_with_horizon(
        true,
        100.0,
        1.0,
        10.0,
        3,
        110.0,
        90.0,
        PositionHorizon::Continuous
    ));
    arena.used_margin.store(10.0, Ordering::Relaxed);
    reconcile_arena(&[remote(-1.0, "BOTH", 120.0)], &arena, 4);
    assert!(slot.is_open());
    assert!(
        slot.is_long.load(Ordering::Relaxed),
        "unresolved reversal must preserve the known local state"
    );
    assert!(!slot.exchange_confirmed.load(Ordering::Relaxed));
    assert_eq!(slot.entry_price.load(Ordering::Relaxed), 100.0);
}
