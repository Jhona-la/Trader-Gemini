//! Source contracts only: do not execute the host or access an account.
#[test]
fn bounded_income_adapter_uses_the_tested_collector() {
    let source = include_str!("../src/executor.rs");
    assert!(source.contains("collect_income_window("));
    assert!(source.contains("income_page_query("));
}

#[test]
fn fee_breaker_checks_currency_before_aggregation() {
    let source = include_str!("../../../src/bin/god_engine.rs");
    let body = source
        .split(".fetch_income_paged(&[], window_start, 4)")
        .nth(1)
        .unwrap();
    let before_aggregation = body.split("struct Agg {").next().unwrap();
    assert!(before_aggregation.contains("partition_legacy_fee_evidence(&entries)"));
    assert!(body.contains("fee_partition.same_asset_by_symbol.values().flatten()"));
}

#[test]
fn report_does_not_claim_gross_row_hit_rate_is_net_trade_win_rate() {
    let source = include_str!("../../../src/bin/income_report.rs");
    assert!(!source.contains("WR bruto = WR neto por trade"));
    assert!(source.contains("aggregate_income_by_asset("));
}
