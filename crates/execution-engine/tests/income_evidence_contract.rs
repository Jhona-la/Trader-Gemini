//! Pure/mocked contracts: no account, environment, journal or network access.
use execution_engine::income_evidence::*;
use execution_engine::order_types::IncomeEntry;

fn row(id: u64, time: u64, income: f64) -> IncomeEntry {
    IncomeEntry {
        symbol: "TESTUSDT".into(),
        income_type: "REALIZED_PNL".into(),
        income,
        asset: "USDT".into(),
        time,
        tran_id: id,
        trade_id: id.to_string(),
    }
}
fn kind(id: u64, income: f64, name: &str, asset: &str) -> IncomeEntry {
    let mut e = row(id, 10, income);
    e.income_type = name.into();
    e.asset = asset.into();
    e
}
async fn collect(
    pages: Vec<Vec<IncomeEntry>>,
    size: u32,
    budget: u32,
) -> Result<IncomeWindow, IncomeEvidenceError> {
    collect_income_window(1, 100, budget, size, |page| {
        std::future::ready(
            pages
                .get((page - 1) as usize)
                .cloned()
                .ok_or_else(|| "unexpected fetch".into()),
        )
    })
    .await
}

#[test]
fn query_freezes_interval_and_uses_numbered_pages() {
    assert_eq!(
        income_page_query(10, 99, 2, 1000, &[], 100).unwrap(),
        "startTime=10&endTime=99&page=2&limit=1000&timestamp=100"
    );
    assert!(income_page_query(10, 99, 3, 10, &["COMMISSION"], 101)
        .unwrap()
        .ends_with("&incomeType=COMMISSION"));
}
#[test]
fn query_rejects_invalid_ranges_sizes_timestamps_and_filters() {
    assert_eq!(
        income_page_query(2, 1, 1, 1, &[], 3),
        Err(IncomeEvidenceError::InvalidRange)
    );
    assert_eq!(
        income_page_query(0, u64::MAX, 1, 1, &[], 3),
        Err(IncomeEvidenceError::InvalidRange)
    );
    assert_eq!(
        income_page_query(0, 1, 0, 1, &[], 3),
        Err(IncomeEvidenceError::InvalidPage)
    );
    for size in [0, 1001] {
        assert_eq!(
            income_page_query(0, 1, 1, size, &[], 3),
            Err(IncomeEvidenceError::InvalidPageSize)
        );
    }
    for ts in [0, u64::MAX] {
        assert_eq!(
            income_page_query(0, 1, 1, 1, &[], ts),
            Err(IncomeEvidenceError::InvalidTimestamp)
        );
    }
    for filter in [
        "",
        "COMMISSION&limit=1",
        "COMMISSION,REALIZED_PNL",
        "commission",
        "FUNDING FEE",
    ] {
        assert_eq!(
            income_page_query(0, 1, 1, 1, &[filter], 3),
            Err(IncomeEvidenceError::UnsupportedFilter)
        );
    }
    assert_eq!(
        income_page_query(0, 1, 1, 1, &["COMMISSION", "FUNDING_FEE"], 3),
        Err(IncomeEvidenceError::UnsupportedFilter)
    );
}
#[tokio::test]
async fn more_than_one_thousand_events_at_one_millisecond_are_preserved() {
    let first: Vec<_> = (1..=1000).map(|id| row(id, 50, 1.0)).collect();
    let w = collect(vec![first, vec![row(1001, 50, 1.0)]], 1000, 4)
        .await
        .unwrap();
    assert_eq!(w.pages_read, 2);
    assert_eq!(w.coverage, IncomeCoverage::PageExhausted);
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 1001);
}
#[tokio::test]
async fn full_last_allowed_page_is_not_implicitly_complete() {
    let w = collect(vec![vec![row(1, 10, 1.0)]], 1, 1).await.unwrap();
    assert_eq!(
        w.into_exhausted_entries().unwrap_err(),
        IncomeEvidenceError::Incomplete(IncomeCoverage::PageBudgetExceeded)
    );
}
#[tokio::test]
async fn exact_multiple_requires_an_empty_terminal_page() {
    let w = collect(vec![vec![row(1, 10, 1.0)], vec![]], 1, 2)
        .await
        .unwrap();
    assert_eq!(w.pages_read, 2);
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 1);
}
#[tokio::test]
async fn repeated_page_is_no_progress_even_when_short() {
    let w = collect(
        vec![
            vec![row(1, 10, 1.0), row(2, 10, 2.0)],
            vec![row(1, 10, 1.0)],
        ],
        2,
        3,
    )
    .await
    .unwrap();
    assert_eq!(w.coverage, IncomeCoverage::NoProgress);
    assert_eq!(w.pages_read, 2);
    assert!(w.into_exhausted_entries().is_err());
}
#[tokio::test]
async fn same_visible_identity_with_a_different_amount_is_a_conflict() {
    let err = collect(vec![vec![row(1, 10, 1.0)], vec![row(1, 10, -1.0)]], 1, 2)
        .await
        .unwrap_err();
    assert_eq!(err, IncomeEvidenceError::ConflictingRecord);
}
#[tokio::test]
async fn asset_and_trade_id_prevent_false_duplicate_collapse() {
    let a = row(1, 10, 1.0);
    let mut b = a.clone();
    b.asset = "BNB".into();
    let mut c = a.clone();
    c.trade_id = "different".into();
    let w = collect(vec![vec![a, b, c]], 10, 1).await.unwrap();
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 3);
}
#[tokio::test]
async fn identical_overlap_is_consumed_once_and_zero_sign_is_not_a_conflict() {
    let mut minus = row(1, 10, 0.0);
    minus.income = -0.0;
    let w = collect(vec![vec![row(1, 10, 0.0), minus, row(2, 10, -1.0)]], 10, 1)
        .await
        .unwrap();
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 2);
}
#[tokio::test]
async fn invalid_missing_and_outside_window_rows_are_not_admitted() {
    let mut invalids = vec![];
    let mut x = row(1, 10, 1.0);
    x.asset.clear();
    invalids.push(x);
    let mut x = row(1, 10, 1.0);
    x.income_type.clear();
    invalids.push(x);
    invalids.push(row(0, 10, 1.0));
    invalids.push(row(1, 0, 1.0));
    invalids.push(row(1, 10, f64::NAN));
    invalids.push(row(1, 10, f64::INFINITY));
    for x in invalids {
        assert_eq!(
            collect(vec![vec![x]], 10, 1).await.unwrap_err(),
            IncomeEvidenceError::InvalidRecord
        );
    }
    assert_eq!(
        collect(vec![vec![row(1, 101, 1.0)]], 10, 1)
            .await
            .unwrap_err(),
        IncomeEvidenceError::OutOfRange
    );
    assert_eq!(
        collect(vec![vec![row(1, 10, 1.0), row(2, 10, 1.0)]], 1, 1)
            .await
            .unwrap_err(),
        IncomeEvidenceError::OversizedPage
    );
}
#[tokio::test]
async fn invalid_request_does_not_invoke_fetcher() {
    for (start, end, budget, size) in [(2, 1, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1001)] {
        let mut calls = 0;
        let result = collect_income_window(start, end, budget, size, |_| {
            calls += 1;
            std::future::ready(Ok(vec![]))
        })
        .await;
        assert!(result.is_err());
        assert_eq!(calls, 0);
    }
}
#[tokio::test]
async fn transport_error_after_partial_success_is_not_empty_or_complete_evidence() {
    let result = collect_income_window(1, 100, 3, 1, |p| {
        std::future::ready(if p == 1 {
            Ok(vec![row(1, 10, 1.0)])
        } else {
            Err("offline".into())
        })
    })
    .await;
    assert_eq!(
        result.unwrap_err(),
        IncomeEvidenceError::Transport("offline".into())
    );
}
#[test]
fn single_asset_guard_never_converts_or_adds_currencies() {
    assert_eq!(single_income_asset(&[]), Ok(None));
    assert_eq!(single_income_asset(&[row(1, 10, 1.0)]), Ok(Some("USDT")));
    assert_eq!(
        single_income_asset(&[row(1, 10, 1.0), kind(2, -1.0, "COMMISSION", "BNB")]),
        Err(IncomeEvidenceError::MixedAssets)
    );
}
#[test]
fn aggregation_separates_assets_preserves_signed_costs_and_exposes_other_flows() {
    let s = aggregate_income_by_asset(&[
        kind(1, 10.0, "REALIZED_PNL", "USDT"),
        kind(2, -2.0, "COMMISSION", "USDT"),
        kind(3, 0.5, "COMMISSION", "USDT"),
        kind(4, -1.0, "FUNDING_FEE", "USDT"),
        kind(5, 100.0, "TRANSFER", "USDT"),
        kind(6, 3.0, "REALIZED_PNL", "BNB"),
    ])
    .unwrap();
    assert_eq!(s.by_asset.len(), 2);
    assert_eq!(s.by_symbol.len(), 2);
    let t = &s.by_asset["USDT"];
    assert_eq!(t.selected_net().unwrap(), 7.5);
    assert_eq!(t.commission, -1.5);
    assert_eq!(t.other_income, 100.0);
    assert_eq!(t.other_rows, 1);
    assert_eq!(s.by_asset["BNB"].selected_net().unwrap(), 3.0);
}

#[test]
fn mixed_fee_currency_is_quarantined_only_for_the_affected_symbol() {
    let a = row(1, 10, 1.0);
    let b = kind(2, -0.1, "COMMISSION", "BNB");
    let mut c = row(3, 10, 2.0);
    c.symbol = "OTHERUSDT".into();
    let partition = partition_legacy_fee_evidence(&[a, b, c]);
    assert_eq!(
        partition.rejected_by_symbol["TESTUSDT"],
        IncomeEvidenceError::MixedAssets
    );
    assert!(!partition.same_asset_by_symbol.contains_key("TESTUSDT"));
    assert_eq!(partition.same_asset_by_symbol["OTHERUSDT"].len(), 1);
}

#[test]
fn global_transfer_does_not_suppress_single_currency_fee_evidence() {
    let mut transfer = kind(2, 1.0, "TRANSFER", "BTC");
    transfer.symbol.clear();
    let partition = partition_legacy_fee_evidence(&[row(1, 10, 1.0), transfer]);
    assert!(partition.rejected_by_symbol.is_empty());
    assert_eq!(partition.same_asset_by_symbol["TESTUSDT"].len(), 1);
}

#[test]
fn symbols_in_distinct_settlement_assets_can_each_be_reviewed_separately() {
    let mut other = kind(2, 1.0, "REALIZED_PNL", "USDC");
    other.symbol = "OTHERUSDC".into();
    let partition = partition_legacy_fee_evidence(&[row(1, 10, 1.0), other]);
    assert!(partition.rejected_by_symbol.is_empty());
    assert_eq!(partition.same_asset_by_symbol.len(), 2);
}
#[test]
fn positive_row_fraction_includes_flat_rows_but_does_not_claim_trade_win_rate() {
    let s =
        aggregate_income_by_asset(&[row(1, 10, 10.0), row(2, 10, 0.0), row(3, 10, -1.0)]).unwrap();
    assert_eq!(s.by_asset["USDT"].pnl_rows, 3);
    assert_eq!(s.by_asset["USDT"].positive_row_fraction(), Some(1.0 / 3.0));
    assert_eq!(IncomeTotals::default().positive_row_fraction(), None);
}
#[test]
fn ratios_are_unit_invariant_and_undefined_is_not_zero() {
    for scale in [1e-12, 1.0, 100.0] {
        let s = aggregate_income_by_asset(&[
            kind(1, 10.0 * scale, "REALIZED_PNL", "USDT"),
            kind(2, -2.0 * scale, "COMMISSION", "USDT"),
        ])
        .unwrap();
        assert!((s.by_asset["USDT"].cost_to_abs_realized_ratio().unwrap() + 0.2).abs() < 1e-14);
    }
    let t = IncomeTotals {
        commission: -1.0,
        ..Default::default()
    };
    assert_eq!(t.cost_to_abs_realized_ratio(), None);
}
#[test]
fn finite_rows_cannot_overflow_components_or_selected_net_silently() {
    for rows in [
        vec![row(1, 10, f64::MAX), row(2, 10, f64::MAX)],
        vec![
            row(1, 10, f64::MAX),
            kind(2, f64::MAX, "COMMISSION", "USDT"),
        ],
        vec![
            kind(1, f64::MAX, "TRANSFER", "USDT"),
            kind(2, f64::MAX, "TRANSFER", "USDT"),
        ],
    ] {
        assert_eq!(
            aggregate_income_by_asset(&rows).unwrap_err(),
            IncomeEvidenceError::NonFiniteAggregate
        );
    }
}
#[test]
fn aggregation_is_permutation_invariant_for_exactly_representable_fixture() {
    let rows = vec![
        row(1, 10, 2.0),
        row(2, 10, -1.0),
        kind(3, -0.5, "COMMISSION", "USDT"),
    ];
    let a = aggregate_income_by_asset(&rows).unwrap();
    let mut reversed = rows;
    reversed.reverse();
    let b = aggregate_income_by_asset(&reversed).unwrap();
    assert_eq!(
        a.by_asset["USDT"].selected_net(),
        b.by_asset["USDT"].selected_net()
    );
}
#[test]
fn lookback_rejects_overflow_zero_and_pre_epoch_instead_of_wrapping() {
    assert_eq!(income_lookback_start(2 * 86_400_000, 1), Ok(86_400_000));
    for (now, days) in [(100, 0), (100, 1), (u64::MAX, u64::MAX)] {
        assert!(income_lookback_start(now, days).is_err());
    }
}
#[tokio::test]
async fn real_adapter_paper_mode_is_explicitly_not_exchange_evidence() {
    let mut exec =
        execution_engine::executor::OrderExecutor::new("fixture".into(), "fixture".into(), true);
    exec.set_paper_trading(true);
    let w = exec.fetch_income_window(&[], 1, 100, 2).await.unwrap();
    assert_eq!(w.coverage, IncomeCoverage::Simulated);
    assert_eq!(w.pages_read, 0);
    assert!(w.into_exhausted_entries().is_err());
    assert!(exec.fetch_income_paged(&[], 1, 2).await.is_err());
    assert!(exec.fetch_income_window(&[], 2, 1, 2).await.is_err());
    assert!(exec.fetch_income_window(&[], 1, 100, 0).await.is_err());
}
