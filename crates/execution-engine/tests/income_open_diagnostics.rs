//! OPEN: limitations of paged public evidence, not desired financial guarantees.
use execution_engine::{income_evidence::*, order_types::IncomeEntry};
fn row(id: u64, time: u64, income: f64) -> IncomeEntry {
    IncomeEntry {
        symbol: "TEST".into(),
        asset: "USDT".into(),
        income_type: "REALIZED_PNL".into(),
        tran_id: id,
        trade_id: id.to_string(),
        time,
        income,
    }
}
#[tokio::test]
async fn open_observed_exhaustion_does_not_establish_provider_snapshot() {
    // A late backfill inserts id4 before the page already read. Page2 overlaps
    // one identity and contains a new row. No snapshot/version is in this API.
    let pages = vec![
        vec![row(1, 10, 1.0), row(2, 20, 1.0)],
        vec![row(2, 20, 1.0), row(3, 30, 1.0)],
        vec![],
    ];
    let w = collect_income_window(1, 100, 3, 2, |p| {
        std::future::ready(Ok(pages[(p - 1) as usize].clone()))
    })
    .await
    .unwrap();
    assert_eq!(w.coverage, IncomeCoverage::PageExhausted);
    assert_eq!(w.entries.len(), 3);
    assert!(!w.entries.iter().any(|e| e.tran_id == 4));
}
#[tokio::test]
async fn open_revision_of_identity_fields_needs_provider_reconciliation() {
    // Same transaction changes timestamp and amount. Visible-tuple identity
    // cannot establish whether this is a correction or a distinct event.
    let w = collect_income_window(1, 100, 1, 10, |_| {
        std::future::ready(Ok(vec![row(1, 10, 1.0), row(1, 11, 2.0)]))
    })
    .await
    .unwrap();
    assert_eq!(w.into_exhausted_entries().unwrap().len(), 2);
}
