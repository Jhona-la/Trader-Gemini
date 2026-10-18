//! Contract test for F1-006 (Ola Ω152):
//!
//! Verifies that AuditorInterno and VerificadorResultados calculate independent
//! metrics with pure accounting:
//! 1. `realized_pnl` is gross PnL, `fee_usd` is total exchange commission paid.
//! 2. Net profit is `gross_pnl - fee_usd`.
//! 3. Equity is incremented by `net` exactly once (no double-subtraction of fees).
//! 4. Win rate considers trades with `net > 0.0`.

use audit_engine::auditor_interno::{AuditorInterno, ExchangeFillRecord};

#[test]
fn test_f1_006_no_double_fee_subtraction_contract() {
    let mut auditor = AuditorInterno::with_capital(13.0);

    // Trade 1: gross +1.00 USD, fee 0.04 USD => net +0.96 USD (Win)
    auditor.record_fill(ExchangeFillRecord {
        timestamp_ns: 1_000_000_000,
        symbol_id: 0,
        is_buyer: true,
        price: 60_000.0,
        qty: 0.001,
        fee_usd: 0.04,
        realized_pnl: 1.00,
        is_closed_position: true,
    });

    // Trade 2: gross +0.02 USD, fee 0.04 USD => net -0.02 USD (Loss due to fee!)
    auditor.record_fill(ExchangeFillRecord {
        timestamp_ns: 2_000_000_000,
        symbol_id: 0,
        is_buyer: false,
        price: 60_020.0,
        qty: 0.001,
        fee_usd: 0.04,
        realized_pnl: 0.02,
        is_closed_position: true,
    });

    // Trade 3: gross -0.50 USD, fee 0.04 USD => net -0.54 USD (Loss)
    auditor.record_fill(ExchangeFillRecord {
        timestamp_ns: 3_000_000_000,
        symbol_id: 0,
        is_buyer: true,
        price: 59_500.0,
        qty: 0.001,
        fee_usd: 0.04,
        realized_pnl: -0.50,
        is_closed_position: true,
    });

    let metrics = auditor.compute_independent_metrics();

    // 1 win (Trade 1), 2 losses (Trade 2 and 3)
    // Win rate must be 1/3 ~ 0.3333:
    assert!((metrics.calculated_win_rate - (1.0 / 3.0)).abs() < 1e-4,
        "Win rate must reflect net positive trades (1/3), got {}", metrics.calculated_win_rate);

    // Total net PnL = +0.96 - 0.02 - 0.54 = +0.40 USD.
    // Base capital = 13.0 USD => Peak equity = 13.96 USD (after Trade 1).
    // Final equity = 13.40 USD.
    // Max DD from peak: (13.96 - 13.40) / 13.96 = 0.56 / 13.96 ~ 0.04011.
    assert!((metrics.calculated_max_drawdown - (0.56 / 13.96)).abs() < 1e-4,
        "Max DD must be calculated from genuine net equity, got {}", metrics.calculated_max_drawdown);
}
