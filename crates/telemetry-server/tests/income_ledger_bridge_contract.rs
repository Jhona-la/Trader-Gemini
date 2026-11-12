//! Contrato F6: puente income→ledger y aserto de reconciliación.
//!
//! Fixture: 5 ciclos reales de Demo según `glm.md`.
//!  - Ciclos 1, 4, 5: bruto documentado en glm.md.
//!  - Ciclos 2, 3: glm.md dice "ídem patrón" (taker/taker, comisión ≈ −0.0043);
//!    su bruto se infiere del neto declarado menos esa comisión y se marca como
//!    inferido. NO se inventa ningún bruto nuevo: es el mismo patrón afirmado.
//!
//! GREEN: el puente completo reconcilia bruto/comisión/funding con el exchange.
//! RED:   sin puente (registro vacío) o puente parcial => la reconciliación
//!        detecta la discrepancia (el aserto es significativo).
use telemetry_server::trade_ledger::{
    exchange_income_totals, ledger_from_income, reconcile_ledger_vs_income, IncomeRow,
    ReconcileField, TradeLedger,
};

fn row(symbol: &str, income_type: &str, income: f64, trade_id: &str, time_ms: u64) -> IncomeRow {
    IncomeRow {
        symbol: symbol.to_string(),
        income_type: income_type.to_string(),
        income,
        time_ms,
        tran_id: 0,
        trade_id: trade_id.to_string(),
        asset: "USDT".to_string(),
    }
}

/// Los 5 ciclos reales de Demo (glm.md). Ciclos 2 y 3 infieren el bruto del
/// patrón "ídem" declarado (comisión −0.0043); ver comentario arriba.
fn five_real_cycles() -> Vec<IncomeRow> {
    vec![
        // 1) BCHUSDT 11-06 18:17:36  bruto −0.0006  neto −0.0049  comisión −0.0043
        row(
            "BCHUSDT",
            "REALIZED_PNL",
            -0.0006,
            "a3b-c1",
            1_784_000_000_000,
        ),
        row(
            "BCHUSDT",
            "COMMISSION",
            -0.0043,
            "a3b-c1",
            1_784_000_000_000,
        ),
        // 2) DOGEUSDT 11-08 05:08:01 "ídem patrón" neto −0.0049 => bruto −0.0006 (inferido)
        row(
            "DOGEUSDT",
            "REALIZED_PNL",
            -0.0006,
            "a3b-c2",
            1_785_000_000_000,
        ),
        row(
            "DOGEUSDT",
            "COMMISSION",
            -0.0043,
            "a3b-c2",
            1_785_000_000_000,
        ),
        // 3) BCHUSDT 11-09 06:04:53 "ídem" neto −0.0044 => bruto −0.0001 (inferido)
        row(
            "BCHUSDT",
            "REALIZED_PNL",
            -0.0001,
            "a3c-c3",
            1_786_000_000_000,
        ),
        row(
            "BCHUSDT",
            "COMMISSION",
            -0.0043,
            "a3c-c3",
            1_786_000_000_000,
        ),
        // 4) ARBUSDT 11-10 17:36:14 bruto −0.0033  neto −0.0115  comisión −0.0082
        row(
            "ARBUSDT",
            "REALIZED_PNL",
            -0.0033,
            "a3g-c4",
            1_787_000_000_000,
        ),
        row(
            "ARBUSDT",
            "COMMISSION",
            -0.0082,
            "a3g-c4",
            1_787_000_000_000,
        ),
        // 5) LDOUSDT 11-10 18:27:26 bruto 0  neto −0.0041  comisión −0.0041
        row("LDOUSDT", "REALIZED_PNL", 0.0, "a3g-c5", 1_788_000_000_000),
        row(
            "LDOUSDT",
            "COMMISSION",
            -0.0041,
            "a3g-c5",
            1_788_000_000_000,
        ),
    ]
}

#[test]
fn f6_green_full_bridge_reconciles() {
    let rows = five_real_cycles();
    let ledger = ledger_from_income(&rows);
    let result = reconcile_ledger_vs_income(&ledger, &rows, 1e-9);
    assert!(
        result.is_ok(),
        "F6 GREEN: el registro debe reconciliar con el exchange: {:?}",
        result
    );
}

#[test]
fn f6_green_known_totals() {
    let rows = five_real_cycles();
    let ledger = ledger_from_income(&rows);
    let t = ledger.totals();
    assert!(
        (t.gross_pnl + 0.0046).abs() < 1e-12,
        "gross = -0.0046, got {}",
        t.gross_pnl
    );
    assert!(
        (t.fees + 0.0252).abs() < 1e-12,
        "commission = -0.0252, got {}",
        t.fees
    );
    assert!(t.funding.abs() < 1e-12, "funding = 0, got {}", t.funding);
    // Income no lleva slippage => N/D (NaN); no se fabrica 0.00.
    assert!(t.slippage.is_nan(), "slippage debe ser N/D");
    // Bruto/comisión son medidos (finites) => no N/D.
    assert!(t.gross_pnl.is_finite());
    assert!(t.fees.is_finite());
}

#[test]
fn f6_red_no_bridge_mismatch() {
    // Sin puente (registro vacío) vs income real => 3 discrepancias.
    let rows = five_real_cycles();
    let empty: TradeLedger = TradeLedger::new();
    let result = reconcile_ledger_vs_income(&empty, &rows, 1e-9);
    let mismatches = result.expect_err("F6 RED: sin puente debe fallar la reconciliación");
    assert_eq!(
        mismatches.len(),
        3,
        "bruto/comisión/funding ausentes vs medidos"
    );
    let fields: Vec<ReconcileField> = mismatches.iter().map(|m| m.field).collect();
    assert!(fields.contains(&ReconcileField::Gross));
    assert!(fields.contains(&ReconcileField::Commission));
    assert!(fields.contains(&ReconcileField::Funding));
}

#[test]
fn f6_red_partial_bridge_mismatch() {
    // Puente parcial: faltan los ciclos 4 (bruto −0.0033, comisión −0.0082).
    let rows = five_real_cycles();
    let partial: Vec<IncomeRow> = rows
        .into_iter()
        .filter(|r| r.trade_id != "a3g-c4")
        .collect();
    let ledger = ledger_from_income(&partial);
    let result = reconcile_ledger_vs_income(&ledger, &five_real_cycles(), 1e-9);
    let mismatches = result.expect_err("F6 RED: puente parcial debe fallar");
    assert!(!mismatches.is_empty());
    let fields: Vec<ReconcileField> = mismatches.iter().map(|m| m.field).collect();
    assert!(fields.contains(&ReconcileField::Gross));
    assert!(fields.contains(&ReconcileField::Commission));
    // Funding no cambia entre parcial y completo => no debe discrepar.
    assert!(!fields.contains(&ReconcileField::Funding));
}

#[test]
fn f6_exchange_totals_consistent() {
    let rows = five_real_cycles();
    let ex = exchange_income_totals(&rows);
    assert!((ex.gross_pnl + 0.0046).abs() < 1e-12);
    assert!((ex.fees + 0.0252).abs() < 1e-12);
    assert!(ex.funding.abs() < 1e-12);
}
