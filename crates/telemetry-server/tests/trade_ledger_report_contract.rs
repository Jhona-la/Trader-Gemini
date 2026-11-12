//! M1 paso 2 — Contrato del informe: ROI/PnL/WR antes y después de fees,
//! con periodo y datos usados.
//!
//! Fixtures sintéticas SÓLO para comportamiento; NO son evidencia financiera,
//! ni ROI/PnL/WR medidos, ni OOS. Sin mocks ni `unwrap`/`expect`/`panic`.

use telemetry_server::trade_ledger::{
    DataProvenance, FinancialReport, ReportPeriod, TradeLedger, TradeLedgerEntry,
};

const START: u64 = 1_000;
const END: u64 = 5_000;

fn period() -> ReportPeriod {
    ReportPeriod::new(START, END)
}

fn fixture() -> TradeLedger {
    let mut ledger = TradeLedger::new();
    // gana bruto y neto
    ledger.append(TradeLedgerEntry::from_measured(
        1_000, "BTCUSDT", 100.0, 10.0, 2.0, 3.0,
    ));
    // pierde bruto y neto
    ledger.append(TradeLedgerEntry::from_measured(
        2_000, "ETHUSDT", -40.0, 5.0, 1.0, 1.0,
    ));
    // gana bruto, pierde neto: las fees cambian el WR
    ledger.append(TradeLedgerEntry::from_measured(
        3_000, "SOLUSDT", 20.0, 30.0, 0.0, 0.0,
    ));
    ledger
}

fn real(dataset: &str) -> DataProvenance {
    DataProvenance::Real {
        dataset: dataset.to_string(),
    }
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// PRIMERA PRUEBA RED: sin datos reales, el informe es N/D, NO ceros ni un
/// capital fijo. El patrón defectuoso publica cifras y `dataset` inventado.
#[test]
fn report_without_real_data_is_nd_not_fabricated() {
    let r = FinancialReport::build(&fixture(), period(), 1_000.0, DataProvenance::Unavailable);
    assert_eq!(r.dataset(), None, "sin datos reales no hay dataset");
    assert!(!r.has_evidence());
    let m = r.metrics();
    assert!(m.pnl_before_fees().is_nan(), "no se fabrica PnL");
    assert!(m.pnl_after_fees().is_nan(), "no se fabrica PnL neto");
    assert!(m.roi_before_fees_pct().is_nan(), "no se fabrica ROI");
    assert!(m.roi_after_fees_pct().is_nan(), "no se fabrica ROI neto");
    assert!(m.win_rate_before_fees().is_nan(), "no se fabrica WR");
    assert!(m.win_rate_after_fees().is_nan(), "no se fabrica WR neto");
    let w = r.wire();
    assert_eq!(w.dataset, None);
    assert_eq!(w.rows, None);
    assert_eq!(w.pnl_before_fees, None);
    assert_eq!(w.roi_before_fees_pct, None);
    assert_eq!(w.wallet_capital, None);
    // El periodo SIEMPRE se informa, aunque no haya datos.
    assert_eq!(w.period_start_ns, START);
    assert_eq!(w.period_end_ns, END);
}

/// Informe con datos reales: PnL y ROI antes y después de fees, WR antes y
/// después de fees, periodo, dataset y filas usadas.
#[test]
fn report_has_roi_pnl_wr_before_and_after_fees_with_period_and_dataset() {
    let capital = 1_000.0; // capital sintético del llamador, nunca fijo en código
    let r = FinancialReport::build(&fixture(), period(), capital, real("demo-wallet-fixture"));
    assert_eq!(r.dataset(), Some("demo-wallet-fixture"));
    assert_eq!(r.period(), period());
    assert_eq!(r.rows(), 3, "filas usadas del periodo");
    assert_eq!(r.wallet_capital(), capital);
    assert!(r.has_evidence());

    let m = r.metrics();
    assert!(approx(m.pnl_before_fees(), 80.0), "PnL bruto 100-40+20");
    assert!(approx(m.pnl_after_fees(), 28.0), "PnL neto 85-47-10");
    assert!(approx(m.fees(), 45.0));
    assert!(approx(m.funding(), 3.0));
    assert!(approx(m.slippage(), 4.0));
    assert!(approx(m.roi_before_fees_pct(), 8.0));
    assert!(approx(m.roi_after_fees_pct(), 2.8));
    assert!(approx(m.win_rate_before_fees(), 2.0 / 3.0));
    assert!(
        approx(m.win_rate_after_fees(), 1.0 / 3.0),
        "las fees bajan el WR"
    );
    assert!(
        m.win_rate_before_fees() > m.win_rate_after_fees(),
        "WR antes y después de fees deben poder diferir"
    );
    assert!(m.pnl_before_fees() > m.pnl_after_fees());
    assert!(m.roi_before_fees_pct() > m.roi_after_fees_pct());
}

/// El capital lo aporta el llamador: dos capitales distintos dan ROI distinto.
/// Guarda contra el capital fijo (13 USD) prohibido por el operador.
#[test]
fn roi_uses_caller_wallet_capital_not_a_fixed_value() {
    let ledger = fixture();
    let capital_a = 1_000.0;
    let capital_b = 4_000.0;
    let a = FinancialReport::build(&ledger, period(), capital_a, real("dataset-a"));
    let b = FinancialReport::build(&ledger, period(), capital_b, real("dataset-b"));
    assert_eq!(a.wallet_capital(), capital_a);
    assert_eq!(b.wallet_capital(), capital_b);
    assert!(a.metrics().roi_before_fees_pct() > b.metrics().roi_before_fees_pct());
    assert!(approx(
        a.metrics().roi_before_fees_pct(),
        b.metrics().roi_before_fees_pct() * 4.0
    ));
    // PnL no depende del capital.
    assert!(approx(
        a.metrics().pnl_before_fees(),
        b.metrics().pnl_before_fees()
    ));
}

/// Disponibilidad independiente: sin capital medido, SÓLO los ROI son N/D;
/// PnL y WR medidos se conservan.
#[test]
fn missing_capital_leaves_only_roi_nd() {
    for capital in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let r = FinancialReport::build(&fixture(), period(), capital, real("dataset"));
        let m = r.metrics();
        assert!(m.roi_before_fees_pct().is_nan());
        assert!(m.roi_after_fees_pct().is_nan());
        assert!(approx(m.pnl_before_fees(), 80.0));
        assert!(approx(m.pnl_after_fees(), 28.0));
        assert!(approx(m.win_rate_before_fees(), 2.0 / 3.0));
        assert!(approx(m.win_rate_after_fees(), 1.0 / 3.0));
    }
}

/// Guarda de división por cero: capital nulo o negativo no produce ROI.
#[test]
fn zero_or_negative_capital_does_not_divide_by_zero() {
    for capital in [0.0, -1.0, -f64::MAX] {
        let r = FinancialReport::build(&fixture(), period(), capital, real("dataset"));
        assert!(r.metrics().roi_before_fees_pct().is_nan());
        assert!(r.metrics().roi_after_fees_pct().is_nan());
    }
}

/// El periodo filtra: operaciones fuera de la ventana no entran en las cifras.
#[test]
fn period_excludes_entries_outside_the_window() {
    let mut ledger = fixture();
    ledger.append(TradeLedgerEntry::from_measured(
        9_999, "OUTUSDT", 1_000.0, 0.0, 0.0, 0.0,
    ));
    let r = FinancialReport::build(&ledger, period(), 1_000.0, real("dataset"));
    assert_eq!(r.rows(), 3, "la operación fuera de periodo no se usa");
    assert!(approx(r.metrics().pnl_before_fees(), 80.0));
    assert!(approx(r.metrics().pnl_after_fees(), 28.0));
}

/// Periodo inválido o sin operaciones => N/D, sin ceros fabricados.
#[test]
fn invalid_or_empty_period_is_nd() {
    let invalid = FinancialReport::build(
        &fixture(),
        ReportPeriod::new(END, START),
        1_000.0,
        real("dataset"),
    );
    assert!(invalid.metrics().pnl_before_fees().is_nan());
    assert!(invalid.metrics().win_rate_before_fees().is_nan());
    assert!(!invalid.has_evidence());
    // Un periodo rechazado NO se presenta como "0 filas medidas" ni con
    // capital medido: todo ello sería una cifra inventada sobre datos no usados.
    let iw = invalid.wire();
    assert_eq!(iw.rows, None);
    assert_eq!(iw.wallet_capital, None);
    assert_eq!(iw.pnl_before_fees, None);

    let empty = FinancialReport::build(&TradeLedger::new(), period(), 1_000.0, real("dataset"));
    assert_eq!(empty.rows(), 0, "cero filas es un dato medido");
    assert!(empty.metrics().pnl_before_fees().is_nan());
    assert!(empty.metrics().roi_before_fees_pct().is_nan());
    assert!(empty.metrics().win_rate_before_fees().is_nan());
    assert_eq!(empty.wire().rows, Some(0), "0 filas != N/D");
}

/// El límite superior del periodo es inclusivo; un periodo real sin ninguna
/// operación dentro informa `Some(0)` filas y métricas N/D (no ceros).
#[test]
fn end_boundary_is_inclusive_and_empty_real_period_is_some_zero_rows() {
    let mut ledger = TradeLedger::new();
    ledger.append(TradeLedgerEntry::from_measured(
        END, "EDGEUSDT", 50.0, 5.0, 0.0, 0.0,
    ));
    let edge = FinancialReport::build(&ledger, period(), 1_000.0, real("dataset"));
    assert_eq!(edge.rows(), 1, "end_ns inclusive");
    assert!(approx(edge.metrics().pnl_before_fees(), 50.0));

    let outside = FinancialReport::build(
        &ledger,
        ReportPeriod::new(START, END - 1),
        1_000.0,
        real("dataset"),
    );
    assert_eq!(outside.rows(), 0);
    assert_eq!(
        outside.wire().rows,
        Some(0),
        "0 filas es dato medido, no N/D"
    );
    assert!(outside.metrics().pnl_before_fees().is_nan());
    assert!(!outside.has_evidence());
}

/// Ausencia por campo: un coste no medido contamina SÓLO su agregado, el neto
/// y el WR neto; el bruto y el WR bruto medidos se conservan.
#[test]
fn absent_component_poisons_only_its_own_metric() {
    let mut ledger = TradeLedger::new();
    ledger.append(TradeLedgerEntry::from_measured(
        1_000,
        "BTCUSDT",
        100.0,
        f64::NAN,
        0.0,
        0.0,
    ));
    ledger.append(TradeLedgerEntry::from_measured(
        2_000, "ETHUSDT", -40.0, 5.0, 1.0, 1.0,
    ));
    let r = FinancialReport::build(&ledger, period(), 1_000.0, real("dataset"));
    let m = r.metrics();
    assert!(approx(m.pnl_before_fees(), 60.0));
    assert!(m.fees().is_nan());
    assert!(m.pnl_after_fees().is_nan());
    assert!(m.roi_after_fees_pct().is_nan());
    assert!(m.win_rate_after_fees().is_nan());
    assert!(approx(m.roi_before_fees_pct(), 6.0));
    assert!(approx(m.win_rate_before_fees(), 0.5));
}

/// El cable representa N/D como `None` y un cero medido como `Some(0.0)`.
#[test]
fn wire_nd_is_none_and_measured_zero_is_some() {
    let mut ledger = TradeLedger::new();
    ledger.append(TradeLedgerEntry::from_measured(
        1_000, "BTCUSDT", 0.0, 0.0, 0.0, 0.0,
    ));
    let r = FinancialReport::build(&ledger, period(), 1_000.0, real("dataset"));
    let w = r.wire();
    assert_eq!(w.pnl_before_fees, Some(0.0));
    assert_eq!(w.pnl_after_fees, Some(0.0));
    assert_eq!(w.roi_before_fees_pct, Some(0.0));
    assert_eq!(
        w.win_rate_before_fees,
        Some(0.0),
        "0 victorias de 1 es 0.0 medido"
    );
    assert_eq!(w.dataset.as_deref(), Some("dataset"));
    assert_eq!(w.rows, Some(1));

    let nd = FinancialReport::build(&ledger, period(), 1_000.0, DataProvenance::Unavailable);
    assert_eq!(nd.wire().win_rate_before_fees, None);
}

/// Desborde de la suma => N/D, no un total infinito ni parcial.
#[test]
fn aggregate_overflow_is_nd() {
    let mut ledger = TradeLedger::new();
    ledger.append(TradeLedgerEntry::from_measured(
        1_000,
        "BTCUSDT",
        f64::MAX,
        0.0,
        0.0,
        0.0,
    ));
    ledger.append(TradeLedgerEntry::from_measured(
        2_000,
        "ETHUSDT",
        f64::MAX,
        0.0,
        0.0,
        0.0,
    ));
    let r = FinancialReport::build(&ledger, period(), 1_000.0, real("dataset"));
    assert!(r.metrics().pnl_before_fees().is_nan());
    assert!(r.metrics().roi_before_fees_pct().is_nan());
}
