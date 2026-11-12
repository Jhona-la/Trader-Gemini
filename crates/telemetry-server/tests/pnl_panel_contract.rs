//! M1 paso 3 — Contrato de la conexión panel <-> registro de operaciones.
//!
//! Fixtures sintéticas SÓLO para comportamiento; NO son evidencia financiera.
//! Sin mocks ni `unwrap`/`expect`/`panic`.

use telemetry_server::trade_ledger::{
    DataProvenance, FinancialReport, ReportPeriod, TradeLedger, TradeLedgerEntry,
    pnl_panel_from_ledger_jsonl, pnl_panel_lines,
};

const START: u64 = 1_000;
const END: u64 = 5_000;

fn period() -> ReportPeriod {
    ReportPeriod::new(START, END)
}

fn real() -> DataProvenance {
    DataProvenance::Real {
        dataset: "behaviour-fixture".to_string(),
    }
}

fn jsonl() -> &'static str {
    concat!(
        r#"{"timestamp_ns":1000,"symbol":"BTCUSDT","gross_pnl":100.0,"fees":10.0,"funding":2.0,"slippage":3.0}"#,
        "\n",
        r#"{"timestamp_ns":2000,"symbol":"ETHUSDT","gross_pnl":-40.0,"fees":5.0,"funding":1.0,"slippage":1.0}"#,
        "\n",
        r#"{"timestamp_ns":3000,"symbol":"SOLUSDT","gross_pnl":20.0,"fees":30.0,"funding":0.0,"slippage":0.0}"#,
        "\n",
    )
}

/// PRIMERA PRUEBA RED: sin procedencia real, el panel es N/D en TODO,
/// nunca `0.00` ni un capital/dataset inventado.
#[test]
fn without_real_data_panel_is_nd_not_zeros() {
    let lines =
        pnl_panel_from_ledger_jsonl(jsonl(), period(), 1_000.0, DataProvenance::Unavailable);
    for field in [
        &lines.gross_pnl,
        &lines.fees,
        &lines.funding,
        &lines.slippage,
        &lines.net_pnl,
        &lines.roi_before_fees_pct,
        &lines.roi_after_fees_pct,
        &lines.win_rate_before_fees,
        &lines.win_rate_after_fees,
        &lines.period,
        &lines.dataset,
        &lines.rows,
    ] {
        assert_eq!(field, "N/D", "sin datos reales no se imprime ninguna cifra");
    }
}

/// Con procedencia real declarada, el panel muestra bruto y neto, ROI y WR
/// antes y después de fees, periodo, dataset y filas usadas.
#[test]
fn real_data_shows_pre_and_post_fees_with_period_dataset_and_rows() {
    let lines = pnl_panel_from_ledger_jsonl(jsonl(), period(), 1_000.0, real());
    assert_eq!(lines.gross_pnl, "80.00");
    assert_eq!(lines.fees, "45.00");
    assert_eq!(lines.funding, "3.00");
    assert_eq!(lines.slippage, "4.00");
    assert_eq!(lines.net_pnl, "28.00");
    assert_eq!(lines.roi_before_fees_pct, "8.00");
    assert_eq!(lines.roi_after_fees_pct, "2.80");
    assert_eq!(lines.win_rate_before_fees, "66.7%");
    assert_eq!(lines.win_rate_after_fees, "33.3%", "las fees bajan el WR");
    assert_eq!(lines.dataset, "behaviour-fixture");
    assert_eq!(lines.rows, "3");
    assert_eq!(lines.period, "1000..5000 ns");
}

/// El capital lo aporta el llamador: dos capitales dan ROI distinto y el PnL
/// no cambia. Guarda contra capital fijo (13 USD) en el panel.
#[test]
fn capital_comes_from_caller_not_hardcoded() {
    let a = pnl_panel_from_ledger_jsonl(jsonl(), period(), 1_000.0, real());
    let b = pnl_panel_from_ledger_jsonl(jsonl(), period(), 4_000.0, real());
    assert_eq!(a.gross_pnl, b.gross_pnl);
    assert_eq!(a.net_pnl, b.net_pnl);
    assert_eq!(a.roi_before_fees_pct, "8.00");
    assert_eq!(b.roi_before_fees_pct, "2.00");
}

/// Un componente ausente en el JSONL es N/D en ese campo, no `0.00`,
/// y no crea un beneficio sin comisiones.
#[test]
fn absent_component_is_nd_not_zero_and_not_free_profit() {
    let text = concat!(
        r#"{"timestamp_ns":1000,"symbol":"BTCUSDT","gross_pnl":100.0,"funding":0.0,"slippage":0.0}"#,
        "\n",
    );
    let lines = pnl_panel_from_ledger_jsonl(text, period(), 1_000.0, real());
    assert_eq!(lines.gross_pnl, "100.00");
    assert_eq!(lines.fees, "N/D");
    assert_eq!(lines.net_pnl, "N/D");
    assert_eq!(lines.roi_after_fees_pct, "N/D");
    assert_eq!(lines.win_rate_after_fees, "N/D");
    assert_eq!(lines.win_rate_before_fees, "100.0%");
}

/// Un cero medido se imprime `0.00`, no `N/D`.
#[test]
fn measured_zero_is_printed_as_zero() {
    let text = concat!(
        r#"{"timestamp_ns":1000,"symbol":"BTCUSDT","gross_pnl":0.0,"fees":0.0,"funding":0.0,"slippage":0.0}"#,
        "\n",
    );
    let lines = pnl_panel_from_ledger_jsonl(text, period(), 1_000.0, real());
    assert_eq!(lines.gross_pnl, "0.00");
    assert_eq!(lines.fees, "0.00");
    assert_eq!(lines.net_pnl, "0.00");
    assert_eq!(lines.roi_before_fees_pct, "0.00");
    assert_eq!(lines.win_rate_before_fees, "0.0%");
}

/// Periodo inválido o sin filas => N/D, sin `0` filas ni capital inventados.
#[test]
fn invalid_period_or_no_rows_is_nd() {
    let invalid =
        pnl_panel_from_ledger_jsonl(jsonl(), ReportPeriod::new(END, START), 1_000.0, real());
    assert_eq!(invalid.gross_pnl, "N/D");
    assert_eq!(invalid.rows, "N/D");
    assert_eq!(invalid.dataset, "behaviour-fixture");

    let empty = pnl_panel_from_ledger_jsonl("", period(), 1_000.0, real());
    assert_eq!(empty.gross_pnl, "N/D");
    assert_eq!(empty.rows, "0", "0 filas es dato medido");
    assert_eq!(empty.period, "1000..5000 ns");
}

/// Línea malformada => fail-closed: panel completo N/D.
#[test]
fn malformed_line_fails_closed() {
    let text = concat!(
        r#"{"timestamp_ns":1000,"symbol":"BTCUSDT","gross_pnl":100.0,"fees":10.0,"funding":0.0,"slippage":0.0}"#,
        "\n",
        "esto no es json\n",
    );
    assert_eq!(
        pnl_panel_from_ledger_jsonl(text, period(), 1_000.0, real()),
        pnl_panel_lines(&FinancialReport::build(
            &TradeLedger::new(),
            period(),
            1_000.0,
            DataProvenance::Unavailable,
        )),
        "un registro corrupto no debe publicar cifras parciales"
    );
}

/// `pnl_panel_lines` no imprime `NaN`/`inf` bajo ninguna entrada no finita.
#[test]
fn never_prints_nan_or_inf_strings() {
    let mut ledger = TradeLedger::new();
    ledger.append(TradeLedgerEntry::from_measured(
        1_000,
        "BTCUSDT",
        f64::NAN,
        15.0,
        f64::INFINITY,
        0.0,
    ));
    let report = FinancialReport::build(&ledger, period(), 1_000.0, real());
    let lines = pnl_panel_lines(&report);
    for field in [
        &lines.gross_pnl,
        &lines.fees,
        &lines.funding,
        &lines.slippage,
        &lines.net_pnl,
        &lines.roi_before_fees_pct,
        &lines.roi_after_fees_pct,
        &lines.win_rate_before_fees,
        &lines.win_rate_after_fees,
    ] {
        assert_ne!(field, "NaN");
        assert_ne!(field, "inf");
        assert!(!field.contains("NaN"), "N/D explícito: {field}");
        assert!(!field.contains("inf"), "N/D explícito: {field}");
    }
}
