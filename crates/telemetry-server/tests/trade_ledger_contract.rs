//! M1 paso 1 — Contrato del ledger por operación.
//!
//! Datos sintéticos SÓLO para comportamiento; no son evidencia financiera,
//! ni ROI/PnL/WR medidos, ni OOS. Sin mocks ni `unwrap`/`expect`/`panic`.

use telemetry_server::trade_ledger::{TradeLedger, TradeLedgerEntry};

const TS: u64 = 1_763_000_000_000_000_000;

fn entry(gross: f64, fees: f64, funding: f64, slippage: f64) -> TradeLedgerEntry {
    TradeLedgerEntry::from_measured(TS, "BTCUSDT", gross, fees, funding, slippage)
}

/// PRIMERA PRUEBA RED: un bruto no medido es ausencia, no una pérdida de fees.
/// El patrón defectuoso coerción a 0.0 publica `gross=0, fees=15, net=-15`.
#[test]
fn absent_gross_is_nd_not_a_fabricated_loss() {
    let e = entry(f64::NAN, 15.0, 0.0, 0.0);
    assert!(
        e.gross_pnl().is_nan(),
        "bruto no medido debe ser N/D, no 0.0"
    );
    assert!(
        e.net_pnl().is_nan(),
        "neto no debe fabricar una pérdida de -15"
    );
    assert_eq!(e.fees(), 15.0, "las fees medidas se conservan");
    assert_eq!(e.funding(), 0.0);
    assert_eq!(e.slippage(), 0.0);
    assert!(!e.gross_is_measured());
    assert!(!e.net_is_measured());
}

/// Control: un cero MEDIDO sigue siendo cero real, nunca N/D.
#[test]
fn measured_zero_stays_zero() {
    let e = entry(0.0, 0.0, 0.0, 0.0);
    assert_eq!(e.gross_pnl(), 0.0);
    assert_eq!(e.fees(), 0.0);
    assert_eq!(e.funding(), 0.0);
    assert_eq!(e.slippage(), 0.0);
    assert_eq!(e.net_pnl(), 0.0);
    assert!(e.net_is_measured());
}

/// Los cinco importes se registran y el neto se deriva, no se inventa.
#[test]
fn all_five_components_are_recorded_and_net_is_derived() {
    let e = entry(150.0, 15.0, 2.0, 3.0);
    assert_eq!(e.gross_pnl(), 150.0);
    assert_eq!(e.fees(), 15.0);
    assert_eq!(e.funding(), 2.0);
    assert_eq!(e.slippage(), 3.0);
    assert_eq!(e.net_pnl(), 130.0);
    assert_eq!(e.symbol(), "BTCUSDT");
    assert_eq!(e.timestamp_ns(), TS);
}

/// Sin heurística de signos: funding recibido y rebates son datos medidos.
#[test]
fn negative_funding_and_rebate_are_measured_data() {
    let e = entry(90.0, -0.5, -1.5, 0.5);
    assert_eq!(e.funding(), -1.5);
    assert_eq!(e.fees(), -0.5);
    assert_eq!(e.net_pnl(), 91.5);
    assert!(e.net_is_measured());
}

/// Desborde del neto: ausencia, no pérdida infinita.
#[test]
fn net_overflow_is_absence_not_infinite_loss() {
    let e = entry(-f64::MAX, f64::MAX, 0.0, 0.0);
    assert_eq!(e.gross_pnl(), -f64::MAX);
    assert_eq!(e.fees(), f64::MAX);
    assert!(e.net_pnl().is_nan());
}

/// Ausencia independiente por campo: fees/funding/slippage no medidos no
/// crean un beneficio sin costes.
#[test]
fn absent_cost_is_nd_not_free_profit() {
    for (fees, funding, slippage) in [
        (f64::NAN, 0.0, 0.0),
        (0.0, f64::INFINITY, 0.0),
        (0.0, 0.0, f64::NEG_INFINITY),
    ] {
        let e = entry(150.0, fees, funding, slippage);
        assert_eq!(e.gross_pnl(), 150.0);
        assert!(e.net_pnl().is_nan());
    }
}

/// Capital real de la wallet desde el llamador; guarda de división por cero.
/// Los importes son fixtures sintéticos de comportamiento, NO un balance
/// fijo: el mismo neto produce ROI distinto con cada capital recibido.
#[test]
fn roi_uses_real_wallet_capital_and_guards_division_by_zero() {
    let mut ledger = TradeLedger::new();
    ledger.append(entry(150.0, 15.0, 2.0, 3.0));
    let capital_a = 650.0;
    let capital_b = 1300.0;
    assert_eq!(ledger.roi_pct(capital_a), 20.0);
    assert_eq!(ledger.roi_pct(capital_b), 10.0);
    assert!(ledger.roi_pct(0.0).is_nan());
    assert!(ledger.roi_pct(-1.0).is_nan());
    assert!(ledger.roi_pct(f64::NAN).is_nan());
    assert!(ledger.roi_pct(f64::INFINITY).is_nan());
}

/// Sin operaciones, los agregados son N/D; un asiento ausente contamina
/// SÓLO su campo agregado y no publica una suma parcial como total medido.
#[test]
fn empty_totals_are_nd_and_aggregates_fail_closed() {
    let empty = TradeLedger::new();
    assert!(empty.is_empty());
    let t = empty.totals();
    assert_eq!(t.operations, 0);
    assert!(t.gross_pnl.is_nan());
    assert!(t.fees.is_nan());
    assert!(t.funding.is_nan());
    assert!(t.slippage.is_nan());
    assert!(t.net_pnl.is_nan());

    let mut ledger = TradeLedger::new();
    ledger.append(entry(f64::NAN, 15.0, 0.0, 0.0));
    ledger.append(entry(150.0, 15.0, 2.0, 3.0));
    assert_eq!(ledger.len(), 2);
    let totals = ledger.totals();
    assert_eq!(totals.operations, 2);
    assert!(totals.gross_pnl.is_nan());
    assert!(totals.net_pnl.is_nan());
    assert_eq!(totals.fees, 30.0);
    assert_eq!(totals.funding, 2.0);
    assert_eq!(totals.slippage, 3.0);
    assert!(ledger.roi_pct(1000.0).is_nan());
}

/// El cable representa N/D como `None`, nunca como `null`→`0.00`, y un cero
/// medido sigue siendo `Some(0.0)`. Cubre el hallazgo de serde NaN→null.
#[test]
fn wire_represents_nd_as_none_and_measured_zero_as_some() {
    let absent = entry(f64::NAN, 15.0, 0.0, 0.0);
    let w = absent.wire_amounts();
    assert_eq!(w.gross_pnl, None);
    assert_eq!(w.net_pnl, None);
    assert_eq!(w.fees, Some(15.0));
    assert_eq!(w.funding, Some(0.0));
    assert_eq!(w.slippage, Some(0.0));

    let zero = entry(0.0, 0.0, 0.0, 0.0);
    let wz = zero.wire_amounts();
    assert_eq!(wz.gross_pnl, Some(0.0));
    assert_eq!(wz.net_pnl, Some(0.0));

    let mut ledger = TradeLedger::new();
    ledger.append(absent);
    let tw = ledger.totals().wire_amounts();
    assert_eq!(tw.gross_pnl, None);
    assert_eq!(tw.net_pnl, None);
    assert_eq!(tw.fees, Some(15.0));
}

/// El neto del asiento SIEMPRE es coherente con sus cuatro componentes: el
/// único constructor disponible deriva el neto, así que no se puede inyectar
/// un beneficio fabricado. Cubre el hallazgo de invariante no forzado.
#[test]
fn entry_net_is_always_consistent_with_its_components() {
    let cases = [
        (150.0, 15.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 0.0),
        (90.0, -0.5, -1.5, 0.5),
        (f64::NAN, 15.0, 0.0, 0.0),
        (150.0, f64::NAN, 0.0, 0.0),
        (-f64::MAX, f64::MAX, 0.0, 0.0),
    ];
    for (g, f, u, s) in cases {
        let e = entry(g, f, u, s);
        let measured = e.gross_pnl().is_finite()
            && e.fees().is_finite()
            && e.funding().is_finite()
            && e.slippage().is_finite();
        if measured {
            let expected = e.gross_pnl() - e.fees() - e.funding() - e.slippage();
            assert!(
                e.net_pnl().is_nan() || e.net_pnl() == expected,
                "neto inconsistente con sus componentes"
            );
        } else {
            assert!(e.net_pnl().is_nan(), "con datos ausentes el neto es N/D");
        }
    }
}
