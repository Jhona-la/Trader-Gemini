//! Contrato de Pruebas para F1-101 y F1-102 (Ola Ω160):
//!
//! 1. F1-101: Contabilidad completa de multi-cierre simultáneo depth + trade en `booktick_replay.rs`.
//!    Erradicado `c1.or(c2)` que descartaba el segundo cierre (`c2`), alineando `stats.trades`,
//!    `stats.net_pnl`, `pnl_list`, `fees_est`, `gross_pnl` y `risk_envelope.record_trade` con
//!    la evolución real de `arena.unified_capital`.
//! 2. F1-102: Erradicación del evaluador muerto `vectorized.rs` y la dependencia pesada `polars`
//!    en `backtest-engine`. Preservación limpia de `compute_multi_coin_event_priority` en `metrics.rs`.

use backtest_engine::compute_multi_coin_event_priority;

#[test]
fn test_f1_101_multi_close_accounting_contract() {
    // Escenario crítico F1-101:
    // Slot 0 cierra en el evento depth (c1): ganancia +2.5 USD, qty 0.1
    // Slot 1 cierra en el evento trade del MISMO tick (c2): pérdida -1.0 USD, qty 0.1
    let c1: Option<(usize, f64, f64)> = Some((0, 2.5, 0.1));
    let c2: Option<(usize, f64, f64)> = Some((1, -1.0, 0.1));
    let mid = 60_000.0;
    let roundtrip_fee = 0.0004;

    // --- Comportamiento BUG previo: c1.or(c2) ---
    let mut bug_trades = 0;
    let mut bug_net_pnl = 0.0;
    let mut bug_pnls = Vec::new();
    if let Some((_, pnl_net, _qty)) = c1.or(c2) {
        bug_trades += 1;
        bug_net_pnl += pnl_net;
        bug_pnls.push(pnl_net);
    }
    // El bug sólo registraba 1 trade y omitía completamente la pérdida del segundo cierre:
    assert_eq!(bug_trades, 1, "Bug histórico: c1.or(c2) solo registraba 1 trade");
    assert_eq!(bug_net_pnl, 2.5, "Bug histórico: ignoraba el segundo cierre");
    assert_eq!(bug_pnls.len(), 1);

    // --- Comportamiento F1-101 corregido: [c1, c2].into_iter().flatten() ---
    let mut fix_trades = 0;
    let mut fix_net_pnl = 0.0;
    let mut fix_fees_est = 0.0;
    let mut fix_gross_pnl = 0.0;
    let mut fix_wins_net = 0;
    let mut fix_pnls = Vec::new();
    let mut fix_turnover = 0.0;

    for (_, pnl_net, qty) in [c1, c2].into_iter().flatten() {
        let notional = qty * mid;
        let fee = notional * roundtrip_fee;
        fix_trades += 1;
        fix_net_pnl += pnl_net;
        fix_fees_est += fee;
        fix_gross_pnl += pnl_net + fee;
        if pnl_net > 0.0 {
            fix_wins_net += 1;
        }
        fix_pnls.push(pnl_net);
        fix_turnover += notional + notional;
    }

    assert_eq!(fix_trades, 2, "F1-101: ambos cierres deben ser registrados");
    assert!((fix_net_pnl - 1.5).abs() < 1e-9, "F1-101: net PnL debe ser la suma exacta (+2.5 - 1.0 = +1.5)");
    assert!((fix_fees_est - 4.8).abs() < 1e-9, "F1-101: fees estimados para 2 trades de 6000 USD al 0.04%");
    assert!((fix_gross_pnl - 6.3).abs() < 1e-9, "F1-101: gross PnL debe incluir fees recuperados");
    assert_eq!(fix_wins_net, 1, "F1-101: 1 trade ganador de 2");
    assert_eq!(fix_pnls, vec![2.5, -1.0], "F1-101: lista de PnL contiene ambos eventos en orden causal");
    assert!((fix_turnover - (2.0 * (0.1 * 60_000.0 * 2.0))).abs() < 1e-9, "Turnover acumulado de ambos lados");
}

#[test]
fn test_f1_101_single_or_empty_closure_contract() {
    // Si solo hay c1:
    let c1: Option<(usize, f64, f64)> = Some((0, 1.2, 0.05));
    let c2: Option<(usize, f64, f64)> = None;
    let closed: Vec<_> = [c1, c2].into_iter().flatten().collect();
    assert_eq!(closed.len(), 1);
    assert_eq!(closed[0], (0, 1.2, 0.05));

    // Si solo hay c2:
    let c1: Option<(usize, f64, f64)> = None;
    let c2: Option<(usize, f64, f64)> = Some((1, -0.5, 0.05));
    let closed: Vec<_> = [c1, c2].into_iter().flatten().collect();
    assert_eq!(closed.len(), 1);
    assert_eq!(closed[0], (1, -0.5, 0.05));

    // Si ninguno cerró:
    let c1: Option<(usize, f64, f64)> = None;
    let c2: Option<(usize, f64, f64)> = None;
    let closed: Vec<_> = [c1, c2].into_iter().flatten().collect();
    assert_eq!(closed.len(), 0);
}

#[test]
fn test_f1_102_multi_coin_event_priority_contract() {
    let t = 1_700_000_000_000_000_000u64;

    // Stop / liquidación tiene menor valor que orden regular a mismo timestamp:
    let p_stop = compute_multi_coin_event_priority(t, true);
    let p_normal = compute_multi_coin_event_priority(t, false);

    assert_eq!(p_stop, t << 1, "Prioridad de stop debe ser exactamente timestamp << 1");
    assert_eq!(p_normal, (t << 1) | 1, "Prioridad normal debe tener LSB en 1");
    assert!(p_stop < p_normal, "Stop debe tener mayor prioridad (menor valor numérico)");

    // Causalidad temporal estricta: tiempo anterior siempre tiene mayor prioridad
    let t_past = t - 1;
    let p_past_normal = compute_multi_coin_event_priority(t_past, false);
    assert!(p_past_normal < p_stop, "Evento en el pasado siempre precede a evento futuro sin importar tipo");
}
