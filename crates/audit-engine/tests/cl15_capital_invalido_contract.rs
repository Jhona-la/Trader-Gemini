//! CL-15 (GLM 144): el capital ausente NO se inventa.
//!
//! Erradicación de los fallbackos que reinyectaban 13.0 ante capital
//! inválido: la evidencia no anclable se declara (métricas vacías, tribunal
//! armado — consistente con F412: sin fuente independiente el veto no decae),
//! jamás se fabrican retornos de una cuenta inexistente.
use audit_engine::auditor_interno::{AuditorInterno, ExchangeFillRecord, VerificadorResultados};

fn cierre(ts: u64, pnl: f64) -> ExchangeFillRecord {
    ExchangeFillRecord {
        timestamp_ns: ts,
        symbol_id: 0,
        is_buyer: true,
        price: 100.0,
        qty: 1.0,
        fee_usd: 0.0,
        realized_pnl: pnl,
        is_closed_position: true,
    }
}

/// Capital inválido (NaN/0/negativo) ⇒ evidencia NO anclable ⇒ métricas
/// vacías, idéntico a fills vacíos. El 13.0 inventado fabricaría win_rate,
/// sharpe y drawdown de una cuenta que no existe.
#[test]
fn cl15_capital_invalido_deja_metricas_vacias() {
    let fills = vec![cierre(1, 0.5), cierre(2, -0.2), cierre(3, 0.5)];
    for capital_malo in [f64::NAN, 0.0, -5.0, f64::INFINITY * -1.0] {
        let m = VerificadorResultados::compute_metrics_with_capital(&fills, capital_malo);
        assert_eq!(m.calculated_sharpe, 0.0, "sin capital no hay sharpe");
        assert_eq!(m.calculated_win_rate, 0.0, "sin capital no hay wr anclado");
        assert_eq!(m.calculated_max_drawdown, 0.0, "sin capital no hay dd");
    }
    // La MISMA evidencia con capital válido sí ancla (2 de 3 cierres netos > 0).
    let m = VerificadorResultados::compute_metrics_with_capital(&fills, 100.0);
    assert!(
        (m.calculated_win_rate - 2.0 / 3.0).abs() < 1e-9,
        "con capital real el wr es el de los fills: {}",
        m.calculated_win_rate
    );
}

/// El Default de AuditorInterno NO declara capital: sin `with_capital`
/// explícito, la métrica del canal independiente queda sin anclar aunque
/// existan fills — el host vivo debe declarar el capital real de la API.
#[test]
fn cl15_default_no_declara_capital_las_metricas_no_anclan() {
    let mut a = AuditorInterno::default();
    for i in 0..5u64 {
        a.record_fill_independiente(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, 0.5));
    }
    let m = a.compute_independent_metrics_from_host();
    assert_eq!(
        m.calculated_win_rate, 0.0,
        "sin with_capital la evidencia del canal no ancla (ausencia honesta)"
    );
    // Con capital explícito la misma evidencia ancla (5/5 ganadores).
    let mut b = AuditorInterno::with_capital(100.0);
    for i in 0..5u64 {
        b.record_fill_independiente(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, 0.5));
    }
    let mb = b.compute_independent_metrics_from_host();
    assert!(
        (mb.calculated_win_rate - 1.0).abs() < 1e-9,
        "con capital declarado el wr ancla: {}",
        mb.calculated_win_rate
    );
}
