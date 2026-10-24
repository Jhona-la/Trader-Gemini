//! LG-33 (Línea G, ola 10; hallazgo LG10-EXE-02, confirmado por un escéptico):
//! la discrepancia «evidenciada» del Auditor Interno se publicaba con UN solo
//! fill del canal independiente (`tiene_fuente_independiente` = canal no
//! vacío). En vivo ese canal sólo recibe los cierres por bracket (TP/SL algo)
//! y lo comparaba con el WR de todos los cierres del núcleo: el primer SL por
//! bracket daba WR independiente 0/1, discrepancia |0,6 − 0| = 0,6 > 0,20 y
//! el consejo vetaba la moneda. Como el valor sólo se recalcula cuando esa
//! moneda cierra, y el veto le impide abrir, el veto era absorbente.
//!
//! La discrepancia queda sujeta a la misma puerta de evidencia que las sombras
//! del auditor (`MIN_FILLS_SOMBRA` cierres independientes): con menos no hay
//! evidencia y no se publica nada.

use audit_engine::auditor_interno::{AuditorInterno, ExchangeFillRecord, MIN_FILLS_SOMBRA};

fn cierre(i: usize, pnl: f64) -> ExchangeFillRecord {
    ExchangeFillRecord {
        timestamp_ns: 1_700_000_000_000_000_000 + i as u64 * 1_000_000_000,
        symbol_id: 0,
        is_buyer: true,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: 0.0,
        realized_pnl: pnl,
        is_closed_position: true,
    }
}

#[test]
fn lg33_un_solo_cierre_independiente_no_publica_discrepancia() {
    let mut a = AuditorInterno::with_capital(13.0);
    a.record_fill_independiente(cierre(0, -0.2)); // primer SL por bracket
    assert_eq!(
        a.compute_internal_discrepancy_evidenciada(0.0, 0.6, 0.0),
        None,
        "con 1 cierre independiente no hay evidencia: no se publica discrepancia"
    );
}

#[test]
fn lg33_la_puerta_se_abre_exactamente_en_min_fills_sombra() {
    let mut a = AuditorInterno::with_capital(13.0);
    for i in 0..MIN_FILLS_SOMBRA - 1 {
        a.record_fill_independiente(cierre(i, if i % 2 == 0 { 0.5 } else { -0.2 }));
    }
    assert_eq!(a.compute_internal_discrepancy_evidenciada(0.0, 0.9, 0.0), None, "n = MIN − 1");
    a.record_fill_independiente(cierre(MIN_FILLS_SOMBRA - 1, 0.5));
    let d = a.compute_internal_discrepancy_evidenciada(0.0, 0.9, 0.0);
    assert!(d.is_some_and(f64::is_finite), "n = MIN: hay evidencia ({d:?})");
}

/// Los fills que no cierran posición no cuentan como evidencia de win rate.
#[test]
fn lg33_fills_sin_cierre_no_cuentan() {
    let mut a = AuditorInterno::with_capital(13.0);
    for i in 0..(3 * MIN_FILLS_SOMBRA) {
        let mut f = cierre(i, 0.0);
        f.is_closed_position = false;
        a.record_fill_independiente(f);
    }
    assert_eq!(a.compute_internal_discrepancy_evidenciada(0.0, 0.9, 0.0), None);
}
