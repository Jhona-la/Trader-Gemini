//! SOL (2026-10-20, censo C-F7b §3-4): LA PATA DE WIN RATE DEL VETO SE MIDE
//! CON SU INTERVALO. Con evidencia suficiente (n ≥ MIN_FILLS_SOMBRA) la
//! discrepancia de WR entra al máximo de las tres patas por su COTA INFERIOR
//! (`|disc| − ee` por bloques): el ruido de muestreo deja de vetar y sólo
//! veta la discrepancia que sobrevive a su propio intervalo. Con menos
//! cierres, la pata queda puntual (comportamiento previo): el ajuste no
//! inventa precisión que no existe.
//!
//! Cifras exactas (verificables a mano):
//! - 24 cierres alternados (+0,5 / −0,2, fee 0) ⇒ WR real 0,5, bloques
//!   homogéneos ⇒ ee = 0 ⇒ la pata NO cambia (disc 0,4 exacto con claim 0,9).
//! - 24 cierres agrupados (12 wins, luego 12 loses) ⇒ bloques [1,1,0,0] ⇒
//!   ee = √(1/12) ≈ 0,2887. Claim 0,9 ⇒ pata 0,4 ⇒ cota 0,4 − 0,2887 =
//!   0,1113 < 0,20 ⇒ NO veta. Claim 1,0 ⇒ pata 0,5 ⇒ cota 0,2113 > 0,20 ⇒
//!   veta (la discrepancia real sobrevive a su ruido).
//! - 19 cierres ⇒ sin sombra (n < 20) ⇒ pata puntual 0,4 (como antes).

use audit_engine::auditor_interno::{
    evaluate_tribunal_verdict, AuditorInterno, ExchangeFillRecord, TribunalThresholds,
    TribunalVerdict, MIN_FILLS_SOMBRA,
};

fn cierre(ts_ns: u64, pnl: f64) -> ExchangeFillRecord {
    ExchangeFillRecord {
        timestamp_ns: ts_ns,
        symbol_id: 0,
        is_buyer: false,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: 0.0,
        realized_pnl: pnl,
        is_closed_position: true,
    }
}

/// 24 cierres alternados: bloques homogéneos ⇒ ee = 0.
fn alternados() -> AuditorInterno {
    let mut a = AuditorInterno::new();
    for i in 0..24u64 {
        let pnl = if i % 2 == 0 { 0.5 } else { -0.2 };
        a.record_fill_independiente(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, pnl));
    }
    a
}

/// 24 cierres agrupados (12 wins, luego 12 loses): bloques [1,1,0,0] ⇒ ee = √(1/12).
fn agrupados() -> AuditorInterno {
    let mut a = AuditorInterno::new();
    for i in 0..24u64 {
        let pnl = if i < 12 { 0.5 } else { -0.2 };
        a.record_fill_independiente(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, pnl));
    }
    a
}

const EE_AGRUPADO: f64 = 0.288_675_134_594_812_9; // √(1/12)

#[test]
fn ee_cero_no_cambia_la_pata_puntual() {
    let a = alternados();
    let disc = a
        .compute_internal_discrepancy_evidenciada(0.0, 0.9, 0.0)
        .expect("canal con evidencia");
    assert!(
        (disc - 0.4).abs() < 1e-12,
        "con ee 0 la pata es puntual: 0,4 exacto, obtuvo {disc}"
    );
}

#[test]
fn el_ruido_de_muestreo_ya_no_veta() {
    let a = agrupados();
    let disc = a
        .compute_internal_discrepancy_evidenciada(0.0, 0.9, 0.0)
        .expect("canal con evidencia");
    let esperado = 0.4 - EE_AGRUPADO;
    assert!(
        (disc - esperado).abs() < 1e-9,
        "cota inferior = 0,4 − √(1/12) ≈ {esperado}, obtuvo {disc}"
    );
    assert!(disc < 0.20, "0,1113 < 0,20 ⇒ el ruido no dispara autoengaño");
    // 0,1113 cae entre recuperable (0,05) y autoengaño (0,20): queda
    // clasificado como recuperable, pero NO como autoengaño.
    let v = evaluate_tribunal_verdict(disc, &TribunalThresholds::default());
    assert_eq!(
        v,
        TribunalVerdict::CulpableRecuperable,
        "ruido de muestreo ⇒ recuperable, jamás autoengaño"
    );
    assert_ne!(v, TribunalVerdict::AutoengañoDetectado);
}

#[test]
fn la_discrepancia_real_sobrevive_a_su_ruido_y_veta() {
    let a = agrupados();
    let disc = a
        .compute_internal_discrepancy_evidenciada(0.0, 1.0, 0.0)
        .expect("canal con evidencia");
    let esperado = 0.5 - EE_AGRUPADO;
    assert!(
        (disc - esperado).abs() < 1e-9,
        "cota inferior = 0,5 − √(1/12) ≈ {esperado}, obtuvo {disc}"
    );
    assert!(disc > 0.20, "0,2113 > 0,20 ⇒ veta");
    let v = evaluate_tribunal_verdict(disc, &TribunalThresholds::default());
    assert_eq!(v, TribunalVerdict::AutoengañoDetectado, "discrepancia real ⇒ veta");
}

#[test]
fn con_menos_de_n_min_la_pata_queda_puntual() {
    let mut a = AuditorInterno::new();
    for i in 0..19u64 {
        let pnl = if i < 10 { 0.5 } else { -0.2 };
        a.record_fill_independiente(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, pnl));
    }
    assert!(a.sombra_wr_disc(0.9).is_none(), "19 < MIN_FILLS_SOMBRA ⇒ sin sombra");
    let disc = a
        .compute_internal_discrepancy_evidenciada(0.0, 0.9, 0.0)
        .expect("canal con fills");
    // WR real 10/19 ≈ 0,5263 ⇒ pata |0,9 − 0,5263| ≈ 0,3737 SIN ajuste.
    let esperado = (0.9 - 10.0 / 19.0).abs();
    assert!(
        (disc - esperado).abs() < 1e-9,
        "sin evidencia suficiente no hay ajuste: {esperado}, obtuvo {disc}"
    );
    assert_eq!(MIN_FILLS_SOMBRA, 20, "el umbral del contrato");
}

#[test]
fn las_otras_patas_siguen_dominando_cuando_son_mayores() {
    let a = agrupados();
    // Sharpe interno 10 contra un sharpe real de pocos céntimos ⇒
    // sharpe_disc ≈ 1,0 domina a la cota de WR (0,1113).
    let disc = a
        .compute_internal_discrepancy_evidenciada(10.0, 0.9, 0.0)
        .expect("canal con evidencia");
    assert!(
        disc > 0.5,
        "la pata de sharpe domina a la cota de WR (0,1113): {disc}"
    );
}

#[test]
fn claim_honesto_no_veta_ni_con_ruido() {
    let a = agrupados();
    let disc = a
        .compute_internal_discrepancy_evidenciada(0.0, 0.5, 0.0)
        .expect("canal con evidencia");
    assert_eq!(disc, 0.0, "pata 0, cota max(0 − ee, 0) = 0");
}

#[test]
fn la_cota_nunca_es_negativa() {
    let a = agrupados();
    // Claim 0,2 contra real 0,5: pata 0,3; cota 0,3 − 0,2887 = 0,0113 ≥ 0.
    let disc = a
        .compute_internal_discrepancy_evidenciada(0.0, 0.2, 0.0)
        .expect("canal con evidencia");
    assert!(
        (disc - (0.3 - EE_AGRUPADO)).abs() < 1e-9 && disc >= 0.0,
        "cota = 0,3 − ee, obtuvo {disc}"
    );
}
