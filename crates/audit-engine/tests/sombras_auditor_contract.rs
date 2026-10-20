//! SOL (2026-10-19, censo C-F7b §3-2): SOMBRAS DEL AUDITOR INTERNO —
//! `sombra_wr_disc` (pata de win rate, F413) y `sombra_friccion_real_menos_modelada_bp`
//! (fricción real vs modelada por el gate).
//!
//! Doctrina (buzón 2026-10-19, acuse de cierre PR #84):
//! - Sombra = telemetría: ninguna decisión la consume hoy. Mide, con
//!   intervalo, lo que antes iba al veto sin evidencia.
//! - Con intervalo: media ± error estándar por BLOQUES (los fills son
//!   serie correlacionada: no vale el binomial iid).
//! - Con n_min por fuente: menos de MIN_FILLS_SOMBRA cierres/fills ⇒ `None`.
//!   Ausencia de dato nunca es señal.
//! - Con dos mitades OOS: la sombra publica su valor en cada mitad temporal;
//!   la estabilidad se verifica, no se asume.
//! - Con guardas de estabilidad numérica: claims no finitos se rechazan
//!   (`None`), fills con nocional 0 o fee no finito se ignoran; nada de NaN
//!   ni infinitos dentro del motor.
//!
//! Cifras exactas (verificables a mano):
//! - fill cn precio 50 000 y qty 0,001 ⇒ nocional 50; fee 0,025 ⇒ 5 bp.
//! - 24 cierres alternados (12 wins +0,5 / 12 loses −0,2, fee 0) ⇒ WR 0,5.
//! - interno 0,9 vs real 0,5 ⇒ disc 0,4 exacto en global y en cada mitad.
//! - alternados ⇒ cada bloque proporción 0,5 ⇒ varianza 0 ⇒ EE 0 exacto.
//! - agrupados (12 wins luego 12 loses) ⇒ bloques [1,1,0,0] ⇒ sd 1/√3,
//!   EE = √(var/4) con var muestral = 1/3 ⇒ EE = √(1/12).
//! - fricción: 5 bp reales vs 4 bp modelo ⇒ media +1 bp exacto.

use audit_engine::auditor_interno::{
    ExchangeFillRecord, AuditorInterno, MIN_FILLS_SOMBRA,
};

/// Cierre del canal independiente: nocional 50 (50 000 × 0,001).
fn cierre(ts_ns: u64, pnl: f64, fee: f64) -> ExchangeFillRecord {
    ExchangeFillRecord {
        timestamp_ns: ts_ns,
        symbol_id: 0,
        is_buyer: false,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: fee,
        realized_pnl: pnl,
        is_closed_position: true,
    }
}

/// Fill de apertura (no cierra): para la sombra de fricción cuenta toda
/// ejecución; para la de WR sólo los cierres.
fn apertura(ts_ns: u64, fee: f64) -> ExchangeFillRecord {
    ExchangeFillRecord {
        timestamp_ns: ts_ns,
        symbol_id: 0,
        is_buyer: false,
        price: 50_000.0,
        qty: 0.001,
        fee_usd: fee,
        realized_pnl: 0.0,
        is_closed_position: false,
    }
}

/// 24 cierres alternados ganar/perder: WR real 0,5 y bloques homogéneos.
fn auditor_alternado() -> AuditorInterno {
    let mut a = AuditorInterno::new();
    for i in 0..24u64 {
        let (pnl, fee) = if i % 2 == 0 { (0.5, 0.0) } else { (-0.2, 0.0) };
        a.record_fill_independiente(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, pnl, fee));
    }
    a
}

#[test]
fn sin_canal_las_sombras_no_se_publican() {
    let a = AuditorInterno::new();
    assert!(a.sombra_wr_disc(0.9).is_none(), "sin cierres no hay sombra WR");
    assert!(
        a.sombra_friccion_bp(4.0).is_none(),
        "sin fills no hay sombra de fricción"
    );
}

#[test]
fn wr_disc_exacto_con_mitades_y_ee_alternado() {
    let a = auditor_alternado();
    let s = a.sombra_wr_disc(0.9).expect("24 cierres ≥ MIN_FILLS_SOMBRA");
    assert_eq!(s.n, 24);
    assert!((s.wr_real - 0.5).abs() < 1e-12, "WR real 0,5 exacto, obtuvo {}", s.wr_real);
    assert!((s.disc - 0.4).abs() < 1e-12, "interno 0,9 vs real 0,5 ⇒ 0,4 exacto");
    assert!((s.disc_mitad_a - 0.4).abs() < 1e-12, "mitad A estable: {}", s.disc_mitad_a);
    assert!((s.disc_mitad_b - 0.4).abs() < 1e-12, "mitad B estable: {}", s.disc_mitad_b);
    assert!(
        s.ee_real.abs() < 1e-12,
        "alternados ⇒ bloques homogéneos ⇒ EE 0 exacto, obtuvo {}",
        s.ee_real
    );
}

#[test]
fn wr_disc_honesto_y_ee_no_cero_con_series_agrupadas() {
    // Claim honesto vs el mismo canal ⇒ disc ≈ 0 (la sombra confirma, no acusa).
    let a = auditor_alternado();
    let s = a.sombra_wr_disc(0.5).expect("canal suficiente");
    assert!(s.disc.abs() < 1e-12, "afirmación honesta ⇒ disc 0, obtuvo {}", s.disc);

    // 12 wins seguidos de 12 loses: bloques [1,1,0,0] ⇒ var muestral 1/3 ⇒
    // EE = √((1/3)/4) = √(1/12). El EE mide la correlación, no la proporción.
    let mut b = AuditorInterno::new();
    for i in 0..24u64 {
        let pnl = if i < 12 { 0.5 } else { -0.2 };
        b.record_fill_independiente(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, pnl, 0.0));
    }
    let s = b.sombra_wr_disc(0.9).expect("canal suficiente");
    assert!((s.disc - 0.4).abs() < 1e-12);
    let ee_esperado = (1.0_f64 / 12.0).sqrt();
    assert!(
        (s.ee_real - ee_esperado).abs() < 1e-9,
        "EE por bloques = √(1/12) ≈ {}, obtuvo {}",
        ee_esperado,
        s.ee_real
    );
}

#[test]
fn n_min_y_guardas_de_estabilidad_numerica() {
    // 19 cierres (< 20) ⇒ None aunque el claim sea inflado.
    let mut a = AuditorInterno::new();
    for i in 0..19u64 {
        let pnl = if i % 2 == 0 { 0.5 } else { -0.2 };
        a.record_fill_independiente(cierre(1_700_000_000_000_000_000 + i * 1_000_000_000, pnl, 0.0));
    }
    assert!(a.sombra_wr_disc(0.9).is_none(), "19 < MIN_FILLS_SOMBRA ⇒ sin sombra");
    assert!(a.sombra_friccion_bp(4.0).is_none());

    // Claim interno no finito ⇒ None (jamás claim inventado).
    let b = auditor_alternado();
    assert!(b.sombra_wr_disc(f64::NAN).is_none(), "WR interno NaN ⇒ None");
    assert!(b.sombra_wr_disc(f64::INFINITY).is_none(), "WR interno ∞ ⇒ None");

    // Claim interno fuera de rango se satura a [0,1] (no se inventa, se acota).
    let s = b.sombra_wr_disc(1.5).expect("canal suficiente");
    assert!((s.wr_interno - 1.0).abs() < 1e-12, "1,5 satura a 1,0");
    assert!((s.disc - 0.5).abs() < 1e-12, "|1 − 0,5| = 0,5");

    // Fricción modelo no finita o negativa ⇒ None.
    assert!(b.sombra_friccion_bp(f64::NAN).is_none());
    assert!(b.sombra_friccion_bp(-1.0).is_none());
}

#[test]
fn friccion_real_menos_modelada_exacta() {
    // 24 fills de 5 bp reales vs 4 bp modelo ⇒ media +1 bp, EE 0 (homogéneo).
    let mut a = AuditorInterno::new();
    for i in 0..24u64 {
        a.record_fill_independiente(apertura(1_700_000_000_000_000_000 + i * 1_000_000_000, 0.025));
    }
    let f = a.sombra_friccion_bp(4.0).expect("24 fills ≥ MIN_FILLS_SOMBRA");
    assert_eq!(f.n, 24);
    assert!((f.media_bp - 1.0).abs() < 1e-9, "5 bp − 4 bp = +1 bp exacto, obtuvo {}", f.media_bp);
    assert!(f.ee_bp.abs() < 1e-9, "homogéneo ⇒ EE 0, obtuvo {}", f.ee_bp);
    assert!((f.media_mitad_a_bp - 1.0).abs() < 1e-9, "mitad A = +1 bp");
    assert!((f.media_mitad_b_bp - 1.0).abs() < 1e-9, "mitad B = +1 bp");
}

#[test]
fn friccion_descarta_fills_insalvables_sin_contaminar() {
    // 22 válidos de 5 bp + precio 0 (nocional 0) + fee NaN + fee negativa:
    // los insalvables se ignoran; los 22 válidos sostienen la sombra.
    let mut a = AuditorInterno::new();
    for i in 0..22u64 {
        a.record_fill_independiente(apertura(1_700_000_000_000_000_000 + i * 1_000_000_000, 0.025));
    }
    let mut roto_precio = apertura(1_700_000_000_100_000_000, 0.025);
    roto_precio.price = 0.0;
    a.record_fill_independiente(roto_precio);
    let mut roto_fee = apertura(1_700_000_000_200_000_000, f64::NAN);
    roto_fee.fee_usd = f64::NAN;
    a.record_fill_independiente(roto_fee);
    let mut roto_fee_neg = apertura(1_700_000_000_300_000_000, 0.025);
    roto_fee_neg.fee_usd = -1.0;
    a.record_fill_independiente(roto_fee_neg);

    let f = a.sombra_friccion_bp(4.0).expect("22 válidos ≥ MIN_FILLS_SOMBRA");
    assert_eq!(f.n, 22, "sólo los fills salvables cuentan");
    assert!((f.media_bp - 1.0).abs() < 1e-9, "los insalvables no contaminan la media");
    assert!(f.media_bp.is_finite() && f.ee_bp.is_finite(), "nada de NaN en la sombra");
}

#[test]
fn friccion_mitades_divergentes_se_ven_en_la_sombra() {
    // 12 fills de 6 bp + 12 de 2 bp vs modelo 4 ⇒ media 0 pero mitades +2/−2:
    // la divergencia OOS queda expuesta en lugar de promediarse.
    let mut a = AuditorInterno::new();
    for i in 0..24u64 {
        let fee = if i < 12 { 0.03 } else { 0.01 }; // 6 bp y 2 bp
        a.record_fill_independiente(apertura(1_700_000_000_000_000_000 + i * 1_000_000_000, fee));
    }
    let f = a.sombra_friccion_bp(4.0).expect("canal suficiente");
    assert!(f.media_bp.abs() < 1e-9, "media 0 exacto, obtuvo {}", f.media_bp);
    assert!((f.media_mitad_a_bp - 2.0).abs() < 1e-9, "mitad A +2 bp, obtuvo {}", f.media_mitad_a_bp);
    assert!((f.media_mitad_b_bp + 2.0).abs() < 1e-9, "mitad B −2 bp, obtuvo {}", f.media_mitad_b_bp);
    assert!(f.ee_bp > 1.0, "la serie agrupada tiene EE alto, obtuvo {}", f.ee_bp);
}

#[test]
fn wr_solo_mira_cierres_no_aperturas() {
    // 24 aperturas con PnL positivo (NO cierran) + 0 cierres ⇒ la sombra WR
    // es None: una apertura no es evidencia de resultado.
    let mut a = AuditorInterno::new();
    for i in 0..24u64 {
        a.record_fill_independiente(apertura(1_700_000_000_000_000_000 + i * 1_000_000_000, 0.0));
    }
    assert!(a.sombra_wr_disc(0.9).is_none(), "sin cierres no hay WR que medir");
    // Pero la de fricción SÍ ve esas aperturas (toda ejecución paga).
    assert!(a.sombra_friccion_bp(4.0).is_some());
}

#[test]
fn min_constante_coincide_con_el_contrato_del_buzon() {
    assert_eq!(MIN_FILLS_SOMBRA, 20, "n_min por fuente = 20 (buzón §3-2)");
}
