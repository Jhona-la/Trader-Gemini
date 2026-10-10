//! QS-R2 — SIMETRÍA ESPEJO DEL CONSEJO (contrato G-TEO §30.1, punto 3).
//!
//! Si el mercado se refleja (largo ↔ corto), cada asiento debe dar la opinión
//! reflejada: misma confianza, misma decisión de veto y dirección opuesta. Una
//! asimetría sin un dato asimétrico medido que la justifique es un sesgo
//! estructural contra un lado (caso real: #708 penaliza cortos por la
//! semántica de su entrada; ver §30 del plan de sincronización).
//!
//! Espejo de la carga:
//! - firmadas, se niegan: `intended_direction`, `book_imbalance`,
//!   `fused_score`, `yang_mills_current`, `prospect_pressure`;
//! - cocientes compra/venta, se invierten: `crowd_ls_ratio`,
//!   `crowd_taker_ratio`;
//! - magnitudes y regímenes sin lado, no cambian: Hurst, persistencia (CL-30:
//!   sin lado), ATR, drawdown, slippage, τ, ballena (|z|), liquidaciones, OI,
//!   spoofing, Hodge, Reynolds, frescura macro.
//!
//! `ml_prob` queda FUERA: la carga sólo trae P(TP del LARGO antes que SL), y su
//! espejo (la del corto) no se puede derivar de ella. Por eso aquí vale su base
//! en los dos lados (el asiento ML se abstiene). Que el consejo juzgue un corto
//! con la probabilidad del largo es una limitación de diseño documentada en
//! el ledger QS-R1, no algo que este test oculte.

use metacortex_engine::consejo_seniors::{
    ConsejoDeliberacion, MarketSnapshotPayload, SeniorOpinion, TradingHorizon,
};

const TOL: f64 = 1e-9;

fn espejo(p: &MarketSnapshotPayload) -> MarketSnapshotPayload {
    let mut m = p.clone();
    m.intended_direction = -p.intended_direction;
    m.book_imbalance = -p.book_imbalance;
    m.fused_score = -p.fused_score;
    m.yang_mills_current = -p.yang_mills_current;
    m.prospect_pressure = -p.prospect_pressure;
    m.crowd_ls_ratio = 1.0 / p.crowd_ls_ratio;
    m.crowd_taker_ratio = 1.0 / p.crowd_taker_ratio;
    m
}

/// Rejilla de cargas con signos y magnitudes variados en todas las entradas
/// firmadas; el resto en valores vivos plausibles.
fn casos() -> Vec<MarketSnapshotPayload> {
    let mut out = Vec::new();
    for &obi in &[0.40, -0.30, 0.0] {
        for &fused in &[0.55, -0.20] {
            for &pers in &[0.35, -0.35] {
                for &ym in &[0.40, -0.10] {
                    for &(ls, tk) in &[(2.5, 1.6), (0.5, 0.7)] {
                        for &pk in &[-3.0, 2.0, 0.0] {
                            // 140 pb ejerce el veto del asiento de ejecución
                            // (límite 35..100 pb según τ).
                            for &slip in &[12.0, 140.0] {
                                out.push(MarketSnapshotPayload {
                                    horizon: TradingHorizon::Continuous,
                                    book_imbalance: obi,
                                    hurst_exponent: 0.62,
                                    ml_prob: 0.25,
                                    fused_score: fused,
                                    persistence: pers,
                                    atr_pct: 0.002,
                                    loss_streak: 1,
                                    intended_direction: 1.0,
                                    do_calculus_risk: 0.10,
                                    causal_veto_threshold: 0.75,
                                    current_drawdown_pct: 0.02,
                                    estimated_slippage_bps: slip,
                                    dominant_tau_ms: 300_000.0,
                                    whale_burst_z: 2.0,
                                    liquidation_severity: 0.2,
                                    open_interest_norm: 0.4,
                                    spoof_score: 0.1,
                                    crowd_ls_ratio: ls,
                                    crowd_taker_ratio: tk,
                                    ml_model_base: 0.25,
                                    hodge_curl_share: 0.3,
                                    yang_mills_current: ym,
                                    macro_staleness_ms: 5_000,
                                    navier_reynolds_number: 1.5,
                                    navier_laminar_share: 0.3,
                                    prospect_pressure: pk,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn por_rol<'a>(ops: &'a [SeniorOpinion], o: &SeniorOpinion) -> &'a SeniorOpinion {
    ops.iter()
        .find(|x| x.role == o.role)
        .expect("el espejo debe tener los mismos asientos")
}

#[test]
fn qs_r2_cada_asiento_da_la_opinion_espejada() {
    let consejo = ConsejoDeliberacion::new();
    let mut asimetrias = Vec::new();
    // No trivialidad: el contrato sólo vale si los asientos OPINAN (dirección
    // no nula) y las decisiones varían entre casos.
    let (mut con_direccion, mut aprobados, mut vetados) = (0usize, 0usize, 0usize);
    for (i, p) in casos().iter().enumerate() {
        let q = espejo(p);
        assert!(
            p.validate().is_ok() && q.validate().is_ok(),
            "caso {i}: carga inválida"
        );
        let a = consejo.deliberar_traced(p, 0.55, None);
        let b = consejo.deliberar_traced(&q, 0.55, None);
        assert_eq!(a.opinions.len(), b.opinions.len(), "caso {i}");
        aprobados += a.consensus.approved as usize;
        vetados += a.consensus.vetoed_by.is_some() as usize;
        con_direccion += a
            .opinions
            .iter()
            .filter(|o| o.signal_direction.abs() > 0.05)
            .count();
        for oa in &a.opinions {
            let ob = por_rol(&b.opinions, oa);
            let dir_ok = (oa.signal_direction + ob.signal_direction).abs() < TOL;
            let conf_ok = (oa.confidence - ob.confidence).abs() < TOL;
            let veto_ok = oa.is_veto == ob.is_veto;
            if !(dir_ok && conf_ok && veto_ok) {
                asimetrias.push(format!(
                    "caso {i} {:?}: dir {:+.6}/{:+.6} conf {:.6}/{:.6} veto {}/{}",
                    oa.role,
                    oa.signal_direction,
                    ob.signal_direction,
                    oa.confidence,
                    ob.confidence,
                    oa.is_veto,
                    ob.is_veto
                ));
            }
        }
    }
    let n = casos().len();
    assert!(
        con_direccion >= 2 * n,
        "asientos casi mudos: {con_direccion} opiniones con dirección en {n} casos"
    );
    assert!(
        aprobados > 0 && aprobados < n,
        "la rejilla no varía la aprobación: {aprobados}/{n}"
    );
    assert!(vetados > 0, "la rejilla no ejerce ningún veto");
    eprintln!("QS-R2: {n} casos, {con_direccion} opiniones con dirección, {aprobados} aprobados, {vetados} vetados");
    assert!(
        asimetrias.is_empty(),
        "{} opiniones no simétricas bajo espejo:\n{}",
        asimetrias.len(),
        asimetrias
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn qs_r2_el_consenso_es_espejado() {
    let consejo = ConsejoDeliberacion::new();
    for (i, p) in casos().iter().enumerate() {
        let a = consejo.deliberar_traced(p, 0.55, None).consensus;
        let b = consejo.deliberar_traced(&espejo(p), 0.55, None).consensus;
        assert_eq!(
            a.approved, b.approved,
            "caso {i}: aprobación distinta por lado"
        );
        assert_eq!(a.vetoed_by, b.vetoed_by, "caso {i}: veto distinto por lado");
        assert!(
            (a.final_signal + b.final_signal).abs() < TOL,
            "caso {i}: señal final {:+.6} vs espejo {:+.6}",
            a.final_signal,
            b.final_signal
        );
        assert!(
            (a.total_consensus_pct - b.total_consensus_pct).abs() < TOL,
            "caso {i}: consenso {:.6} vs espejo {:.6}",
            a.total_consensus_pct,
            b.total_consensus_pct
        );
    }
}
