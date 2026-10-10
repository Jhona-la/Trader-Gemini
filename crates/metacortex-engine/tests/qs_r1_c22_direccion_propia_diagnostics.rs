//! QS-R1 / C-22 — ¿puede el consejo aprobar una entrada cuyos asientos
//! DIRECCIONALES están, en neto, en contra?
//!
//! `final_signal` promedia TODOS los asientos con señal, incluidos los
//! moduladores Volatilidad (peso 0,9), Riesgo (1,5) y Ente (1,0), que no
//! generan dirección: votan `intended_direction`. La aprobación exige
//! `consenso_direccional ≥ 0,35 ∧ signo(final_signal) = lado`
//! (consejo_seniors.rs:1297-1301). El comentario de 1186-1192 excluye a los
//! moduladores del consenso «o el consejo rubber-stamp-earía su propia
//! entrada», pero siguen decidiendo el signo de `final_signal`. Con
//! moduladores tranquilos suman hasta +3,4 de capacidad a favor del lado
//! pedido.
//!
//! Este diagnóstico busca en una rejilla casos aprobados cuya señal
//! direccional neta (media ponderada de los cinco asientos con dirección
//! propia) es de signo CONTRARIO al lado aprobado. Está marcado `#[ignore]`:
//! documenta un hallazgo ABIERTO (contrato OPEN) sin poner la CI en rojo. Lo
//! cierra quien cambie la regla de aprobación, quitando el `#[ignore]`.

use metacortex_engine::consejo_seniors::{
    ConsejoDeliberacion, MarketSnapshotPayload, SeniorOpinion, SeniorRole, TradingHorizon,
};

fn direccional(o: &SeniorOpinion) -> bool {
    !matches!(
        o.role,
        SeniorRole::Causal
            | SeniorRole::Riesgo
            | SeniorRole::Ejecucion
            | SeniorRole::Volatilidad
            | SeniorRole::AuditorInterno
            | SeniorRole::EnteMercado
    )
}

/// Media ponderada de la dirección de los asientos con dirección propia, con
/// la misma ponderación que usa el consejo (señal · confianza · peso).
fn senal_direccional(ops: &[SeniorOpinion]) -> f64 {
    let (num, den) = ops
        .iter()
        .filter(|o| direccional(o) && o.signal_direction != 0.0 && o.confidence > 0.0)
        .fold((0.0, 0.0), |(n, d), o| {
            (
                n + o.signal_direction * o.confidence * o.weight,
                d + o.confidence * o.weight,
            )
        });
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

fn carga(obi: f64, fused: f64, pers: f64, ml: f64) -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        horizon: TradingHorizon::Continuous,
        book_imbalance: obi,
        hurst_exponent: 0.62,
        ml_prob: ml,
        fused_score: fused,
        persistence: pers,
        atr_pct: 0.0008,
        loss_streak: 0,
        intended_direction: 1.0,
        do_calculus_risk: 0.10,
        causal_veto_threshold: 0.75,
        current_drawdown_pct: 0.0,
        estimated_slippage_bps: 3.0,
        dominant_tau_ms: 300_000.0,
        whale_burst_z: 0.0,
        liquidation_severity: 0.0,
        open_interest_norm: 0.0,
        spoof_score: 0.0,
        crowd_ls_ratio: 1.0,
        crowd_taker_ratio: 1.0,
        ml_model_base: 0.25,
        hodge_curl_share: 0.0,
        yang_mills_current: 0.0,
        macro_staleness_ms: 0,
        navier_reynolds_number: 0.0,
        navier_laminar_share: 1.0,
        prospect_pressure: 0.0,
    }
}

#[test]
#[ignore = "QS-R1 C-22 OPEN: la aprobación usa el signo de una media que incluye moduladores que votan el lado pedido"]
fn qs_r1_c22_no_se_aprueba_un_largo_con_asientos_direccionales_netos_en_contra() {
    let consejo = ConsejoDeliberacion::new();
    let rejilla = [-0.9, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.9];
    let mls = [0.02, 0.10, 0.20, 0.30, 0.45, 0.60];
    let mut contraejemplos = Vec::new();
    let mut aprobados = 0usize;
    for &obi in &rejilla {
        for &fused in &rejilla {
            for &pers in &[0.4, -0.4] {
                for &ml in &mls {
                    let p = carga(obi, fused, pers, ml);
                    let d = consejo.deliberar_traced(&p, 0.55, None);
                    if !d.consensus.approved {
                        continue;
                    }
                    aprobados += 1;
                    let dir = senal_direccional(&d.opinions);
                    if dir < 0.0 {
                        contraejemplos.push(format!(
                            "obi {obi:+.1} fused {fused:+.1} pers {pers:+.1} ml {ml:.2}: \
                             aprobado LARGO con señal direccional {dir:+.3}, \
                             final {:+.3}, consenso largo {:.2}",
                            d.consensus.final_signal, d.consensus.total_consensus_pct
                        ));
                    }
                }
            }
        }
    }
    eprintln!(
        "QS-R1 C-22: {aprobados} largos aprobados, {} con asientos direccionales netos en contra",
        contraejemplos.len()
    );
    for c in contraejemplos.iter().take(10) {
        eprintln!("  {c}");
    }
    assert!(
        contraejemplos.is_empty(),
        "{} largos aprobados con señal direccional neta bajista (ver salida)",
        contraejemplos.len()
    );
}
