//! CAPITAL-UNICO V4B (pedido del coordinador, punto 4; hallazgo de Qoder en
//! `qoder_n1_en_el_log.md`): el veto del puente no pasaba por `[EMBUDO]` ni
//! tenía contador. Ahora el riesgo y el OMS cuentan cada entrada vetada por
//! `capital_incompleto` y el host lo escribe cada 60 s. Solo medida: la
//! decisión es la misma. No cambia d* ni umbrales.

use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn larga() -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    }
}

#[test]
fn v4b_cada_entrada_vetada_por_el_puente_se_cuenta_y_la_decision_no_cambia() {
    let a = GlobalArena::build_in_own_stack(100.0);
    assert_eq!(a.vetos_capital_riesgo.load(Relaxed), 0);
    a.capital_incompleto.store(true, Relaxed);
    for _ in 0..3 {
        let orden = RiskEngine::new(100.0).evaluate_quantum_order(0, &larga(), &a);
        assert_eq!(orden.signal, SignalType::Flat);
        assert_eq!(risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]), Some("drawdown"));
    }
    assert_eq!(a.vetos_capital_riesgo.load(Relaxed), 3);
    // Control: sin el veto del puente no se cuenta nada más.
    a.capital_incompleto.store(false, Relaxed);
    let _ = RiskEngine::new(100.0).evaluate_quantum_order(0, &larga(), &a);
    assert_eq!(a.vetos_capital_riesgo.load(Relaxed), 3);
}

#[test]
fn v4b_contrato_de_fuente_del_contador() {
    let host_oms = include_str!("../../../src/host_oms.rs").replace("\r\n", "\n");
    let veto = &host_oms[host_oms.find("if arena.capital_incompleto.load(Ordering::Acquire)").unwrap()..];
    assert!(veto[..200].contains("arena.vetos_capital_oms.fetch_add(1, Ordering::Relaxed)"));
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert!(host.contains("[CAPITAL] veto del puente, 60 s: entradas vetadas riesgo={} oms={}"));
    assert!(host.contains("arena_rs.vetos_capital_riesgo.swap(0, Ordering::Relaxed)"));
    assert!(host.contains("puente_rs.tomar_diagnostico()"));
}
