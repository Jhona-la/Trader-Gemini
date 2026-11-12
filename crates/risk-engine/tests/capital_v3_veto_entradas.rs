//! CAPITAL-UNICO V3 (dictamen de Codex QRA sobre 1757709c4, P1.2 y P1.3):
//! sin saldo vigente de la era del puente de la cuenta, o con su equity
//! incompleta, el host pone `arena.capital_incompleto` y el riesgo no admite
//! entradas. Los cierres no pasan por esta puerta. No cambia d* ni umbrales.

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
fn v3_capital_incompleto_veta_las_entradas_y_al_soltarse_deja_de_vetar_por_eso() {
    let a = GlobalArena::build_in_own_stack(100.0);
    assert!(!a.capital_incompleto.load(Relaxed), "nace en false: backtest y simulación no tienen puente");

    a.capital_incompleto.store(true, Relaxed);
    let orden = RiskEngine::new(100.0).evaluate_quantum_order(0, &larga(), &a);
    assert_eq!(orden.signal, SignalType::Flat);
    assert_eq!(risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]), Some("drawdown"));

    // Sin la bandera, esta misma orden no se rechaza por esta puerta (puede
    // rechazarse por otra, pero no por drawdown con el arena intacto).
    a.capital_incompleto.store(false, Relaxed);
    let _ = RiskEngine::new(100.0).evaluate_quantum_order(0, &larga(), &a);
    assert_ne!(risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]), Some("drawdown"));
}

#[test]
fn v3_contrato_de_fuente_veto_solo_de_entradas() {
    let riesgo = include_str!("../src/lib.rs");
    assert!(riesgo.contains("arena.capital_incompleto.load(Ordering::Acquire)"));
    let oms = include_str!("../../../src/host_oms.rs");
    let entrada = oms.split("fn check_entry_locked(").nth(1).expect("check_entry_locked");
    assert!(entrada[..entrada.find("pub fn begin_refresh").unwrap()].contains("arena.capital_incompleto.load(Ordering::Acquire)"));
    // Los cierres (aplanado del inmune y salidas del ejecutor) no la leen.
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert_eq!(host.matches("capital_incompleto").count(), 1, "el host sólo la escribe desde el puente");
    let aplanado = include_str!("../../execution-engine/src/aplanado_inmune.rs");
    assert!(!aplanado.contains("capital_incompleto"));
    let ejecutor = include_str!("../../execution-engine/src/executor.rs");
    assert!(!ejecutor.contains("capital_incompleto"));
}
