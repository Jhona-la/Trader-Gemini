//! CAPITAL-UNICO V4E (P1-b, paso 4 del pedido): el cero de una cuenta vacia
//! llega publicado al arena con el veto del puente suelto (el puente no veta un
//! saldo valido), y Risk no admite la entrada por su regla de siempre
//! (`current_capital <= 0.0`, `REJ_INVALID_INPUT`). Sin atajo en el puente y
//! sin regla nueva. Control: con el saldo publicado positivo, esa regla no es la
//! que rechaza.

use quantum_arena::GlobalArena;
use risk_engine::{RiskEngine, REJ_INVALID_INPUT};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::{Relaxed, Release};

fn larga() -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    }
}

/// Lo que hace el publicador del host en el alza: saldo y DESPUES el veto.
fn publicar_alza(a: &GlobalArena, saldo: f64) {
    a.unified_capital.store(saldo, Relaxed);
    a.capital_incompleto.store(false, Release);
}

#[test]
fn v4e_cero_de_cuenta_vacia_publicado_no_admite_entradas() {
    let a = GlobalArena::build_in_own_stack(100.0);
    a.capital_incompleto.store(true, Release);
    publicar_alza(&a, 0.0);
    let orden = RiskEngine::new(100.0).evaluate_quantum_order(0, &larga(), &a);
    assert_eq!(orden.signal, SignalType::Flat, "la cuenta vacia no admite entradas");
    assert_eq!(risk_engine::ultimo_rechazo(), Some(REJ_INVALID_INPUT), "por la regla de siempre: capital <= 0");
}

#[test]
fn v4e_control_saldo_positivo_publicado_no_cae_en_esa_regla() {
    let a = GlobalArena::build_in_own_stack(100.0);
    a.capital_incompleto.store(true, Release);
    publicar_alza(&a, 10.0);
    let _ = RiskEngine::new(100.0).evaluate_quantum_order(0, &larga(), &a);
    assert_ne!(risk_engine::ultimo_rechazo(), Some(REJ_INVALID_INPUT));
}
