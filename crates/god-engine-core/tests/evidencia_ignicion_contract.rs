//! Contratos del contrato de Evidencia de Ignición (FASE A-A2, GLM-203;
//! diseño: docs/audit/CONTRATO_EVIDENCIA_IGNICION_A2_2026-11-05.md,
//! auditoría base: docs/audit/IGNICION_A1_2026-11-05.md).
use god_engine_core::evidencia_ignicion::{EvidenciaIgnicion, Venue, VeredictoDarwin};

/// Base válida: cuenta viva (Testnet), warmup 1000 (lo que BOOT:255
/// carga), entrenamiento real con métrica finita, OMS del snapshot.
fn base_valida() -> EvidenciaIgnicion {
    EvidenciaIgnicion {
        klines_warmup_por_simbolo: 1000,
        minimo_klines_por_simbolo: 500,
        venue: Venue::CuentaViva,
        epochs_entrenamiento_efectivas: 3,
        metrica_entrenamiento: Some(0.42),
        oms_reconciled: true,
    }
}

#[test]
fn training_no_op_rechaza() {
    // EL test que mata al BLOQUEANTE #1: hoy 120 ticks + un training
    // NO-OP (BOOT:287-294) bastan para aprobar Darwin (GE:1716). Con el
    // contrato, epochs=0/métrica=None RECHAZA aunque todo lo demás esté
    // perfecto — el sistema NO se habilita a sí mismo.
    let mut ev = base_valida();
    ev.epochs_entrenamiento_efectivas = 0;
    ev.metrica_entrenamiento = None;
    assert_eq!(
        ev.aprueba_darwin(),
        VeredictoDarwin::Rechazado("training_no_op")
    );
    // Y la métrica ausente con epochs>0 también es no-op encubierto.
    ev.epochs_entrenamiento_efectivas = 5;
    assert_eq!(
        ev.aprueba_darwin(),
        VeredictoDarwin::Rechazado("metrica_entrenamiento_invalida")
    );
}

#[test]
fn papel_nunca_reconcilia() {
    // A1 MEDIO #6: en papel local oms_reconciled=TRUE sin consultar
    // nada (OMS:199) era indistinguible de reconciliado. El contrato
    // rechaza la INCOHERENCIA: papel que dice reconciliar miente.
    let mut ev = base_valida();
    ev.venue = Venue::PapelLocal;
    ev.oms_reconciled = true; // incoherencia deliberada
    assert_eq!(
        ev.aprueba_darwin(),
        VeredictoDarwin::Rechazado("papel_local_no_reconcilia")
    );
    // Papel COHERENTE (reconciled=false) con entrenamiento real aprueba:
    // el papel es legítimo para A/B y warmup, sólo no puede "reconciliar".
    ev.oms_reconciled = false;
    assert_eq!(ev.aprueba_darwin(), VeredictoDarwin::Aprobado);
}

#[test]
fn cuenta_viva_reconciliada_con_entrenamiento_aprueba() {
    // Happy path Testnet: exactamente la evidencia que la corrida A3
    // debe producir para habilitar la fase de trading.
    assert_eq!(base_valida().aprueba_darwin(), VeredictoDarwin::Aprobado);
    // Y cuenta viva SIN snapshot del conjunto (OMS:619-629) no aprueba.
    let mut ev = base_valida();
    ev.oms_reconciled = false;
    assert_eq!(
        ev.aprueba_darwin(),
        VeredictoDarwin::Rechazado("oms_no_reconciliado")
    );
}

#[test]
fn warmup_insuficiente_y_config_invalida_rechazan() {
    // Warmup CONTADO por debajo del mínimo: rechazado con la razón.
    let mut ev = base_valida();
    ev.klines_warmup_por_simbolo = 499;
    assert_eq!(
        ev.aprueba_darwin(),
        VeredictoDarwin::Rechazado("warmup_insuficiente")
    );
    // Mínimo cero = config del host inválida (el contrato no esconde
    // literales: un mínimo 0 apruebaría cualquier warmup).
    ev.klines_warmup_por_simbolo = 1000;
    ev.minimo_klines_por_simbolo = 0;
    assert_eq!(
        ev.aprueba_darwin(),
        VeredictoDarwin::Rechazado("config_minimo_klines_cero")
    );
}

#[test]
fn metrica_no_finita_falla_cerrado() {
    // NaN y ±Inf en la métrica: rechazados, jamás aprobados, sin panic.
    for mala in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut ev = base_valida();
        ev.metrica_entrenamiento = Some(mala);
        assert_eq!(
            ev.aprueba_darwin(),
            VeredictoDarwin::Rechazado("metrica_entrenamiento_invalida")
        );
    }
}
