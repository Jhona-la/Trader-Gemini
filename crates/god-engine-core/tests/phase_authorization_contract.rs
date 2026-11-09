use god_engine_core::orchestrator::{PhaseOrchestrator, SystemPhase};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[test]
fn waiting_is_not_genomic_approval() {
    let approval = Arc::new(AtomicBool::new(false));
    let mut phase = PhaseOrchestrator::new(1, false, approval);
    for _ in 0..6000 {
        phase.on_tick();
    }
    assert_eq!(phase.current_phase, SystemPhase::GenomicAudit);
    assert!(!phase.is_trading_allowed());
}

#[test]
fn revocation_blocks_admission_immediately_and_returns_to_audit() {
    for demo in [false, true] {
        let approval = Arc::new(AtomicBool::new(true));
        let mut phase = PhaseOrchestrator::new(1, demo, approval.clone());
        // GLM-216: en Producción la puerta exige certificado demo (bloqueante
        // #2 del A1); este contrato prueba la REVOCACIÓN, así que se aporta
        // evidencia válida para llegar a operar como antes.
        if !demo {
            phase.set_evidencia_demo(Some(god_engine_core::orchestrator::EvidenciaDemo {
                cierres: 30,
                ev_neto_por_trade: 0.001,
                wins: 20,
            }));
        }
        for _ in 0..4 {
            phase.on_tick();
        }
        assert!(phase.is_trading_allowed());
        approval.store(false, Ordering::Release);
        assert!(!phase.is_trading_allowed());
        phase.on_tick();
        assert_eq!(phase.current_phase, SystemPhase::GenomicAudit);
        approval.store(true, Ordering::Release);
        phase.on_tick();
        assert_eq!(phase.current_phase, SystemPhase::DemoVerify);
        assert!(!phase.is_trading_allowed());
        phase.on_tick();
        assert!(phase.is_trading_allowed());
    }
}

/// GLM-216 (bloqueante #2 del A1): la transición DemoVerify→
/// ProductionMainnet ya no es inmediata. Exige certificado demo persistido
/// con cierres ≥ mínimo y EV neto post-fees > 0 (criterio global). Los casos
/// con los números REALES de la corrida A3 al día 2 (EV −0.0048, n=3):
/// ese sistema NO pasaría a Producción — exactamente lo que el operador
/// quiere que la puerta impida.
#[test]
fn glm216_produccion_exige_certificado_demo() {
    use god_engine_core::orchestrator::EvidenciaDemo;

    let llega_a_demo_verify = |mut p: PhaseOrchestrator| {
        for _ in 0..3 {
            p.on_tick();
        }
        assert_eq!(p.current_phase, SystemPhase::DemoVerify);
        p
    };

    // Sin certificado: retenido para siempre (ningún timer aprueba solo).
    let approval = Arc::new(AtomicBool::new(true));
    let mut p = llega_a_demo_verify(PhaseOrchestrator::new(1, false, approval.clone()));
    for _ in 0..10_000 {
        p.on_tick();
    }
    assert_eq!(p.current_phase, SystemPhase::DemoVerify);
    assert!(!p.is_trading_allowed());

    // Certificado con cierres insuficientes (3 < 30): retenido.
    let mut p = llega_a_demo_verify(PhaseOrchestrator::new(1, false, approval.clone()));
    p.set_evidencia_demo(Some(EvidenciaDemo {
        cierres: 3,
        ev_neto_por_trade: 0.001,
        wins: 2,
    }));
    p.on_tick();
    assert_eq!(p.current_phase, SystemPhase::DemoVerify);

    // Certificado con EV neto ≤ 0 (el caso REAL del día 2: −0.0048): retenido.
    let mut p = llega_a_demo_verify(PhaseOrchestrator::new(1, false, approval.clone()));
    p.set_evidencia_demo(Some(EvidenciaDemo {
        cierres: 50,
        ev_neto_por_trade: -0.004_825,
        wins: 10,
    }));
    p.on_tick();
    assert_eq!(p.current_phase, SystemPhase::DemoVerify);
    assert!(!p.is_trading_allowed());

    // Certificado válido (≥ 30 cierres, EV > 0): transiciona y opera.
    let mut p = llega_a_demo_verify(PhaseOrchestrator::new(1, false, approval.clone()));
    p.set_evidencia_demo(Some(EvidenciaDemo {
        cierres: 30,
        ev_neto_por_trade: 0.001_2,
        wins: 18,
    }));
    p.on_tick();
    assert_eq!(p.current_phase, SystemPhase::ProductionMainnet);
    assert!(p.is_trading_allowed());

    // Demo nunca exige certificado: su fin es GENERARLO.
    let mut p = llega_a_demo_verify(PhaseOrchestrator::new(1, true, approval.clone()));
    p.on_tick();
    assert_eq!(p.current_phase, SystemPhase::PaperTrading);
    assert!(p.is_trading_allowed());
}

#[test]
fn revocation_between_audit_and_demo_verify_does_not_authorize() {
    let approval = Arc::new(AtomicBool::new(true));
    let mut phase = PhaseOrchestrator::new(1, false, approval.clone());
    for _ in 0..3 {
        phase.on_tick();
    }
    assert_eq!(phase.current_phase, SystemPhase::DemoVerify);
    approval.store(false, Ordering::Release);
    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::GenomicAudit);
    assert!(!phase.is_trading_allowed());
}

#[test]
fn qra_boot_01_oms_reconciliation_required_for_live_ignition() {
    let approval = Arc::new(AtomicBool::new(true));
    let oms_reconciled = Arc::new(AtomicBool::new(false));
    let mut phase = PhaseOrchestrator::new_with_oms(1, false, approval.clone(), oms_reconciled.clone());
    // GLM-216: certificado demo válido para que el foco de ESTE contrato
    // siga siendo la reconciliación OMS, no la puerta de Producción.
    phase.set_evidencia_demo(Some(god_engine_core::orchestrator::EvidenciaDemo {
        cierres: 30,
        ev_neto_por_trade: 0.001,
        wins: 20,
    }));

    // Tick 1: Init -> DataWarmup
    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::DataWarmup);

    // Warmup ticks elapse: DataWarmup -> GenomicAudit
    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::GenomicAudit);

    // Even if Darwin is approved and thousands of ticks elapse,
    // if OMS reconciliation has NOT been confirmed (oms_reconciled = false),
    // it MUST NOT enter DemoVerify or ProductionMainnet, and trading MUST be blocked.
    for _ in 0..1000 {
        phase.on_tick();
        assert_eq!(phase.current_phase, SystemPhase::GenomicAudit);
        assert!(!phase.is_trading_allowed());
    }

    // Now simulate successful OMS position snapshot & reconciliation
    oms_reconciled.store(true, Ordering::Release);
    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::DemoVerify);
    assert!(!phase.is_trading_allowed());

    // Next tick in production transitions to ProductionMainnet with trading authorized
    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::ProductionMainnet);
    assert!(phase.is_trading_allowed());

    // If OMS reconciliation is lost/invalidated, trading is immediately revoked
    oms_reconciled.store(false, Ordering::Release);
    assert!(!phase.is_trading_allowed());
    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::GenomicAudit);
    assert!(!phase.is_trading_allowed());
}

#[test]
fn qra_boot_01_oms_reconciliation_required_for_demo_testnet_mode_when_configured() {
    let approval = Arc::new(AtomicBool::new(true));
    let oms_reconciled = Arc::new(AtomicBool::new(false));
    // Testnet / Demo mode with remote executor configured
    let mut phase = PhaseOrchestrator::new_with_oms(1, true, approval.clone(), oms_reconciled.clone());

    phase.on_tick(); // Init -> DataWarmup
    phase.on_tick(); // DataWarmup -> GenomicAudit

    // In demo mode with unconfirmed OMS, trading MUST be blocked
    for _ in 0..100 {
        phase.on_tick();
        assert_eq!(phase.current_phase, SystemPhase::GenomicAudit);
        assert!(!phase.is_trading_allowed());
    }

    // Snapshot confirmed -> unlock DemoVerify -> PaperTrading
    oms_reconciled.store(true, Ordering::Release);
    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::DemoVerify);
    assert!(!phase.is_trading_allowed());

    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::PaperTrading);
    assert!(phase.is_trading_allowed());

    // Revocation in demo mode also immediately returns to GenomicAudit and blocks trading
    oms_reconciled.store(false, Ordering::Release);
    assert!(!phase.is_trading_allowed());
    phase.on_tick();
    assert_eq!(phase.current_phase, SystemPhase::GenomicAudit);
    assert!(!phase.is_trading_allowed());
}
