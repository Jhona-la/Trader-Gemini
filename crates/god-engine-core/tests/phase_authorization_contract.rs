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
