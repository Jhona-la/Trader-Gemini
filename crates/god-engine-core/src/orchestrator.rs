use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemPhase {
    Initialization,
    DataWarmup,
    GenomicAudit,
    DemoVerify,
    PaperTrading,
    ProductionMainnet,
}

/// GLM-216 (BLOQUEANTE #2 del A1): evidencia de rentabilidad demo persistida.
/// La transición DemoVerify→ProductionMainnet deja de ser inmediata: exige
/// cierres reales suficientes y EV neto post-fees > 0 medido del income del
/// exchange (criterio global; mandato §23: «sólo tras certificar
/// automáticamente la rentabilidad neta en Demo, el sistema conmuta a
/// Producción»). El bin la persiste en data/demo/evidencia_demo.json desde
/// el agregado del FEE-BREAKER (GLM-214) y la inyecta al orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EvidenciaDemo {
    /// Cierres reales (REALIZED_PNL) de la ventana demo.
    pub cierres: u64,
    /// EV neto post-fees por trade del income REAL (gross + fees firmados).
    pub ev_neto_por_trade: f64,
    /// Wins de la ventana (para el WR de la evidencia).
    pub wins: u64,
}

/// Cierres mínimos para autorizar Producción. 30: suficiente para que el EV
/// neto empiece a significar algo sin exigir la serie completa del criterio
/// global (7 d) — el breaker Prod→Demo (bloqueante #3) acota el resto.
pub const MIN_CIERRES_PARA_PROD: u64 = 30;

pub struct PhaseOrchestrator {
    pub current_phase: SystemPhase,
    pub darwin_approved: Arc<AtomicBool>,
    pub oms_reconciled: Arc<AtomicBool>,
    pub warmup_ticks_required: u64,
    pub current_ticks: u64,
    pub is_demo_mode: bool,
    pub evidencia_demo: Option<EvidenciaDemo>,
    pub min_cierres_para_prod: u64,
}

impl PhaseOrchestrator {
    pub fn new(
        warmup_ticks_required: u64,
        is_demo_mode: bool,
        darwin_approved: Arc<AtomicBool>,
    ) -> Self {
        Self::new_with_oms(
            warmup_ticks_required,
            is_demo_mode,
            darwin_approved,
            Arc::new(AtomicBool::new(true)),
        )
    }

    pub fn new_with_oms(
        warmup_ticks_required: u64,
        is_demo_mode: bool,
        darwin_approved: Arc<AtomicBool>,
        oms_reconciled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            current_phase: SystemPhase::Initialization,
            darwin_approved,
            oms_reconciled,
            warmup_ticks_required: warmup_ticks_required.max(1),
            current_ticks: 0,
            is_demo_mode,
            evidencia_demo: None,
            min_cierres_para_prod: MIN_CIERRES_PARA_PROD,
        }
    }

    /// Inyecta la evidencia demo persistida (el bin la lee de
    /// data/demo/evidencia_demo.json al arrancar en modo Producción, o la
    /// actualiza en vivo en modo demo). `None` = sin certificado: la puerta
    /// a Producción permanece cerrada (fail-closed).
    pub fn set_evidencia_demo(&mut self, evidencia: Option<EvidenciaDemo>) {
        self.evidencia_demo = evidencia;
    }

    /// Llamado en cada tick del Market Data. Mueve la máquina de estados.
    pub fn on_tick(&mut self) -> SystemPhase {
        // Readiness is revocable. Elapsed ticks cannot replace authorization or OMS reconciliation.
        // Both trade admission and phase transition strictly require confirmed OMS reconciliation.
        let darwin_ok = self.darwin_approved.load(Ordering::Acquire);
        let oms_ok = self.oms_reconciled.load(Ordering::Acquire);

        if matches!(
            self.current_phase,
            SystemPhase::DemoVerify | SystemPhase::PaperTrading | SystemPhase::ProductionMainnet
        ) && (!darwin_ok || !oms_ok)
        {
            self.current_phase = SystemPhase::GenomicAudit;
            return self.current_phase;
        }
        match self.current_phase {
            SystemPhase::Initialization => {
                // Inmediatamente a Warmup después de arrancar
                println!("🚀 [ORCHESTRATOR] Fase: Initialization -> DataWarmup");
                self.current_phase = SystemPhase::DataWarmup;
            }
            SystemPhase::DataWarmup => {
                self.current_ticks = self.current_ticks.saturating_add(1);
                if self.current_ticks >= self.warmup_ticks_required {
                    println!(
                        "🚀 [ORCHESTRATOR] Fase: DataWarmup -> GenomicAudit (Ticks: {})",
                        self.current_ticks
                    );
                    self.current_phase = SystemPhase::GenomicAudit;
                }
            }
            SystemPhase::GenomicAudit => {
                self.current_ticks = self.current_ticks.saturating_add(1);
                // A timeout may be diagnosed, but never grants trading permission.
                // Both Darwin approval and OMS reconciliation are required in live trading.
                if darwin_ok && oms_ok {
                    println!(
                        "🚀 [ORCHESTRATOR] Fase: GenomicAudit -> DemoVerify (approval & OMS reconciliation confirmed)"
                    );
                    self.current_phase = SystemPhase::DemoVerify;
                }
            }
            SystemPhase::DemoVerify => {
                if self.is_demo_mode {
                    // En demo el fin del flujo es PaperTrading: generar la
                    // evidencia que la puerta de Producción exige.
                    println!("🚀 [ORCHESTRATOR] Fase: DemoVerify -> PaperTrading");
                    self.current_phase = SystemPhase::PaperTrading;
                } else {
                    // GLM-216 (BLOQUEANTE #2 del A1): la transición a
                    // Producción EXIGE certificado demo persistido — cierres
                    // ≥ mínimo y EV neto post-fees > 0 del income REAL.
                    // Antes transicionaba inmediatamente sin métrica alguna.
                    // Fail-closed: sin evidencia o con EV ≤ 0, PERMANECE en
                    // DemoVerify (sin operar) — ningún timer aprueba solo.
                    match self.evidencia_demo {
                        Some(ev)
                            if ev.cierres >= self.min_cierres_para_prod
                                && ev.ev_neto_por_trade > 0.0
                                && ev.ev_neto_por_trade.is_finite() =>
                        {
                            println!(
                                "🚀 [ORCHESTRATOR] DemoVerify -> ProductionMainnet (certificado demo: {} cierres, EV neto {:+.6} USDT/trade)",
                                ev.cierres, ev.ev_neto_por_trade
                            );
                            self.current_phase = SystemPhase::ProductionMainnet;
                        }
                        Some(ev) => {
                            println!(
                                "⏸️ [ORCHESTRATOR] DemoVerify RETENIDO: certificado demo insuficiente ({} cierres < {}, EV {:+.6}) — sin rentabilidad neta demo no hay Producción",
                                ev.cierres, self.min_cierres_para_prod, ev.ev_neto_por_trade
                            );
                        }
                        None => {
                            println!(
                                "⏸️ [ORCHESTRATOR] DemoVerify RETENIDO: sin certificado demo persistido (data/demo/evidencia_demo.json) — fail-closed"
                            );
                        }
                    }
                }
            }
            SystemPhase::PaperTrading => {
                // Estado final si DEMO_MODE = true
            }
            SystemPhase::ProductionMainnet => {
                // Estado final si DEMO_MODE = false
            }
        }
        self.current_phase
    }

    pub fn is_trading_allowed(&self) -> bool {
        matches!(
            self.current_phase,
            SystemPhase::PaperTrading | SystemPhase::ProductionMainnet
        ) && self.darwin_approved.load(Ordering::Acquire)
            && self.oms_reconciled.load(Ordering::Acquire)
    }

    pub fn is_paper_trading(&self) -> bool {
        self.current_phase == SystemPhase::PaperTrading
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_orchestrator_demo_flow() {
        let darwin = Arc::new(AtomicBool::new(true));
        let mut orch = PhaseOrchestrator::new(3, true, darwin);

        assert_eq!(orch.current_phase, SystemPhase::Initialization);
        assert!(!orch.is_trading_allowed());

        // Tick 1: Init -> DataWarmup
        orch.on_tick();
        assert_eq!(orch.current_phase, SystemPhase::DataWarmup);

        // Tick 2, 3, 4: Warmup -> GenomicAudit
        orch.on_tick();
        orch.on_tick();
        orch.on_tick();
        assert_eq!(orch.current_phase, SystemPhase::GenomicAudit);

        // Tick 5: GenomicAudit -> DemoVerify (darwin approved)
        orch.on_tick();
        assert_eq!(orch.current_phase, SystemPhase::DemoVerify);

        // Tick 6: DemoVerify -> PaperTrading (demo mode)
        orch.on_tick();
        assert_eq!(orch.current_phase, SystemPhase::PaperTrading);
        assert!(orch.is_trading_allowed());
        assert!(orch.is_paper_trading());
    }

    #[test]
    fn test_phase_orchestrator_production_flow_with_darwin_approval() {
        let darwin = Arc::new(AtomicBool::new(false));
        let mut orch = PhaseOrchestrator::new(1, false, darwin.clone());
        // GLM-216: certificado demo válido — este contrato prueba el flujo
        // Darwin, no la puerta de Producción (ésta tiene su contrato propio
        // en tests/phase_authorization_contract.rs).
        orch.set_evidencia_demo(Some(EvidenciaDemo {
            cierres: 30,
            ev_neto_por_trade: 0.001,
            wins: 20,
        }));

        orch.on_tick(); // Init -> DataWarmup
        orch.on_tick(); // DataWarmup -> GenomicAudit
        assert_eq!(orch.current_phase, SystemPhase::GenomicAudit);

        // Darwin approves
        darwin.store(true, Ordering::Relaxed);
        orch.on_tick(); // GenomicAudit -> DemoVerify
        assert_eq!(orch.current_phase, SystemPhase::DemoVerify);

        orch.on_tick(); // DemoVerify -> ProductionMainnet
        assert_eq!(orch.current_phase, SystemPhase::ProductionMainnet);
        assert!(orch.is_trading_allowed());
        assert!(!orch.is_paper_trading());
    }

    #[test]
    fn test_phase_orchestrator_production_blocks_when_oms_not_reconciled() {
        let darwin = Arc::new(AtomicBool::new(true));
        let oms_reconciled = Arc::new(AtomicBool::new(false));
        let mut orch = PhaseOrchestrator::new_with_oms(1, false, darwin.clone(), oms_reconciled.clone());
        // GLM-216: ídem — el foco de este contrato es la reconciliación OMS.
        orch.set_evidencia_demo(Some(EvidenciaDemo {
            cierres: 30,
            ev_neto_por_trade: 0.001,
            wins: 20,
        }));

        orch.on_tick(); // Init -> DataWarmup
        orch.on_tick(); // DataWarmup -> GenomicAudit
        assert_eq!(orch.current_phase, SystemPhase::GenomicAudit);

        // Darwin is approved, but OMS is NOT reconciled: must stay in GenomicAudit and forbid trading
        for _ in 0..150 {
            orch.on_tick();
            assert_eq!(orch.current_phase, SystemPhase::GenomicAudit);
            assert!(!orch.is_trading_allowed());
        }

        // Once OMS snapshot succeeds and reconciles:
        oms_reconciled.store(true, Ordering::Release);
        orch.on_tick(); // GenomicAudit -> DemoVerify
        assert_eq!(orch.current_phase, SystemPhase::DemoVerify);

        orch.on_tick(); // DemoVerify -> ProductionMainnet
        assert_eq!(orch.current_phase, SystemPhase::ProductionMainnet);
        assert!(orch.is_trading_allowed());

        // Revoking OMS reconciliation immediately locks the system back to GenomicAudit
        oms_reconciled.store(false, Ordering::Release);
        assert!(!orch.is_trading_allowed());
        orch.on_tick();
        assert_eq!(orch.current_phase, SystemPhase::GenomicAudit);
        assert!(!orch.is_trading_allowed());
    }
}
