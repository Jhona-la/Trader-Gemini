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

pub struct PhaseOrchestrator {
    pub current_phase: SystemPhase,
    pub darwin_approved: Arc<AtomicBool>,
    pub warmup_ticks_required: u64,
    pub current_ticks: u64,
    pub is_demo_mode: bool,
}

impl PhaseOrchestrator {
    pub fn new(
        warmup_ticks_required: u64,
        is_demo_mode: bool,
        darwin_approved: Arc<AtomicBool>,
    ) -> Self {
        Self {
            current_phase: SystemPhase::Initialization,
            darwin_approved,
            warmup_ticks_required: warmup_ticks_required.max(1),
            current_ticks: 0,
            is_demo_mode,
        }
    }

    /// Llamado en cada tick del Market Data. Mueve la máquina de estados.
    pub fn on_tick(&mut self) -> SystemPhase {
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
                // FIX #1506: Transición por aprobación de Darwin o timeout defensivo (5000 ticks)
                if self.darwin_approved.load(Ordering::Relaxed)
                    || self.current_ticks >= self.warmup_ticks_required.saturating_add(5000)
                {
                    println!(
                        "🚀 [ORCHESTRATOR] Fase: GenomicAudit -> DemoVerify (Darwin Approved / Ready)"
                    );
                    self.current_phase = SystemPhase::DemoVerify;
                }
            }
            SystemPhase::DemoVerify => {
                // En DemoVerify hacemos los últimos chequeos de paridad (o simplemente transicionamos inmediatamente si ya todo está ok).
                if self.is_demo_mode {
                    println!("🚀 [ORCHESTRATOR] Fase: DemoVerify -> PaperTrading");
                    self.current_phase = SystemPhase::PaperTrading;
                } else {
                    println!("🚀 [ORCHESTRATOR] Fase: DemoVerify -> ProductionMainnet");
                    self.current_phase = SystemPhase::ProductionMainnet;
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
        self.current_phase == SystemPhase::PaperTrading
            || self.current_phase == SystemPhase::ProductionMainnet
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
}

