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
    pub fn new(warmup_ticks_required: u64, is_demo_mode: bool, darwin_approved: Arc<AtomicBool>) -> Self {
        Self {
            current_phase: SystemPhase::Initialization,
            darwin_approved,
            warmup_ticks_required,
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
                self.current_ticks += 1;
                if self.current_ticks >= self.warmup_ticks_required {
                    println!("🚀 [ORCHESTRATOR] Fase: DataWarmup -> GenomicAudit (Ticks: {})", self.current_ticks);
                    self.current_phase = SystemPhase::GenomicAudit;
                }
            }
            SystemPhase::GenomicAudit => {
                // En GenomicAudit esperamos a que Darwin apruebe el inicio
                if self.darwin_approved.load(Ordering::Relaxed) {
                    println!("🚀 [ORCHESTRATOR] Fase: GenomicAudit -> DemoVerify (Darwin Approved)");
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
        self.current_phase == SystemPhase::PaperTrading || self.current_phase == SystemPhase::ProductionMainnet
    }

    pub fn is_paper_trading(&self) -> bool {
        self.current_phase == SystemPhase::PaperTrading
    }
}
