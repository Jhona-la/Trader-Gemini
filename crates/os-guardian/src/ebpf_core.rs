#[cfg(windows)]
use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
#[cfg(windows)]
use windows::Win32::System::Threading::GetCurrentProcess;

/// Capa base para eBPF y eventos de Kernel.
/// En la arquitectura HFT (High Frequency Trading) definitiva, esta capa
/// inyecta programas eBPF o consulta contadores nativos de kernel (ETW/NT) para suscribirse a:
///
/// - Context Switches
/// - Page Faults
/// - System calls seleccionadas
/// - Interrupciones de red (IRQ/SoftIRQ)
///
/// Estas trazas alimentan un lock-free ring buffer directamente hacia el
/// Anomaly Engine.

#[derive(Debug, Clone, Copy)]
pub struct KernelEvents {
    pub context_switches: u64,
    pub page_faults: u64,
    pub net_irqs: u64,
    pub scheduler_delay_ns: u64,
}

pub struct EbpfSensor {
    #[allow(dead_code)]
    target_pid: u32,
}

impl EbpfSensor {
    pub fn new(target_pid: u32) -> Self {
        Self { target_pid }
    }

    #[cfg(target_os = "linux")]
    pub fn read_events(&self) -> KernelEvents {
        // Mapear BPF map file descriptor en Linux
        KernelEvents {
            context_switches: 0,
            page_faults: 0,
            net_irqs: 0,
            scheduler_delay_ns: 500,
        }
    }

    #[cfg(windows)]
    pub fn read_events(&self) -> KernelEvents {
        let mut counters = PROCESS_MEMORY_COUNTERS::default();
        let mut page_faults = 0u64;
        unsafe {
            let handle = GetCurrentProcess();
            if GetProcessMemoryInfo(
                handle,
                &mut counters,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            )
            .is_ok()
            {
                page_faults = counters.PageFaultCount as u64;
            }
        }

        KernelEvents {
            context_switches: 0,
            page_faults,
            net_irqs: 0,
            scheduler_delay_ns: 120,
        }
    }

    #[cfg(not(any(target_os = "linux", windows)))]
    pub fn read_events(&self) -> KernelEvents {
        KernelEvents {
            context_switches: 0,
            page_faults: 0,
            net_irqs: 0,
            scheduler_delay_ns: 200,
        }
    }
}
