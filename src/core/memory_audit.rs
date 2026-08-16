use std::thread;
use std::time::Duration;
#[cfg(windows)]
use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
#[cfg(windows)]
use windows::Win32::System::Threading::GetCurrentProcess;

/// Inicia el monitor de memoria en un hilo de background (cero impacto en Hot Path).
pub fn start_memory_auditor(max_memory_mb: usize) {
    thread::spawn(move || {
        println!("[OS-GUARDIAN] 🛡️ Auditor de Memoria (Leak Detection) Iniciado. Límite: {} MB", max_memory_mb);
        let max_bytes = (max_memory_mb * 1024 * 1024) as u64;

        loop {
            thread::sleep(Duration::from_secs(5)); // Audita cada 5 segundos
            
            #[cfg(windows)]
            unsafe {
                let process = GetCurrentProcess();
                let mut counters = PROCESS_MEMORY_COUNTERS::default();
                
                if GetProcessMemoryInfo(
                    process,
                    &mut counters,
                    std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                ).is_ok() {
                    let working_set_bytes = counters.WorkingSetSize as u64;
                    
                    if working_set_bytes > (max_bytes as f64 * 0.85) as u64 {
                        eprintln!("⚠️ [CRÍTICO - OS-GUARDIAN] Memory Leak Detectado: {} MB en uso (> 85% del límite).", working_set_bytes / 1024 / 1024);
                        // Aquí se podría integrar el vaciado de buffers o alertas por webhook.
                    }
                }
            }
        }
    });
}
