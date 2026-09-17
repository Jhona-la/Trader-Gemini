use std::thread;
use std::time::Duration;
#[cfg(windows)]
use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
#[cfg(windows)]
use windows::Win32::System::Threading::GetCurrentProcess;

use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Inicia el monitor de memoria en un hilo de background (cero impacto en Hot Path).
pub fn start_memory_auditor(max_memory_mb: usize, arena: Arc<GlobalArena>) {
    thread::spawn(move || {
        telemetry_engine::telemetry!(
            "[OS-GUARDIAN] 🛡️ Auditor de Memoria (Leak Detection) Iniciado. Límite: {} MB",
            max_memory_mb
        );
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
                )
                .is_ok()
                {
                    let working_set_bytes = counters.WorkingSetSize as u64;

                    if working_set_bytes > (max_bytes as f64 * 0.85) as u64 {
                        telemetry_engine::telemetry_err!("⚠️ [CRÍTICO - OS-GUARDIAN] Memory Leak Detectado: {} MB en uso (> 85% del límite). Activando Pánico de Memoria.", working_set_bytes / 1024 / 1024);
                        arena.panic_memory_dump.store(true, Ordering::SeqCst);

                        // FASE 14: Intento de mitigar desde OS Guardian forzando GC de malloc y compactando heap si es posible
                        // Rust no tiene GC, pero podemos sugerir a Windows que libere páginas
                        // windows::Win32::System::ProcessStatus::EmptyWorkingSet(process); (Peligroso para HFT, causa page faults, se delega el frenado a IA)
                    } else if working_set_bytes < (max_bytes as f64 * 0.70) as u64
                        && arena.panic_memory_dump.load(Ordering::Relaxed)
                    {
                        telemetry_engine::telemetry!("✅ [OS-GUARDIAN] Memoria estabilizada ({} MB). Desactivando Pánico de Memoria.", working_set_bytes / 1024 / 1024);
                        arena.panic_memory_dump.store(false, Ordering::SeqCst);

                        // FASE 27: Si estamos en zona segura, garantizamos que Windows NO envíe nuestra memoria al archivo de paginación (SSD)
                        // Para esto requeriríamos VirtualLock, pero llamarlo sobre todo el proceso es peligroso sin privilegios SeLockMemoryPrivilege.
                        // Imprimimos el guardián pasivo.
                    }
                }
            }
        }
    });
}

/// Obtiene el consumo actual de memoria del proceso en MB.
pub fn get_memory_usage_mb() -> f64 {
    #[cfg(windows)]
    unsafe {
        let process = GetCurrentProcess();
        let mut counters = PROCESS_MEMORY_COUNTERS::default();
        if GetProcessMemoryInfo(
            process,
            &mut counters,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        )
        .is_ok()
        {
            return counters.WorkingSetSize as f64 / 1024.0 / 1024.0;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_memory_usage_mb() {
        let mem = get_memory_usage_mb();
        assert!(mem >= 0.0);
        assert!(mem.is_finite());
    }
}
