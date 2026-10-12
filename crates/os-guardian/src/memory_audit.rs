use std::thread;
use std::time::Duration;
#[cfg(windows)]
use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
#[cfg(windows)]
use windows::Win32::System::Threading::GetCurrentProcess;

use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// LG-14 (Línea G) — VETO DE MEMORIA CON HISTÉRESIS SOBRE EL COMMIT.
///
/// El tope duro del JobObject limita el COMMIT del proceso; el veto medía el
/// working set, otra magnitud (y la compactación lo recortaba): el commit
/// podía llegar al tope sin que el veto saltara. Ahora el veto mira la misma
/// magnitud que el tope. Devuelve el nuevo estado del pánico, o `None` si no
/// cambia: se arma por encima del 85 % del límite y se suelta por debajo del
/// 70 %.
pub fn transicion_panico_memoria(commit_bytes: u64, max_bytes: u64, panico_actual: bool) -> Option<bool> {
    if max_bytes == 0 {
        return None;
    }
    let fraccion = commit_bytes as f64 / max_bytes as f64;
    if !panico_actual && fraccion > 0.85 {
        Some(true)
    } else if panico_actual && fraccion < 0.70 {
        Some(false)
    } else {
        None
    }
}

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
                    // LG-14: el commit (PagefileUsage) es lo que limita el JobObject.
                    let working_set_bytes = counters.PagefileUsage as u64;
                    let transicion = transicion_panico_memoria(
                        working_set_bytes,
                        max_bytes,
                        arena.panic_memory_dump.load(Ordering::Relaxed),
                    );

                    if transicion == Some(true) {
                        telemetry_engine::telemetry_err!("⚠️ [CRÍTICO - OS-GUARDIAN] Memory Leak Detectado: {} MB de commit (> 85% del límite). Activando Pánico de Memoria.", working_set_bytes / 1024 / 1024);
                        arena.panic_memory_dump.store(true, Ordering::SeqCst);

                        // FASE 14: Intento de mitigar desde OS Guardian forzando GC de malloc y compactando heap si es posible
                        // Rust no tiene GC, pero podemos sugerir a Windows que libere páginas
                        // windows::Win32::System::ProcessStatus::EmptyWorkingSet(process); (Peligroso para HFT, causa page faults, se delega el frenado a IA)
                    } else if transicion == Some(false) {
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

#[cfg(test)]
mod tests_lg14 {
    use super::*;

    #[test]
    fn lg14_el_veto_de_memoria_tiene_histeresis_sobre_el_commit() {
        let max = 1_000u64;
        assert_eq!(transicion_panico_memoria(860, max, false), Some(true));
        assert_eq!(transicion_panico_memoria(800, max, true), None, "entre 70 % y 85 % se mantiene");
        assert_eq!(transicion_panico_memoria(690, max, true), Some(false));
        assert_eq!(transicion_panico_memoria(500, max, false), None);
        assert_eq!(transicion_panico_memoria(900, 0, false), None);
    }

    #[test]
    fn lg14_el_job_limita_por_proceso_y_el_veto_mide_el_commit() {
        let lib = include_str!("lib.rs");
        let por_job = ["LimitFlags = JOB_OBJECT_", "LIMIT_JOB_MEMORY"].concat();
        assert!(!lib.contains(&por_job), "el tope no debe sumar el commit del entrenador hijo");
        assert!(lib.contains("JOB_OBJECT_LIMIT_PROCESS_MEMORY"));
        let audit = include_str!("memory_audit.rs");
        assert!(audit.contains("counters.PagefileUsage as u64;"));
    }
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
