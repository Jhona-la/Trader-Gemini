pub mod anomaly_detector;
pub mod ebpf_core;
pub mod large_pages;
pub mod memory_audit;
pub mod memory_compaction;
pub mod observability_plane;
pub mod pmu_sensor;
pub mod silicon_topology;
pub mod simd_crypto;
pub mod subnormal_control;
pub mod telemetry;

pub use large_pages::{LargePageArena, CANONICAL_LARGE_PAGE_SIZE, STANDARD_PAGE_SIZE};
pub use silicon_topology::{CoreType, CpuTopologyInfo};
pub use simd_crypto::{constant_time_eq_32, sha256_hmac_stack, sha256_stack, verify_payload_integrity};
pub use subnormal_control::{SubnormalHardwareController, SubnormalStatus};

use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
};
use windows::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThread, SetPriorityClass, SetProcessAffinityMask,
    SetProcessWorkingSetSize, SetThreadPriority, HIGH_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS,
    THREAD_PRIORITY_HIGHEST,
};

use std::ffi::c_void;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

use quantum_arena::GlobalArena;
use std::sync::Arc;

/// Inicializa la protección del sistema operativo Windows.
/// Configura la prioridad del proceso y la afinidad de núcleos.
pub fn init_guardian(affinity_mask: usize, max_memory_mb: usize, arena: Arc<GlobalArena>) {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return; // Ya inicializado
    }

    #[cfg(windows)]
    unsafe {
        let process = GetCurrentProcess();

        // 1. Establecer prioridad de proceso a HIGH_PRIORITY_CLASS (Fase 24)
        if let Err(e) = SetPriorityClass(process, HIGH_PRIORITY_CLASS) {
            eprintln!("[OS-GUARDIAN] Error estableciendo prioridad HIGH: {:?}", e);
            // Fallback
            let _ = SetPriorityClass(process, NORMAL_PRIORITY_CLASS);
        } else {
            println!("[OS-GUARDIAN] Prioridad del proceso establecida en HIGH_PRIORITY_CLASS.");
        }

        // 2. Establecer afinidad de CPU
        if SetProcessAffinityMask(process, affinity_mask).is_err() {
            eprintln!("[OS-GUARDIAN] Error estableciendo afinidad de CPU.");
        } else {
            println!(
                "[OS-GUARDIAN] Afinidad de CPU establecida con máscara: {:#X}.",
                affinity_mask
            );
        }

        // 3. Establecer límites de Memoria vía JobObject
        if let Ok(job) = CreateJobObjectW(None, None) {
            // LG-14 (Línea G): límite POR PROCESO. Con JOB_OBJECT_LIMIT_JOB_MEMORY
            // el commit del entrenador hijo (que hereda el job) se sumaba al del
            // proceso vivo y la asignación fallida abortaba el motor con
            // posiciones abiertas, sin pasar por el veto de memoria.
            let mut limit_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limit_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            limit_info.ProcessMemoryLimit = max_memory_mb * 1024 * 1024;

            let result = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &limit_info as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );

            if result.is_ok() {
                if AssignProcessToJobObject(job, process).is_ok() {
                    println!(
                        "[OS-GUARDIAN] Job Object Memory Limit establecido a {} MB.",
                        max_memory_mb
                    );
                } else {
                    eprintln!("[OS-GUARDIAN] Error asignando proceso al JobObject.");
                }
            } else {
                eprintln!("[OS-GUARDIAN] Error configurando Memory Limit en JobObject.");
            }
        }

        // 4. Forzar memoria física pura (Lock RAM, Disable Pagefile swapping)
        let min_working_set = (max_memory_mb / 2) * 1024 * 1024;
        let max_working_set = max_memory_mb * 1024 * 1024;
        if let Err(e) = SetProcessWorkingSetSize(process, min_working_set, max_working_set) {
            eprintln!(
                "⚠️ [OS-GUARDIAN] No se pudo fijar Working Set (Privilegios insuficientes?): {:?}",
                e
            );
        } else {
            println!("🔒 [OS-GUARDIAN] RAM Physical Lock: {} MB - {} MB (Pagefile swap disabled para latencia cero).", min_working_set / 1_048_576, max_working_set / 1_048_576);
        }
    }

    // Iniciar el auditor dinámico
    memory_audit::start_memory_auditor(max_memory_mb, arena);

    #[cfg(windows)]
    let tid = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
    #[cfg(not(windows))]
    let tid = std::process::id();

    // Iniciar Hardware-Assisted Out-of-Band Observability Plane
    let obs_plane = observability_plane::ObservabilityPlane::new(
        std::process::id(), // process ID
        tid,                // Native Hot-Path Thread ID
    );
    // Asignar al último CPU core lógico disponible (fuera del rango de HFT) según topología de hardware (BUG-678)
    let total_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let obs_core = if total_cpus > 1 {
        Some(total_cpus - 1)
    } else {
        None
    };
    obs_plane.spawn_isolated(obs_core);

    // 5. Supresión activa de subnormales de hardware (FTZ + DAZ)
    if SubnormalHardwareController::enable_suppression() {
        println!("⚡ [OS-GUARDIAN] Supresión de subnormales de silicio (FTZ/DAZ) activada exitosamente.");
    }

    #[cfg(not(windows))]
    {
        println!("[OS-GUARDIAN] Ejecutando en modo NO-Windows. Guardián inactivo.");
    }
}

/// FASE 8 / FASE 24: Configura el hilo actual para operar en máxima prioridad física y con supresión de subnormales.
/// Solo debe llamarse desde el bucle de decisión continuo en tiempo real (GodEngineCore).
pub fn set_current_thread_time_critical() {
    // 1. Activar supresión de subnormales de hardware (FTZ + DAZ) en el hilo actual
    SubnormalHardwareController::enable_suppression();

    #[cfg(windows)]
    unsafe {
        let thread = GetCurrentThread();
        // MOD6/8-025 (INFORME DECIMOCUARTO): SetThreadIdealProcessor es un
        // HINT blando, no pinning — el scheduler puede migrar el hilo de
        // todos modos; el "anclaje a L3 con garantía" era un comentario
        // falso. Hint blando, no pinning — MOD6/8-025.
        // let _ = SetThreadIdealProcessor(thread, 1);

        // MOD6/8-025: THREAD_PRIORITY_TIME_CRITICAL en el hilo de DECISIÓN
        // podía STARVAR los hilos tokio que EJECUTAN el maker-chase — el
        // hilo que decide postergaba al hilo que ejecuta (prioridades
        // invertidas). HIGHEST sigue POR ENCIMA de normal, pero ya no
        // puede monopolizar el scheduler.
        if let Err(e) = SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST) {
            eprintln!(
                "⚠️ [OS-GUARDIAN] No se pudo asignar THREAD_PRIORITY_HIGHEST: {:?}",
                e
            );
        } else {
            println!("⚡ [OS-GUARDIAN] Hilo de decisión en THREAD_PRIORITY_HIGHEST (sin monopolizar el scheduler — MOD6/8-025).");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_current_thread_time_critical() {
        set_current_thread_time_critical();
    }
}
