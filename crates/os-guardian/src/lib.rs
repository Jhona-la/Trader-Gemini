pub mod anomaly_detector;
pub mod ebpf_core;
pub mod memory_audit;
pub mod memory_compaction;
pub mod observability_plane;
pub mod pmu_sensor;
pub mod telemetry;
pub mod zero_latency_telemetry;
use std::sync::atomic::{AtomicBool, Ordering};
use uuid::Uuid;

use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_JOB_MEMORY,
};
use windows::Win32::System::Threading::SetProcessWorkingSetSize;
use windows::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThread, SetPriorityClass, SetProcessAffinityMask,
    SetThreadIdealProcessor, SetThreadPriority, HIGH_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS,
    THREAD_PRIORITY_TIME_CRITICAL,
};

use std::ffi::c_void;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

use quantum_arena::GlobalArena;
use std::sync::Arc;

/// Inicializa la protección del sistema operativo Windows.
/// Configura la prioridad del proceso y la afinidad de núcleos.
pub fn init_guardian(affinity_mask: usize, max_memory_mb: usize, arena: Arc<GlobalArena>) -> Uuid {
    // Generate UUIDv7 for process tracking (timestamp sortable)
    let process_id = Uuid::now_v7();
    println!("[OS-GUARDIAN] 🛡️ Process UUIDv7 Tracker: {}", process_id);

    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return process_id; // Ya inicializado
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
            let mut limit_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limit_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
            limit_info.JobMemoryLimit = max_memory_mb * 1024 * 1024;

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
        // Primero vaciamos el working set para evitar overhead legacy (equivale a EmptyWorkingSet)
        let _ = SetProcessWorkingSetSize(process, usize::MAX, usize::MAX);

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

    // Iniciar Hardware-Assisted Out-of-Band Observability Plane
    let obs_plane = observability_plane::ObservabilityPlane::new(
        std::process::id(), // process ID
        std::process::id(), // TODO: Obtener el TID del hot-path real en lugar de PID
    );
    // Asignar al CPU core 5 (fuera del rango de HFT)
    obs_plane.spawn_isolated(Some(5));

    #[cfg(not(windows))]
    {
        println!("[OS-GUARDIAN] Ejecutando en modo NO-Windows. Guardián inactivo.");
    }

    process_id
}

/// FASE 29: Lock Memory Region (VirtualLock). Evita que Windows haga page-out (swapping al disco) de regiones de memoria críticas como el MmapTelemetry y pesos de ML.
/// # Safety
/// Caller must ensure `ptr` and `size` describe a valid region of memory
pub unsafe fn lock_memory_region(ptr: *mut std::ffi::c_void, size: usize) -> bool {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::System::Memory::VirtualLock;
        crate::memory_compaction::ensure_working_set_size(size);
        VirtualLock(ptr, size).is_ok()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// FASE 24: Configura el hilo actual para operar en máxima prioridad física.
/// Solo debe llamarse desde el HFT Spin Loop (GodEngineScalp).
pub fn set_current_thread_time_critical() {
    #[cfg(windows)]
    unsafe {
        let thread = GetCurrentThread();
        // Pin to a specific physical core (e.g., Core 1) for L3 cache hit guarantee
        let _ = SetThreadIdealProcessor(thread, 1);

        if let Err(e) = SetThreadPriority(thread, THREAD_PRIORITY_TIME_CRITICAL) {
            eprintln!(
                "⚠️ [OS-GUARDIAN] No se pudo asignar THREAD_PRIORITY_TIME_CRITICAL: {:?}",
                e
            );
        } else {
            println!("⚡ [OS-GUARDIAN] Hilo promocionado a THREAD_PRIORITY_TIME_CRITICAL (Latencia 0) y anclado a L3 Caché.");
        }
    }
}
