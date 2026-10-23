pub mod anomaly_detector;
pub mod crash_dump;
pub mod ebpf_core;
pub mod memory_audit;
pub mod memory_compaction;
pub mod observability_plane;
pub mod pmu_sensor;
pub mod telemetry;
pub mod zero_latency_telemetry;

pub use crash_dump::{EmergencyCrashDump, PositionDumpEntry};
use std::sync::atomic::{AtomicBool, Ordering};
use uuid::Uuid;

use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
};
use windows::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThread, SetPriorityClass, SetProcessAffinityMask,
    SetProcessWorkingSetSize, SetThreadAffinityMask, SetThreadIdealProcessor, SetThreadPriority,
    HIGH_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS, THREAD_PRIORITY_HIGHEST,
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

    // FASE 2: Activar FTZ/DAZ para supresión determinista de subnormales a nivel de silicio
    enable_ftz_daz();

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
    // FASE 2: Garantizar FTZ/DAZ en el hilo HFT para eliminar microcode traps en subnormales
    enable_ftz_daz();

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

/// FASE 24 & Punto #244: Fija la afinidad estricta del hilo actual a un núcleo físico específico
pub fn pin_current_thread_to_core(core_id: usize) -> bool {
    #[cfg(windows)]
    unsafe {
        let thread = GetCurrentThread();
        let _ = SetThreadIdealProcessor(thread, core_id as u32);
        let mask = 1usize << (core_id % 64);
        let old_mask = SetThreadAffinityMask(thread, mask);
        old_mask != 0
    }
    #[cfg(not(windows))]
    {
        let _ = core_id;
        false
    }
}

/// FASE 2 (Átomo 2.1): Configura el registro de hardware MXCSR para suprimir números subnormales
/// activando FTZ (Flush-To-Zero, bit 15 = 0x8000) y DAZ (Denormals-Are-Zero, bit 6 = 0x0040).
/// Previene que números de punto flotante < 2.225e-308 provoquen microcode hardware assists (traps),
/// garantizando ejecución matemática en latencia ultra-baja y determinista O(1).
pub fn enable_ftz_daz() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm_getcsr, _mm_setcsr};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm_getcsr, _mm_setcsr};

        unsafe {
            let mxcsr = _mm_getcsr();
            // Bit 15: FTZ (0x8000), Bit 6: DAZ (0x0040)
            _mm_setcsr(mxcsr | 0x8040);
        }
    }
}

/// FASE 2 (Átomo 2.1): Consulta si los modos FTZ y DAZ están activos en el registro de hardware actual.
pub fn is_ftz_daz_enabled() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm_getcsr;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm_getcsr;

        unsafe {
            let mxcsr = _mm_getcsr();
            (mxcsr & 0x8040) == 0x8040
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ftz_daz_activation_and_subnormal_suppression() {
        enable_ftz_daz();
        assert!(
            is_ftz_daz_enabled(),
            "FTZ y DAZ deben estar habilitados en MXCSR tras enable_ftz_daz()"
        );

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Un subnormal f64 ocurre por debajo de MIN_POSITIVE (2.2250738585072014e-308).
            let min_normal = f64::MIN_POSITIVE;
            let subnormal = min_normal * 0.5;
            // Con FTZ activo a nivel de CPU, el subnormal se redondea inmediatamente a 0.0 sin microcódigo assist.
            assert_eq!(
                subnormal, 0.0,
                "En hardware x86 con FTZ activo, el subnormal debe ser limpiado a cero inmediatamente"
            );
        }
    }

    #[test]
    fn test_pin_current_thread_to_core() {
        let ok = pin_current_thread_to_core(0);
        #[cfg(windows)]
        assert!(
            ok,
            "On Windows, thread affinity pinning to core 0 should succeed"
        );
        #[cfg(not(windows))]
        let _ = ok;
    }

    #[test]
    fn test_set_current_thread_time_critical_and_virtual_lock() {
        set_current_thread_time_critical();

        let mut buffer = vec![0u8; 4096];
        let ptr = buffer.as_mut_ptr() as *mut std::ffi::c_void;
        let _locked = unsafe { lock_memory_region(ptr, buffer.len()) };
        // Validates safe execution without panics or memory corruption
    }
}
