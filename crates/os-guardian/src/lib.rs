pub mod memory_audit;

use std::sync::atomic::{AtomicBool, Ordering};
use uuid::Uuid;

use windows::Win32::System::Threading::{
    GetCurrentProcess, SetPriorityClass, SetProcessAffinityMask, NORMAL_PRIORITY_CLASS,
};
use windows::Win32::System::JobObjects::{
    CreateJobObjectW, AssignProcessToJobObject, SetInformationJobObject,
    JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_JOB_MEMORY
};

use std::ffi::c_void;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Inicializa la protección del sistema operativo Windows.
/// Configura la prioridad del proceso y la afinidad de núcleos.
pub fn init_guardian(affinity_mask: usize, max_memory_mb: usize) -> Uuid {
    // Generate UUIDv7 for process tracking (timestamp sortable)
    let process_id = Uuid::now_v7();
    println!("[OS-GUARDIAN] 🛡️ Process UUIDv7 Tracker: {}", process_id);

    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return process_id; // Ya inicializado
    }
    
    #[cfg(windows)]
    unsafe {
        let process = GetCurrentProcess();
        
        // 1. Establecer prioridad normal
        if let Err(e) = SetPriorityClass(process, NORMAL_PRIORITY_CLASS) {
            eprintln!("[OS-GUARDIAN] Error estableciendo prioridad normal: {:?}", e);
        } else {
            println!("[OS-GUARDIAN] Prioridad del proceso establecida en NORMAL.");
        }
        
        // 2. Establecer afinidad de CPU
        if SetProcessAffinityMask(process, affinity_mask).is_err() {
            eprintln!("[OS-GUARDIAN] Error estableciendo afinidad de CPU.");
        } else {
            println!("[OS-GUARDIAN] Afinidad de CPU establecida con máscara: {:#X}.", affinity_mask);
        }

        // 3. Establecer límites de Memoria vía JobObject
        if let Ok(job) = CreateJobObjectW(None, None) {
            let mut limit_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limit_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
            limit_info.JobMemoryLimit = (max_memory_mb * 1024 * 1024);

            let result = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &limit_info as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );

            if result.is_ok() {
                if AssignProcessToJobObject(job, process).is_ok() {
                    println!("[OS-GUARDIAN] Job Object Memory Limit establecido a {} MB.", max_memory_mb);
                } else {
                    eprintln!("[OS-GUARDIAN] Error asignando proceso al JobObject.");
                }
            } else {
                eprintln!("[OS-GUARDIAN] Error configurando Memory Limit en JobObject.");
            }
        }
    }
    
    // Iniciar el auditor dinámico
    memory_audit::start_memory_auditor(max_memory_mb);

    #[cfg(not(windows))]
    {
        println!("[OS-GUARDIAN] Ejecutando en modo NO-Windows. Guardián inactivo.");
    }
    
    process_id
}
