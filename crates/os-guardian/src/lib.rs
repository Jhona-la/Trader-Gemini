use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(windows)]
use windows::Win32::System::Threading::{
    GetCurrentProcess, SetPriorityClass, SetProcessAffinityMask, NORMAL_PRIORITY_CLASS,
};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Inicializa la protección del sistema operativo Windows.
/// Configura la prioridad del proceso y la afinidad de núcleos.
/// 
/// `affinity_mask`: Máscara de bits para la afinidad. 
/// Ejemplo: Si el Ryzen 7 5700U tiene 16 hilos (0-15), y queremos usar del 8 al 15:
/// Máscara = (1 << 8) | (1 << 9) | ... | (1 << 15) = 0xFF00 = 65280
pub fn init_guardian(affinity_mask: usize) {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return; // Ya inicializado
    }
    
    #[cfg(windows)]
    unsafe {
        let process = GetCurrentProcess();
        
        // 1. Establecer prioridad normal (Axioma XX: No robar ciclos al DWM)
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
    }
    
    #[cfg(not(windows))]
    {
        println!("[OS-GUARDIAN] Ejecutando en modo NO-Windows. Guardián inactivo.");
    }
}
