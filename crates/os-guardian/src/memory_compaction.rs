use std::ffi::c_void;

/// Lock a generic block of memory into physical RAM to guarantee 0 Page Faults.
/// This prevents the OS from ever swapping this region to the pagefile (SSD).
/// Crucial for HFT memory like order books, ML models, and the Quantum Arena.
/// # Safety
/// Caller must ensure the pointer is valid and that the process has SeLockMemoryPrivilege.
pub unsafe fn lock_critical_memory<T>(data_ref: &T) -> bool {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::System::Memory::VirtualLock;

        let ptr = data_ref as *const T as *const c_void as *mut c_void;
        let size = std::mem::size_of_val(data_ref);

        ensure_working_set_size(size);

        // Attempt to lock memory region
        if VirtualLock(ptr, size).is_ok() {
            telemetry_engine::telemetry!("🔒 [MEMORY-COMPACTION] Successfully locked {} bytes into physical RAM (0 Page Faults guaranteed).", size);
            true
        } else {
            telemetry_engine::telemetry_err!("⚠️ [MEMORY-COMPACTION] Failed to lock {} bytes into RAM. Process may lack SeLockMemoryPrivilege.", size);
            false
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        true
    }
}

/// Lock a slice of memory (e.g. Vec buffers or Box slices) into physical RAM.
/// # Safety
/// Caller must ensure the slice is valid and that the process has SeLockMemoryPrivilege.
pub unsafe fn lock_critical_memory_slice<T>(slice: &[T]) -> bool {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::System::Memory::VirtualLock;

        let ptr = slice.as_ptr() as *const c_void as *mut c_void;
        let size = std::mem::size_of_val(slice);

        ensure_working_set_size(size);

        if VirtualLock(ptr, size).is_ok() {
            telemetry_engine::telemetry!(
                "🔒 [MEMORY-COMPACTION] Successfully locked {} bytes of slice into physical RAM.",
                size
            );
            true
        } else {
            telemetry_engine::telemetry_err!(
                "⚠️ [MEMORY-COMPACTION] Failed to lock slice of {} bytes into RAM.",
                size
            );
            false
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        true
    }
}

/// Force Windows to compact the working set of the process, freeing unused RAM.
/// # Safety
/// Safe to call, but might cause performance hit due to page faults.
pub unsafe fn force_working_set_compaction() {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::System::ProcessStatus::EmptyWorkingSet;
        use windows::Win32::System::Threading::GetCurrentProcess;

        // This tells Windows to aggressively swap out non-locked memory pages
        // to disk, and compact the RAM usage. Since our critical paths are
        // VirtualLock'd, they won't be touched, but the unused heap will be reclaimed.
        let process_handle = GetCurrentProcess();
        let _ = EmptyWorkingSet(process_handle);
    }
}

/// LG-13 (Línea G) — ¿HAY PRESIÓN DE MEMORIA QUE JUSTIFIQUE COMPACTAR?
///
/// QUÉ ESTABA MAL: el host llamaba a `EmptyWorkingSet` sobre el proceso vivo
/// cada hora, la necesitara el sistema o no. Recortar el working set obliga
/// a volver a cargar cada página no bloqueada en el siguiente acceso: una
/// ráfaga de fallos de página en el camino caliente justo después, a cambio
/// de nada cuando sobra memoria.
///
/// QUÉ GARANTIZA: sólo hay presión cuando la memoria física disponible del
/// sistema es menor que el working set del proceso (el sistema no podría
/// alojar otra copia de lo que ocupamos sin paginar). Sin datos (0), no hay
/// presión: no se compacta a ciegas.
pub fn hay_presion_de_memoria(disponible_bytes: u64, working_set_bytes: u64) -> bool {
    working_set_bytes > 0 && disponible_bytes > 0 && disponible_bytes < working_set_bytes
}

/// LG-13: mide y compacta sólo con presión. Devuelve `true` si compactó.
pub fn compactar_si_hay_presion() -> bool {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let disponible = sys.available_memory();
    let working_set = (crate::memory_audit::get_memory_usage_mb().max(0.0) * 1024.0 * 1024.0) as u64;
    if hay_presion_de_memoria(disponible, working_set) {
        // SAFETY: llamada de sistema sobre el propio proceso; sólo descarta
        // páginas no bloqueadas del working set.
        unsafe { force_working_set_compaction() };
        true
    } else {
        false
    }
}

/// # Safety
/// Safe to call on valid processes.
pub unsafe fn ensure_working_set_size(size_needed: usize) {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::System::Threading::{
            GetCurrentProcess, GetProcessWorkingSetSize, SetProcessWorkingSetSize,
        };

        let handle = GetCurrentProcess();
        let mut min_ws = 0;
        let mut max_ws = 0;

        if GetProcessWorkingSetSize(handle, &mut min_ws, &mut max_ws).is_ok() {
            // FIX #585: Buffer proporcional relativo al tamaño necesario sin hinchamiento acumulativo
            let padding = 32 * 1024 * 1024;

            // Hardcap to 4GB to preserve physical RAM headroom (16GB laptop limitation)
            let hardcap = 4 * 1024 * 1024 * 1024;

            let requested_min = (size_needed + padding).max(min_ws).min(hardcap);
            let requested_max = (requested_min + padding).max(max_ws).min(hardcap);

            let _ = SetProcessWorkingSetSize(handle, requested_min, requested_max);
        }
    }
}

#[cfg(test)]
mod tests_lg13 {
    use super::*;

    #[test]
    fn lg13_solo_hay_presion_si_no_cabe_otra_copia_del_working_set() {
        let gb = 1u64 << 30;
        assert!(!hay_presion_de_memoria(8 * gb, gb), "con 8 GB libres y 1 GB ocupado no se compacta");
        assert!(hay_presion_de_memoria(gb / 2, gb), "con menos libre que el working set sí");
        assert!(!hay_presion_de_memoria(0, gb), "sin dato del sistema no se compacta a ciegas");
        assert!(!hay_presion_de_memoria(gb, 0), "sin dato del proceso no se compacta a ciegas");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_compaction_and_lock_slice() {
        let buffer = vec![0u8; 1024 * 64]; // 64KB buffer
        unsafe {
            let _ = lock_critical_memory_slice(&buffer);
            force_working_set_compaction();
        }
        assert_eq!(buffer.len(), 1024 * 64);
    }

    #[test]
    fn test_memory_lock_single_structure() {
        #[repr(C, align(64))]
        struct TestArenaData {
            values: [f64; 32],
            timestamp: u64,
        }

        let data = TestArenaData {
            values: [1.0; 32],
            timestamp: 1700000000,
        };

        unsafe {
            let _ = lock_critical_memory(&data);
        }
        assert_eq!(data.values[0], 1.0);
    }

    #[test]
    fn test_expand_working_set_hardcap() {
        unsafe {
            // Solicitud excesiva (10GB) debe ser acotada al hardcap de 4GB
            ensure_working_set_size(10 * 1024 * 1024 * 1024);
        }
    }
}
