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
            telemetry_engine::telemetry!("🔒 [MEMORY-COMPACTION] Successfully locked {} bytes of slice into physical RAM.", size);
            true
        } else {
            telemetry_engine::telemetry_err!("⚠️ [MEMORY-COMPACTION] Failed to lock slice of {} bytes into RAM.", size);
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

/// # Safety
/// Safe to call on valid processes.
pub unsafe fn ensure_working_set_size(size_needed: usize) {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessWorkingSetSize, SetProcessWorkingSetSize};
        
        let handle = GetCurrentProcess();
        let mut min_ws = 0;
        let mut max_ws = 0;
        
        if GetProcessWorkingSetSize(handle, &mut min_ws, &mut max_ws).is_ok() {
            // Give ourselves a nice buffer: requested + 32MB padding to avoid OS thrashing
            let padding = 32 * 1024 * 1024;
            
            // Hardcap to 6GB to prevent total OS freezing (16GB laptop limitation)
            let hardcap = 6 * 1024 * 1024 * 1024;
            
            let requested_min = (min_ws + size_needed + padding).min(hardcap);
            let requested_max = (max_ws + size_needed + padding * 2).min(hardcap);
            
            let _ = SetProcessWorkingSetSize(handle, requested_min, requested_max);
        }
    }
}
