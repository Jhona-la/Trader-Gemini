pub mod parsers;
pub mod dark_alpha_router;
pub mod dark_alpha_sniffer;
pub mod trailing;
pub mod quantum_arena;
pub mod orderbook;
pub mod config;
pub mod dashboard;
pub mod multi_asset_orchestrator;
pub use quantum_arena::{QuantumRingBuffer, FEATURE_SIZE, QuantumStateArena};
pub use trailing::{evaluate_quantum_trailing, TrailingResult};


#[no_mangle]
pub extern "C" fn quantum_ring_new() -> *mut QuantumRingBuffer {
    let b = Box::new(QuantumRingBuffer::new());
    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn quantum_ring_free(ptr: *mut QuantumRingBuffer) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}



#[no_mangle]
pub extern "C" fn quantum_ring_read_tick(
    ring: *const QuantumRingBuffer,
    read_idx: usize,
    out_payload: *mut f32
) -> bool {
    if ring.is_null() || out_payload.is_null() {
        return false;
    }
    
    unsafe {
        let out_slice = std::slice::from_raw_parts_mut(out_payload, FEATURE_SIZE);
        let mut temp_payload = [0.0f32; FEATURE_SIZE];
        let success = (*ring).read_tick(read_idx, &mut temp_payload);
        if success {
            out_slice.copy_from_slice(&temp_payload);
        }
        success
    }
}




