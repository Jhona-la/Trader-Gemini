import os

code = """
// =========================================================
// MATH KERNELS EXPORTS (FFI)
// =========================================================

#[no_mangle]
pub extern "C" fn ffi_compute_kelly_fraction(
    p: f64, b: f64, apply_mult: bool, kelly_mult: f64, stress_score: f64, max_exposure: f64
) -> f64 {
    math_kernels::compute_kelly_fraction(p, b, apply_mult, kelly_mult, stress_score, max_exposure)
}

#[no_mangle]
pub extern "C" fn ffi_extract_kelly_stats(
    pnl_ptr: *const f64, pnl_len: usize, 
    is_win_ptr: *const bool, is_win_len: usize,
    out_p: *mut f64, out_b: *mut f64
) {
    if pnl_ptr.is_null() || is_win_ptr.is_null() || out_p.is_null() || out_b.is_null() || pnl_len != is_win_len {
        return;
    }
    let pnl_slice = unsafe { std::slice::from_raw_parts(pnl_ptr, pnl_len) };
    let is_win_slice = unsafe { std::slice::from_raw_parts(is_win_ptr, is_win_len) };
    
    let (p, b) = math_kernels::extract_kelly_stats(pnl_slice, is_win_slice);
    unsafe {
        *out_p = p;
        *out_b = b;
    }
}

#[no_mangle]
pub extern "C" fn ffi_compute_cvar(
    loss_ptr: *const f64, loss_len: usize, confidence_level: f64
) -> f64 {
    if loss_ptr.is_null() || loss_len == 0 {
        return 0.0;
    }
    let loss_slice = unsafe { std::slice::from_raw_parts(loss_ptr, loss_len) };
    math_kernels::compute_cvar(loss_slice, confidence_level)
}
"""

with open(r'core\rust_engine\src\lib.rs', 'a', encoding='utf-8') as f:
    f.write(code)
