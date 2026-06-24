import os
path = 'core/rust_engine/src/lib.rs'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

if 'pub mod execution;' not in text:
    text = 'pub mod execution;\n' + text

ffi_append = '''

#[no_mangle]
pub unsafe extern "C" fn ffi_sign_binance_payload(
    secret_ptr: *const c_char,
    payload_ptr: *const c_char,
    out_ptr: *mut c_char,
    max_len: usize
) -> bool {
    if secret_ptr.is_null() || payload_ptr.is_null() || out_ptr.is_null() { return false; }
    
    let secret = CStr::from_ptr(secret_ptr).to_str().unwrap_or("");
    let payload = CStr::from_ptr(payload_ptr).to_str().unwrap_or("");
    
    let sig = execution::sign_binance_payload(secret, payload);
    let sig_bytes = sig.as_bytes();
    
    if sig_bytes.len() >= max_len { return false; }
    
    std::ptr::copy_nonoverlapping(sig_bytes.as_ptr(), out_ptr as *mut u8, sig_bytes.len());
    *out_ptr.add(sig_bytes.len()) = 0; // Null terminator
    
    true
}
'''

if 'ffi_sign_binance_payload' not in text:
    text += ffi_append

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
