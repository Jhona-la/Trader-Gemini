use std::ffi::CStr;
use std::os::raw::c_char;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use std::sync::Mutex;
use lazy_static::lazy_static;
use serde_json::Value;

// Static runtime for network execution to avoid thread creation overhead
lazy_static! {
    static ref RT: Runtime = Runtime::new().expect("Failed to create Tokio runtime");
    static ref GLOBAL_RECEIVER: Mutex<Option<mpsc::Receiver<Value>>> = Mutex::new(None);
}

#[no_mangle]
pub extern "C" fn ffi_start_ws_client(symbols_ptr: *const c_char) -> bool {
    if symbols_ptr.is_null() {
        return false;
    }
    
    let c_str = unsafe { CStr::from_ptr(symbols_ptr) };
    let symbols_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };
    
    let symbols: Vec<String> = symbols_str.split(',').map(|s| s.to_string()).collect();
    
    let (tx, rx) = mpsc::channel(10000);
    
    // Store receiver globally
    *GLOBAL_RECEIVER.lock().unwrap() = Some(rx);
    
    // Start networking client
    RT.spawn(async move {
        let client = crate::networking::BinanceWsClient::new(symbols, tx);
        client.start().await;
    });
    
    true
}

#[no_mangle]
pub extern "C" fn ffi_poll_ws_event(out_buffer: *mut c_char, max_len: usize) -> bool {
    let mut guard = GLOBAL_RECEIVER.lock().unwrap();
    if let Some(rx) = guard.as_mut() {
        if let Ok(val) = rx.try_recv() {
            let json_str = val.to_string();
            let bytes = json_str.as_bytes();
            if bytes.len() < max_len {
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buffer as *mut u8, bytes.len());
                    *out_buffer.add(bytes.len()) = 0; // null terminator
                }
                return true;
            }
        }
    }
    false
}

#[no_mangle]
pub extern "C" fn ffi_execute_order(
    api_key_ptr: *const c_char,
    secret_key_ptr: *const c_char,
    symbol_ptr: *const c_char,
    side_ptr: *const c_char,
    order_type_ptr: *const c_char,
    quantity: f64,
    price: f64,
) -> bool {
    // Safety checks and parsing
    if api_key_ptr.is_null() || secret_key_ptr.is_null() || symbol_ptr.is_null() || side_ptr.is_null() || order_type_ptr.is_null() {
        return false;
    }

    let api_key = unsafe { CStr::from_ptr(api_key_ptr) }.to_string_lossy().into_owned();
    let secret_key = unsafe { CStr::from_ptr(secret_key_ptr) }.to_string_lossy().into_owned();
    let symbol = unsafe { CStr::from_ptr(symbol_ptr) }.to_string_lossy().into_owned();
    let side = unsafe { CStr::from_ptr(side_ptr) }.to_string_lossy().into_owned();
    let order_type = unsafe { CStr::from_ptr(order_type_ptr) }.to_string_lossy().into_owned();
    
    let price_opt = if price <= 0.0 { None } else { Some(price) };

    RT.spawn(async move {
        let executor = crate::executor::BinanceRestExecutor::new(api_key, secret_key, false);
        let _ = executor.create_order(&symbol, &side, &order_type, quantity, price_opt).await;
    });

    true
}
