pub mod atomic_telemetry;

/// Initialize the telemetry engine with a fixed capacity ring buffer
pub fn init_telemetry(capacity: usize) {
    atomic_telemetry::init_binary_telemetry(capacity);
}

pub fn pop_telemetry() -> Option<String> {
    atomic_telemetry::pop_binary_telemetry()
        .map(|(buf, len)| String::from_utf8_lossy(&buf[..len]).into_owned())
}

/// A zero-latency replacement for `println!` that writes directly to a stack buffer
/// and pushes to the lock-free ring buffer without heap allocations.
#[macro_export]
macro_rules! telemetry {
    ($($arg:tt)*) => {
        {
            use core::fmt::Write;
            let mut buf = $crate::atomic_telemetry::StackBuffer { buf: [0; 512], len: 0 };
            let _ = write!(&mut buf, $($arg)*);
            $crate::atomic_telemetry::push_binary_telemetry(&buf.buf[..buf.len]);
        }
    };
}

#[macro_export]
macro_rules! telemetry_err {
    ($($arg:tt)*) => {
        {
            use core::fmt::Write;
            let mut buf = $crate::atomic_telemetry::StackBuffer { buf: [0; 512], len: 0 };
            let _ = write!(&mut buf, "⚠️ ERROR: ");
            let _ = write!(&mut buf, $($arg)*);
            $crate::atomic_telemetry::push_binary_telemetry(&buf.buf[..buf.len]);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_telemetry_stack_buffer_and_pop() {
        init_telemetry(32);
        telemetry!("TEST_SYSTEM_NOMINAL_TICK: {}", 42);
        telemetry_err!("TEST_SIMULATED_REJECTION: code={}", 500);

        let msg1 = pop_telemetry();
        assert!(msg1.is_some());
        assert!(msg1.unwrap().contains("TEST_SYSTEM_NOMINAL_TICK"));

        let msg2 = pop_telemetry();
        assert!(msg2.is_some());
        assert!(msg2.unwrap().contains("ERROR: TEST_SIMULATED_REJECTION"));
    }
}
