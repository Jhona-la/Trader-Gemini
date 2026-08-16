use sysinfo::System;
use std::sync::Mutex;
use lazy_static::lazy_static;

lazy_static! {
    static ref SYS: Mutex<System> = Mutex::new(System::new_all());
}

#[derive(Debug, Clone)]
pub struct SystemTelemetry {
    pub cpu_usage: f32, // percentage
    pub memory_used_mb: f64,
    pub total_memory_mb: f64,
}

pub fn get_system_telemetry() -> SystemTelemetry {
    let mut sys = SYS.lock().unwrap();
    sys.refresh_cpu_all();
    sys.refresh_memory();
    
    let cpu_usage = sys.global_cpu_usage();
    let memory_used = sys.used_memory() as f64 / (1024.0 * 1024.0);
    let total_memory = sys.total_memory() as f64 / (1024.0 * 1024.0);
    
    SystemTelemetry {
        cpu_usage,
        memory_used_mb: memory_used,
        total_memory_mb: total_memory,
    }
}
