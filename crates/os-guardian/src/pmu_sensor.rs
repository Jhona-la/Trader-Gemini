/// Abstracción para Performance Monitoring Units (PMU).
///
/// En Linux, esto se conectaría a `perf_event_open` para leer:
/// - Instrucciones
/// - Ciclos de CPU
/// - Cache Misses (L1/LLC)
/// - Branch Mispredictions
///
/// Actualmente expone una interfaz "Mock" para entornos no-Linux (ej. Windows)
/// que simula la recolección de HW counters para el motor de anomalías sin bloquear
/// el hot path.

#[derive(Debug, Clone, Copy)]
pub struct PmuVector {
    pub ipc: f64,            // Instructions Per Cycle
    pub l1_miss_rate: f64,   // L1 Cache miss rate
    pub llc_miss_rate: f64,  // Last Level Cache miss rate
    pub branch_mispred: f64, // Branch misprediction rate
    pub cycles: u64,         // Ciclos totales consumidos en la ventana
    pub instructions: u64,   // Instrucciones totales ejecutadas
}

pub struct PmuSensor {
    #[allow(dead_code)]
    target_tid: u32,
}

impl PmuSensor {
    pub fn new(target_tid: u32) -> Self {
        Self { target_tid }
    }

    /// Lee los contadores PMU del TID (Thread ID) objetivo.
    /// Esta operación debe hacerse fuera de banda (Out-of-Band) en un núcleo dedicado.
    #[cfg(target_os = "linux")]
    pub fn sample(&self) -> PmuVector {
        // TODO: Implementar syscall `perf_event_open` nativo para eBPF/perf
        PmuVector {
            ipc: 2.1,
            l1_miss_rate: 0.02,
            llc_miss_rate: 0.005,
            branch_mispred: 0.01,
            cycles: 50_000,
            instructions: 105_000,
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn sample(&self) -> PmuVector {
        // En Windows devolvemos un vector de simulación con ligero ruido estocástico
        // para alimentar el Anomaly Engine
        let base_cycles = 50_000;
        let noise = (unsafe { std::arch::x86_64::_rdtsc() } % 1000) as u64;

        PmuVector {
            ipc: 1.8 + (noise as f64 / 10000.0),
            l1_miss_rate: 0.03 + (noise as f64 / 50000.0),
            llc_miss_rate: 0.01,
            branch_mispred: 0.02,
            cycles: base_cycles + noise,
            instructions: (base_cycles as f64 * 1.8) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pmu_sensor_sample() {
        let sensor = PmuSensor::new(1234);
        let sample = sensor.sample();
        assert!(sample.ipc.is_finite() && sample.ipc > 0.0);
        assert!(sample.l1_miss_rate.is_finite() && sample.l1_miss_rate >= 0.0);
        assert!(sample.cycles > 0);
    }

    #[test]
    fn test_pmu_sensor_vector_finite_metrics() {
        let sensor = PmuSensor::new(5678);
        for _ in 0..5 {
            let sample = sensor.sample();
            assert!(sample.llc_miss_rate.is_finite());
            assert!(sample.branch_mispred.is_finite());
            assert!(sample.instructions > 0);
        }
    }
}
