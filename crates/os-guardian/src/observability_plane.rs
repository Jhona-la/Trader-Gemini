use crate::anomaly_detector::StatisticalAnomalyDetector;
use crate::ebpf_core::EbpfSensor;
use crate::pmu_sensor::PmuSensor;
use std::thread;
use std::time::Duration;

/// The Hardware-Assisted Out-of-Band Observability Plane.
/// This runs on its own isolated CPU core (e.g. CPU 5).
/// It loops reading PMU and eBPF kernel events without interfering
/// with the HFT Hot Path (CPU 0-3).
pub struct ObservabilityPlane {
    pmu: PmuSensor,
    ebpf: EbpfSensor,
    detector: StatisticalAnomalyDetector,
}

impl ObservabilityPlane {
    pub fn new(target_pid: u32, target_tid: u32) -> Self {
        Self {
            pmu: PmuSensor::new(target_tid),
            ebpf: EbpfSensor::new(target_pid),
            detector: StatisticalAnomalyDetector::new(0.01), // Alpha for EWMA
        }
    }

    /// Spawns the dedicated telemetry thread, optionally pinned to a CPU core.
    pub fn spawn_isolated(self, _core_id: Option<usize>) {
        thread::Builder::new()
            .name("OOB-Observability-Plane".into())
            .spawn(move || {
                #[cfg(target_os = "linux")]
                if let Some(core) = core_id {
                    // Set core affinity using libc (Linux)
                    unsafe {
                        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
                        libc::CPU_ZERO(&mut cpuset);
                        libc::CPU_SET(core, &mut cpuset);
                        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
                    }
                }

                // MOD6/8-026 (INFORME DECIMOCUARTO): en Windows el pinning es
                // cfg(linux) y PMU/eBPF son MOCKS — el hilo flotaba libre
                // despertando 1000×/s para alimentar un detector EWMA con
                // constantes: telemetría fantasma Y fuente de jitter para el
                // hot path. mock data — polling reducido para no injectar
                // jitter (MOD6/8-026): 10 Hz en Windows, 1 kHz donde hay
                // sensores reales + core aislado.
                #[cfg(target_os = "windows")]
                let poll_interval = Duration::from_millis(100);
                #[cfg(not(target_os = "windows"))]
                let poll_interval = Duration::from_millis(1);

                // Polling loop inside the isolated core
                loop {
                    let pmu_vec = self.pmu.sample();
                    let kernel_ev = self.ebpf.read_events();

                    let anomaly_score = self.detector.observe(&pmu_vec, &kernel_ev);

                    if anomaly_score.is_anomaly {
                        println!(
                            "[OBSERVABILITY-ALERT] Riesgo en HFT Detectado! Score: {:.2}",
                            anomaly_score.score
                        );
                        println!("[OBSERVABILITY-CAUSAL-HINT] {}", anomaly_score.causal_hint);

                        // Aquí se inyectaría la mitigación en el genoma o se activaría un kill switch en el engine
                    }

                    thread::sleep(poll_interval);
                }
            })
            .expect("Failed to spawn ObservabilityPlane");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observability_plane_new() {
        let plane = ObservabilityPlane::new(100, 200);
        let pmu = plane.pmu.sample();
        assert!(pmu.ipc.is_finite());
    }
}
