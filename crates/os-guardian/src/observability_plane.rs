use crate::anomaly_detector::{AnomalyScore, StatisticalAnomalyDetector};
use crate::ebpf_core::{EbpfSensor, KernelEvents};
use crate::pmu_sensor::{PmuSensor, PmuVector};
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
    pub fn spawn_isolated(self, core_id: Option<usize>) {
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

                    // Dormir 1 milisegundo para no saturar el core aislado (resolución de 1ms)
                    thread::sleep(Duration::from_millis(1));
                }
            })
            .expect("Failed to spawn ObservabilityPlane");
    }
}
