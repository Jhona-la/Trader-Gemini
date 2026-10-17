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

                // Polling loop inside the isolated core.
                // GLM 133 (política anti-mock del operador + CL-15): el
                // detector SOLO observa con muestras REALES. Sin sensor
                // PMU (ninguna plataforma lo tiene hoy) o sin eventos
                // eBPF (Linux sin mapa BPF), el tick se cuenta como
                // ausencia y NO se fabrica score — el EWMA ya no aprende
                // de constantes ficticias.
                let mut ticks_sin_sensor: u64 = 0;
                loop {
                    match (self.pmu.sample(), self.ebpf.read_events()) {
                        (Some(pmu_vec), Some(kernel_ev)) => {
                            let anomaly_score = self.detector.observe(&pmu_vec, &kernel_ev);
                            if anomaly_score.is_anomaly {
                                println!(
                                    "[OBSERVABILITY-ALERT] Riesgo en HFT Detectado! Score: {:.2}",
                                    anomaly_score.score
                                );
                                println!(
                                    "[OBSERVABILITY-CAUSAL-HINT] {}",
                                    anomaly_score.causal_hint
                                );
                            }
                        }
                        _ => {
                            ticks_sin_sensor += 1;
                            // Telemetría de ausencia cada 100k ticks: la
                            // honestidad del plano es observable.
                            if ticks_sin_sensor % 100_000 == 0 {
                                println!(
                                    "[OBSERVABILITY] {ticks_sin_sensor} ticks sin sensor real (PMU: ausente; eBPF: {:?}) — sin observación fabricada",
                                    cfg!(target_os = "windows")
                                );
                            }
                        }
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

    /// GLM 133: sin sensores reales, el plano NO fabrica observación.
    /// El sample es ausencia (None) — contrato de la política anti-mock.
    #[test]
    fn test_observability_plane_sin_sensor_no_opina() {
        let plane = ObservabilityPlane::new(100, 200);
        assert!(
            plane.pmu.sample().is_none(),
            "PMU sin backend real ⇒ None (CL-15)"
        );
        // eBPF: en Windows hay eventos reales (page_faults); el contrato
        // es Option — la ausencia nunca es KernelEvents fabricado.
        let _ = plane.ebpf.read_events();
    }
}
