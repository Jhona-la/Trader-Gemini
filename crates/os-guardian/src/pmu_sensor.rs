/// Abstracción para Performance Monitoring Units (PMU).
///
/// En Linux, esto se conectaría a `perf_event_open` para leer:
/// - Instrucciones
/// - Ciclos de CPU
/// - Cache Misses (L1/LLC)
/// - Branch Mispredictions
///
/// GLM 133 (política del operador: CERO código simulado en producción):
/// este sensor devolvía CONSTANTES FICTICIAS en ambas plataformas (y en
/// Windows «ruido» generado con `_rdtsc() % 1000`) que alimentaban el
/// detector de anomalías con datos que no existen — telemetría fantasma
/// que además fabricaba baseline EWMA (MOD6/8-026 ya lo marcaba).
/// Doctrina CL-15 del repo: SIN contador real NO HAY muestra —
/// `sample()` devuelve `None` hasta que exista un backend
/// `perf_event_open` real. La ausencia es honesta; el número inventado
/// no vuelve.

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
    target_tid: u32,
}

impl PmuSensor {
    pub fn new(target_tid: u32) -> Self {
        Self { target_tid }
    }

    /// Lee los contadores PMU del TID objetivo. `None` = sin sensor real
    /// disponible en esta plataforma/build — el consumidor NO debe
    /// fabricar observación (ausencia honesta, CL-15). Cuando se
    /// implemente `perf_event_open` nativo (Linux) este método pasará a
    /// devolver `Some` con contadores reales; hasta entonces NO existe
    /// medición PMU en el sistema y ningún componente debe comportarse
    /// como si existiera.
    pub fn sample(&self) -> Option<PmuVector> {
        let _ = self.target_tid;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GLM 133: sin backend real NO HAY muestra — jamás constantes.
    /// Este test es el contrato de la política anti-mock del operador
    /// para este sensor: si algún día empieza a devolver Some, debe ser
    /// porque existe un `perf_event_open` real cableado (y el test se
    /// actualiza a la par con verificación de plausibilidad de medidas
    /// reales, no de literales).
    #[test]
    fn sin_backend_real_el_sensor_no_opina() {
        let sensor = PmuSensor::new(1234);
        assert!(
            sensor.sample().is_none(),
            "CL-15: sin contador real la muestra es AUSENCIA (None), no literales"
        );
    }

    /// El vector sigue siendo construible por backends reales (contrato
    /// de forma para el futuro perf_event_open).
    #[test]
    fn pmu_vector_construible_para_backend_real() {
        let v = PmuVector {
            ipc: 1.9,
            l1_miss_rate: 0.01,
            llc_miss_rate: 0.002,
            branch_mispred: 0.01,
            cycles: 1_000,
            instructions: 1_900,
        };
        assert!(v.ipc.is_finite() && v.cycles > 0);
    }
}
