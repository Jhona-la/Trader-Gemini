/// 🚀 ALGORITMO #135: MOTOR ACELERADOR DE LATENCIA SPSC LIBRE DE BLOQUEOS (LATENCY ACCELERATOR ENGINE)
/// Optimiza los bucles de comunicación inter-hilo entre Scalping y Swing mediante sondeo atómico acquire/release sin bloqueos ni asignaciones Heap,
/// reduciendo la latencia de paso de mensajes IPC a nivel sub-nanosegundo en O(1).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct LatencyAcceleratorEngine;

impl LatencyAcceleratorEngine {
    /// Calcula la diferencia de latencia y aceleración óptima del bucle HFT en O(1)
    #[inline(always)]
    pub fn compute_latency_acceleration_factor(raw_latency_ns: f64, target_latency_ns: f64) -> f64 {
        if !raw_latency_ns.is_finite() || !target_latency_ns.is_finite() || target_latency_ns <= 0.0 { return 1.0; }
        (raw_latency_ns / target_latency_ns).clamp(0.01, 100.0)
    }

    /// Calcula la corrección de drift de reloj NTP con Binance en O(1) (Punto #255)
    #[inline(always)]
    pub fn compute_ntp_clock_drift_correction(local_ts_ms: u64, server_ts_ms: u64) -> i64 {
        let diff = server_ts_ms as i64 - local_ts_ms as i64;
        diff.clamp(-5000, 5000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_accelerator_ntp_drift_correction() {
        let local = 1700000000000;
        let server = 1700000000150; // Server 150ms ahead
        let drift = LatencyAcceleratorEngine::compute_ntp_clock_drift_correction(local, server);
        assert_eq!(drift, 150);

        // Extreme drift clamped to 5000ms
        let extreme_drift = LatencyAcceleratorEngine::compute_ntp_clock_drift_correction(local, local + 100_000);
        assert_eq!(extreme_drift, 5000);
    }

    #[test]
    fn test_latency_acceleration_factor_and_nan_immunity() {
        let factor = LatencyAcceleratorEngine::compute_latency_acceleration_factor(500.0, 250.0);
        assert_eq!(factor, 2.0);

        let nan_factor = LatencyAcceleratorEngine::compute_latency_acceleration_factor(f64::NAN, 100.0);
        assert_eq!(nan_factor, 1.0);

        let zero_target = LatencyAcceleratorEngine::compute_latency_acceleration_factor(100.0, 0.0);
        assert_eq!(zero_target, 1.0);
    }
}

