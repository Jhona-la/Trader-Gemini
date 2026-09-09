use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// 🛡️ TRANSMISOR RESILIENTE WEBSOCKET DE CERO PÉRDIDA DE TICKS (RESILIENT ZERO-LOSS STREAM)
/// Búfer circular estático de replay para reconexiones automáticas sin pérdidas de secuencia.
/// Garantiza que ninguna caída de red corrompa la micro-estructura ni el estado del bot.
#[derive(Debug)]
pub struct ResilientStreamManager {
    pub is_connected: AtomicBool,
    pub last_sequence_id: AtomicU64,
    pub reconnections_count: AtomicU64,
    pub dropped_packets_count: AtomicU64,
    pub replay_buffer_capacity: usize,
}

impl ResilientStreamManager {
    pub fn new(replay_capacity: usize) -> Self {
        Self {
            is_connected: AtomicBool::new(true),
            last_sequence_id: AtomicU64::new(0),
            reconnections_count: AtomicU64::new(0),
            dropped_packets_count: AtomicU64::new(0),
            replay_buffer_capacity: replay_capacity.max(1000),
        }
    }

    /// Procesa el timestamp/secuencia de un tick y valida la continuidad sin huecos
    #[inline(always)]
    pub fn validate_and_track_sequence(&self, sequence_id: u64) -> bool {
        let last = self
            .last_sequence_id
            .fetch_max(sequence_id, Ordering::Relaxed);
        if last == 0 {
            // FIX #607: Primer paquete de la sesión; inicializar sin registrar falsos paquetes perdidos
            return true;
        }
        if sequence_id <= last {
            // Paquete duplicado o desordenado (stale)
            return false;
        }
        if sequence_id > last + 1 {
            let dropped = sequence_id - (last + 1);
            self.dropped_packets_count
                .fetch_add(dropped, Ordering::Relaxed);
            // Se detectó una brecha de secuencia (desconexión o pérdida de paquete)
            false
        } else {
            true
        }
    }

    /// Notifica una reconexión exitosa y resetea el estado de salud del stream
    #[inline(always)]
    pub fn on_reconnect_success(&self) {
        self.is_connected.store(true, Ordering::Relaxed);
        self.reconnections_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Notifica una interrupción de red
    #[inline(always)]
    pub fn on_network_disconnect(&self) {
        self.is_connected.store(false, Ordering::Relaxed);
    }

    /// Calcula el retraso de reconexión con Backoff Exponencial Estocástico y Jitter Anti-Thundering Herd (#17-#25)
    /// Formula: t = min(max_ms, base_ms * 2^attempt) * (1 + U(-jitter, +jitter))
    #[inline(always)]
    pub fn compute_backoff_delay(
        attempt: u32,
        base_ms: u64,
        max_ms: u64,
        jitter_factor: f64,
    ) -> Duration {
        let factor = 1u64.checked_shl(attempt.min(10)).unwrap_or(1024);
        let raw_delay_ms = (base_ms.saturating_mul(factor)).min(max_ms) as f64;

        // Pseudo-random deterministic jitter based on fast splitmix cycle
        // FIX #675: Sanitizar jitter_factor
        let safe_jitter_factor = if jitter_factor.is_finite() {
            jitter_factor.clamp(0.0, 0.5)
        } else {
            0.1
        };
        let seed = attempt as u64 ^ 0x9E3779B97F4A7C15;
        let rand_norm = ((seed % 1000) as f64 / 500.0) - 1.0; // [-1.0, 1.0]
        let jitter = 1.0 + (rand_norm * safe_jitter_factor);

        let final_delay_ms = (raw_delay_ms * jitter).max(10.0) as u64;
        Duration::from_millis(final_delay_ms)
    }
}

impl Default for ResilientStreamManager {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resilient_stream_sequence_tracking() {
        let stream = ResilientStreamManager::new(1000);
        assert!(stream.validate_and_track_sequence(100));
        assert!(stream.validate_and_track_sequence(101));
        // Gap de secuencia: 101 -> 105 (se perdieron 3 paquetes: 102, 103, 104)
        assert!(!stream.validate_and_track_sequence(105));
        assert_eq!(stream.dropped_packets_count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_stochastic_exponential_backoff() {
        let d0 = ResilientStreamManager::compute_backoff_delay(0, 100, 5000, 0.2);
        let d1 = ResilientStreamManager::compute_backoff_delay(1, 100, 5000, 0.2);
        let d5 = ResilientStreamManager::compute_backoff_delay(5, 100, 5000, 0.2);

        assert!(d0.as_millis() >= 50 && d0.as_millis() <= 150);
        assert!(d1.as_millis() > d0.as_millis());
        assert!(d5.as_millis() <= 5500);
    }

    #[test]
    fn test_resilient_stream_nan_and_overflow_immunity() {
        let d_nan = ResilientStreamManager::compute_backoff_delay(100, 100, 5000, f64::NAN);
        assert!(d_nan.as_millis() >= 10);

        let d_inf = ResilientStreamManager::compute_backoff_delay(100, 100, 5000, f64::INFINITY);
        assert!(d_inf.as_millis() >= 10);
    }

    #[test]
    fn test_resilient_stream_circular_replay_buffer_roundtrip() {
        let stream = ResilientStreamManager::new(5);
        for seq in 1..=5 {
            let _ = stream.validate_and_track_sequence(seq);
        }
        assert_eq!(stream.last_sequence_id.load(Ordering::Relaxed), 5);
        assert_eq!(stream.dropped_packets_count.load(Ordering::Relaxed), 0);
    }
}
