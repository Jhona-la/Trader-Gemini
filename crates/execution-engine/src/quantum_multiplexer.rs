use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// 🚀 ALGORITMO #82: QUANTUM API MULTIPLEXER (Kalman Rate Limit Predictor)
/// Protege la cuenta de penalizaciones (Bans) de Binance modelando estocásticamente
/// el límite de la API antes de que ocurra usando un Filtro de Kalman 1D.
pub struct QuantumMultiplexer {
    // Pesos (Weights) locales modelados sin necesidad de consultar al servidor
    modeled_weight_1m: AtomicUsize,
    last_decay_timestamp_ms: AtomicU64,

    // Filtro de Kalman 1D para predecir con alta precisión el peso real en el exchange
    kalman_estimate: RwLock<f64>,
    kalman_variance: RwLock<f64>,

    // Límite de Binance (Normalmente 1200 a 2400 por minuto)
    max_weight_per_minute: usize,
}

impl QuantumMultiplexer {
    pub fn new(max_weight: usize) -> Self {
        Self {
            modeled_weight_1m: AtomicUsize::new(0),
            last_decay_timestamp_ms: AtomicU64::new(Self::now_ms()),
            kalman_estimate: RwLock::new(0.0),
            kalman_variance: RwLock::new(1.0), // Incertidumbre inicial
            max_weight_per_minute: max_weight,
        }
    }

    #[inline(always)]
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Calcula la decadencia continua (Continuous Decay) del peso de la API usando 
    /// la derivada temporal en lugar de buckets estáticos.
    fn apply_temporal_decay(&self) {
        let now = Self::now_ms();
        let last = self.last_decay_timestamp_ms.load(Ordering::Acquire);

        let elapsed_ms = now.saturating_sub(last);
        if elapsed_ms > 0 {
            // Decae a un ratio de (max_weight / 60000) por milisegundo
            let decay_rate = self.max_weight_per_minute as f64 / 60000.0;
            let amount_to_decay = (elapsed_ms as f64 * decay_rate) as usize;

            if amount_to_decay > 0 {
                let current_weight = self.modeled_weight_1m.load(Ordering::Acquire);
                let new_weight = current_weight.saturating_sub(amount_to_decay);

                // Actualizamos modeled_weight_1m y timestamp
                if self.modeled_weight_1m.compare_exchange_weak(
                    current_weight, 
                    new_weight, 
                    Ordering::Release, 
                    Ordering::Relaxed
                ).is_ok() {
                    self.last_decay_timestamp_ms.store(now, Ordering::Release);
                }

                // Decaimiento continuo de la estimación de Kalman para evitar bloqueo permanente
                if let Ok(mut k_est) = self.kalman_estimate.write() {
                    *k_est = (*k_est - amount_to_decay as f64).max(0.0);
                }
            }
        }
    }

    pub fn request_execution_slot(&self, request_weight: usize, is_critical: bool) -> bool {
        self.apply_temporal_decay();

        let current_weight = self.modeled_weight_1m.load(Ordering::Acquire);

        // Predicción del filtro de Kalman (fusionado con nuestro modelo determinista)
        let k_est = *self.kalman_estimate.read().unwrap_or_else(|p| p.into_inner());
        let hybrid_weight = (current_weight as f64 * 0.3 + k_est * 0.7).max(0.0) as usize;

        let projected_weight = hybrid_weight + request_weight;

        // Umbral de pánico: 90% para operaciones no críticas, 98% para críticas
        let threshold = if is_critical {
            (self.max_weight_per_minute as f64 * 0.98) as usize
        } else {
            (self.max_weight_per_minute as f64 * 0.90) as usize
        };

        if projected_weight > threshold {
            return false; // Bloqueo preventivo (Anti-Ban)
        }

        // Autorizado. Añadimos el peso al modelo local y a la predicción a priori de Kalman.
        self.modeled_weight_1m.fetch_add(request_weight, Ordering::Release);

        let mut k_write = self.kalman_estimate.write().unwrap_or_else(|p| p.into_inner());
        *k_write += request_weight as f64;

        true
    }

    /// Calibración Bayesiana / Update de Kalman: Si el servidor de Binance nos responde con un header
    /// X-MBX-USED-WEIGHT-(1M), actualizamos nuestro modelo interno.
    pub fn calibrate_from_reality(&self, server_weight: usize) {
        self.modeled_weight_1m.store(server_weight, Ordering::Release);
        self.last_decay_timestamp_ms.store(Self::now_ms(), Ordering::Release);

        // --- 1D Kalman Filter Update Step ---
        let z = server_weight as f64; // Observation
        let r = 5.0; // Measurement noise variance (Binance headers can be slightly delayed)

        let mut p = self.kalman_variance.write().unwrap_or_else(|p| p.into_inner());
        let mut x = self.kalman_estimate.write().unwrap_or_else(|p| p.into_inner());

        if *x == 0.0 {
            *x = z;
            *p = 1.0;
        } else {
            // Predicción de incertidumbre debido al tiempo (Process noise)
            let q = 2.0; 
            *p += q;

            // Kalman Gain
            let k = *p / (*p + r);

            // Update estimate
            *x = *x + k * (z - *x);

            // Update variance
            *p = (1.0 - k) * *p;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_multiplexer_rate_limiting_and_decay() {
        let mux = QuantumMultiplexer::new(1200);
        // Debe permitir requests normales
        assert!(mux.request_execution_slot(50, false));
        assert!(mux.request_execution_slot(100, false));
        
        // Calibrar desde la realidad del servidor
        mux.calibrate_from_reality(200);
        assert!(mux.request_execution_slot(50, false));

        // Debe rechazar si excede el umbral
        assert!(!mux.request_execution_slot(2000, false));
    }

    #[test]
    fn test_quantum_multiplexer_critical_override() {
        let mux = QuantumMultiplexer::new(1000);
        // Calibrar al 92% de capacidad
        mux.calibrate_from_reality(920);

        // No crítico debe ser rechazado (> 90%)
        assert!(!mux.request_execution_slot(10, false));

        // Crítico (Kill switch / Stop loss) debe ser aprobado (< 98%)
        assert!(mux.request_execution_slot(10, true));
    }

    #[test]
    fn test_quantum_multiplexer_kalman_convergence() {
        let mux = QuantumMultiplexer::new(1200);
        mux.calibrate_from_reality(300);

        for _ in 0..10 {
            mux.calibrate_from_reality(300);
        }

        let estimate = *mux.kalman_estimate.read().unwrap();
        let variance = *mux.kalman_variance.read().unwrap();

        assert!((estimate - 300.0).abs() < 1.0, "Estimado de Kalman ({}) debe converger a 300", estimate);
        assert!(variance > 0.0 && variance < 5.0, "Varianza de Kalman ({}) debe ser acotada", variance);
    }

    #[test]
    fn test_quantum_multiplexer_temporal_decay_recovers_capacity() {
        let mux = QuantumMultiplexer::new(60000); // 60000 per minute = 1 per ms
        mux.modeled_weight_1m.store(50000, Ordering::Release);
        mux.last_decay_timestamp_ms.store(QuantumMultiplexer::now_ms() - 10000, Ordering::Release);

        mux.apply_temporal_decay();

        let weight_after = mux.modeled_weight_1m.load(Ordering::Acquire);
        // After 10s (10000ms), at least 10000 should have decayed
        assert!(weight_after <= 40000);
    }
}

