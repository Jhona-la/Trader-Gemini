/// Fase B: Continuous Online Learning
/// Implementa un mecanismo de aprendizaje en línea (Online Gradient Descent / Hebbian)
/// que ajusta los pesos de un modelo en memoria en tiempo de ejecución, basado en
/// la diferencia temporal (td_error) o el error residual de la predicción.

#[derive(Debug, Clone)]
pub struct OnlineLearningModule {
    pub weights: [f32; 64],
    pub learning_rate: f32,
    pub momentum: f32,
    pub velocity: [f32; 64],

    // Kalman Filter (Aproximación diagonal por recursos limitados 16GB)
    pub p_covariance: [f32; 64], // Matriz de Incertidumbre (Diagonal)
    pub q_noise: f32,            // Process Noise (evolucionable)
    pub r_noise: f32,            // Measurement Noise (evolucionable)
    pub l2_decay: f32,           // Regularización L2 (evolucionable, antes hardcoded 0.99999)
}

impl Default for OnlineLearningModule {
    fn default() -> Self {
        Self {
            weights: [0.0; 64],
            learning_rate: 0.001,
            momentum: 0.9,
            velocity: [0.0; 64],
            p_covariance: [1.0; 64],
            q_noise: 1e-4,
            r_noise: 1e-2,
            l2_decay: 0.99999,
        }
    }
}

impl OnlineLearningModule {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            weights: [0.0; 64], // Inicialización a 0
            learning_rate,
            momentum,
            velocity: [0.0; 64],
            p_covariance: [1.0; 64],
            q_noise: 1e-4,
            r_noise: 1e-2,
            l2_decay: 0.99999,
        }
    }

    /// Realiza una inferencia lineal rápida con el tensor actual
    #[inline(always)]
    pub fn predict(&self, features: &[f32; 64]) -> f32 {
        let mut sum = 0.0;
        for i in 0..64 {
            let f = features[i];
            if f.is_finite() {
                sum += self.weights[i] * f;
            }
        }
        if sum.is_finite() { sum.clamp(-100.0, 100.0) } else { 0.0 }
    }

    /// Actualiza los pesos de forma continua usando un Filtro de Kalman Tensorial
    /// td_error es el (Reward Observado - Predicción).
    /// lyapunov_chaos determina si la actualización es segura (filtro de drift).
    #[inline(always)]
    pub fn update_weights_with_kalman(
        &mut self,
        features: &[f32; 64],
        td_error: f32,
        lyapunov_chaos: f32,
    ) {
        if !td_error.is_finite() || !lyapunov_chaos.is_finite() {
            return;
        }

        // [Fase XXXIX] Tensorized Chaos Damping (Continuous Online Learning)
        // En lugar del corte estricto `if lyapunov_chaos > 1.5 { return; }`,
        // usamos una curva continua que amortigua asintóticamente la capacidad de aprendizaje
        // cuando el mercado se vuelve altamente caótico, protegiendo los pesos de ruido estocástico.
        let chaos_damping = (1.0 - (lyapunov_chaos / 2.0)).clamp(0.01, 1.0);

        for i in 0..64 {
            let x = features[i];
            if !x.is_finite() || x == 0.0 {
                continue;
            } // Omitir features inactivos para rendimiento

            // 1. Prediction Step (Kalman)
            self.p_covariance[i] += self.q_noise;

            // 2. Update Step (Kalman Gain)
            let s = self.p_covariance[i] * x * x + self.r_noise;
            let kalman_gain = (self.p_covariance[i] * x) / s;

            // La innovación en el contexto de Q-learning online es el td_error
            let innovation = td_error;

            // Actualización del peso modulada por la confianza (inversamente proporcional al caos)
            let weight_update = kalman_gain * innovation * chaos_damping;

            // Integrando Momentum con la ganancia de Kalman directamente
            self.velocity[i] = self.momentum * self.velocity[i] + weight_update;
            self.weights[i] += self.velocity[i];

            // 3. Actualizar la covarianza (protegida contra colapso numérico e incertidumbre congelada)
            let updated_cov = ((1.0 - kalman_gain * x) * self.p_covariance[i]).clamp(1e-4, 10.0);
            self.p_covariance[i] = if updated_cov < 1e-3 {
                updated_cov + self.q_noise * 2.0
            } else {
                updated_cov
            };

            // Decadencia de pesos suave (Regularización L2 — evolucionable)
            self.weights[i] *= self.l2_decay;
        }
    }

    /// Constructor extendido con parámetros Kalman evolucionables
    pub fn new_with_kalman(
        learning_rate: f32,
        momentum: f32,
        q_noise: f32,
        r_noise: f32,
        l2_decay: f32,
    ) -> Self {
        Self {
            weights: [0.0; 64],
            learning_rate,
            momentum,
            velocity: [0.0; 64],
            p_covariance: [1.0; 64],
            q_noise,
            r_noise,
            l2_decay,
        }
    }
}

use std::sync::Arc;
use storage_engine::MmapTelemetryReader;
use tokio::sync::Mutex;

/// Inicia el consumidor asíncrono en background para autoevolucionar leyendo el MmapTelemetryBus.
pub fn spawn_telemetry_consumer(
    learning_module: Arc<Mutex<OnlineLearningModule>>,
    mmap_path: &'static str,
) {
    tokio::spawn(async move {
        let mut reader = MmapTelemetryReader::new(mmap_path);

        // FASE 23: Stateful Correlation Buffer for True PnL Online Learning
        // Correlate Decision features (Frame 1) with their actual market results (Frame 13)
        let mut last_features = [[0.0_f32; 64]; 30];
        let mut last_entropy = [1.0_f32; 30];

        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            if let Ok(frames) = reader.read_latest_frames() {
                if !frames.is_empty() {
                    let mut module = learning_module.lock().await;
                    for frame in frames {
                        let coin_id = frame.payload[0] as usize;
                        if coin_id >= 30 {
                            continue;
                        }

                        // Frame type 1 = Decision Trace (Features)
                        if frame.frame_type == 1 {
                            // Extraer payload: [coin_id, latency, holistic, hawkes, entropy, ml_prob]
                            let hawkes = frame.payload[3] as f32;
                            let entropy = frame.payload[4] as f32;
                            let ml_prob = frame.payload[5] as f32;

                            // Buffer the features for when the trade closes
                            last_features[coin_id][0] = hawkes;
                            last_features[coin_id][1] = entropy;
                            last_features[coin_id][2] = ml_prob;
                            last_entropy[coin_id] = entropy;
                        }
                        // Frame type 13 = ROI Metrics (Real Reward)
                        else if frame.frame_type == 13 {
                            // Extraer payload: [coin_id, gross_pnl, net_pnl, maker_fee, taker_fee, win_flag]
                            let net_pnl = frame.payload[2] as f32;

                            // TD-Error exacto de Bellman: Reward observado menos predicción previa
                            let prior_pred = module.predict(&last_features[coin_id]);
                            let td_error = net_pnl - prior_pred;

                            // Evolucionamos usando las features de la última decisión y el resultado real
                            module.update_weights_with_kalman(
                                &last_features[coin_id],
                                td_error,
                                last_entropy[coin_id],
                            );
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_learning_kalman_update() {
        let mut module = OnlineLearningModule::new(0.01, 0.9);
        let mut features = [0.0f32; 64];
        features[0] = 1.0;
        features[1] = 0.5;

        let pred_initial = module.predict(&features);
        assert_eq!(pred_initial, 0.0);

        let reward = 0.05f32;
        let td_error = reward - pred_initial;
        module.update_weights_with_kalman(&features, td_error, 0.2);

        let pred_updated = module.predict(&features);
        assert!(pred_updated > 0.0, "Weight should adapt towards positive reward");
    }
}
