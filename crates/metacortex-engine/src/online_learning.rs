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
            sum += self.weights[i] * features[i];
        }
        sum
    }

    /// Actualiza los pesos de forma continua usando un Filtro de Kalman Tensorial
    /// td_error es el (Reward Observado - Predicción). 
    /// lyapunov_chaos determina si la actualización es segura (filtro de drift).
    #[inline(always)]
    pub fn update_weights_with_kalman(&mut self, features: &[f32; 64], td_error: f32, lyapunov_chaos: f32) {
        // [Fase XXXIX] Tensorized Chaos Damping (Continuous Online Learning)
        // En lugar del corte estricto `if lyapunov_chaos > 1.5 { return; }`,
        // usamos una curva continua que amortigua asintóticamente la capacidad de aprendizaje
        // cuando el mercado se vuelve altamente caótico, protegiendo los pesos de ruido estocástico.
        let chaos_damping = (1.0 - (lyapunov_chaos / 2.0)).clamp(0.01, 1.0);

        for i in 0..64 {
            let x = features[i];
            if x == 0.0 { continue; } // Omitir features inactivos para rendimiento

            // 1. Prediction Step (Kalman)
            self.p_covariance[i] += self.q_noise;

            // 2. Update Step (Kalman Gain)
            let s = self.p_covariance[i] * x * x + self.r_noise;
            let kalman_gain = (self.p_covariance[i] * x) / s;

            // La innovación en el contexto de Q-learning online es el td_error
            let innovation = td_error; 

            // Actualización del peso modulada por la confianza (inversamente proporcional al caos)
            let weight_update = kalman_gain * innovation * chaos_damping;
            
            // Integrando Momentum (Adam/SGD híbrido con Kalman)
            self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * weight_update;
            self.weights[i] += self.velocity[i];

            // 3. Actualizar la covarianza
            self.p_covariance[i] = (1.0 - kalman_gain * x) * self.p_covariance[i];
            
            // Decadencia de pesos suave (Regularización L2 — evolucionable)
            self.weights[i] *= self.l2_decay;
        }
    }

    /// Constructor extendido con parámetros Kalman evolucionables
    pub fn new_with_kalman(learning_rate: f32, momentum: f32, q_noise: f32, r_noise: f32, l2_decay: f32) -> Self {
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

use storage_engine::MmapTelemetryReader;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Inicia el consumidor asíncrono en background para autoevolucionar leyendo el MmapTelemetryBus.
pub fn spawn_telemetry_consumer(
    learning_module: Arc<Mutex<OnlineLearningModule>>,
    mmap_path: &'static str
) {
    tokio::spawn(async move {
        let reader_result = MmapTelemetryReader::new(mmap_path);
        if reader_result.is_err() {
            println!("⚠️ [METACORTEX] No se pudo inicializar MmapTelemetryReader para Online Learning.");
            return;
        }
        let mut reader = reader_result.unwrap();
        
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
                        if coin_id >= 30 { continue; }
                        
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
                            
                            // True TD Error is the actual Net PnL percentage
                            let td_error = net_pnl; 
                            
                            // Evolucionamos usando las features de la última decisión y el resultado real
                            module.update_weights_with_kalman(&last_features[coin_id], td_error, last_entropy[coin_id]);
                        }
                    }
                }
            }
        }
    });
}
