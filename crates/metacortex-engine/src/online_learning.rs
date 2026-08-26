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

    /// Modula el ruido de medición R_noise en función de la volatilidad instantánea
    /// y el exponente de Lyapunov / caos para desacoplar el aprendizaje del ruido de microestructura.
    #[inline(always)]
    pub fn compute_dynamic_r_noise(&self, instantaneous_volatility: f32, lyapunov_chaos: f32) -> f32 {
        let base_r = if self.r_noise.is_finite() && self.r_noise > 0.0 {
            self.r_noise
        } else {
            1e-2
        };

        let safe_vol = if instantaneous_volatility.is_finite() && instantaneous_volatility > 0.0 {
            instantaneous_volatility
        } else {
            0.01
        };

        let safe_chaos = if lyapunov_chaos.is_finite() && lyapunov_chaos > 0.0 {
            lyapunov_chaos
        } else {
            0.0
        };

        // Escalamiento del ruido de medición:
        // A mayor volatilidad de microestructura y mayor caos, R_noise aumenta,
        // amortiguando la Ganancia de Kalman K = (P*x) / (P*x^2 + R) y protegiendo los pesos de oscilaciones espurias.
        let vol_factor = 1.0 + (safe_vol * 100.0).clamp(0.0, 50.0);
        let chaos_factor = 1.0 + (safe_chaos * safe_chaos).clamp(0.0, 20.0);

        (base_r * vol_factor * chaos_factor).clamp(1e-5, 100.0)
    }

    /// Actualiza los pesos de forma continua usando un Filtro de Kalman Tensorial con modulación de volatilidad instantánea
    /// td_error es el (Reward Observado - Predicción).
    /// lyapunov_chaos determina si la actualización es segura (filtro de drift).
    /// instantaneous_volatility modula dinámicamente R_noise para prevenir sobreajuste a ruido microestructural.
    #[inline(always)]
    pub fn update_weights_with_kalman_adaptive_vol(
        &mut self,
        features: &[f32; 64],
        td_error: f32,
        lyapunov_chaos: f32,
        instantaneous_volatility: f32,
    ) {
        if !td_error.is_finite() || !lyapunov_chaos.is_finite() {
            return;
        }

        let dynamic_r = self.compute_dynamic_r_noise(instantaneous_volatility, lyapunov_chaos);

        // [Fase XXXIX] Tensorized Chaos Damping (Continuous Online Learning)
        // Amortigua asintóticamente la capacidad de aprendizaje cuando el mercado se vuelve altamente caótico.
        let chaos_damping = (1.0 - (lyapunov_chaos / 2.0)).clamp(0.01, 1.0);

        for i in 0..64 {
            let x = features[i];
            if !x.is_finite() || x == 0.0 {
                continue;
            } // Omitir features inactivos para rendimiento

            // 1. Prediction Step (Kalman)
            self.p_covariance[i] += self.q_noise;

            // 2. Update Step (Kalman Gain con R_noise dinámico adaptativo)
            let s = self.p_covariance[i] * x * x + dynamic_r;
            let kalman_gain = (self.p_covariance[i] * x) / s;

            // La innovación en el contexto de Q-learning online es el td_error
            let innovation = td_error;

            // Actualización del peso modulada por la ganancia y el amortiguador de caos
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
        // Estimar volatilidad instantánea a partir de la dispersión de features y magnitud de innovación
        let mut sum = 0.0;
        let mut count = 0.0;
        for &f in features {
            if f.is_finite() && f != 0.0 {
                sum += f.abs();
                count += 1.0;
            }
        }
        let feat_dispersion = if count > 0.0 { sum / count } else { 0.01 };
        let inst_vol = (feat_dispersion * 0.5 + td_error.abs() * 0.5).clamp(1e-4, 10.0);

        self.update_weights_with_kalman_adaptive_vol(features, td_error, lyapunov_chaos, inst_vol);
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

use crate::consejo_seniors::TradingHorizon;

/// Inicia el consumidor asíncrono en background para autoevolucionar leyendo el MmapTelemetryBus.
pub fn spawn_telemetry_consumer(
    scalping_module: Arc<Mutex<OnlineLearningModule>>,
    swing_module: Arc<Mutex<OnlineLearningModule>>,
    mmap_path: &'static str,
) {
    tokio::spawn(async move {
        let mut reader = MmapTelemetryReader::new(mmap_path);

        // FASE 23: Stateful Correlation Buffer for True PnL Online Learning
        // Correlate Decision features (Frame 1) with their actual market results (Frame 13)
        let mut last_features_scalping = [[0.0_f32; 64]; 30];
        let mut last_features_swing = [[0.0_f32; 64]; 30];
        let mut last_entropy_scalping = [1.0_f32; 30];
        let mut last_entropy_swing = [1.0_f32; 30];

        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

            if let Ok(frames) = reader.read_latest_frames() {
                if !frames.is_empty() {
                    let mut s_mod = scalping_module.lock().await;
                    let mut sw_mod = swing_module.lock().await;
                    
                    for frame in frames {
                        let coin_id = frame.payload[0] as usize;
                        if coin_id >= 30 {
                            continue;
                        }
                        
                        // Infer horizon from payload if possible, for now we will simulate split by bit flag or payload[6] if added.
                        // Assuming payload[6] has the horizon flag (0 = Scalping, 1 = Swing)
                        let horizon = if frame.payload.len() > 6 && frame.payload[6] > 0.0 {
                            TradingHorizon::Swing
                        } else {
                            TradingHorizon::Scalping
                        };

                        // Frame type 1 = Decision Trace (Features)
                        if frame.frame_type == 1 {
                            // Extraer payload: [coin_id, latency, holistic, hawkes, entropy, ml_prob, horizon]
                            let hawkes = frame.payload[3] as f32;
                            let entropy = frame.payload[4] as f32;
                            let ml_prob = frame.payload[5] as f32;

                            match horizon {
                                TradingHorizon::Scalping => {
                                    last_features_scalping[coin_id][0] = hawkes;
                                    last_features_scalping[coin_id][1] = entropy;
                                    last_features_scalping[coin_id][2] = ml_prob;
                                    last_entropy_scalping[coin_id] = entropy;
                                },
                                TradingHorizon::Swing => {
                                    last_features_swing[coin_id][0] = hawkes;
                                    last_features_swing[coin_id][1] = entropy;
                                    last_features_swing[coin_id][2] = ml_prob;
                                    last_entropy_swing[coin_id] = entropy;
                                }
                            }
                        }
                        // Frame type 13 = ROI Metrics (Real Reward)
                        else if frame.frame_type == 13 {
                            // Extraer payload: [coin_id, gross_pnl, net_pnl, maker_fee, taker_fee, win_flag, horizon]
                            let net_pnl = frame.payload[2] as f32;

                            match horizon {
                                TradingHorizon::Scalping => {
                                    let prior_pred = s_mod.predict(&last_features_scalping[coin_id]);
                                    let td_error = net_pnl - prior_pred;
                                    s_mod.update_weights_with_kalman(
                                        &last_features_scalping[coin_id],
                                        td_error,
                                        last_entropy_scalping[coin_id],
                                    );
                                },
                                TradingHorizon::Swing => {
                                    let prior_pred = sw_mod.predict(&last_features_swing[coin_id]);
                                    let td_error = net_pnl - prior_pred;
                                    sw_mod.update_weights_with_kalman(
                                        &last_features_swing[coin_id],
                                        td_error,
                                        last_entropy_swing[coin_id],
                                    );
                                }
                            }
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

    #[test]
    fn test_online_learning_r_noise_volatility_modulation() {
        let mut module_calm = OnlineLearningModule::new(0.01, 0.0);
        let mut module_volatile = OnlineLearningModule::new(0.01, 0.0);

        let mut features = [0.0f32; 64];
        features[0] = 1.0;

        let td_error = 0.10f32;

        // Calm market: low volatility (0.001)
        module_calm.update_weights_with_kalman_adaptive_vol(&features, td_error, 0.0, 0.001);

        // Volatile / noisy market: high instantaneous volatility (0.50)
        module_volatile.update_weights_with_kalman_adaptive_vol(&features, td_error, 0.0, 0.50);

        // In high volatility market, R_noise is higher, reducing Kalman gain and damping weight change to prevent overfitting
        assert!(
            module_volatile.weights[0] < module_calm.weights[0],
            "Volatile market must dampen weight updates: volatile weight {}, calm weight {}",
            module_volatile.weights[0],
            module_calm.weights[0]
        );
        assert!(module_volatile.weights[0] > 0.0);
    }

    #[test]
    fn test_online_learning_nan_and_infinite_volatility_immunity() {
        let mut module = OnlineLearningModule::new(0.01, 0.9);
        let mut features = [0.0f32; 64];
        features[0] = 1.0;

        // Pass NaN and Inf in volatility, chaos, and features
        module.update_weights_with_kalman_adaptive_vol(&features, 0.05, f32::NAN, f32::INFINITY);
        assert!(module.weights[0].is_finite());

        let dyn_r_nan = module.compute_dynamic_r_noise(f32::NAN, f32::NAN);
        assert!(dyn_r_nan.is_finite());
        assert!(dyn_r_nan > 0.0);
    }
}
