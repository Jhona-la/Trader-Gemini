use std::f64;
use std::sync::atomic::Ordering;

pub mod atomic_compat {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    #[repr(transparent)]
    pub struct AtomicF64(AtomicU64);
    
    impl AtomicF64 {
        pub const fn new(val: f64) -> Self {
            Self(AtomicU64::new(val.to_bits()))
        }
        pub fn load(&self, order: Ordering) -> f64 {
            f64::from_bits(self.0.load(order))
        }
        pub fn store(&self, val: f64, order: Ordering) {
            self.0.store(val.to_bits(), order)
        }
    }
}

use atomic_compat::AtomicF64;

/// 🧬 MOTOR DE APRENDIZAJE POR REFUERZO CONTINUO PPO (ONLINE PPO POLICY ENGINE)
/// Algoritmo Proximal Policy Optimization (PPO) en vivo para la adaptación continua de pesos de estrategia.
/// Actualiza la política en nanosegundos evaluando el gradiente de la recompensa por Sharpe Ratio.
/// FASE 25: 100% Lock-Free usando AtomicF64 y EMA.
pub struct OnlinePpoPolicyEngine {
    pub weights: [AtomicF64; 5], // Pesos adaptativos para (OFI, OBI, Hawkes, LeadLag, MarketRegime)
    pub reward_ema: AtomicF64,  // Media Móvil Exponencial (EMA) de la recompensa
}

impl OnlinePpoPolicyEngine {
    pub fn new(init_weights: [f64; 5]) -> Self {
        Self {
            weights: [
                AtomicF64::new(init_weights[0]),
                AtomicF64::new(init_weights[1]),
                AtomicF64::new(init_weights[2]),
                AtomicF64::new(init_weights[3]),
                AtomicF64::new(init_weights[4]),
            ],
            reward_ema: AtomicF64::new(0.0),
        }
    }

    /// Actualiza la política PPO basada en el resultado de la operación realizada (reward)
    /// `plasticity_multiplier`: Multiplicador de aprendizaje basado en la entropía del mercado.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn update_policy(&self, reward: f64, state_features: &[f64; 5], plasticity_multiplier: f64, alpha_ema: f64, dynamic_learning_rate: f64, dynamic_clip_eps: f64, dynamic_weight_min_clip: f64) {
        // Actualizamos EMA de la recompensa usando el alpha proporcionado dinámicamente
        let alpha = alpha_ema;
        let prev_ema = self.reward_ema.load(Ordering::Relaxed);
        let new_ema = prev_ema + alpha * (reward - prev_ema);
        self.reward_ema.store(new_ema, Ordering::Relaxed);

        let advantage = reward - new_ema;

        // Actualización de pesos por gradiente PPO recortado, modulado por la plasticidad
        let effective_learning_rate = dynamic_learning_rate * plasticity_multiplier;

        let mut current_weights = [0.0; 5];
        let mut sum_w = 0.0;

        for (i, &feat) in state_features.iter().enumerate().take(5) {
            let grad = feat * advantage;
            let ratio = 1.0 + effective_learning_rate * grad;
            let clipped_ratio = ratio.clamp(1.0 - dynamic_clip_eps, 1.0 + dynamic_clip_eps);

            let old_w = self.weights[i].load(Ordering::Relaxed);
            let new_w = (old_w * clipped_ratio).max(dynamic_weight_min_clip);
            current_weights[i] = new_w;
            sum_w += new_w;
        }

        // Renormalizar suma de pesos a 1.0 y almacenar lock-free
        if sum_w > 0.0 {
            for (i, &w) in current_weights.iter().enumerate() {
                self.weights[i].store(w / sum_w, Ordering::Relaxed);
            }
        }
    }

    /// Calcula la puntuación combinada ponderada por la política PPO
    #[inline(always)]
    pub fn evaluate_policy(&self, state_features: &[f64; 5]) -> f64 {
        state_features[0] * self.weights[0].load(Ordering::Relaxed)
            + state_features[1] * self.weights[1].load(Ordering::Relaxed)
            + state_features[2] * self.weights[2].load(Ordering::Relaxed)
            + state_features[3] * self.weights[3].load(Ordering::Relaxed)
            + state_features[4] * self.weights[4].load(Ordering::Relaxed)
    }
}

impl Default for OnlinePpoPolicyEngine {
    fn default() -> Self {
        Self::new([0.20, 0.20, 0.20, 0.20, 0.20])
    }
}
