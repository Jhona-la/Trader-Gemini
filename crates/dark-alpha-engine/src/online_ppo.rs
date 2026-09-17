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
        pub fn fetch_update<F>(
            &self,
            set_order: Ordering,
            fetch_order: Ordering,
            mut f: F,
        ) -> Result<f64, f64>
        where
            F: FnMut(f64) -> Option<f64>,
        {
            let mut prev_bits = self.0.load(fetch_order);
            loop {
                let prev_val = f64::from_bits(prev_bits);
                let next_val = match f(prev_val) {
                    Some(v) => v,
                    None => return Err(prev_val),
                };
                match self.0.compare_exchange_weak(
                    prev_bits,
                    next_val.to_bits(),
                    set_order,
                    fetch_order,
                ) {
                    Ok(_) => return Ok(prev_val),
                    Err(actual) => prev_bits = actual,
                }
            }
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
    pub reward_ema: AtomicF64,   // Media Móvil Exponencial (EMA) de la recompensa
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
    pub fn update_policy(
        &self,
        reward: f64,
        state_features: &[f64; 5],
        action_sign: f64, // +1.0 for Long, -1.0 for Short
        plasticity_multiplier: f64,
        alpha_ema: f64,
        dynamic_learning_rate: f64,
        dynamic_clip_eps: f64,
        dynamic_weight_min_clip: f64,
    ) {
        // FIX #668: Descartar actualizaciones con reward corrupto o no finito
        if !reward.is_finite() {
            return;
        }

        // FIX #581: Actualizamos EMA de la recompensa usando el valor previo atómico garantizado por el CAS
        let safe_alpha = if alpha_ema.is_finite() {
            alpha_ema.clamp(0.001, 1.0)
        } else {
            0.05
        };
        let _ = self
            .reward_ema
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |prev| {
                let next = prev + safe_alpha * (reward - prev);
                if next.is_finite() {
                    Some(next)
                } else {
                    Some(prev)
                }
            });
        let new_ema = self.reward_ema.load(Ordering::Relaxed);

        // Ventaja normalizada con tanh para evitar saturaciones instantáneas de ratios
        let raw_advantage = reward - new_ema;
        let advantage = (raw_advantage * 10.0).tanh();

        // Actualización de pesos por gradiente PPO recortado, modulado por la plasticidad y la dirección
        // FIX #1433: Sanitización de plasticidad, learning rate y features individuales
        let safe_plasticity = if plasticity_multiplier.is_finite() && plasticity_multiplier > 0.0 {
            plasticity_multiplier.clamp(0.1, 5.0)
        } else {
            1.0
        };
        let safe_lr = if dynamic_learning_rate.is_finite() && dynamic_learning_rate > 0.0 {
            dynamic_learning_rate.clamp(0.0001, 1.0)
        } else {
            0.01
        };
        let effective_learning_rate = safe_lr * safe_plasticity;
        let safe_clip_eps = if dynamic_clip_eps.is_finite() && dynamic_clip_eps > 0.0 {
            dynamic_clip_eps.clamp(0.01, 0.50)
        } else {
            0.20
        };
        let safe_min_clip = if dynamic_weight_min_clip.is_finite() && dynamic_weight_min_clip > 0.0
        {
            dynamic_weight_min_clip.clamp(0.0001, 1.0)
        } else {
            0.01
        };

        let sign = if action_sign >= 0.0 { 1.0 } else { -1.0 };
        for (i, &feat) in state_features.iter().enumerate().take(5) {
            let safe_feat = if feat.is_finite() { feat } else { 0.0 };
            let grad = safe_feat * sign * advantage;
            let ratio = 1.0 + effective_learning_rate * grad;
            let clipped_ratio = ratio.clamp(1.0 - safe_clip_eps, 1.0 + safe_clip_eps);

            // FIX #862 & #1102: Actualización atómica CAS con techo superior (1000.0) para prevenir desbordamiento infinito
            let _ = self.weights[i].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                let updated = (cur * clipped_ratio).clamp(safe_min_clip, 1000.0);
                if updated.is_finite() && updated > 0.0 {
                    Some(updated)
                } else {
                    Some(safe_min_clip)
                }
            });
        }
    }

    /// Calcula la puntuación combinada ponderada por la política PPO
    #[inline(always)]
    pub fn evaluate_policy(&self, state_features: &[f64; 5]) -> f64 {
        let w0 = self.weights[0].load(Ordering::Relaxed);
        let w1 = self.weights[1].load(Ordering::Relaxed);
        let w2 = self.weights[2].load(Ordering::Relaxed);
        let w3 = self.weights[3].load(Ordering::Relaxed);
        let w4 = self.weights[4].load(Ordering::Relaxed);
        let sum_w = w0 + w1 + w2 + w3 + w4;
        let inv_sum = if sum_w > 1e-12 && sum_w.is_finite() {
            1.0 / sum_w
        } else {
            0.20
        };

        let f0 = if state_features[0].is_finite() {
            state_features[0]
        } else {
            0.0
        };
        let f1 = if state_features[1].is_finite() {
            state_features[1]
        } else {
            0.0
        };
        let f2 = if state_features[2].is_finite() {
            state_features[2]
        } else {
            0.0
        };
        let f3 = if state_features[3].is_finite() {
            state_features[3]
        } else {
            0.0
        };
        let f4 = if state_features[4].is_finite() {
            state_features[4]
        } else {
            0.0
        };

        let res = (f0 * w0 + f1 * w1 + f2 * w2 + f3 * w3 + f4 * w4) * inv_sum;
        if res.is_finite() {
            res
        } else {
            0.0
        }
    }

    /// Calcula el clipping ratio dinámico modulado por la volatilidad no lineal (#47-#58)
    /// $\epsilon(\sigma) = \epsilon_0 \cdot (1.0 + \tanh((\sigma - \bar{\sigma}) / \bar{\sigma}))$
    #[inline(always)]
    pub fn compute_dynamic_clipping_ratio(
        base_eps: f64,
        current_volatility: f64,
        mean_volatility: f64,
    ) -> f64 {
        let safe_base = if base_eps.is_finite() { base_eps } else { 0.20 };
        if !mean_volatility.is_finite()
            || mean_volatility <= 1e-6
            || !current_volatility.is_finite()
            || current_volatility < 0.0
        {
            return safe_base.clamp(0.05, 0.30);
        }
        let vol_ratio = (current_volatility - mean_volatility) / mean_volatility;
        let modulation = 1.0 + vol_ratio.tanh() * 0.5;
        (safe_base * modulation).clamp(0.05, 0.35)
    }
}

impl Default for OnlinePpoPolicyEngine {
    fn default() -> Self {
        Self::new([0.20, 0.20, 0.20, 0.20, 0.20])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_ppo_update_and_dynamic_clipping() {
        let engine = OnlinePpoPolicyEngine::default();
        let features = [1.0, 0.5, 0.2, 0.1, 0.05];

        let score_before = engine.evaluate_policy(&features);
        assert!(score_before > 0.0);

        // Actualizar política con reward positivo en Long
        let eps = OnlinePpoPolicyEngine::compute_dynamic_clipping_ratio(0.2, 0.02, 0.01);
        assert!(eps >= 0.05 && eps <= 0.35);

        engine.update_policy(
            0.05, // reward
            &features, 1.0,  // Long
            1.0,  // plasticity
            0.1,  // alpha_ema
            0.01, // lr
            eps, 0.01, // min_clip
        );

        let score_after = engine.evaluate_policy(&features);
        assert!(score_after.is_finite());
    }

    #[test]
    fn test_online_ppo_nan_reward_and_zero_weight_immunity() {
        let engine = OnlinePpoPolicyEngine::new([0.0, 0.0, 0.0, 0.0, 0.0]);
        let features = [f64::NAN, 0.5, 0.2, 0.1, 0.05];

        let score = engine.evaluate_policy(&features);
        assert!(score.is_finite());

        // Update con reward NaN debe mantenerse acotado
        engine.update_policy(
            f64::NAN,
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            -1.0,
            1.0,
            0.1,
            0.01,
            0.2,
            0.01,
        );

        for w_atomic in &engine.weights {
            let w = w_atomic.load(Ordering::Relaxed);
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_online_ppo_short_action_and_atomic_compat() {
        let engine = OnlinePpoPolicyEngine::default();
        let features = [0.8, -0.4, 0.3, -0.1, 0.5];

        // Update policy on successful short
        engine.update_policy(
            0.10, &features, -1.0, // Short
            1.5, 0.05, 0.02, 0.2, 0.01,
        );

        let eval = engine.evaluate_policy(&features);
        assert!(eval.is_finite());

        // Test AtomicF64 fetch_update
        let atomic = atomic_compat::AtomicF64::new(10.0);
        assert_eq!(atomic.load(Ordering::Relaxed), 10.0);
        atomic.store(20.0, Ordering::Relaxed);
        assert_eq!(atomic.load(Ordering::Relaxed), 20.0);

        let updated = atomic.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |prev| {
            Some(prev + 5.0)
        });
        assert_eq!(updated, Ok(20.0));
        assert_eq!(atomic.load(Ordering::Relaxed), 25.0);
    }

    #[test]
    fn test_online_ppo_dynamic_clipping_nan_and_negative_volatility() {
        let eps_nan_mean =
            OnlinePpoPolicyEngine::compute_dynamic_clipping_ratio(0.2, 0.05, f64::NAN);
        assert!((0.05..=0.35).contains(&eps_nan_mean));

        let eps_neg_cur = OnlinePpoPolicyEngine::compute_dynamic_clipping_ratio(0.2, -0.05, 0.02);
        assert!((0.05..=0.35).contains(&eps_neg_cur));

        let eps_nan_base =
            OnlinePpoPolicyEngine::compute_dynamic_clipping_ratio(f64::NAN, 0.05, 0.02);
        assert!((0.05..=0.35).contains(&eps_nan_base));
    }

    #[test]
    fn test_online_ppo_update_and_eval_nan_features_immunity() {
        let engine = OnlinePpoPolicyEngine::default();
        let nan_features = [f64::NAN, f64::INFINITY, -f64::INFINITY, 0.5, 0.2];

        let score = engine.evaluate_policy(&nan_features);
        assert!(score.is_finite());

        engine.update_policy(
            0.05,
            &nan_features,
            1.0,
            f64::NAN, // will use safe default
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        );

        let score_after = engine.evaluate_policy(&nan_features);
        assert!(score_after.is_finite());
    }
}
