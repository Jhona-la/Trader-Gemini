use std::f64;
use std::collections::VecDeque;

/// 🧬 PREDICCIÓN CONFORMAL NO PARAMÉTRICA (CONFORMAL PREDICTION BANDS)
/// Construye intervalos de confianza dinámicos con garantía de cobertura probabilística (e.g. 95%).
/// Elimina límites fijos o estáticos de TP y SL en el sistema.
#[derive(Debug, Clone)]
pub struct ConformalPredictor {
    pub alpha: f64,               // Nivel de significancia (e.g. 0.05 para 95% de cobertura)
    pub residual_history: VecDeque<f64>, // Historial circular de residuos no paramétricos
    pub max_history: usize,
}

impl ConformalPredictor {
    pub fn new(alpha: f64, max_history: usize) -> Self {
        Self {
            alpha,
            residual_history: VecDeque::with_capacity(max_history),
            max_history: max_history.max(50),
        }
    }

    #[inline(always)]
    pub fn update(&mut self, actual_return: f64, predicted_return: f64) {
        if !actual_return.is_finite() || !predicted_return.is_finite() {
            return;
        }
        let residual = (actual_return - predicted_return).abs();
        if self.residual_history.len() >= self.max_history {
            self.residual_history.pop_front();
        }
        self.residual_history.push_back(residual);
    }

    /// Calcula el valor q-cuantil no paramétrico q_{1-\alpha} sin alocación en el heap
    #[inline(always)]
    pub fn compute_conformal_quantile(&self) -> f64 {
        let n = self.residual_history.len();
        if n < 10 {
            return 0.0030; // Fallback inicial dinámico si no hay historial
        }

        let mut sorted = [0.0f64; 256];
        let n_clamped = n.min(256);
        // FIX #563: Muestrear los 256 residuos MÁS RECIENTES usando .iter().rev()
        for (i, &v) in self.residual_history.iter().rev().take(n_clamped).enumerate() {
            sorted[i] = v;
        }
        let slice = &mut sorted[..n_clamped];
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Cuantil ajustado por muestra finita: (1 - alpha) * (1 + 1/n)
        let q_idx = (((1.0 - self.alpha) * (n_clamped as f64 + 1.0)).ceil() as usize).saturating_sub(1);
        slice[q_idx.min(n_clamped - 1)]
    }

    /// Retorna los objetivos dinámicos (TP, SL) garantizados probabilísticamente
    #[inline(always)]
    pub fn compute_dynamic_tp_sl(&self, atr_pct: f64, maker_fee: f64, taker_fee: f64, is_scalp: bool, config: &quantum_arena::config::QuantumConfig) -> (f64, f64) {
        let safe_atr = if atr_pct.is_finite() && atr_pct > 0.0 { atr_pct } else { 0.005 };
        let q = self.compute_conformal_quantile();
        let base_vol = safe_atr.max(if q.is_finite() && q > 0.0 { q } else { 0.003 });

        // FIX #689: Sanitizar comisiones
        let safe_maker = if maker_fee.is_finite() && maker_fee >= 0.0 { maker_fee } else { 0.0002 };
        let safe_taker = if taker_fee.is_finite() && taker_fee >= 0.0 { taker_fee } else { 0.0005 };
        let _roundtrip_fee = safe_maker + safe_taker;

        use std::sync::atomic::Ordering;
        let temporal_scale = config.temporal_scale.load(Ordering::Relaxed).clamp(0.0, 1.0);
        let s = if is_scalp {
            0.0
        } else {
            temporal_scale.max(0.5)
        };

        let scalp_tp_mult = config.tp_rr_ratio_btc.load(Ordering::Relaxed);
        let safe_scalp_tp = if scalp_tp_mult.is_finite() && scalp_tp_mult > 0.0 { scalp_tp_mult.max(1.0) } else { 1.5 };
        let swing_tp_mult = config.swing_trail_atr_mult_base.load(Ordering::Relaxed);
        let safe_swing_tp = if swing_tp_mult.is_finite() && swing_tp_mult > 0.0 { swing_tp_mult.max(2.0) } else { 3.0 };

        let sl_mult = config.sl_atr_multiplier.load(Ordering::Relaxed);
        let safe_scalp_sl = if sl_mult.is_finite() && sl_mult > 0.0 { sl_mult.max(0.5) } else { 1.0 };
        let safe_swing_sl = if sl_mult.is_finite() && sl_mult > 0.0 { sl_mult.max(1.0) } else { 1.5 };

        // D-337: Interpolación continua suave en homotopía s in [0, 1]
        let eff_tp_mult = safe_scalp_tp * (1.0 - s) + safe_swing_tp * s;
        let eff_sl_mult = safe_scalp_sl * (1.0 - s) + safe_swing_sl * s;

        let raw_tp = (base_vol * eff_tp_mult).clamp(0.001, 0.50);
        let raw_sl = (base_vol * eff_sl_mult).clamp(0.001, 0.50);
        (raw_tp, raw_sl)
    }
}

impl Default for ConformalPredictor {
    fn default() -> Self {
        Self::new(0.05, 200)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_predictor_update_and_quantile() {
        let mut predictor = ConformalPredictor::new(0.05, 100);
        assert_eq!(predictor.compute_conformal_quantile(), 0.0030); // fallback under 10 samples

        for i in 1..=20 {
            predictor.update(i as f64 * 0.001, 0.0);
        }

        let q = predictor.compute_conformal_quantile();
        assert!(q > 0.015 && q <= 0.020);
    }

    #[test]
    fn test_conformal_predictor_dynamic_tp_sl_scalp_vs_swing() {
        let mut predictor = ConformalPredictor::new(0.05, 100);
        for _ in 0..20 {
            predictor.update(0.005, 0.0);
        }
        let config = quantum_arena::config::QuantumConfig::new(13.0);

        let (scalp_tp, scalp_sl) = predictor.compute_dynamic_tp_sl(0.005, 0.0002, 0.0005, true, &config);
        let (swing_tp, swing_sl) = predictor.compute_dynamic_tp_sl(0.005, 0.0002, 0.0005, false, &config);

        assert!(swing_tp > scalp_tp, "Swing TP must be larger than Scalp TP");
        assert!(scalp_tp > 0.0 && scalp_sl > 0.0);
        assert!(swing_tp > 0.0 && swing_sl > 0.0);
    }

    #[test]
    fn test_conformal_predictor_nan_and_negative_immunity() {
        let mut predictor = ConformalPredictor::new(0.05, 100);
        predictor.update(f64::NAN, 0.0);
        assert_eq!(predictor.residual_history.len(), 0);

        let config = quantum_arena::config::QuantumConfig::new(13.0);
        let (tp, sl) = predictor.compute_dynamic_tp_sl(f64::NAN, f64::NAN, f64::NAN, true, &config);
        assert!(tp.is_finite() && sl.is_finite());
        assert!(tp > 0.0 && sl > 0.0);
    }
}


