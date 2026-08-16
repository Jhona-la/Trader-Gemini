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
        let residual = (actual_return - predicted_return).abs();
        if self.residual_history.len() >= self.max_history {
            self.residual_history.pop_front();
        }
        self.residual_history.push_back(residual);
    }

    /// Calcula el valor q-cuantil no paramétrico q_{1-\alpha}
    #[inline(always)]
    pub fn compute_conformal_quantile(&self) -> f64 {
        let n = self.residual_history.len();
        if n < 10 {
            return 0.0030; // Fallback inicial dinámico si no hay historial
        }

        let mut sorted: Vec<f64> = self.residual_history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Cuantil ajustado por muestra finita: (1 - alpha) * (1 + 1/n)
        let q_idx = (((1.0 - self.alpha) * (n as f64 + 1.0)).ceil() as usize).saturating_sub(1);
        let q_val = sorted[q_idx.min(n - 1)];
        q_val // Pure prediction sin forzar inflación
    }

    /// Retorna los objetivos dinámicos (TP, SL) garantizados probabilísticamente
    #[inline(always)]
    pub fn compute_dynamic_tp_sl(&self, atr_pct: f64, maker_fee: f64, taker_fee: f64, is_scalp: bool, config: &quantum_arena::config::QuantumConfig) -> (f64, f64) {
        let q = self.compute_conformal_quantile();
        let base_vol = atr_pct.max(q);

        let _roundtrip_fee = maker_fee + taker_fee;

        if is_scalp {
            use std::sync::atomic::Ordering;
            let tp_mult = config.tp_rr_ratio_btc.load(Ordering::Relaxed).max(1.0);
            let sl_mult = config.sl_atr_multiplier.load(Ordering::Relaxed).max(0.5);
            let raw_tp = base_vol * tp_mult;
            let raw_sl = base_vol * sl_mult;
            (raw_tp, raw_sl)
        } else {
            use std::sync::atomic::Ordering;
            let tp_mult = config.swing_trail_atr_mult_base.load(Ordering::Relaxed).max(2.0);
            let sl_mult = config.sl_atr_multiplier.load(Ordering::Relaxed).max(1.0);
            let raw_tp = base_vol * tp_mult;
            let raw_sl = base_vol * sl_mult;
            (raw_tp, raw_sl)
        }
    }
}

impl Default for ConformalPredictor {
    fn default() -> Self {
        Self::new(0.05, 200)
    }
}
